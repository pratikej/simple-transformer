"""A compact decoder-only transformer language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from simple_transformer.config import TransformerConfig
from simple_transformer.data import IGNORE_INDEX


@dataclass
class StaticKVCache:
    """
    Fixed-size key/value cache for one transformer layer.

    k/v shape: [batch, n_heads, max_seq_len, head_dim]

    pos: Next write position shared by the batch.
    """

    k: torch.Tensor
    v: torch.Tensor
    pos: int = 0

    def reset(self) -> None:
        self.pos = 0


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for attention queries and keys."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        positions shape: [seq_len]
        inv_freq shape: [head_dim / 2]
        """
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = freqs.cos().to(dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype)[None, None, :, :]
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch, n_heads, seq_len, head_dim]
        """
        cos, sin = self._cos_sin(positions, x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError("RoPE positions exceed max_seq_len")
        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=q.device,
        )
        return self._apply_rope(q, positions), self._apply_rope(k, positions)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.value = nn.Linear(d_model, d_ff, bias=bias)
        self.out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.value(x)
        return self.dropout(self.out(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional KV caching."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout
        self.force_flash = config.force_flash

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

    def new_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> StaticKVCache:
        cache_shape = (batch_size, self.n_heads, self.max_seq_len, self.head_dim)
        return StaticKVCache(
            k=torch.empty(cache_shape, device=device, dtype=dtype),
            v=torch.empty(cache_shape, device=device, dtype=dtype),
        )

    def _offset_causal_mask(
        self,
        q_len: int,
        k_len: int,
        start_pos: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_positions = torch.arange(
            start_pos,
            start_pos + q_len,
            device=device,
        )[:, None]
        k_positions = torch.arange(k_len, device=device)[None, :]
        return k_positions <= q_positions

    def forward(
        self,
        x: torch.Tensor,
        cache: StaticKVCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        q, k_new, v_new = self.qkv(x).chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = k_new.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
            1,
            2,
        )
        v_new = v_new.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
            1,
            2,
        )

        if cache is None:
            q, k = self.rope(q, k_new, start_pos=0)
            v = v_new
            attn_mask = None
            is_causal = True
        else:
            start_pos = cache.pos
            end_pos = start_pos + seq_len
            if end_pos > self.max_seq_len:
                raise ValueError("KV cache exceeded max_seq_len")

            q, k_new = self.rope(q, k_new, start_pos=start_pos)
            cache.k[:, :, start_pos:end_pos, :] = k_new
            cache.v[:, :, start_pos:end_pos, :] = v_new
            cache.pos = end_pos

            k = cache.k[:, :, :end_pos, :]
            v = cache.v[:, :, :end_pos, :]
            attn_mask = None
            is_causal = seq_len > 1 and start_pos == 0
            if seq_len > 1 and start_pos > 0:
                attn_mask = self._offset_causal_mask(
                    q_len=seq_len,
                    k_len=end_pos,
                    start_pos=start_pos,
                    device=x.device,
                )

        dropout_p = self.dropout if self.training else 0.0

        def run_sdpa():
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        if self.force_flash:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                y = run_sdpa()
        else:
            y = run_sdpa()
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.out(y))


class TransformerBlock(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.mlp = SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            bias=config.bias,
        )

    def new_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> StaticKVCache:
        return self.attn.new_cache(batch_size, device, dtype)

    def forward(
        self,
        x: torch.Tensor,
        cache: StaticKVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cache=cache)
        return x + self.mlp(self.norm2(x))


class SimpleTransformerLM(nn.Module):
    """Decoder-only transformer for character-level next-token prediction."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[StaticKVCache]:
        param = next(self.parameters())
        device = param.device if device is None else device
        dtype = param.dtype if dtype is None else dtype
        return [
            block.new_cache(batch_size=batch_size, device=device, dtype=dtype)
            for block in self.blocks
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cache: list[StaticKVCache] | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input_ids length exceeds config.max_seq_len")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("cache must have one entry per transformer block")

        x = self.dropout(self.token_embedding(input_ids))
        for index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[index]
            x = block(x, cache=layer_cache)

        logits = self.lm_head(self.norm(x))
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return output

    @torch.inference_mode()
    def infer(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cache: list[StaticKVCache] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass with autograd and version tracking disabled."""

        return self(input_ids, labels=labels, cache=cache)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids

        cache = self.new_cache(
            batch_size=input_ids.size(0),
            device=input_ids.device,
        )
        logits = self(input_ids, cache=cache)["logits"][:, -1, :]
        finished = torch.zeros(
            input_ids.size(0), dtype=torch.bool, device=input_ids.device
        )

        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.config.max_seq_len:
                raise ValueError("generated sequence exceeded config.max_seq_len")

            # Greedy decoding: always pick the highest-logit next token.
            next_token = logits.argmax(dim=-1, keepdim=True)
            if eos_token_id is not None:
                eos_tokens = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished[:, None], eos_tokens, next_token)
                finished |= next_token.squeeze(1) == eos_token_id

            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(finished):
                break
            if input_ids.size(1) < self.config.max_seq_len:
                logits = self(next_token, cache=cache)["logits"][:, -1, :]

        return input_ids

    @torch.inference_mode()
    def generate_batch(
        self,
        input_ids: list[list[int]] | list[torch.Tensor],
        eos_token_id: int | None = None,
    ) -> list[torch.Tensor]:
        """Generate for variable-length prompts by batching same-length prompts."""

        param = next(self.parameters())
        buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for index, prompt in enumerate(input_ids):
            prompt_ids = torch.as_tensor(prompt, dtype=torch.long, device=param.device)
            if prompt_ids.ndim != 1:
                raise ValueError("each prompt must be a 1D sequence of token ids")
            if prompt_ids.numel() == 0:
                raise ValueError("prompts must be non-empty")
            if prompt_ids.numel() > self.config.max_seq_len:
                raise ValueError("prompt length exceeds config.max_seq_len")

            buckets.setdefault(prompt_ids.numel(), []).append((index, prompt_ids))

        outputs: dict[int, torch.Tensor] = {}
        for bucket in buckets.values():
            bucket_indices = [index for index, _ in bucket]
            bucket_input_ids = torch.stack([prompt for _, prompt in bucket])
            bucket_max_new_tokens = self.config.max_seq_len - bucket_input_ids.size(1)
            bucket_output_ids = self.generate(
                bucket_input_ids,
                max_new_tokens=bucket_max_new_tokens,
                eos_token_id=eos_token_id,
            )
            for index, output_ids in zip(bucket_indices, bucket_output_ids):
                outputs[index] = output_ids

        return [outputs[index] for index in range(len(input_ids))]


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
