"""Flax NNX implementation of the project's decoder-only transformer."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from simple_transformer.config import TransformerConfig


Array = jax.Array
KERNEL_INIT = jax.nn.initializers.normal(stddev=0.02)


def _apply_rope(x: Array, positions: Array, base: float) -> Array:
    """Apply rotary embeddings to ``[batch, heads, sequence, head_dim]``."""

    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    frequencies = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos = jnp.cos(frequencies).astype(x.dtype)[None, None, :, :]
    sin = jnp.sin(frequencies).astype(x.dtype)[None, None, :, :]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1)
    return rotated.reshape(x.shape)


def _linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool,
    rngs: nnx.Rngs,
) -> nnx.Linear:
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=use_bias,
        kernel_init=KERNEL_INIT,
        bias_init=jax.nn.initializers.zeros,
        rngs=rngs,
    )


class SwiGLU(nnx.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs) -> None:
        self.gate = _linear(
            config.d_model, config.d_ff, use_bias=config.bias, rngs=rngs
        )
        self.value = _linear(
            config.d_model, config.d_ff, use_bias=config.bias, rngs=rngs
        )
        self.out = _linear(config.d_ff, config.d_model, use_bias=config.bias, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = jax.nn.silu(self.gate(x)) * self.value(x)
        return self.dropout(self.out(x))


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention without an inference KV cache."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        self.qkv = _linear(
            config.d_model, 3 * config.d_model, use_bias=config.bias, rngs=rngs
        )
        self.out = _linear(
            config.d_model, config.d_model, use_bias=config.bias, rngs=rngs
        )
        self.attention_dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.residual_dropout = nnx.Dropout(config.dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        batch_size, seq_len, channels = x.shape
        head_dim = self.config.d_model // self.config.n_heads
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)

        def split_heads(tensor: Array) -> Array:
            return tensor.reshape(
                batch_size, seq_len, self.config.n_heads, head_dim
            ).transpose(0, 2, 1, 3)

        q, k, v = map(split_heads, (q, k, v))
        positions = jnp.arange(seq_len)
        q = _apply_rope(q, positions, self.config.rope_base)
        k = _apply_rope(k, positions, self.config.rope_base)

        scale = jnp.asarray(head_dim, dtype=q.dtype) ** -0.5
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -jnp.inf)
        weights = self.attention_dropout(jax.nn.softmax(scores, axis=-1))
        output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, channels)
        return self.residual_dropout(self.out(output))


class TransformerBlock(nnx.Module):
    """One pre-norm transformer block."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.attention = CausalSelfAttention(config, rngs=rngs)
        self.norm2 = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.mlp = SwiGLU(config, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class JaxSimpleTransformerLM(nnx.Module):
    """Decoder-only NNX language model for character-level prediction."""

    def __init__(self, config: TransformerConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        self.token_embedding = nnx.Embed(
            config.vocab_size,
            config.d_model,
            embedding_init=KERNEL_INIT,
            rngs=rngs,
        )
        self.embedding_dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.blocks = nnx.List(
            [TransformerBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        )
        self.norm = nnx.RMSNorm(config.d_model, rngs=rngs)
        if not config.tie_embeddings:
            self.lm_head = _linear(
                config.d_model,
                config.vocab_size,
                use_bias=False,
                rngs=rngs,
            )

    def __call__(self, input_ids: Array) -> Array:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input_ids length exceeds config.max_seq_len")

        x = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.config.tie_embeddings:
            return jnp.einsum("btd,vd->btv", x, self.token_embedding.embedding)
        return self.lm_head(x)


def count_parameters(model: nnx.Module) -> int:
    params = nnx.state(model, nnx.Param)
    return sum(x.size for x in jax.tree.leaves(params))
