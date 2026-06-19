"""Model configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass

from simple_transformer.data import ADDITION_VOCAB, max_addition_text_length


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for a small decoder-only transformer."""

    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10000.0
    force_flash: bool = False
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.n_heads < 1:
            raise ValueError("n_heads must be positive")
        if self.d_ff < 1:
            raise ValueError("d_ff must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")


def small_addition_config(max_digits: int = 3) -> TransformerConfig:
    """Create a roughly 1M parameter config for addition experiments."""

    return TransformerConfig(
        vocab_size=len(ADDITION_VOCAB),
        max_seq_len=max_addition_text_length(max_digits),
        d_model=128,
        n_layers=5,
        n_heads=4,
        d_ff=384,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
    )
