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


def small_model_config(
    *,
    max_digits: int = 3,
    device: str = "cpu",
) -> TransformerConfig:
    """Create a roughly 1M parameter config for experiments."""

    return TransformerConfig(
        vocab_size=len(ADDITION_VOCAB),
        max_seq_len=max_addition_text_length(max_digits),
        d_model=192,
        n_layers=5,
        n_heads=6,
        d_ff=448,
        dropout=0.0,
        bias=False,
        force_flash=device.startswith("cuda"),
        tie_embeddings=True,
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a simple local training run."""

    max_digits: int = 3
    train_examples: int = 2048
    val_examples: int = 512
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    grad_clip_norm: float | None = 1.0
    seed: int = 42
    device: str = "cpu"
    use_amp: bool = False
    compile_model: bool = False
    use_fused_optimizer: bool = False
    pin_memory: bool = False
    log_every: int = 20

    def __post_init__(self) -> None:
        if self.max_digits < 1:
            raise ValueError("max_digits must be at least 1")
        if self.train_examples < 1:
            raise ValueError("train_examples must be positive")
        if self.val_examples < 1:
            raise ValueError("val_examples must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0.0, 1.0]")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when set")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")


def local_training_config(
    *,
    max_digits: int = 3,
    device: str = "cpu",
) -> TrainingConfig:
    """Small training config meant to run quickly on a laptop CPU."""

    use_cuda = device.startswith("cuda")
    return TrainingConfig(
        max_digits=max_digits,
        train_examples=16_384,
        val_examples=512,
        batch_size=64,
        epochs=20,
        learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=10,
        device=device,
        use_amp=use_cuda,
        compile_model=False,
        use_fused_optimizer=use_cuda,
        pin_memory=use_cuda,
    )
