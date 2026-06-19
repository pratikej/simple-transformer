"""Training and validation helpers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch.utils.data import DataLoader

from simple_transformer.config import TrainingConfig
from simple_transformer.data import (
    IGNORE_INDEX,
    AdditionTokenizer,
    make_addition_dataloader,
)
from simple_transformer.model import SimpleTransformerLM


@dataclass(frozen=True)
class TrainMetrics:
    """Metrics collected for one train or validation pass."""

    loss: float
    accuracy: float
    learning_rate: float
    grad_norm: float | None = None


@dataclass(frozen=True)
class FitResult:
    """Per-epoch training history."""

    train: list[TrainMetrics]
    validation: list[TrainMetrics]


def make_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Create the default optimizer for this project."""

    decay_params = []
    nodecay_params = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_params.append(parameter)
        else:
            nodecay_params.append(parameter)

    fused = config.use_fused_optimizer and torch.device(config.device).type == "cuda"
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        fused=fused,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Create a linear-warmup cosine scheduler when requested."""

    max_steps = config.epochs * steps_per_epoch
    if max_steps < 1 or (config.warmup_steps == 0 and config.min_lr_ratio == 1.0):
        return None

    def lr_lambda(step: int) -> float:
        if config.warmup_steps > 0 and step < config.warmup_steps:
            return step / max(1, config.warmup_steps)

        progress = (step - config.warmup_steps) / max(1, max_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_train_val_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, AdditionTokenizer]:
    """Create train and validation loaders from a training config."""

    train_loader, tokenizer = make_addition_dataloader(
        num_examples=config.train_examples,
        max_digits=config.max_digits,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
        pin_memory=config.pin_memory,
    )
    val_loader, _ = make_addition_dataloader(
        num_examples=config.val_examples,
        max_digits=config.max_digits,
        batch_size=config.batch_size,
        seed=config.seed + 1,
        shuffle=False,
        pin_memory=config.pin_memory,
    )
    return train_loader, val_loader, tokenizer


def prepare_model(
    model: SimpleTransformerLM,
    config: TrainingConfig,
) -> SimpleTransformerLM:
    """Move the model to its device and optionally compile it."""

    model = model.to(config.device)
    if config.compile_model:
        model = torch.compile(model)
    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    config: TrainingConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    on_step: Callable[[int, TrainMetrics], None] | None = None,
) -> TrainMetrics:
    """Train for one epoch and return aggregate metrics."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    last_grad_norm: float | None = None

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(config.device, non_blocking=config.pin_memory)
        labels = batch["labels"].to(config.device, non_blocking=config.pin_memory)

        optimizer.zero_grad(set_to_none=True)
        with _amp_context(config):
            output = model(input_ids, labels=labels)
            loss = output["loss"]

        loss.backward()
        if config.grad_clip_norm is not None:
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip_norm,
                )
            )

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        token_count = metrics_token_count(labels)
        total_loss += float(loss.item()) * token_count
        total_correct += _correct_count(output["logits"], labels)
        total_tokens += token_count

        if on_step is not None and step % config.log_every == 0:
            on_step(
                step,
                TrainMetrics(
                    loss=float(loss.item()),
                    accuracy=_correct_count(output["logits"], labels) / token_count,
                    learning_rate=_learning_rate(optimizer),
                    grad_norm=last_grad_norm,
                ),
            )

    return TrainMetrics(
        loss=total_loss / total_tokens,
        accuracy=total_correct / total_tokens,
        learning_rate=_learning_rate(optimizer),
        grad_norm=last_grad_norm,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    config: TrainingConfig,
    learning_rate: float = 0.0,
) -> TrainMetrics:
    """Evaluate on a validation loader."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(config.device, non_blocking=config.pin_memory)
        labels = batch["labels"].to(config.device, non_blocking=config.pin_memory)
        with _amp_context(config):
            output = model(input_ids, labels=labels)

        token_count = metrics_token_count(labels)
        total_loss += float(output["loss"].item()) * token_count
        total_correct += _correct_count(output["logits"], labels)
        total_tokens += token_count

    return TrainMetrics(
        loss=total_loss / total_tokens,
        accuracy=total_correct / total_tokens,
        learning_rate=learning_rate,
    )


def fit(
    model: SimpleTransformerLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    on_epoch: Callable[[int, TrainMetrics, TrainMetrics], None] | None = None,
) -> FitResult:
    """Run a full training loop."""

    torch.manual_seed(config.seed)
    model = prepare_model(model, config)
    optimizer = make_optimizer(model, config) if optimizer is None else optimizer
    if scheduler is None:
        scheduler = make_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    train_history: list[TrainMetrics] = []
    val_history: list[TrainMetrics] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            config=config,
            scheduler=scheduler,
        )
        val_metrics = validate(
            model,
            val_loader,
            config=config,
            learning_rate=_learning_rate(optimizer),
        )
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if on_epoch is not None:
            on_epoch(epoch, train_metrics, val_metrics)

    return FitResult(train=train_history, validation=val_history)


def metrics_token_count(labels: torch.Tensor) -> int:
    """Count non-padding target tokens."""

    return int((labels != IGNORE_INDEX).sum().item())


def _correct_count(logits: torch.Tensor, labels: torch.Tensor) -> int:
    predictions = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX
    return int((predictions[mask] == labels[mask]).sum().item())


def _learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _amp_context(config: TrainingConfig):
    if not config.use_amp or torch.device(config.device).type == "cpu":
        return nullcontext()
    device_type = torch.device(config.device).type
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
