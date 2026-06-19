"""Training and validation helpers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from simple_transformer.checkpoint import CheckpointManager
from simple_transformer.config import TrainingConfig
from simple_transformer.metrics import (
    FitResult,
    MetricsAccumulator,
    TrainMetrics,
    TrainingObserver,
    correct_count,
    learning_rate,
    make_step_metrics,
    now,
    sync_if_cuda,
    token_count,
)
from simple_transformer.model import SimpleTransformerLM


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
    epoch: int = 1,
    global_step_start: int = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    on_step: Callable[[int, TrainMetrics], None] | None = None,
    observer: TrainingObserver | None = None,
) -> TrainMetrics:
    """Train for one epoch and return aggregate metrics."""

    model.train()
    accumulator = MetricsAccumulator()
    last_grad_norm: float | None = None
    data_start = now()

    for step, batch in enumerate(loader, start=1):
        data_time_s = now() - data_start
        compute_start = now()
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
        sync_if_cuda(config.device)

        tokens = token_count(labels)
        correct = correct_count(output["logits"], labels)
        loss_value = float(loss.item())
        compute_time_s = now() - compute_start
        accumulator.update(loss=loss_value, correct=correct, tokens=tokens)
        global_step = global_step_start + step

        if on_step is not None and step % config.log_every == 0:
            on_step(
                step,
                TrainMetrics(
                    loss=loss_value,
                    accuracy=correct / tokens,
                    learning_rate=learning_rate(optimizer),
                    grad_norm=last_grad_norm,
                ),
            )
        if observer is not None and step % config.log_every == 0:
            observer.log_step(
                make_step_metrics(
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    phase="train",
                    loss=loss_value,
                    correct=correct,
                    tokens=tokens,
                    examples=input_ids.size(0),
                    learning_rate=learning_rate(optimizer),
                    data_time_s=data_time_s,
                    compute_time_s=compute_time_s,
                    grad_norm=last_grad_norm,
                    device=config.device,
                )
            )
        data_start = now()

    return accumulator.to_metrics(
        learning_rate=learning_rate(optimizer),
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
    accumulator = MetricsAccumulator()

    for batch in loader:
        input_ids = batch["input_ids"].to(config.device, non_blocking=config.pin_memory)
        labels = batch["labels"].to(config.device, non_blocking=config.pin_memory)
        with _amp_context(config):
            output = model(input_ids, labels=labels)

        tokens = token_count(labels)
        accumulator.update(
            loss=float(output["loss"].item()),
            correct=correct_count(output["logits"], labels),
            tokens=tokens,
        )

    return accumulator.to_metrics(learning_rate=learning_rate)


def fit(
    model: SimpleTransformerLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    on_epoch: Callable[[int, TrainMetrics, TrainMetrics], None] | None = None,
    observer: TrainingObserver | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    resume_from: str | Path | None = None,
) -> FitResult:
    """Run a full training loop."""

    torch.manual_seed(config.seed)
    model = prepare_model(model, config)
    optimizer = make_optimizer(model, config) if optimizer is None else optimizer
    if scheduler is None:
        scheduler = make_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    train_history: list[TrainMetrics] = []
    val_history: list[TrainMetrics] = []
    global_step = 0
    start_epoch = 1

    if resume_from is not None:
        loader = checkpoint_manager or CheckpointManager()
        checkpoint = loader.load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=config.device,
        )
        train_history = checkpoint["train_history"]
        val_history = checkpoint["val_history"]
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            config=config,
            epoch=epoch,
            global_step_start=global_step,
            scheduler=scheduler,
            observer=observer,
        )
        global_step += len(train_loader)
        val_metrics = validate(
            model,
            val_loader,
            config=config,
            learning_rate=learning_rate(optimizer),
        )
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if on_epoch is not None:
            on_epoch(epoch, train_metrics, val_metrics)
        if observer is not None:
            observer.log_epoch(epoch, train_metrics, val_metrics)
        if checkpoint_manager is not None:
            checkpoint_manager.save_epoch(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_history=train_history,
                val_history=val_history,
                training_config=config,
                model_config=getattr(model, "config", None),
            )

    return FitResult(train=train_history, validation=val_history)


def _amp_context(config: TrainingConfig):
    if not config.use_amp or torch.device(config.device).type == "cpu":
        return nullcontext()
    device_type = torch.device(config.device).type
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
