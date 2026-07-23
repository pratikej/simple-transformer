"""Optax training and validation helpers for the Flax NNX model."""

from __future__ import annotations

from collections.abc import Callable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader

from simple_transformer.config import TrainingConfig
from simple_transformer.data import IGNORE_INDEX
from simple_transformer.jax.model import JaxSimpleTransformerLM
from simple_transformer.metrics import FitResult, TrainMetrics


Array = jax.Array
Batch = dict[str, Array]


def masked_loss_and_accuracy(
    logits: Array, labels: Array
) -> tuple[Array, Array, Array]:
    """Return mean cross-entropy, correct count, and active-token count."""

    mask = labels != IGNORE_INDEX
    safe_labels = jnp.where(mask, labels, 0)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
    token_count = jnp.sum(mask)
    denominator = jnp.maximum(token_count, 1)
    loss = jnp.sum(jnp.where(mask, token_losses, 0.0)) / denominator
    predictions = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(jnp.logical_and(mask, predictions == safe_labels))
    return loss, correct, token_count


def make_learning_rate_schedule(
    config: TrainingConfig,
    steps_per_epoch: int,
) -> optax.Schedule:
    max_steps = config.epochs * steps_per_epoch

    if config.warmup_steps > max_steps:
        raise ValueError("warmup_steps cannot exceed total training steps")

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=max_steps,
        end_value=config.learning_rate * config.min_lr_ratio,
    )


def create_optimizer(
    model: JaxSimpleTransformerLM,
    config: TrainingConfig,
    *,
    steps_per_epoch: int,
) -> nnx.Optimizer:
    """Create the default Optax optimizer over the model's NNX parameters."""

    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    schedule = make_learning_rate_schedule(config, steps_per_epoch)

    def decay_mask(params):
        # Build this from the raw parameter pytree Optax supplies. Precomputing
        # it from ``nnx.state`` would retain NNX Param wrappers around booleans.
        return jax.tree.map(lambda parameter: parameter.ndim >= 2, params)

    transformations: list[optax.GradientTransformation] = []
    if config.grad_clip_norm is not None:
        transformations.append(optax.clip_by_global_norm(config.grad_clip_norm))
    transformations.append(
        optax.adamw(
            learning_rate=schedule,
            b1=config.betas[0],
            b2=config.betas[1],
            weight_decay=config.weight_decay,
            mask=decay_mask,
        )
    )
    return nnx.Optimizer(model, optax.chain(*transformations), wrt=nnx.Param)


@nnx.jit
def train_step(
    model: JaxSimpleTransformerLM,
    optimizer: nnx.Optimizer,
    batch: Batch,
) -> dict[str, Array]:
    """Run one compiled optimizer step, mutating model and optimizer state."""

    def loss_fn(model: JaxSimpleTransformerLM) -> tuple[Array, tuple[Array, Array]]:
        logits = model(batch["input_ids"])
        loss, correct, tokens = masked_loss_and_accuracy(logits, batch["labels"])
        return loss, (correct, tokens)

    (loss, (correct, tokens)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    )
    optimizer.update(model, grads)
    return {
        "loss": loss,
        "correct": correct,
        "tokens": tokens,
        "grad_norm": grad_norm,
    }


@nnx.jit
def validation_step(
    model: JaxSimpleTransformerLM,
    batch: Batch,
) -> dict[str, Array]:
    """Run one compiled deterministic validation step."""

    logits = model(batch["input_ids"])
    loss, correct, tokens = masked_loss_and_accuracy(logits, batch["labels"])
    return {"loss": loss, "correct": correct, "tokens": tokens}


def fit_jax(
    model: JaxSimpleTransformerLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    *,
    optimizer: nnx.Optimizer | None = None,
    on_epoch: Callable[[int, TrainMetrics, TrainMetrics], None] | None = None,
) -> tuple[nnx.Optimizer, FitResult]:
    """Train a mutable NNX model using the existing Torch DataLoaders."""

    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1 or len(val_loader) < 1:
        raise ValueError("train and validation loaders must be non-empty")
    if optimizer is None:
        optimizer = create_optimizer(model, config, steps_per_epoch=steps_per_epoch)
    schedule = make_learning_rate_schedule(config, steps_per_epoch)
    # RandomSampler uses Torch's global generator unless one is supplied to the
    # DataLoader. Seed it here so the shared input pipeline is reproducible too.
    torch.manual_seed(config.seed)
    train_history: list[TrainMetrics] = []
    val_history: list[TrainMetrics] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_tokens = 0
        last_grad_norm: float | None = None
        for torch_batch in train_loader:
            metrics = train_step(model, optimizer, _jax_batch(torch_batch))
            loss, correct, tokens, grad_norm = jax.device_get(
                (
                    metrics["loss"],
                    metrics["correct"],
                    metrics["tokens"],
                    metrics["grad_norm"],
                )
            )
            token_value = int(tokens)
            train_loss_sum += float(loss) * token_value
            train_correct += int(correct)
            train_tokens += token_value
            last_grad_norm = float(grad_norm)

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        validation_tokens = 0
        for torch_batch in val_loader:
            metrics = validation_step(model, _jax_batch(torch_batch))
            loss, correct, tokens = jax.device_get(
                (metrics["loss"], metrics["correct"], metrics["tokens"])
            )
            token_value = int(tokens)
            validation_loss_sum += float(loss) * token_value
            validation_correct += int(correct)
            validation_tokens += token_value

        if train_tokens == 0 or validation_tokens == 0:
            raise ValueError(
                "loaders must contain at least one non-ignored target token"
            )
        step = jnp.maximum(optimizer.step[...] - 1, 0)
        learning_rate = float(jax.device_get(schedule(step)))
        train_metrics = TrainMetrics(
            loss=train_loss_sum / train_tokens,
            accuracy=train_correct / train_tokens,
            learning_rate=learning_rate,
            grad_norm=last_grad_norm,
        )
        val_metrics = TrainMetrics(
            loss=validation_loss_sum / validation_tokens,
            accuracy=validation_correct / validation_tokens,
            learning_rate=learning_rate,
        )
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        if on_epoch is not None:
            on_epoch(epoch, train_metrics, val_metrics)

    return optimizer, FitResult(train=train_history, validation=val_history)


def _jax_batch(torch_batch: dict[str, object]) -> Batch:
    """Convert a CPU Torch batch to NumPy at the JAX backend boundary."""

    return {
        "input_ids": np.asarray(torch_batch["input_ids"].numpy(), dtype=np.int32),
        "labels": np.asarray(torch_batch["labels"].numpy(), dtype=np.int32),
    }
