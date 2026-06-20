"""Training metrics and TensorBoard logging helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from simple_transformer.data import IGNORE_INDEX


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


@dataclass(frozen=True)
class StepMetrics:
    """Detailed metrics collected for one training step."""

    epoch: int
    step: int
    global_step: int
    phase: str
    loss: float
    accuracy: float
    learning_rate: float
    tokens: int
    examples: int
    data_time_s: float
    compute_time_s: float
    step_time_s: float
    grad_norm: float | None = None
    cpu_percent: float | None = None
    memory_percent: float | None = None
    process_rss_mb: float | None = None
    gpu_allocated_mb: float | None = None
    gpu_reserved_mb: float | None = None
    gpu_max_allocated_mb: float | None = None
    gpu_utilization_percent: float | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.step_time_s if self.step_time_s > 0 else 0.0

    @property
    def examples_per_second(self) -> float:
        return self.examples / self.step_time_s if self.step_time_s > 0 else 0.0


@dataclass
class MetricsAccumulator:
    """Accumulate token-weighted loss and accuracy over a pass."""

    total_loss: float = 0.0
    total_correct: int = 0
    total_tokens: int = 0

    def update(self, *, loss: float, correct: int, tokens: int) -> None:
        self.total_loss += loss * tokens
        self.total_correct += correct
        self.total_tokens += tokens

    def to_metrics(
        self,
        *,
        learning_rate: float,
        grad_norm: float | None = None,
    ) -> TrainMetrics:
        return TrainMetrics(
            loss=self.total_loss / self.total_tokens,
            accuracy=self.total_correct / self.total_tokens,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
        )


class TrainingObserver:
    """No-op base observer used by the training loop."""

    def log_step(self, metrics: StepMetrics) -> None:
        pass

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Any,
        validation_metrics: Any,
    ) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardTrainingObserver(TrainingObserver):
    """Write training, validation, system, and throughput metrics to TensorBoard."""

    def __init__(
        self,
        log_dir: str | Path = "runs/local-arithmetic",
        *,
        flush_secs: int = 10,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def log_config(
        self,
        *,
        training_config: Any | None = None,
        model_config: Any | None = None,
        parameter_count: int | None = None,
    ) -> None:
        lines = []
        if training_config is not None:
            lines.append(_markdown_config("training", training_config))
        if model_config is not None:
            lines.append(_markdown_config("model", model_config))
        if parameter_count is not None:
            lines.append(f"\nparameter_count: {parameter_count:,}\n")
        if lines:
            self.writer.add_text("config", "\n".join(lines), global_step=0)

    def log_step(self, metrics: StepMetrics) -> None:
        prefix = metrics.phase
        step = metrics.global_step
        self.writer.add_scalar(f"{prefix}/loss", metrics.loss, step)
        self.writer.add_scalar(f"{prefix}/accuracy", metrics.accuracy, step)
        self.writer.add_scalar(f"{prefix}/learning_rate", metrics.learning_rate, step)
        self.writer.add_scalar(f"{prefix}/tokens_per_second", metrics.tokens_per_second, step)
        self.writer.add_scalar(
            f"{prefix}/examples_per_second",
            metrics.examples_per_second,
            step,
        )
        self.writer.add_scalar(f"{prefix}/data_time_ms", metrics.data_time_s * 1000, step)
        self.writer.add_scalar(
            f"{prefix}/compute_time_ms",
            metrics.compute_time_s * 1000,
            step,
        )
        self.writer.add_scalar(f"{prefix}/step_time_ms", metrics.step_time_s * 1000, step)
        if metrics.grad_norm is not None:
            self.writer.add_scalar(f"{prefix}/grad_norm", metrics.grad_norm, step)

        for name in (
            "cpu_percent",
            "memory_percent",
            "process_rss_mb",
            "gpu_allocated_mb",
            "gpu_reserved_mb",
            "gpu_max_allocated_mb",
            "gpu_utilization_percent",
        ):
            value = getattr(metrics, name)
            if value is not None:
                self.writer.add_scalar(f"system/{name}", value, step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Any,
        validation_metrics: Any,
    ) -> None:
        self.writer.add_scalar("epoch/train_loss", train_metrics.loss, epoch)
        self.writer.add_scalar("epoch/train_accuracy", train_metrics.accuracy, epoch)
        self.writer.add_scalar("epoch/validation_loss", validation_metrics.loss, epoch)
        self.writer.add_scalar(
            "epoch/validation_accuracy",
            validation_metrics.accuracy,
            epoch,
        )
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def sample_system_metrics(device: str | torch.device) -> dict[str, float | None]:
    """Collect lightweight CPU, memory, and CUDA memory counters."""

    process = psutil.Process()
    metrics: dict[str, float | None] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "process_rss_mb": process.memory_info().rss / (1024**2),
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
        "gpu_max_allocated_mb": None,
        "gpu_utilization_percent": None,
    }

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        index = torch_device.index
        metrics["gpu_allocated_mb"] = torch.cuda.memory_allocated(index) / (1024**2)
        metrics["gpu_reserved_mb"] = torch.cuda.memory_reserved(index) / (1024**2)
        metrics["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated(index) / (1024**2)
        utilization = getattr(torch.cuda, "utilization", None)
        if utilization is not None:
            try:
                metrics["gpu_utilization_percent"] = float(utilization(index))
            except Exception:
                metrics["gpu_utilization_percent"] = None

    return metrics


def make_step_metrics(
    *,
    epoch: int,
    step: int,
    global_step: int,
    phase: str,
    loss: float,
    correct: int,
    tokens: int,
    examples: int,
    learning_rate: float,
    data_time_s: float,
    compute_time_s: float,
    grad_norm: float | None,
    device: str | torch.device,
) -> StepMetrics:
    """Create a detailed per-step metrics record."""

    return StepMetrics(
        epoch=epoch,
        step=step,
        global_step=global_step,
        phase=phase,
        loss=loss,
        accuracy=correct / tokens,
        learning_rate=learning_rate,
        tokens=tokens,
        examples=examples,
        data_time_s=data_time_s,
        compute_time_s=compute_time_s,
        step_time_s=data_time_s + compute_time_s,
        grad_norm=grad_norm,
        **sample_system_metrics(device),
    )


def token_count(labels: torch.Tensor) -> int:
    """Count non-padding target tokens."""

    return int((labels != IGNORE_INDEX).sum().item())


def correct_count(logits: torch.Tensor, labels: torch.Tensor) -> int:
    predictions = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX
    return int((predictions[mask] == labels[mask]).sum().item())


def learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def sync_if_cuda(device: str | torch.device) -> None:
    """Synchronize CUDA so timings reflect completed GPU work."""

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def now() -> float:
    return time.perf_counter()


def _markdown_config(title: str, config: Any) -> str:
    values = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
    rows = "\n".join(f"- `{key}`: `{value}`" for key, value in values.items())
    return f"## {title}\n{rows}"
