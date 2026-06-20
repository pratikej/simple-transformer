"""Checkpoint helpers for resumable training."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from simple_transformer.metrics import TrainMetrics


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for epoch checkpoints."""

    checkpoint_dir: str | Path = "checkpoints/local-arithmetic"
    keep_last: int = 3
    async_save: bool = True

    def __post_init__(self) -> None:
        if self.keep_last < 1:
            raise ValueError("keep_last must be at least 1")


class CheckpointManager:
    """Save and load epoch checkpoints."""

    def __init__(self, config: CheckpointConfig | None = None) -> None:
        self.config = CheckpointConfig() if config is None else config
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self._executor: ThreadPoolExecutor | None = None
        self._pending: Future | None = None
        if self.config.async_save:
            self._executor = ThreadPoolExecutor(max_workers=1)

    def save_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_history: list[TrainMetrics],
        val_history: list[TrainMetrics],
        training_config: Any | None = None,
        model_config: Any | None = None,
    ) -> Path:
        """Save one epoch checkpoint and prune older checkpoints."""

        self._finish_pending()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"epoch-{epoch:04d}.pt"
        payload = _to_cpu(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "train_history": [_metric_to_dict(metric) for metric in train_history],
                "val_history": [_metric_to_dict(metric) for metric in val_history],
                "training_config": _config_to_dict(training_config),
                "model_config": _config_to_dict(model_config),
            }
        )

        if self._executor is None:
            self._save_and_prune(path, payload)
        else:
            self._pending = self._executor.submit(self._save_and_prune, path, payload)

        return path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load checkpoint state into the provided training objects."""

        checkpoint = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        checkpoint["train_history"] = [
            TrainMetrics(**metric) for metric in checkpoint.get("train_history", [])
        ]
        checkpoint["val_history"] = [
            TrainMetrics(**metric) for metric in checkpoint.get("val_history", [])
        ]
        return checkpoint

    def latest_checkpoint(self) -> Path | None:
        """Return the newest checkpoint path, if any."""

        checkpoints = sorted(self.checkpoint_dir.glob("epoch-*.pt"))
        return checkpoints[-1] if checkpoints else None

    def close(self) -> None:
        """Wait for pending async saves and release worker resources."""

        self._finish_pending()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _save_and_prune(self, path: Path, payload: dict[str, Any]) -> None:
        torch.save(payload, path)
        checkpoints = sorted(self.checkpoint_dir.glob("epoch-*.pt"))
        for old_path in checkpoints[: -self.config.keep_last]:
            old_path.unlink()

    def _finish_pending(self) -> None:
        if self._pending is not None:
            self._pending.result()
            self._pending = None


def _metric_to_dict(metric: TrainMetrics) -> dict[str, Any]:
    return asdict(metric)


def _config_to_dict(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return None


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value
