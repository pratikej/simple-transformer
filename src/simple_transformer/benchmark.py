"""Inference benchmarking helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import statistics
import time
from typing import Any

import torch


@dataclass(frozen=True)
class LatencyMetrics:
    """Latency summary for repeated model inference calls."""

    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples_ms: tuple[float, ...]

    @property
    def throughput_per_second(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


def benchmark_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    repeats: int = 100,
    device: str | torch.device | None = None,
) -> LatencyMetrics:
    """
    Benchmark a zero-argument inference callable.

    Warmup calls are excluded from the reported samples. CUDA and MPS devices are
    synchronized around each timed call so asynchronous kernels do not make the
    latency look artificially low.
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    torch_device = None if device is None else torch.device(device)
    for _ in range(warmup):
        fn()
    _synchronize(torch_device)

    samples_ms = []
    for _ in range(repeats):
        _synchronize(torch_device)
        start = time.perf_counter()
        fn()
        _synchronize(torch_device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return summarize_latency(samples_ms)


def benchmark_forward_latency(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    warmup: int = 10,
    repeats: int = 100,
) -> LatencyMetrics:
    """Benchmark a model forward pass on a fixed input batch."""

    was_training = model.training
    model.eval()
    device = input_ids.device

    def forward() -> Any:
        infer = getattr(model, "infer", None)
        if infer is not None:
            return infer(input_ids, labels=labels)
        return model(input_ids, labels=labels)

    try:
        return benchmark_latency(
            forward,
            warmup=warmup,
            repeats=repeats,
            device=device,
        )
    finally:
        model.train(was_training)


def benchmark_generate_latency(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    warmup: int = 5,
    repeats: int = 50,
) -> LatencyMetrics:
    """Benchmark greedy generation latency on a fixed prompt batch."""

    was_training = model.training
    model.eval()
    device = input_ids.device

    def generate() -> Any:
        return model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    try:
        return benchmark_latency(
            generate,
            warmup=warmup,
            repeats=repeats,
            device=device,
        )
    finally:
        model.train(was_training)


def summarize_latency(samples_ms: list[float] | tuple[float, ...]) -> LatencyMetrics:
    """Summarize latency samples in milliseconds."""

    if not samples_ms:
        raise ValueError("samples_ms must be non-empty")

    samples = tuple(float(sample) for sample in samples_ms)
    sorted_samples = sorted(samples)
    return LatencyMetrics(
        mean_ms=statistics.fmean(samples),
        std_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        median_ms=statistics.median(samples),
        p95_ms=_percentile(sorted_samples, 0.95),
        min_ms=sorted_samples[0],
        max_ms=sorted_samples[-1],
        samples_ms=samples,
    )


def _percentile(sorted_samples: list[float], q: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    index = q * (len(sorted_samples) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_samples) - 1)
    weight = index - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight


def _synchronize(device: torch.device | None) -> None:
    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
