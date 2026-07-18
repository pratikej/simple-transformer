import torch
from pytest import approx

from simple_transformer.benchmark import (
    benchmark_forward_latency,
    benchmark_latency,
    summarize_latency,
)
from simple_transformer.config import TransformerConfig
from simple_transformer.data import ARITHMETIC_VOCAB
from simple_transformer.model import SimpleTransformerLM


def test_summarize_latency_reports_distribution():
    metrics = summarize_latency([1.0, 2.0, 3.0, 4.0])

    assert metrics.mean_ms == 2.5
    assert round(metrics.std_ms, 6) == 1.290994
    assert metrics.median_ms == 2.5
    assert metrics.p95_ms == approx(3.85)
    assert metrics.min_ms == 1.0
    assert metrics.max_ms == 4.0
    assert metrics.throughput_per_second == 400.0


def test_benchmark_latency_runs_warmup_and_repeats():
    calls = 0
    inference_mode_states = []

    def fn():
        nonlocal calls
        calls += 1
        inference_mode_states.append(torch.is_inference_mode_enabled())

    metrics = benchmark_latency(fn, warmup=2, repeats=3)

    assert calls == 5
    assert inference_mode_states == [False, False, False, False, False]
    assert len(metrics.samples_ms) == 3
    assert metrics.mean_ms >= 0.0


def test_benchmark_forward_latency_restores_training_mode():
    model = SimpleTransformerLM(
        TransformerConfig(
            vocab_size=len(ARITHMETIC_VOCAB),
            max_seq_len=11,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
        )
    )
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    model.train()

    metrics = benchmark_forward_latency(model, input_ids, warmup=1, repeats=2)

    assert model.training is True
    assert len(metrics.samples_ms) == 2
    assert metrics.mean_ms > 0.0
