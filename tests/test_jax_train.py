from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("optax")

from simple_transformer.config import TrainingConfig, TransformerConfig
from simple_transformer.data import ARITHMETIC_VOCAB, IGNORE_INDEX, make_train_val_loaders
from simple_transformer.jax.model import JaxSimpleTransformerLM
from simple_transformer.jax.train import (
    create_optimizer,
    fit_jax,
    masked_loss_and_accuracy,
    train_step,
)


def _model(seed=0):
    return JaxSimpleTransformerLM(
        TransformerConfig(
            vocab_size=len(ARITHMETIC_VOCAB),
            max_seq_len=11,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.1,
        ),
        rngs=nnx.Rngs(params=seed, dropout=seed + 1),
    )


def _parameter_arrays(model):
    return jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))


def test_masked_loss_and_accuracy_ignore_requested_labels():
    logits = jnp.array([[[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]])
    labels = jnp.array([[1, IGNORE_INDEX, 0]])

    loss, correct, tokens = masked_loss_and_accuracy(logits, labels)

    assert np.isfinite(loss)
    assert int(tokens) == 2
    assert int(correct) == 1


def test_jitted_train_step_is_finite_and_updates_parameters():
    config = TrainingConfig(
        max_digits=2, train_examples=8, val_examples=4, batch_size=4, epochs=1
    )
    model = _model()
    optimizer = create_optimizer(model, config, steps_per_epoch=2)
    before = [np.asarray(parameter).copy() for parameter in _parameter_arrays(model)]
    batch = {
        "input_ids": jnp.ones((4, 10), dtype=jnp.int32),
        "labels": jnp.ones((4, 10), dtype=jnp.int32),
    }

    metrics = train_step(model, optimizer, batch)

    assert np.isfinite(metrics["loss"])
    after = _parameter_arrays(model)
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))


def test_fit_jax_runs_one_tiny_cpu_epoch_and_is_reproducible():
    config = TrainingConfig(
        max_digits=2,
        train_examples=8,
        val_examples=4,
        batch_size=4,
        epochs=1,
        warmup_steps=1,
        seed=7,
    )
    train_loader, val_loader, _ = make_train_val_loaders(config)
    model1 = _model(seed=config.seed)
    before = [np.asarray(parameter).copy() for parameter in _parameter_arrays(model1)]

    optimizer1, result1 = fit_jax(model1, train_loader, val_loader, config)
    train_loader, val_loader, _ = make_train_val_loaders(config)
    model2 = _model(seed=config.seed)
    optimizer2, result2 = fit_jax(model2, train_loader, val_loader, config)

    assert result1.train[0].loss > 0
    assert result1.validation[0].loss > 0
    assert 0.0 <= result1.validation[0].accuracy <= 1.0
    assert int(optimizer1.step[...]) == len(train_loader)
    assert int(optimizer2.step[...]) == len(train_loader)
    assert any(
        not np.array_equal(first, second)
        for first, second in zip(before, _parameter_arrays(model1))
    )
    assert result1.train[0].loss == pytest.approx(result2.train[0].loss)
    for first, second in zip(_parameter_arrays(model1), _parameter_arrays(model2)):
        assert np.array_equal(first, second)
