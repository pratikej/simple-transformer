from flax import nnx
import jax.numpy as jnp
import pytest

pytest.importorskip("flax")
pytest.importorskip("optax")

from simple_transformer.config import TransformerConfig
from simple_transformer.data import ARITHMETIC_VOCAB
from simple_transformer.jax.model import JaxSimpleTransformerLM


def _config(*, tie_embeddings=True, dropout=0.0):
    return TransformerConfig(
        vocab_size=len(ARITHMETIC_VOCAB),
        max_seq_len=11,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        tie_embeddings=tie_embeddings,
        dropout=dropout,
    )


def _model(*, tie_embeddings=True, dropout=0.0, seed=0):
    return JaxSimpleTransformerLM(
        _config(tie_embeddings=tie_embeddings, dropout=dropout),
        rngs=nnx.Rngs(params=seed, dropout=seed + 1),
    )


def test_nnx_model_initializes_eagerly_and_has_expected_output_shape():
    model = _model()

    logits = model(jnp.ones((2, 10), dtype=jnp.int32))

    assert logits.shape == (2, 10, len(ARITHMETIC_VOCAB))
    assert not hasattr(model, "lm_head")
    assert model.token_embedding.embedding.shape == (
        len(ARITHMETIC_VOCAB),
        model.config.d_model,
    )


def test_untied_nnx_model_has_output_kernel():
    model = _model(tie_embeddings=False)

    assert model.lm_head.kernel.shape == (
        model.config.d_model,
        len(ARITHMETIC_VOCAB),
    )


def test_nnx_attention_is_causal():
    model = _model()
    first = jnp.array([[2, 3, 4, 5]], dtype=jnp.int32)
    second = first.at[0, 3].set(6)

    first_logits = model(first)
    second_logits = model(second)

    assert jnp.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)


def test_nnx_train_and_eval_control_dropout():
    model = _model(dropout=0.5)
    inputs = jnp.ones((1, 4), dtype=jnp.int32)

    model.train()
    first_train = model(inputs)
    second_train = model(inputs)
    model.eval()
    first_eval = model(inputs)
    second_eval = model(inputs)

    assert not jnp.array_equal(first_train, second_train)
    assert jnp.array_equal(first_eval, second_eval)
