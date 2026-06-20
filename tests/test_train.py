from simple_transformer.config import TrainingConfig, TransformerConfig, local_training_config
from simple_transformer.data import ARITHMETIC_VOCAB, make_train_val_loaders
from simple_transformer.model import SimpleTransformerLM
from simple_transformer.train import fit


def test_fit_runs_one_small_cpu_epoch():
    config = TrainingConfig(
        max_digits=2,
        train_examples=16,
        val_examples=8,
        batch_size=8,
        epochs=1,
        warmup_steps=1,
        device="cpu",
    )
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
    train_loader, val_loader, tokenizer = make_train_val_loaders(config)

    result = fit(model, train_loader, val_loader, config)

    assert tokenizer.vocab_size == len(ARITHMETIC_VOCAB)
    assert result.train[0].loss > 0
    assert result.validation[0].loss > 0
    assert 0.0 <= result.validation[0].accuracy <= 1.0


def test_local_training_config_sets_device_optimizations():
    cpu_config = local_training_config(device="cpu")
    cuda_config = local_training_config(device="cuda")

    assert cpu_config.device == "cpu"
    assert cpu_config.use_amp is False
    assert cpu_config.use_fused_optimizer is False
    assert cpu_config.pin_memory is False
    assert cuda_config.device == "cuda"
    assert cuda_config.use_amp is True
    assert cuda_config.use_fused_optimizer is True
    assert cuda_config.pin_memory is True
