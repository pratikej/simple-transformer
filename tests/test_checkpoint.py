from dataclasses import replace

import torch

from simple_transformer.checkpoint import CheckpointConfig, CheckpointManager
from simple_transformer.config import TrainingConfig, TransformerConfig
from simple_transformer.data import ADDITION_VOCAB, make_train_val_loaders
from simple_transformer.model import SimpleTransformerLM
from simple_transformer.train import fit


def test_checkpoint_manager_keeps_only_latest_epochs(tmp_path):
    config = _training_config(epochs=3)
    model = _model()
    train_loader, val_loader, _ = make_train_val_loaders(config)
    checkpoint_manager = CheckpointManager(
        CheckpointConfig(tmp_path, keep_last=2, async_save=False)
    )

    fit(
        model,
        train_loader,
        val_loader,
        config,
        checkpoint_manager=checkpoint_manager,
    )

    assert [path.name for path in sorted(tmp_path.glob("epoch-*.pt"))] == [
        "epoch-0002.pt",
        "epoch-0003.pt",
    ]


def test_async_checkpoint_save_writes_after_close(tmp_path):
    config = _training_config(epochs=1)
    model = _model()
    train_loader, val_loader, _ = make_train_val_loaders(config)
    checkpoint_manager = CheckpointManager(
        CheckpointConfig(tmp_path, keep_last=2, async_save=True)
    )

    fit(
        model,
        train_loader,
        val_loader,
        config,
        checkpoint_manager=checkpoint_manager,
    )
    checkpoint_manager.close()

    latest = checkpoint_manager.latest_checkpoint()
    assert latest is not None
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == len(train_loader)


def test_fit_resumes_from_checkpoint(tmp_path):
    initial_config = _training_config(epochs=1)
    model = _model()
    train_loader, val_loader, _ = make_train_val_loaders(initial_config)
    checkpoint_manager = CheckpointManager(
        CheckpointConfig(tmp_path, keep_last=2, async_save=False)
    )
    initial_result = fit(
        model,
        train_loader,
        val_loader,
        initial_config,
        checkpoint_manager=checkpoint_manager,
    )
    checkpoint_path = checkpoint_manager.latest_checkpoint()

    resumed_config = replace(initial_config, epochs=2)
    resumed_model = _model()
    resumed_train_loader, resumed_val_loader, _ = make_train_val_loaders(resumed_config)
    resumed_result = fit(
        resumed_model,
        resumed_train_loader,
        resumed_val_loader,
        resumed_config,
        checkpoint_manager=checkpoint_manager,
        resume_from=checkpoint_path,
    )

    assert checkpoint_path is not None
    assert len(initial_result.train) == 1
    assert len(resumed_result.train) == 2
    assert len(resumed_result.validation) == 2
    assert checkpoint_manager.latest_checkpoint().name == "epoch-0002.pt"


def _training_config(*, epochs: int) -> TrainingConfig:
    return TrainingConfig(
        max_digits=2,
        train_examples=16,
        val_examples=8,
        batch_size=8,
        epochs=epochs,
        warmup_steps=1,
        device="cpu",
    )


def _model() -> SimpleTransformerLM:
    return SimpleTransformerLM(
        TransformerConfig(
            vocab_size=len(ADDITION_VOCAB),
            max_seq_len=9,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
        )
    )
