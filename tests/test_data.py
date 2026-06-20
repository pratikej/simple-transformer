import pytest
import torch

from simple_transformer.config import TrainingConfig
from simple_transformer.data import (
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    AdditionDataset,
    AdditionExample,
    AdditionTokenizer,
    make_addition_dataloader,
    make_addition_dataset,
    make_train_val_loaders,
)


def test_make_addition_dataset_generates_valid_examples():
    examples = make_addition_dataset(10, 3, seed=1)

    assert examples == make_addition_dataset(10, 3, seed=1)
    for example in examples:
        assert 0 <= example.left <= 999
        assert 0 <= example.right <= 999
        assert example.total == example.left + example.right
        assert example.text == f"{example.left}+{example.right}={example.total}"


@pytest.mark.parametrize("kwargs", [{"num_examples": -1, "max_digits": 1}, {"num_examples": 1, "max_digits": 0}])
def test_make_addition_dataset_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        make_addition_dataset(**kwargs)


def test_tokenizer_adds_and_skips_eos():
    tokenizer = AdditionTokenizer()
    token_ids = tokenizer.encode("12+3=15", add_eos=True)

    assert token_ids[-1] == EOS_TOKEN_ID
    assert tokenizer.decode(token_ids) == "12+3=15"


def test_addition_dataset_returns_shifted_inputs_and_labels():
    dataset = AdditionDataset([AdditionExample(1, 2, 3, "1+2=3")], sequence_length=8)
    batch = dataset[0]

    assert set(batch) == {"input_ids", "labels"}
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].tolist() == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        5,
        EOS_TOKEN_ID,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_make_addition_dataloader_batches_examples():
    loader, tokenizer = make_addition_dataloader(4, 2, batch_size=2, seed=1, shuffle=False)
    batch = next(iter(loader))

    assert tokenizer.vocab_size == 14
    assert batch["input_ids"].shape == (2, 9)
    assert batch["labels"].shape == (2, 9)


def test_make_train_val_loaders_have_distinct_examples():
    config = TrainingConfig(
        max_digits=2,
        train_examples=32,
        val_examples=16,
        batch_size=8,
    )

    train_loader, val_loader, _ = make_train_val_loaders(config)
    train_examples = {example.text for example in train_loader.dataset.examples}
    val_examples = {example.text for example in val_loader.dataset.examples}

    assert train_examples.isdisjoint(val_examples)
