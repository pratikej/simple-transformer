import pytest
import torch

from simple_transformer.config import TrainingConfig
from simple_transformer.data import (
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    ArithmeticDataset,
    ArithmeticExample,
    ArithmeticTokenizer,
    make_arithmetic_dataloader,
    make_arithmetic_dataset,
    make_train_val_loaders,
    max_arithmetic_text_length,
)


def test_make_arithmetic_dataset_generates_valid_examples():
    examples = make_arithmetic_dataset(10, 3, operations=("+",), seed=1)

    assert examples == make_arithmetic_dataset(10, 3, operations=("+",), seed=1)
    for example in examples:
        assert 0 <= example.left <= 999
        assert 0 <= example.right <= 999
        assert example.result == example.left + example.right
        assert example.operation == "+"
        assert example.text == f"{example.left}+{example.right}={example.result}"


@pytest.mark.parametrize(
    "kwargs",
    [{"num_examples": -1, "max_digits": 1}, {"num_examples": 1, "max_digits": 0}],
)
def test_make_arithmetic_dataset_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        make_arithmetic_dataset(**kwargs)


def test_tokenizer_adds_and_skips_eos():
    tokenizer = ArithmeticTokenizer()
    token_ids = tokenizer.encode("12-3=9", add_eos=True)

    assert token_ids[-1] == EOS_TOKEN_ID
    assert tokenizer.decode(token_ids) == "12-3=9"


def test_arithmetic_dataset_returns_shifted_inputs_and_labels():
    dataset = ArithmeticDataset(
        [ArithmeticExample(1, 2, "+", 3, "1+2=3")],
        sequence_length=8,
    )
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


def test_make_arithmetic_dataloader_batches_examples():
    loader, tokenizer = make_arithmetic_dataloader(
        4,
        2,
        batch_size=2,
        operations=("+",),
        seed=1,
        shuffle=False,
    )
    batch = next(iter(loader))

    assert tokenizer.vocab_size == 17
    assert batch["input_ids"].shape == (2, 10)
    assert batch["labels"].shape == (2, 10)


def test_train_val_loaders_include_mixed_operations_with_safe_lengths():
    config = TrainingConfig(
        max_digits=2,
        train_examples=64,
        val_examples=16,
        batch_size=8,
    )

    train_loader, val_loader, tokenizer = make_train_val_loaders(config)
    examples = train_loader.dataset.examples + val_loader.dataset.examples
    operations = {example.operation for example in examples}

    assert operations == {"+", "-", "*", "/"}
    assert tokenizer.vocab_size == 17
    assert all(
        len(example.text) + 1 <= max_arithmetic_text_length(2)
        for example in examples
    )
    assert all(example.right != 0 for example in examples if example.operation == "/")
    for example in examples:
        if example.operation == "+":
            assert example.result == example.left + example.right
        elif example.operation == "-":
            assert example.result == example.left - example.right
        elif example.operation == "*":
            assert example.result == example.left * example.right
        else:
            assert example.result == round(example.left / example.right)


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
