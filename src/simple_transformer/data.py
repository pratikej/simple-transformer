"""Dataset helpers for arithmetic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from simple_transformer.config import TrainingConfig


PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
ADDITION_VOCAB = (PAD_TOKEN, EOS_TOKEN, *tuple("0123456789+="))
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
IGNORE_INDEX = -100


@dataclass(frozen=True)
class AdditionExample:
    """One synthetic addition example."""

    left: int
    right: int
    total: int
    text: str


class AdditionTokenizer:
    """Character tokenizer for addition expressions."""

    def __init__(self, vocab: tuple[str, ...] = ADDITION_VOCAB) -> None:
        self.id_to_token = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.pad_token_id = self.token_to_id[PAD_TOKEN]
        self.eos_token_id = self.token_to_id[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        token_ids = [self.token_to_id[token] for token in text]
        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        special_token_ids = {self.pad_token_id, self.eos_token_id}
        return "".join(
            self.id_to_token[token_id]
            for token_id in token_ids
            if not skip_special_tokens or token_id not in special_token_ids
        )


class AdditionDataset(Dataset):
    """PyTorch dataset for next-token prediction on addition examples."""

    def __init__(
        self,
        examples: list[AdditionExample],
        *,
        tokenizer: AdditionTokenizer | None = None,
        sequence_length: int | None = None,
    ) -> None:
        self.examples = examples
        self.tokenizer = AdditionTokenizer() if tokenizer is None else tokenizer
        self.sequence_length = (
            max(len(example.text) + 1 for example in examples)
            if sequence_length is None and examples
            else sequence_length
        )

        if self.sequence_length is None:
            raise ValueError("sequence_length is required when examples is empty")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        token_ids = self.tokenizer.encode(self.examples[index].text, add_eos=True)
        if len(token_ids) > self.sequence_length:
            raise ValueError(
                f"Example is length {len(token_ids)}, which exceeds "
                f"sequence_length={self.sequence_length}"
            )

        padded_ids = token_ids + [PAD_TOKEN_ID] * (
            self.sequence_length - len(token_ids)
        )
        input_ids = torch.tensor(padded_ids[:-1], dtype=torch.long)
        labels = torch.tensor(padded_ids[1:], dtype=torch.long)
        equals_index = token_ids.index(self.tokenizer.token_to_id["="])
        labels[:equals_index] = IGNORE_INDEX
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def make_addition_dataset(
    num_examples: int,
    max_digits: int,
    *,
    seed: int | None = None,
) -> list[AdditionExample]:
    """Create examples of adding two integers.

    Args:
        num_examples: Number of examples to generate.
        max_digits: Maximum number of digits in both operands.
        seed: Optional random seed for reproducible datasets.

    Returns:
        A list of examples formatted as ``"{left}+{right}={total}"``.
    """

    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")
    if max_digits < 1:
        raise ValueError("max_digits must be at least 1")

    rng = Random(seed)
    min_value, max_value = _max_digit_bounds(max_digits)

    examples: list[AdditionExample] = []
    for _ in range(num_examples):
        left = rng.randint(min_value, max_value)
        right = rng.randint(min_value, max_value)
        total = left + right
        examples.append(
            AdditionExample(
                left=left,
                right=right,
                total=total,
                text=f"{left}+{right}={total}",
            )
        )

    return examples


def make_addition_dataloader(
    num_examples: int,
    max_digits: int,
    batch_size: int,
    *,
    seed: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> tuple[DataLoader, AdditionTokenizer]:
    """Create a DataLoader for next-token prediction on addition examples."""

    examples = make_addition_dataset(
        num_examples=num_examples,
        max_digits=max_digits,
        seed=seed,
    )
    tokenizer = AdditionTokenizer()
    dataset = AdditionDataset(
        examples,
        tokenizer=tokenizer,
        sequence_length=max_addition_text_length(max_digits),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )

    return loader, tokenizer


def make_train_val_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, AdditionTokenizer]:
    """Create train and validation loaders from a training config."""

    train_loader, tokenizer = make_addition_dataloader(
        num_examples=config.train_examples,
        max_digits=config.max_digits,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
        pin_memory=config.pin_memory,
    )
    val_loader, _ = make_addition_dataloader(
        num_examples=config.val_examples,
        max_digits=config.max_digits,
        batch_size=config.batch_size,
        seed=config.seed + 1,
        shuffle=False,
        pin_memory=config.pin_memory,
    )
    return train_loader, val_loader, tokenizer


def max_addition_text_length(max_digits: int) -> int:
    """Maximum token length for ``left+right=total`` plus an EOS token."""

    if max_digits < 1:
        raise ValueError("max_digits must be at least 1")

    return max_digits + 1 + max_digits + 1 + (max_digits + 1) + 1


def _max_digit_bounds(max_digits: int) -> tuple[int, int]:
    return 0, 10**max_digits - 1
