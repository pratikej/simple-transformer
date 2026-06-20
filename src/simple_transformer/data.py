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
ARITHMETIC_OPERATIONS = ("+", "-", "*", "/")
ARITHMETIC_VOCAB = (PAD_TOKEN, EOS_TOKEN, *tuple("0123456789+-*/="))
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ArithmeticExample:
    """One synthetic arithmetic example."""

    left: int
    right: int
    operation: str
    result: int
    text: str


class ArithmeticTokenizer:
    """Character tokenizer for arithmetic expressions."""

    def __init__(self, vocab: tuple[str, ...] = ARITHMETIC_VOCAB) -> None:
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


class ArithmeticDataset(Dataset):
    """PyTorch dataset for next-token prediction on arithmetic examples."""

    def __init__(
        self,
        examples: list[ArithmeticExample],
        *,
        tokenizer: ArithmeticTokenizer | None = None,
        sequence_length: int | None = None,
    ) -> None:
        self.examples = examples
        self.tokenizer = ArithmeticTokenizer() if tokenizer is None else tokenizer
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


def make_arithmetic_dataset(
    num_examples: int,
    max_digits: int,
    *,
    operations: tuple[str, ...] = ARITHMETIC_OPERATIONS,
    seed: int | None = None,
) -> list[ArithmeticExample]:
    """Create arithmetic examples using the requested operations.

    Args:
        num_examples: Number of examples to generate.
        max_digits: Maximum number of digits in both operands.
        operations: Operations to sample from.
        seed: Optional random seed for reproducible datasets.

    Returns:
        A list of examples formatted as ``"{left}<op>{right}={result}"``.
    """

    return _make_unique_arithmetic_examples(
        num_examples=num_examples,
        max_digits=max_digits,
        operations=operations,
        seed=seed,
    )


def make_arithmetic_dataloader(
    num_examples: int,
    max_digits: int,
    batch_size: int,
    *,
    operations: tuple[str, ...] = ARITHMETIC_OPERATIONS,
    seed: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> tuple[DataLoader, ArithmeticTokenizer]:
    """Create a DataLoader for next-token prediction on arithmetic examples."""

    examples = make_arithmetic_dataset(
        num_examples=num_examples,
        max_digits=max_digits,
        operations=operations,
        seed=seed,
    )
    tokenizer = ArithmeticTokenizer()
    dataset = ArithmeticDataset(
        examples,
        tokenizer=tokenizer,
        sequence_length=max_arithmetic_text_length(max_digits),
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
) -> tuple[DataLoader, DataLoader, ArithmeticTokenizer]:
    """Create train and validation loaders from a training config."""

    examples = _make_unique_arithmetic_examples(
        num_examples=config.train_examples + config.val_examples,
        max_digits=config.max_digits,
        operations=config.operations,
        seed=config.seed,
    )

    tokenizer = ArithmeticTokenizer()
    sequence_length = max_arithmetic_text_length(config.max_digits)
    train_dataset = ArithmeticDataset(
        examples[: config.train_examples],
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    val_dataset = ArithmeticDataset(
        examples[config.train_examples :],
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=config.pin_memory,
    )
    return train_loader, val_loader, tokenizer


def max_arithmetic_text_length(max_digits: int) -> int:
    """Maximum token length for ``left<op>right=result`` plus an EOS token."""

    if max_digits < 1:
        raise ValueError("max_digits must be at least 1")

    prompt_length = max_digits + 1 + max_digits + 1
    result_length = max(2 * max_digits, max_digits + 1)
    return prompt_length + result_length + 1


def _max_digit_bounds(max_digits: int) -> tuple[int, int]:
    return 0, 10**max_digits - 1


def _make_unique_arithmetic_examples(
    num_examples: int,
    max_digits: int,
    *,
    operations: tuple[str, ...] = ARITHMETIC_OPERATIONS,
    seed: int | None = None,
) -> list[ArithmeticExample]:
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")
    if max_digits < 1:
        raise ValueError("max_digits must be at least 1")
    if not operations:
        raise ValueError("operations must be non-empty")
    if unsupported := set(operations) - set(ARITHMETIC_OPERATIONS):
        raise ValueError(f"Unsupported operations: {sorted(unsupported)}")

    min_value, max_value = _max_digit_bounds(max_digits)
    value_count = max_value - min_value + 1
    population_size = (
        value_count
        * value_count
        * len([operation for operation in operations if operation != "/"])
    )
    if "/" in operations:
        population_size += value_count * (value_count - 1)
    if num_examples > population_size:
        raise ValueError(
            f"Requested {num_examples} examples, but only {population_size} "
            "unique expressions are available"
        )

    rng = Random(seed)
    expressions: set[tuple[int, str, int]] = set()
    while len(expressions) < num_examples:
        operation = rng.choice(operations)
        left = rng.randint(min_value, max_value)
        right = rng.randint(min_value, max_value)
        if operation == "/" and right == 0:
            continue
        expressions.add((left, operation, right))

    examples = []
    for left, operation, right in expressions:
        result = _evaluate(left, operation, right)
        examples.append(
            ArithmeticExample(
                left=left,
                right=right,
                operation=operation,
                result=result,
                text=f"{left}{operation}{right}={result}",
            )
        )

    rng.shuffle(examples)
    return examples


def _evaluate(left: int, operation: str, right: int) -> int:
    if operation == "+":
        return left + right
    if operation == "-":
        return left - right
    if operation == "*":
        return left * right
    if operation == "/":
        if right == 0:
            return 0
        return round(left / right)
    raise ValueError(f"Unsupported operation: {operation}")
