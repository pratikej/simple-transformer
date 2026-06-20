# simple-transformer

Build a small transformer from scratch for learning arithmetic.

The first target is a small decoder-only transformer that learns synthetic
integer arithmetic expressions. Examples look like `12+7=19`, `8*9=72`,
`10-14=-4`, and `9/2=4`; division targets are rounded integers.

The vocabulary is intentionally tiny: digits `0-9`, operators `+`, `-`, `*`,
`/`, `=`, plus padding and EOS tokens.

## Project Layout

```text
simple-transformer/
├── notebooks/              # Colab/Kaggle/local experiments
├── src/simple_transformer/  # Reusable Python package code
└── tests/                  # Tests for package code
```

Keep notebooks focused on experiments. Put reusable model, data, and training
code under `src/simple_transformer/` so it can be imported from any notebook.

## Local Setup

Conda or Mamba is recommended locally, especially if you later use a local GPU.

```bash
mamba env create -f environment.yml
mamba activate simple-transformer
pip install -e .
```

If you use Conda instead of Mamba, replace `mamba` with `conda`.

## Colab or Kaggle Setup

From a notebook cell in the cloned repo:

```python
%pip install -r requirements.txt
%pip install -e .
```

After that, imports should work like:

```python
import simple_transformer
```

## Git Notes

Do not commit generated datasets, model checkpoints, logs, caches, or notebook
checkpoint folders. The `.gitignore` is set up to keep those out of git.
