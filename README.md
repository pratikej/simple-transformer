# simple-transformer

Build a small transformer from scratch for learning.

The first target is a roughly 10M parameter model that learns addition on
synthetic examples. The vocabulary can stay intentionally tiny: digits `0-9`,
space, `+`, and `=`.

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
