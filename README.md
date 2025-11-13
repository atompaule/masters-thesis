# Masters Thesis

## Prerequisites

-   Python 3.11 or higher
-   [UV](https://github.com/astral-sh/uv) package manager

### Installing UV

If you don't have UV installed, you can install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:

```bash
pip install uv
```

## Environment Setup

1. Install dependencies using UV:

```bash
uv sync
```

This will:

-   Create a virtual environment (if it doesn't exist)
-   Install all dependencies specified in `pyproject.toml`
-   Use the locked versions from `uv.lock` for reproducible builds

2. Use UV to run commands directly without activating the virtual environment:

```bash
uv run python src/hrpo/train_grpo.py
```
