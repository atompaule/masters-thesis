# Masters Thesis

## Cloning the Repository

Clone the repository:

git clone https://github.com/atompaule/masters-thesis.git

## Prerequisites

- Python 3.11 or higher
- [UV](https://github.com/astral-sh/uv) package manager

### Installing UV

If you don't have UV installed, you can install it using:

curl -LsSf https://astral.sh/uv/install.sh | sh

Or using pip:

pip install uv

## Environment Setup

1. Install dependencies using UV:

uv sync

This will:

- Create a virtual environment (if it doesn't exist)
- Install all dependencies specified in `pyproject.toml`
- Use the locked versions from `uv.lock` for reproducible builds

2. Use UV to run scripts directly without activating the virtual environment. For example:

uv run python src/latent_embedding_experiments/training/train_latent_head_linear.py
