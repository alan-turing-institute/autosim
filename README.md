# autosim
Lots of Simulations

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install the package

```bash
uv pip install -e .
```

This installs `autosim` in editable mode along with its runtime dependencies:
- `numpy>=1.24`
- `scipy>=1.10`
- `tqdm>=4.65`
- `torch>=2.0`

### Install development dependencies (includes pytest)

```bash
uv sync --group dev
```

## Running tests

Once dev dependencies are installed:

```bash
uv run pytest
```
