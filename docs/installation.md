# Installation

AutoSim uses [uv](https://docs.astral.sh/uv/) for environment management and
command execution.

Install the package for development:

```bash
uv sync --extra dev
```

Install documentation dependencies:

```bash
uv sync --extra dev --extra docs
```

Run the test suite:

```bash
uv run pytest
```

Build the documentation locally:

```bash
uv run jupyter-book build docs --all
```
