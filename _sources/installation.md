# Installation

AutoSim is not yet available on PyPI, so you must obtain the source code by cloning the repository:

```bash
git clone https://github.com/alan-turing-institute/autosim
cd autosim
```

AutoSim uses [uv](https://docs.astral.sh/uv/) for environment management and
command execution; you will need to install it first.
You can then run

```bash
uv sync
```

to install the dependencies needed to run AutoSim.
To check if the installation was successful, you can run the following command:

```bash
uv run autosim --help
```

If you want AutoSim to generate visualisations of the data (enabled by default), you will also need to [install ffmpeg](https://ffmpeg.org/).

## Contributing

If you want to develop AutoSim, some extra dependencies are needed.
The following will install dependencies needed for testing and for building documentation:

```bash
uv sync --extra dev --extra docs
```

You can run the test suite with:

```bash
uv run pytest
```

To build the documentation locally:

```bash
uv run jupyter-book build docs --all
uv run python -m http.server -d docs/_build/html
```
