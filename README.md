# AutoSim <img src="https://raw.githubusercontent.com/alan-turing-institute/autosim/refs/heads/main/AS.png" align="right" height="138" />
Lots of Simulations

## Installation

You will need to install [uv](https://docs.astral.sh/uv/) and [ffmpeg](https://ffmpeg.org/download.html) to run AutoSim.

```bash
git clone https://github.com/alan-turing-institute/autosim
cd autosim
uv sync
```

AutoSim comes with a number of bundled simulators for different physical phenomena.
You can list these with:

```bash
uv run autosim list
```

## Generate training data

To generate a tiny dataset with the advection-diffusion simulator, run:

```bash
uv run autosim \
  simulator=spatiotemporal/advection_diffusion \
  simulator.n=16 simulator.T=0.2 simulator.dt=0.1 \
  dataset.n_train=1 dataset.n_valid=1 dataset.n_test=1
```

Do check out the [documentation](https://alan-turing-institute.github.io/autosim/) for more information on configuration options and simulators!
