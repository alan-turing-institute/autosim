# Generate a Dataset

AutoSim uses Hydra configs to instantiate simulators and generate train,
validation, and test splits.

List available simulator configs:

```bash
uv run autosim list
```

Generate a small dataset with the advection-diffusion simulator:

```bash
uv run autosim \
  simulator=spatiotemporal/advection_diffusion \
  simulator.n=16 simulator.T=0.2 simulator.dt=0.1 \
  dataset.n_train=1 dataset.n_valid=1 dataset.n_test=1
```

The generated dataset contains `train`, `valid`, and `test` directories with a
`data.pt` payload for each split.
