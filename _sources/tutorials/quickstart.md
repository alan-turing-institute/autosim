# Quickstart

AutoSim uses [Hydra](https://hydra.cc) configurations to instantiate simulators and generate train, validation, and test splits.
These configurations are stored as YAML files in the [`src/autosim/configs`](https://github.com/alan-turing-institute/autosim/tree/main/src/autosim/configs) directory.

You can list available simulator configurations with:

```bash
uv run autosim list
```

All of these represent different systems of equations which can be simulated; you can see [their docstrings](../reference/simulations/index.md) for more information.

## Generate a dataset

The following command will generate a dataset for the `spatiotemporal/advection_diffusion` simulator, with:

- a `16x16` grid (`simulator.n=16`),
- a total simulation time of `0.2` (`simulator.T=0.2`)
- a time step of `0.1` (`simulator.dt=0.1`) (i.e., there will be three time steps),
- a single training example (`dataset.n_train=1`),
- a single validation example (`dataset.n_valid=1`), and
- a single test example (`dataset.n_test=1`).

```bash
uv run autosim \
  simulator=spatiotemporal/advection_diffusion \
  simulator.n=16 simulator.T=0.2 simulator.dt=0.1 \
  dataset.n_train=1 dataset.n_valid=1 dataset.n_test=1
```

You can change the arguments to generate a dataset with different parameters.

By default, the generated dataset is saved to the directory

```
outputs/${now:%Y-%m-%d}/${hydra:runtime.choices.simulator}_${shortuuid:7}
```

so, for example, the above command might generate a dataset in

```
outputs/2026-12-31/spatiotemporal/advection_diffusion_1a2b3c4
```

If desired, you can change the output path by supplying an extra argument:

```
uv run autosim [...] ++dataset.output_dir=my_custom_path
```

## What's in a dataset?

The generated dataset has the following structure:

```
.
├── cli.log
├── examples
│   └── train
│       └── batch_0.mp4
├── resolved_config.yaml
├── stats.yml
├── test
│   └── data.pt
├── train
│   └── data.pt
└── valid
    └── data.pt
```

The `data.pt` files are PyTorch-serialized dictionaries.
The simulation trajectories are stored as tensors under the `data` key, alongside
the `constant_scalars` and `constant_fields` entries.
The `examples` directory contains a video of the training example, which can be used to visualize the simulation.

`cli.log` is just the output of the `uv run autosim` command.

`resolved_config.yaml` contains a copy of the configuration used to generate the dataset.
This is useful for inspecting the parameters used to generate the dataset, and for reproducing the dataset later.

`stats.yml` contains normalisation statistics which are used during subsequent model training.
