# Visualization

As seen on [the previous page](quickstart.md), the `autosim` command generates visualizations of the simulation results by default.

## Disabling visualizations

To disable the generation of visualizations, you can specify the extra argument

```bash
uv run autosim [...] ++visualize.enabled=false
```

## Changing which trajectories are visualized

You can generate visualizations for the 1st and 3rd training trajectories by specifying:

```bash
uv run autosim [...] ++visualize.split=train ++visualize.batch_indices[0,2]
```

## Changing the file output

To generate GIFs instead of videos, you can specify

```bash
uv run autosim [...] ++visualize.file_ext=gif
```
