# Use a Custom Simulator

Custom simulator classes can be passed directly through Hydra without adding
them to a registry.

```bash
uv run autosim \
  simulator._target_=my_package.my_module.MySimulator \
  ++simulator.my_arg=42 \
  simulator.log_level=warning
```

Dataset generation expects the configured simulator to inherit from
`autosim.simulations.base.SpatioTemporalSimulator`. Non-spatiotemporal
simulators can still be used directly through their `forward` and
`forward_batch` APIs.
