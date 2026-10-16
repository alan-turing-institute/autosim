autosim.cli
===========

Parallel CPU generation
-----------------------

For simulators using the shared independent-trajectory batching API, set
``dataset.num_workers`` to a positive integer:

.. code-block:: console

   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     uv run autosim --config-name=generate_data_swe_crps_64 \
     dataset.num_workers=16 visualize.enabled=false

The library API also accepts ``num_workers`` on ``forward_batch`` and on
the corresponding ``forward_samples_spatiotemporal`` methods. Parameter
sampling (including Latin hypercube sampling) remains in the parent process;
only the independent trajectory solves run in workers. Spawned processes
avoid the Python GIL and each uses one Torch thread. Set the environment
variables above before starting Python to also limit BLAS threads.

Omitting the option (or using ``null``) preserves the existing serial
implementation and its RNG stream. Explicit ``num_workers=1`` selects the
new independent-sample RNG policy without spawning a pool; with a fixed
seed it produces the same ordered samples as larger worker counts on the
same runtime. This opt-in policy seeds each sample separately, including
owned Torch, NumPy and Python generators. Its stochastic realizations differ
from the legacy serial stream and must be recorded as a new dataset identity.

Workers receive a fresh simulator for each solve and must support pickling
and CPU execution. Custom simulators must not rely on shared mutable state or
unseeded external randomness. Native vectorized implementations that do not
use the shared trajectory batch helper retain their existing API; unsupported
worker requests fail explicitly. Python scripts that spawn workers need an
``if __name__ == "__main__":`` guard.

Results retain input order, failed solves keep the existing skip/retry policy,
and retry seeds remain separate across dataset splits. At most twice the
worker count is in flight. The parent fills one result buffer rather than
retaining every trajectory and then concatenating a second full copy; the
complete dataset still needs to fit in memory. No extra dataset shards or
file-format changes are required.

.. automodule:: autosim.cli
   :members:
   :undoc-members:
   :show-inheritance:
