from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from autosim.cli import generate_dataset_splits
from autosim.experimental.simulations import ShallowWater2D
from autosim.parallel import forward_batch_cpu
from autosim.simulations.base import Simulator, SpatioTemporalSimulator


class RandomSimulator(Simulator):
    def __init__(self):
        super().__init__({"x": (0.0, 1.0)}, ["y"], log_level="warning")
        self._rng = torch.Generator().manual_seed(123)
        self.numpy_rng = np.random.default_rng(123)
        self.python_rng = random.Random(123)

    def _forward(self, x):
        return x + torch.tensor(
            [
                [
                    random.random()
                    + np.random.random()
                    + torch.rand(()).item()
                    + torch.rand((), generator=self._rng).item()
                    + self.numpy_rng.random()
                    + self.python_rng.random()
                ]
            ]
        )


class IdentitySimulator(Simulator):
    def __init__(self):
        super().__init__({"x": (0.0, 1.0)}, ["y"], log_level="warning")

    def _forward(self, x) -> torch.Tensor | None:
        time.sleep(float(x[0, 0]) * 0.01)
        return 2 * x


class FailingSimulator(IdentitySimulator):
    def _forward(self, x):
        if x[0, 0] < 0.5:
            msg = "Expected sample failure"
            raise ValueError(msg)
        return 2 * x


class AlwaysFailSimulator(IdentitySimulator):
    def _forward(self, x):
        del x


class ThreadSimulator(IdentitySimulator):
    def _forward(self, x):
        del x
        return torch.tensor([[torch.get_num_threads(), os.getpid()]], dtype=torch.int64)


class RetrySimulator(SpatioTemporalSimulator):
    def __init__(self):
        super().__init__({"x": (0.0, 1.0)}, ["y"], log_level="warning")

    def sample_inputs(self, n_samples, random_seed=None, method="lhs"):
        del method
        assert random_seed is not None
        self.last_sampling_seed = random_seed
        return torch.full((n_samples, 1), 0.0 if random_seed % 2 else 1.0)

    def _forward(self, x):
        return x + torch.rand_like(x) if x[0, 0] else None

    def forward_samples_spatiotemporal(
        self, n, random_seed=None, ensure_exact_n=False, *, num_workers=None
    ):
        y, x = self._forward_batch_with_optional_retries(
            n, random_seed, ensure_exact_n, num_workers=num_workers
        )
        return {
            "data": y[:, None, None, None, :],
            "constant_scalars": x,
            "constant_fields": None,
        }


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_seeded_samples_are_independent_and_worker_count_invariant(workers):
    x = torch.arange(5, dtype=torch.float32)[:, None]
    expected, _ = RandomSimulator().forward_batch(x, num_workers=1, random_seed=42)
    actual, valid = RandomSimulator().forward_batch(
        x, num_workers=workers, random_seed=42
    )
    assert torch.equal(actual, expected)
    assert torch.equal(valid, x)
    assert len(torch.unique(actual - x)) == len(x)
    other, _ = RandomSimulator().forward_batch(x, num_workers=workers, random_seed=43)
    assert not torch.equal(actual, other)


def test_native_serial_rng_stream_is_unchanged():
    x = torch.zeros(3, 1)
    first = RandomSimulator()
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    samples = [first.forward(sample[None]) for sample in x]
    assert all(sample is not None for sample in samples)
    expected = torch.cat([sample for sample in samples if sample is not None])
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    actual, _ = RandomSimulator().forward_batch(x)
    assert torch.equal(actual, expected)


def test_ordering_and_deterministic_serial_agreement():
    x = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
    expected, _ = IdentitySimulator().forward_batch(x)
    actual, valid = IdentitySimulator().forward_batch(x, num_workers=2, random_seed=42)
    assert torch.equal(actual, expected)
    assert torch.equal(valid, x)


@pytest.mark.parametrize("workers", [1, 2])
def test_failure_policy_and_empty_batches(workers):
    x = torch.tensor([[0.1], [0.8], [0.2], [0.9]])
    result, valid = FailingSimulator().forward_batch(
        x, num_workers=workers, random_seed=42
    )
    assert torch.equal(valid, x[[1, 3]])
    assert torch.equal(result, 2 * valid)
    with pytest.raises(ValueError, match="Expected sample failure"):
        FailingSimulator().forward_batch(
            x, allow_failures=False, num_workers=workers, random_seed=42
        )
    for sim, values in [(IdentitySimulator(), x[:0]), (AlwaysFailSimulator(), x)]:
        result, valid = sim.forward_batch(values, num_workers=workers, random_seed=42)
        assert result.shape == valid.shape == (0, 1)


def test_retries_are_reproducible_and_splits_are_separate():
    expected = generate_dataset_splits(
        RetrySimulator(), 2, 2, 2, base_seed=11, ensure_exact_n=True, num_workers=1
    )
    actual = generate_dataset_splits(
        RetrySimulator(), 2, 2, 2, base_seed=11, ensure_exact_n=True, num_workers=2
    )
    for split in actual:
        assert actual[split]["data"].shape[0] == 2
        assert torch.equal(actual[split]["data"], expected[split]["data"])
    assert not torch.equal(actual["train"]["data"], actual["valid"]["data"])
    assert not torch.equal(actual["valid"]["data"], actual["test"]["data"])


def test_workers_use_one_thread_and_different_processes():
    sim = ThreadSimulator()
    result, _ = sim.forward_batch(torch.ones(8, 1), num_workers=2, random_seed=42)
    assert (result[:, 0] == 1).all()
    assert (result[:, 1] != os.getpid()).all()


@pytest.mark.parametrize("workers", [0, -1, True, 2.5, "2"])
def test_invalid_worker_counts_rejected(workers):
    with pytest.raises(ValueError, match="num_workers"):
        IdentitySimulator().forward_batch(torch.ones(2, 1), num_workers=workers)


def test_cpu_and_pickle_requirements_fail_clearly():
    with pytest.raises(ValueError, match="CPU"):
        forward_batch_cpu(
            IdentitySimulator(),
            torch.empty(2, 1, device="meta"),
            num_workers=2,
            random_seed=42,
            allow_failures=True,
        )
    sim = IdentitySimulator()
    sim.__dict__["unpickleable"] = lambda value: value
    with pytest.raises(TypeError, match="pickleable"):
        sim.forward_batch(torch.ones(2, 1), num_workers=2)


def test_previous_result_buffer_is_not_copied_to_workers():
    sim = IdentitySimulator()
    # A deliberately unpickleable placeholder proves the cache is stripped.
    sim.__dict__["results_tensor"] = lambda value: value
    result, _ = sim.forward_batch(torch.ones(2, 1), num_workers=2, random_seed=42)
    assert torch.equal(result, torch.full((2, 1), 2.0))


def test_swe_reproducible_with_auxiliary_outputs():
    kwargs: dict[str, Any] = dict(  # noqa: C408 -- compact simulator keyword fixture
        nx=16,
        ny=16,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        T=0.25,
        dt_save=0.25,
        skip_nt=0,
        return_timeseries=True,
        return_additional_input_fields=True,
        return_energy_budget=True,
        log_level="warning",
        initial_condition="balanced_random_pv",
        forcing_type="vortical",
        forcing_correlation_time=0.0,
        forcing_energy_rate=0.0015,
        forcing_wavenumber=3.0,
        forcing_bandwidth=0.7,
        parameters_range={"amp": (0.08, 0.12)},
    )
    expected = ShallowWater2D(**kwargs).forward_samples_spatiotemporal(
        3, random_seed=42, num_workers=1
    )
    actual = ShallowWater2D(**kwargs).forward_samples_spatiotemporal(
        3, random_seed=42, num_workers=2
    )
    for key in expected:
        if expected[key] is None:
            assert actual[key] is None
        else:
            assert torch.equal(actual[key], expected[key]), key
    assert not torch.equal(actual["data"][0], actual["data"][1])


def test_cli_parallel_generation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "autosim.cli",
        "simulator=experimental/shallow_water2d_crps_64",
        "simulator.nx=16",
        "simulator.ny=16",
        "simulator.T=.25",
        "simulator.skip_nt=0",
        "dataset.n_train=2",
        "dataset.n_valid=1",
        "dataset.n_test=1",
        "dataset.num_workers=2",
        "seed=42",
        "visualize.enabled=false",
        f"dataset.output_dir={tmp_path / 'data'}",
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ]
    subprocess.run(command, check=True, cwd=root)
    for split, count in [("train", 2), ("valid", 1), ("test", 1)]:
        payload = torch.load(tmp_path / "data" / split / "data.pt", weights_only=True)
        assert payload["data"].shape == (count, 2, 16, 16, 3)
        assert torch.isfinite(payload["data"]).all()
    assert (tmp_path / "data/stats.yml").is_file()


def test_exact_count_exhaustion_raises(monkeypatch):
    monkeypatch.setattr(RetrySimulator, "_retry_budget", staticmethod(lambda _n: 2))
    sim = RetrySimulator()
    monkeypatch.setattr(
        sim, "sample_inputs", lambda n_samples, _random_seed: torch.zeros(n_samples, 1)
    )
    # The monkeypatched instance is deliberately not pickleable; simulate a
    # completely rejected batch at the parent retry boundary instead.
    monkeypatch.setattr(
        sim, "forward_batch", lambda x, **kwargs: (torch.empty(0, 1), x[:0])
    )
    with pytest.raises(RuntimeError, match=r"exactly n=2.*2 retry rounds"):
        sim._forward_batch_with_optional_retries(
            2, random_seed=42, ensure_exact_n=True, num_workers=2
        )


def test_native_batch_implementation_rejects_worker_option():
    class NativeSimulator(SpatioTemporalSimulator):
        def _forward(self, x):
            return x

        def forward_samples_spatiotemporal(
            self, n, random_seed=None, ensure_exact_n=False
        ):
            del n, random_seed, ensure_exact_n
            pytest.fail("Must reject the request before native generation")

    with pytest.raises(ValueError, match="native batch implementation"):
        generate_dataset_splits(NativeSimulator({}, []), 2, 1, 1, num_workers=2)
