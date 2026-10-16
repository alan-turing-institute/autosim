"""Bounded, process-based execution of independent CPU simulations."""

from __future__ import annotations

import copy
import multiprocessing
import pickle
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from autosim.validation import set_random_seed

if TYPE_CHECKING:
    from collections.abc import Generator

    from autosim.simulations.base import Simulator

_WORKER_SIMULATOR: bytes | None = None


def validate_num_workers(num_workers: int | None) -> None:
    """Reject invalid worker counts without starting processes."""
    if num_workers is not None and (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 1
    ):
        msg = "num_workers must be a positive integer or None (legacy serial)."
        raise ValueError(msg)


def _initialize_worker(simulator: bytes, default_dtype: torch.dtype) -> None:
    global _WORKER_SIMULATOR  # noqa: PLW0603 -- process-local immutable template
    _WORKER_SIMULATOR = simulator
    torch.set_default_dtype(default_dtype)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _run_sample(
    simulator: bytes,
    values: np.ndarray,
    seed: int,
    allow_failures: bool,
) -> np.ndarray | None:
    # A fresh simulator avoids carrying worker-dependent state between samples.
    sim = pickle.loads(simulator)
    set_random_seed(seed % 2**32)
    torch.manual_seed(seed)
    # The fluid solvers can also own a generator rather than use the global RNG.
    for name, generator in vars(sim).items():
        if isinstance(generator, torch.Generator):
            generator.manual_seed(seed)
        elif isinstance(generator, np.random.Generator):
            setattr(sim, name, np.random.Generator(type(generator.bit_generator)(seed)))
        elif isinstance(generator, random.Random):
            generator.seed(seed)
    result = sim.forward(torch.from_numpy(values), allow_failures=allow_failures)
    if result is None:
        return None
    if result.device.type != "cpu":
        msg = "Process-based generation supports CPU simulation outputs only."
        raise ValueError(msg)
    # NumPy transport avoids retaining one shared-memory tensor/FD per result.
    return result.detach().numpy()


def _worker_sample(
    values: np.ndarray, seed: int, allow_failures: bool
) -> np.ndarray | None:
    if _WORKER_SIMULATOR is None:
        msg = "Simulation worker has not been initialized."
        raise RuntimeError(msg)
    return _run_sample(_WORKER_SIMULATOR, values, seed, allow_failures)


def _ordered_results(
    simulator: bytes,
    inputs: torch.Tensor,
    seeds: list[int],
    num_workers: int,
    allow_failures: bool,
    default_dtype: torch.dtype,
) -> Generator[np.ndarray | None, None, None]:
    with ProcessPoolExecutor(
        max_workers=min(num_workers, len(inputs)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(simulator, default_dtype),
    ) as executor:
        pending = deque()
        cursor = 0
        while cursor < len(inputs) or pending:
            while cursor < len(inputs) and len(pending) < 2 * num_workers:
                pending.append(
                    executor.submit(
                        _worker_sample,
                        inputs[cursor : cursor + 1].numpy(),
                        seeds[cursor],
                        allow_failures,
                    )
                )
                cursor += 1
            yield pending.popleft().result()


def _prepare_output(
    output: torch.Tensor | None,
    sample: torch.Tensor,
    *,
    size: int,
    index: int,
) -> torch.Tensor:
    """Allocate or validate the batch buffer before copying a result."""
    if output is None:
        output = torch.empty((size, *sample.shape[1:]), dtype=sample.dtype)
    expected_shape = (1, *output.shape[1:])
    if sample.shape != expected_shape:
        msg = (
            f"Simulation sample {index} returned shape {tuple(sample.shape)}; "
            f"expected {expected_shape}."
        )
        raise ValueError(msg)
    if sample.dtype != output.dtype:
        msg = (
            f"Simulation sample {index} returned dtype {sample.dtype}; "
            f"expected {output.dtype}."
        )
        raise ValueError(msg)
    return output


def forward_batch_cpu(
    sim: Simulator,
    inputs: torch.Tensor,
    *,
    num_workers: int,
    random_seed: int | None,
    allow_failures: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run seeded independent samples, preserving input order and sampling."""
    validate_num_workers(num_workers)
    device = getattr(sim, "device", "cpu")
    if inputs.device.type != "cpu" or torch.device(device).type != "cpu":
        msg = "Process-based generation supports CPU simulators and inputs only."
        raise ValueError(msg)
    if len(inputs) == 0:
        return torch.empty((0, sim.out_dim)), inputs
    template = copy.copy(sim)
    # Do not pickle a previous split's potentially multi-GB result buffer.
    template.__dict__.pop("results_tensor", None)
    try:
        serialized = pickle.dumps(template)
    except (pickle.PickleError, TypeError, AttributeError) as error:
        msg = "Process-based generation requires a pickleable simulator."
        raise TypeError(msg) from error
    entropy = (
        random.SystemRandom().randrange(2**63) if random_seed is None else random_seed
    )
    seeds = [
        int(np.random.SeedSequence([entropy, index]).generate_state(1, np.uint64)[0])
        for index in range(len(inputs))
    ]
    old_threads = torch.get_num_threads()
    default_dtype = torch.get_default_dtype()
    results: Generator[np.ndarray | None, None, None] | None = None
    try:
        if num_workers == 1:
            torch.set_num_threads(1)
            results = (
                _run_sample(serialized, inputs[i : i + 1].numpy(), seed, allow_failures)
                for i, seed in enumerate(seeds)
            )
        else:
            results = _ordered_results(
                serialized, inputs, seeds, num_workers, allow_failures, default_dtype
            )
        output = None
        valid_indices = []
        for index, result in enumerate(
            tqdm(
                results,
                total=len(inputs),
                desc="Running simulations",
                disable=not sim.progress_bar,
                unit="sample",
            )
        ):
            if result is None:
                continue
            sample = torch.from_numpy(result)
            output = _prepare_output(output, sample, size=len(inputs), index=index)
            output[len(valid_indices)].copy_(sample[0])
            valid_indices.append(index)
        if output is None:
            return torch.empty((0, sim.out_dim)), inputs[:0]
        sim.results_tensor = output[: len(valid_indices)]
        return sim.results_tensor, inputs[valid_indices]
    finally:
        try:
            # Validation can stop iteration early; still close the process pool
            # even when the caller retains the resulting exception traceback.
            if results is not None:
                results.close()
        finally:
            if num_workers == 1:
                torch.set_num_threads(old_threads)
