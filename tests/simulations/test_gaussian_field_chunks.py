from __future__ import annotations

import pytest
import torch

from autosim.simulations.synthetic import GaussianField


@pytest.mark.parametrize("chunk_steps", [1, 3, 7, 20])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("rho", [0.0, 0.8, -0.6])
def test_chunks_preserve_eager_values_and_boundaries(
    chunk_steps: int, dtype: str, rho: float
) -> None:
    sim = GaussianField(
        height=5,
        width=6,
        n_channels=3,
        n_steps=7,
        rho=rho,
        spatial_length_scale=0.8,
        channel_correlation=0.4,
        dtype=dtype,
    )
    expected = sim.forward_samples_spatiotemporal(2, random_seed=11)
    before = torch.random.get_rng_state().clone()
    chunks = list(sim.iter_samples_spatiotemporal(2, chunk_steps, random_seed=11))
    assert torch.equal(before, torch.random.get_rng_state())
    assert [(chunk["trajectory_index"], chunk["start_step"]) for chunk in chunks] == [
        (trajectory, start)
        for trajectory in range(2)
        for start in range(0, 7, chunk_steps)
    ]
    for chunk in chunks:
        trajectory, start = chunk["trajectory_index"], chunk["start_step"]
        length = min(chunk_steps, 7 - start)
        assert chunk["data"].shape == (1, length, 5, 6, 3)
        assert torch.equal(
            chunk["data"],
            expected["data"][trajectory : trajectory + 1, start : start + length],
        )
        assert chunk["provenance"] == expected["provenance"]
        assert chunk["constant_scalars"] is None
        assert chunk["constant_fields"] is None


def test_chunks_with_fresh_seed_can_be_replayed() -> None:
    sim = GaussianField(height=3, width=5, n_channels=2, n_steps=5, rho=0.7)
    chunks = list(sim.iter_samples_spatiotemporal(1, chunk_steps=2))
    seed = chunks[0]["provenance"]["random_seed"]
    replay = sim.forward_samples_spatiotemporal(1, random_seed=seed)
    assert torch.equal(
        torch.cat([chunk["data"] for chunk in chunks], dim=1), replay["data"]
    )
    assert all(chunk["provenance"]["random_seed"] == seed for chunk in chunks)


def test_changing_a_consumed_chunk_does_not_change_future_frames() -> None:
    sim = GaussianField(height=3, width=5, n_channels=2, n_steps=6, rho=0.8)
    expected = sim.forward_samples_spatiotemporal(1, random_seed=23)
    chunks = sim.iter_samples_spatiotemporal(1, chunk_steps=2, random_seed=23)
    first = next(chunks)
    first["data"].zero_()
    remaining = list(chunks)
    assert torch.equal(
        torch.cat([chunk["data"] for chunk in remaining], dim=1),
        expected["data"][:, 2:],
    )


def test_chunk_size_guard_does_not_require_the_whole_dataset_to_fit() -> None:
    sim = GaussianField(
        height=3, width=5, n_channels=2, n_steps=7, max_output_bytes=2 * 3 * 5 * 2 * 4
    )
    with pytest.raises(ValueError, match="max_output_bytes"):
        sim.forward_samples_spatiotemporal(2, random_seed=7)
    chunks = list(sim.iter_samples_spatiotemporal(2, chunk_steps=2, random_seed=7))
    assert sum(chunk["data"].shape[1] for chunk in chunks) == 14
    assert all(
        chunk["data"].numel() * chunk["data"].element_size() <= sim.max_output_bytes
        for chunk in chunks
    )


def test_chunk_size_guard_precedes_random_allocation(monkeypatch) -> None:
    sim = GaussianField(max_output_bytes=1)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Random data must not be allocated over the byte limit")

    monkeypatch.setattr(torch, "randn", unexpected_allocation)
    with pytest.raises(ValueError, match="max_output_bytes"):
        next(sim.iter_samples_spatiotemporal(1, chunk_steps=2, random_seed=3))


@pytest.mark.parametrize("chunk_steps", [0, -1, True, 1.5])
def test_invalid_chunk_lengths(chunk_steps) -> None:
    with pytest.raises(ValueError, match="chunk_steps"):
        next(GaussianField().iter_samples_spatiotemporal(1, chunk_steps))


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_invalid_trajectory_counts(n) -> None:
    with pytest.raises(ValueError, match="n must"):
        next(GaussianField().iter_samples_spatiotemporal(n, chunk_steps=2))
