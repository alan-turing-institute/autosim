"""Export OU precipitation ensembles from saved test frames in the SWE layout.

Example from the repository root:
    uv run python -m autosim.experimental.precipitation_redraws DATASET OUTPUT \
        --anchor 0 20 --anchor 0 100 --members 32 --leads 4

In AutoCast, ++datamodule.channel_idxs=[0,1] selects accumulation and rain
from the original four-channel dataset for both inputs and targets.
In each reference/anchor_NNN.pt, states[:, 0] is the shared starting frame
and states[:, 1:] contains the futures in (member, lead, x, y, channel) order.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autosim.experimental.simulations.simple_precip import SimplePrecipitation


def export_redraws(
    dataset: Path,
    output: Path,
    pairs: list[tuple[int, int]],
    members: int,
    leads: int,
    seed: int,
) -> None:
    """Write one shared anchor and observed future per pair, and raw OU members."""
    cfg = OmegaConf.load(dataset / "resolved_config.yaml")
    sim = instantiate(cfg.simulator)
    if not (
        isinstance(sim, SimplePrecipitation)
        and sim.temporal_model == "ou"
        and sim.return_timeseries
        and 2 in sim.output_indices
    ):
        msg = "OU precipitation redraws require a timeseries containing phi."
        raise ValueError(msg)
    if not pairs or members < 1 or leads < 1 or seed < 0:
        msg = "Provide at least one anchor, member and lead, and a nonnegative seed."
        raise ValueError(msg)
    if output.exists():
        raise FileExistsError(f"Redraw output already exists: {output}")
    test = torch.load(
        dataset / "test/data.pt", map_location="cpu", mmap=True, weights_only=True
    )
    fields = test["data"]
    phi_channel = sim.output_indices.index(2)
    for row, frame in pairs:
        if not (0 <= row < len(fields) and 0 <= frame < fields.shape[1] - leads):
            raise ValueError(
                f"Invalid test anchor or insufficient future: {row, frame}"
            )
    rows, frames = zip(*pairs, strict=True)
    anchors = fields[rows, frames]
    scalars = test["constant_scalars"][list(rows)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        bundle = Path(tmp) / "redraws"
        (bundle / "reference").mkdir(parents=True)
        for i, (row, frame) in enumerate(pairs):
            start = (frame + 1) * sim.dt_out
            states = anchors[i].new_empty((members, leads + 1, *anchors[i].shape))
            states[:, 0] = anchors[i]
            seeds = np.random.SeedSequence([seed, row, frame]).generate_state(members)
            for member, member_seed in enumerate(seeds):
                future = sim.simulate_trajectory(
                    scalars[i],
                    np.random.default_rng(member_seed),
                    n_frames=leads,
                    initial_phi=anchors[i, ..., phi_channel],
                    start_time=start,
                )
                states[member, 1:] = torch.from_numpy(np.stack(future, axis=-1))
            torch.save(
                {"states": states, "seeds": seeds.tolist()},
                bundle / "reference" / f"anchor_{i:03d}.pt",
            )
        torch.save(
            {
                "pairs": pairs,
                "states": anchors,
                "observed_futures": torch.stack(
                    [fields[r, f + 1 : f + 1 + leads] for r, f in pairs]
                ),
                "constant_scalars": scalars,
                "param_names": sim.param_names,
                "channel_names": list(sim.output_names),
                "output_indices": list(sim.output_indices),
                "start_times": [(f + 1) * sim.dt_out for f in frames],
                "dt_out": sim.dt_out,
            },
            bundle / "anchors.pt",
        )
        bundle.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--anchor",
        type=int,
        nargs=2,
        action="append",
        required=True,
        metavar=("TEST_ROW", "FRAME"),
        help="Zero-based test trajectory and frame.",
    )
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--leads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    export_redraws(
        args.dataset,
        args.output,
        [tuple(pair) for pair in args.anchor],
        args.members,
        args.leads,
        args.seed,
    )
