from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autosim.simulations.gray_scott import GrayScott


def _expected_snapshots(total_time: float, dt: float, snapshot_dt: float) -> int:
    num_steps = max(1, int(np.ceil(total_time / dt)))
    stride = max(1, int(np.round(snapshot_dt / dt)))
    count = 1 + num_steps // stride
    if num_steps % stride != 0:
        count += 1
    return count


def parse_args() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser(
        description=(
            "Generate one WELL-like Gray-Scott trajectory and save it as a compressed "
            "NumPy archive with data layout [time, x, y, channels]."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/gray_scott_well_like_gliders.npz"),
        help="Destination .npz file.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="gliders",
        choices=["gliders", "bubbles", "maze", "worms", "spirals", "spots"],
        help="Pattern preset for (F, k).",
    )
    parser.add_argument(
        "--initial-condition",
        type=str,
        default="gaussians",
        choices=["gaussians", "fourier", "mixed"],
        help="Initialization family used in the WELL generation scripts.",
    )
    parser.add_argument("--n", type=int, default=128, help="Grid size per direction.")
    parser.add_argument(
        "--t-max",
        type=float,
        default=10000.0,
        help="Final simulation time.",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation time step.")
    parser.add_argument(
        "--snapshot-dt",
        type=float,
        default=10.0,
        help="Storage interval for snapshots.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used by the simulator for initial-condition randomness.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed used for API sampling (mainly relevant when parameters vary).",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D103
    args = parse_args()

    simulator = GrayScott(
        pattern=args.pattern,
        return_timeseries=True,
        n=args.n,
        T=args.t_max,
        dt=args.dt,
        snapshot_dt=args.snapshot_dt,
        initial_condition=args.initial_condition,
        random_seed=args.random_seed,
    )

    x = simulator.sample_inputs(1, random_seed=args.sample_seed)
    y, x_valid = simulator.forward_batch(x, allow_failures=False)

    n_channels = 2
    steps = y.shape[1] // (n_channels * args.n * args.n)
    expected_steps = _expected_snapshots(args.t_max, args.dt, args.snapshot_dt)
    if steps != expected_steps:
        msg = f"Unexpected number of snapshots: got {steps}, expected {expected_steps}."
        raise RuntimeError(msg)

    channel_time_xy = y.reshape(1, n_channels, steps, args.n, args.n)[0].cpu().numpy()
    data = np.transpose(channel_time_xy, (1, 2, 3, 0)).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        data=data,
        constant_scalars=x_valid[0].cpu().numpy().astype(np.float32),
        param_names=np.array(simulator.param_names),
        pattern=np.array(args.pattern),
        initial_condition=np.array(args.initial_condition),
        n=np.array(args.n),
        t_max=np.array(args.t_max),
        dt=np.array(args.dt),
        snapshot_dt=np.array(args.snapshot_dt),
        random_seed=np.array(args.random_seed),
    )

    print("Saved", args.output)
    print("Data shape", data.shape, "(time, x, y, channels)")
    print(
        "Parameters",
        dict(zip(simulator.param_names, x_valid[0].tolist(), strict=False)),
    )


if __name__ == "__main__":
    main()
