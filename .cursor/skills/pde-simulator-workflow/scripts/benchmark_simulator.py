from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from typing import Any


def _parse_scalar(text: str) -> Any:
    lower = text.strip().lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_kwargs(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = _parse_scalar(v)
    return out


def load_object(dotted: str) -> Any:
    if ":" in dotted:
        mod, attr = dotted.split(":", 1)
    elif "." in dotted:
        mod, attr = dotted.rsplit(".", 1)
    else:
        raise ValueError(
            "Expected 'module:attr' or 'module.attr' for dotted reference, "
            f"got {dotted!r}"
        )
    module = importlib.import_module(mod)
    return getattr(module, attr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark autosim SpatioTemporalSimulator forward_samples_spatiotemporal."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Simulator class dotted path (e.g. autosim.simulations.AdvectionDiffusion)",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        help="Constructor kwargs as key=value.",
    )
    parser.add_argument("--n", type=int, default=1, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs.")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs.")
    parser.add_argument(
        "--ensure-exact-n",
        action="store_true",
        help="Retry failed sims until exactly n successes.",
    )
    args = parser.parse_args()

    sim_kwargs = parse_kwargs(args.kwargs)
    cls = load_object(args.target)
    sim = cls(**sim_kwargs)

    # Warmup
    for i in range(max(0, args.warmup)):
        sim.forward_samples_spatiotemporal(
            n=args.n,
            random_seed=args.seed + i,
            ensure_exact_n=args.ensure_exact_n,
        )

    times: list[float] = []
    for i in range(max(1, args.runs)):
        t0 = time.perf_counter()
        sim.forward_samples_spatiotemporal(
            n=args.n,
            random_seed=args.seed + 10_000 + i,
            ensure_exact_n=args.ensure_exact_n,
        )
        times.append(time.perf_counter() - t0)

    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[max(0, math.ceil(0.95 * len(times_sorted)) - 1)]

    out = {
        "simulator_target": args.target,
        "simulator_kwargs": sim_kwargs,
        "n": args.n,
        "warmup": args.warmup,
        "runs": args.runs,
        "p50_s": float(p50),
        "p95_s": float(p95),
        "mean_s": float(sum(times) / len(times)),
        "per_sample_mean_s": float((sum(times) / len(times)) / max(1, args.n)),
        "all_s": [float(t) for t in times],
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

