from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import torch


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
        description="Generate deterministic rollout fixtures for PDE simulators."
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
    parser.add_argument(
        "--ensure-exact-n",
        action="store_true",
        help="Retry failed sims until exactly n successes.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .pt path to write (directories created).",
    )
    args = parser.parse_args()

    sim_kwargs = parse_kwargs(args.kwargs)
    cls = load_object(args.target)
    sim = cls(**sim_kwargs)
    payload = sim.forward_samples_spatiotemporal(
        n=args.n,
        random_seed=args.seed,
        ensure_exact_n=args.ensure_exact_n,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "simulator_target": args.target,
            "simulator_kwargs": sim_kwargs,
            "n": args.n,
            "seed": args.seed,
            "data": payload.get("data"),
            "constant_scalars": payload.get("constant_scalars"),
            "constant_fields": payload.get("constant_fields"),
        },
        out_path,
    )
    print(f"Wrote fixture: {out_path}")


if __name__ == "__main__":
    main()

