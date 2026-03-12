from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

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


def instantiate_target(target: str, kwargs: dict[str, Any]) -> Any:
    cls = load_object(target)
    return cls(**kwargs)


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x_f = x.float()
    return {
        "min": float(x_f.min().item()),
        "max": float(x_f.max().item()),
        "mean": float(x_f.mean().item()),
        "std": float(x_f.std(unbiased=False).item()),
        "l2": float(torch.linalg.vector_norm(x_f).item()),
    }


def delta_stats(data: torch.Tensor) -> dict[str, Any]:
    # data: [batch,time,x,y,channels]
    if data.ndim != 5:
        raise ValueError(f"Expected data.ndim==5, got shape={tuple(data.shape)}")
    if data.shape[1] < 2:
        return {"n_time": int(data.shape[1]), "channels": []}

    deltas = data[:, 1:] - data[:, :-1]  # [b,t-1,x,y,c]
    eps = 1e-12
    out_channels: list[dict[str, float]] = []
    for c in range(data.shape[-1]):
        dc = deltas[..., c]
        xc = data[..., c]
        max_abs_delta = float(dc.abs().max().item())
        max_abs_state = float(xc.abs().max().item())
        out_channels.append(
            {
                "delta_abs_mean": float(dc.abs().mean().item()),
                "delta_abs_std": float(dc.abs().std(unbiased=False).item()),
                "delta_abs_max": max_abs_delta,
                "state_abs_max": max_abs_state,
                "delta_to_state_max_ratio": float(max_abs_delta / (max_abs_state + eps)),
            }
        )

    return {"n_time": int(data.shape[1]), "channels": out_channels}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    summary: dict[str, Any]


def validate_payload(
    payload: dict[str, Any],
    *,
    residual_fn: Callable[..., dict[str, float]] | None = None,
    diagnostics_fn: Callable[..., dict[str, float]] | None = None,
    simulator: Any | None = None,
    simulator_kwargs: dict[str, Any] | None = None,
) -> ValidationResult:
    if "data" not in payload:
        return ValidationResult(ok=False, summary={"error": "payload missing key 'data'"})
    data = payload["data"]
    if not isinstance(data, torch.Tensor):
        return ValidationResult(
            ok=False,
            summary={"error": f"payload['data'] is not a torch.Tensor: {type(data)}"},
        )

    ok = True
    summary: dict[str, Any] = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "finite": bool(torch.isfinite(data).all().item()),
        "stats": tensor_stats(data),
        "delta": delta_stats(data),
    }
    if not summary["finite"]:
        ok = False

    if residual_fn is not None:
        try:
            summary["residual"] = residual_fn(
                payload, simulator=simulator, simulator_kwargs=simulator_kwargs
            )
        except Exception as e:  # noqa: BLE001
            ok = False
            summary["residual_error"] = f"{type(e).__name__}: {e}"

    if diagnostics_fn is not None:
        try:
            summary["diagnostics"] = diagnostics_fn(
                payload, simulator=simulator, simulator_kwargs=simulator_kwargs
            )
        except Exception as e:  # noqa: BLE001
            ok = False
            summary["diagnostics_error"] = f"{type(e).__name__}: {e}"

    return ValidationResult(ok=ok, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate autosim SpatioTemporalSimulator rollouts."
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
        help="Constructor kwargs as key=value (e.g. return_timeseries=true n=16)",
    )
    parser.add_argument("--n", type=int, default=1, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--ensure-exact-n",
        action="store_true",
        help="Retry failed sims until exactly n successes.",
    )
    parser.add_argument(
        "--residual",
        default=None,
        help="Optional dotted callable for PDE residual metrics (module:function).",
    )
    parser.add_argument(
        "--diagnostics",
        default=None,
        help="Optional dotted callable for diagnostics/invariants (module:function).",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional output path to write JSON summary.",
    )
    args = parser.parse_args()

    sim_kwargs = parse_kwargs(args.kwargs)
    sim = instantiate_target(args.target, sim_kwargs)

    residual_fn = load_object(args.residual) if args.residual else None
    diagnostics_fn = load_object(args.diagnostics) if args.diagnostics else None

    t0 = time.perf_counter()
    payload = sim.forward_samples_spatiotemporal(
        n=args.n,
        random_seed=args.seed,
        ensure_exact_n=args.ensure_exact_n,
    )
    elapsed_s = time.perf_counter() - t0

    result = validate_payload(
        payload,
        residual_fn=residual_fn,
        diagnostics_fn=diagnostics_fn,
        simulator=sim,
        simulator_kwargs=sim_kwargs,
    )
    result_dict: dict[str, Any] = {
        "ok": result.ok,
        "simulator_target": args.target,
        "simulator_kwargs": sim_kwargs,
        "n": args.n,
        "seed": args.seed,
        "elapsed_s": float(elapsed_s),
        "elapsed_per_sample_s": float(elapsed_s / max(1, args.n)),
        "validation": result.summary,
    }

    finite = result.summary.get("finite", False)
    shape = result.summary.get("shape", None)
    max_delta_ratio = None
    try:
        channels = result.summary["delta"]["channels"]
        if channels:
            max_delta_ratio = max(ch["delta_to_state_max_ratio"] for ch in channels)
    except Exception:  # noqa: BLE001
        max_delta_ratio = None

    line = {
        "ok": result.ok,
        "finite": finite,
        "shape": shape,
        "elapsed_s": round(elapsed_s, 6),
        "max_delta_ratio": None if max_delta_ratio is None else float(max_delta_ratio),
    }
    print(json.dumps(line))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

