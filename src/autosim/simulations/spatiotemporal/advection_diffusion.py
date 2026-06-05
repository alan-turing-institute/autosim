"""Vorticity-only advection-diffusion simulator."""

from __future__ import annotations

from .advection_diffusion_multichannel import AdvectionDiffusionMultichannel


class AdvectionDiffusion(AdvectionDiffusionMultichannel):
    r"""Vorticity-only advection-diffusion simulator.

    This is the single-channel variant of
    :class:`autosim.simulations.spatiotemporal.AdvectionDiffusionMultichannel`.
    It returns only the vorticity channel.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 50,
        L: float = 10.0,
        T: float = 80.0,
        dt: float = 0.25,
        integrator_kwargs: dict | None = None,
    ) -> None:
        """Initialize the vorticity-only advection-diffusion simulator."""
        super().__init__(
            parameters_range=parameters_range,
            output_names=output_names,
            output_indices=[0],
            return_timeseries=return_timeseries,
            log_level=log_level,
            n=n,
            L=L,
            T=T,
            dt=dt,
            integrator_kwargs=integrator_kwargs,
        )
