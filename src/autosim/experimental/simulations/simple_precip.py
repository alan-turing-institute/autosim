"""Idealised stochastic precipitation simulator."""

import math

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import NumpyLike, TensorLike

DEFAULT_PARAMETERS = {
    "s": 1.0,
    "ell": 4.0,
    "tau_g": 9.0,
    "cx": 1.0,
    "cy": 1.0,
    "cex": 1.0,
    "cey": 1.0,
    "x0": 10.0,
    "y0": 10.0,
    "L_env": 20.0,
    "theta0": 4.0,
    "dtheta": 4.0,
}


class SimplePrecipitation(SpatioTemporalSimulator):
    r"""Simulate a simple model of stochastic precipitation on a periodic domain.

    A latent Gaussian random field :math:`\phi` is advected at velocity
    :math:`\mathbf{c}` and, in the moving frame, has a squared-exponential
    covariance in space (length :math:`\ell`) and time (:math:`\tau_g`). Rain is
    :math:`R = \max(\phi - \theta, 0)`, where the threshold is lowered inside a
    Gaussian envelope moving at :math:`\mathbf{c}_e`.

    Outputs at :math:`t_n = n\,\Delta t_{out}` are the accumulation over
    :math:`(t_{n-1}, t_n]` and the instantaneous rain at :math:`t_n`. The
    accumulation uses the trapezoidal rule on a sub-step of approximately :math:`T_{eff}/m`.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = True,
        log_level: str = "progress_bar",
        Lx: float = 100.0,
        Ly: float = 100.0,
        Nx: int = 64,
        Ny: int = 64,
        T: float = 100.0,
        dt_out: float = 5.0,
        m: float = 3.0,
        dt_sub: float | None = None,
    ):
        """Initialise the SimplePrecipitation simulator.

        Args:
            parameters_range: Dictionary mapping each of the twelve model parameters
                to its (min, max) range.
                The parameters are ``s`` (standard deviation of the latent field),
                ``ell`` (spatial correlation length), ``tau_g`` (correlation time in
                the moving frame), ``cx`` and ``cy`` (velocity of the field), ``cex``
                and ``cey`` (velocity of the envelope), ``x0`` and ``y0`` (initial
                centre of the envelope), ``L_env`` (width of the envelope),
                ``theta0`` (threshold outside the envelope) and ``dtheta`` (drop in
                threshold at its centre).
            output_names: Names of the two output channels.
                Defaults to ``["accumulation", "rain"]``.
            return_timeseries: Whether to return every output time or only the last.
                Defaults to True.
            log_level: Logging level for the simulator, as for other simulators.
            Lx: Domain length in x.
            Ly: Domain length in y.
            Nx: Number of grid points in x.
            Ny: Number of grid points in y.
            T: Length of the run. Must be a whole multiple of ``dt_out``.
            dt_out: Accumulation period, which is also the output interval.
            m: Number of sub-steps per effective timescale :math:`T_{eff}`.
            dt_sub: Sub-step to use instead of the one derived from ``m``. It is
                rounded down so that a whole number of sub-steps fits in
                ``dt_out``.
        """
        if parameters_range is None:
            parameters_range = {k: (v, v) for k, v in DEFAULT_PARAMETERS.items()}
        if set(parameters_range) != set(DEFAULT_PARAMETERS):
            msg = (
                f"parameters_range must contain exactly {list(DEFAULT_PARAMETERS)}, "
                f"got {list(parameters_range)}."
            )
            raise ValueError(msg)
        if output_names is None:
            output_names = ["accumulation", "rain"]
        if len(output_names) != 2:
            raise ValueError("output_names must name exactly two channels.")
        n_out = round(T / dt_out)
        if n_out < 1 or not math.isclose(n_out * dt_out, T):
            raise ValueError(f"T ({T}) must be a whole multiple of dt_out ({dt_out}).")

        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.T = T
        self.dt_out = dt_out
        self.n_out = n_out
        self.m = m
        self.dt_sub = dt_sub

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )
        params = dict(
            zip(self.param_names, x.cpu().numpy()[0].astype(float), strict=True)
        )

        grid_spacing = max(self.Lx / self.Nx, self.Ly / self.Ny)

        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        accumulation, rain = simulate_simple_precipitation(
            params,
            rng,
            self.Lx,
            self.Ly,
            self.Nx,
            self.Ny,
            self.T,
            self.dt_out,
            self.m,
            self.dt_sub,
        )
        if not self.return_timeseries:
            accumulation, rain = accumulation[-1:], rain[-1:]

        concat_array = np.concatenate([accumulation.ravel(), rain.ravel()])
        return torch.tensor(concat_array, dtype=torch.float32).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
        *,
        num_workers: int | None = None,
    ) -> dict:
        """Generate samples and reshape them to spatiotemporal format.

        Args:
            n: Number of samples to generate. With every parameter fixed, these are
                independent members of an ensemble.
            random_seed: Random seed for reproducibility. Defaults to None.
            ensure_exact_n: Whether to resample failed trajectories until exactly
                ``n`` succeed.
            num_workers: Optional independent-sample CPU execution.

        Returns:
            A dictionary whose ``data`` entry has shape
            ``(n, n_times, Nx, Ny, 2)``, with accumulation and rain as the two
            variables, together with the constant scalars and constant fields.
        """
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
            num_workers=num_workers,
        )
        n_times = self.n_out if self.return_timeseries else 1
        data = y.reshape(y.shape[0], 2, n_times, self.Nx, self.Ny).permute(
            0, 2, 3, 4, 1
        )
        return {
            "data": data,
            "constant_scalars": x,
            "constant_fields": None,
        }


def substep(params: dict[str, float], dt_out: float, m: float, dt_sub: float | None):
    """Return the number of sub-steps per output interval and their length.

    Args:
        params: Model parameters, as described in :class:`SimplePrecipitation`.
        dt_out: Accumulation period.
        m: Number of sub-steps per effective timescale.
        dt_sub: Optional sub-step overriding the one derived from ``m``.

    Returns:
        Tuple of ``(n_sub, h)``, with ``n_sub * h == dt_out``.
    """
    if dt_sub is None:
        rate2 = (
            params["tau_g"] ** -2
            + (params["cx"] ** 2 + params["cy"] ** 2) / params["ell"] ** 2
            + (params["cex"] ** 2 + params["cey"] ** 2) / params["L_env"] ** 2
        )
        dt_sub = 1.0 / (m * math.sqrt(rate2))
    n_sub = max(1, math.ceil(dt_out / dt_sub - 1e-9))
    return n_sub, dt_out / n_sub


def simulate_simple_precipitation(
    params: dict[str, float],
    rng: np.random.Generator,
    Lx: float = 100.0,
    Ly: float = 100.0,
    Nx: int = 64,
    Ny: int = 64,
    T: float = 100.0,
    dt_out: float = 5.0,
    m: float = 3.0,
    dt_sub: float | None = None,
) -> tuple[NumpyLike, NumpyLike]:
    """Simulate one realisation of the precipitation model.

    Args:
        params: Model parameters, as described in :class:`SimplePrecipitation`.
        rng: Random number generator supplying the noise.
        Lx: Domain length in x.
        Ly: Domain length in y.
        Nx: Number of grid points in x.
        Ny: Number of grid points in y.
        T: Length of the run, a whole multiple of ``dt_out``.
        dt_out: Accumulation period.
        m: Number of sub-steps per effective timescale.
        dt_sub: Optional sub-step overriding the one derived from ``m``.

    Returns:
        Tuple of ``(accumulation, rain)``, each of shape ``(T / dt_out, Nx, Ny)``.
        ``accumulation[n]`` is the rain integrated over the interval ending at
        ``(n + 1) * dt_out`` and ``rain[n]`` is the instantaneous rate at that time.
    """
    p = params
    n_out = round(T / dt_out)
    n_sub, h = substep(p, dt_out, m, dt_sub)
    Nt = n_out * n_sub + 1  # includes t = 0 and t = T
    t = np.arange(Nt) * h
    pad = math.ceil(5 * p["tau_g"] / h)
    Nt_pad = Nt + 2 * pad

    w = 2 * np.pi * np.fft.fftfreq(Nt_pad, h)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, Ly / Ny)
    kx = 2 * np.pi * np.fft.rfftfreq(Nx, Lx / Nx)
    kx_full = 2 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    KYf, KXf = np.meshgrid(ky, kx_full, indexing="ij")

    def spatial_spectrum(kx_, ky_):
        return np.exp(-(kx_**2 + ky_**2) * p["ell"] ** 2 / 2)

    F = np.exp(-((w * p["tau_g"]) ** 2) / 2)
    norm = p["s"] / np.sqrt(spatial_spectrum(KXf, KYf).mean() * F.mean())

    spec = np.fft.rfftn(rng.standard_normal((Nt_pad, Ny, Nx)))
    spec *= np.sqrt(F)[:, None, None]
    spec *= norm * np.sqrt(spatial_spectrum(KX, KY))[None]
    # moving frame with padding removed
    spec = np.fft.ifft(spec, axis=0)[pad : pad + Nt]
    spec *= np.exp(-1j * (KX * p["cx"] + KY * p["cy"])[None] * t[:, None, None])
    phi = np.fft.irfft2(spec, s=(Ny, Nx))  # (Nt, Ny, Nx)

    x = np.arange(Nx) * Lx / Nx
    y = np.arange(Ny) * Ly / Ny
    dX = _periodic(x[None, None, :] - p["x0"] - p["cex"] * t[:, None, None], Lx)
    dY = _periodic(y[None, :, None] - p["y0"] - p["cey"] * t[:, None, None], Ly)
    envelope = np.exp(-(dX**2 + dY**2) / (2 * p["L_env"] ** 2))
    rain = np.maximum(phi - (p["theta0"] - p["dtheta"] * envelope), 0.0)

    trapezoids = 0.5 * h * (rain[:-1] + rain[1:])
    accumulation = trapezoids.reshape(n_out, n_sub, Ny, Nx).sum(axis=1)
    rain_end = rain[n_sub::n_sub]

    return accumulation.transpose(0, 2, 1), rain_end.transpose(0, 2, 1)


def _periodic(d: NumpyLike, L: float) -> NumpyLike:
    """Map displacements onto [-L/2, L/2) on a periodic domain of length L."""
    return (d + L / 2) % L - L / 2
