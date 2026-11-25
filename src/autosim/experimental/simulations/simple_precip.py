"""Idealised stochastic precipitation simulator."""

import math

import numpy as np
import torch
from scipy.fft import ifft, ifft2

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
    accumulation uses the trapezoidal rule on a sub-step of approximately
    :math:`T_{eff}/m`.
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
        output_indices: list[int] | None = None,
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
            output_names: Optional labels for the selected fields. When supplied
                alone, label the first matching number of fields in canonical order.
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
            output_indices: Select one or more fields in the requested order
                from [accumulation, rain, phi, envelope]. For rain alone use [1].
                Defaults to all four fields when no labels are supplied.
        """
        if parameters_range is None:
            parameters_range = {k: (v, v) for k, v in DEFAULT_PARAMETERS.items()}
        for name, (low, high) in parameters_range.items():
            if not -math.inf < low <= high < math.inf:
                raise ValueError(f"Invalid bounds for {name}: ({low}, {high}).")
        field_names = ("accumulation", "rain", "phi", "envelope")
        if output_indices is None:
            count = len(field_names) if output_names is None else len(output_names)
            output_indices = list(range(count))
        if (
            not output_indices
            or any(
                isinstance(i, bool)
                or not isinstance(i, int | np.integer)
                or i not in range(4)
                for i in output_indices
            )
            or len(set(output_indices)) != len(output_indices)
        ):
            msg = "output_indices must contain distinct channel indices from 0 to 3."
            raise ValueError(msg)
        self.output_indices = tuple(output_indices)
        if output_names is None:
            output_names = [field_names[i] for i in output_indices]
        if len(output_names) != len(output_indices) or len(set(output_names)) != len(
            output_names
        ):
            msg = "Provide one distinct output name per selected channel."
            raise ValueError(msg)
        lower = {name: low for name, (low, _) in parameters_range.items()}
        n_out = _validate_setup(
            lower, (Nx, Ny), T, dt_out, Lx=Lx, Ly=Ly, m=m, dt_sub=dt_sub
        )

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

        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        fields = simulate_simple_precipitation(
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
        fields = tuple(fields[i] for i in self.output_indices)
        if not self.return_timeseries:
            fields = tuple(f[-1:] for f in fields)

        concat_array = np.concatenate([f.ravel() for f in fields])

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
            ``(n, n_times, Nx, Ny, len(output_names))``, together with
            constant scalars and constant fields.
        """
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
            num_workers=num_workers,
        )
        n_times = self.n_out if self.return_timeseries else 1
        data = y.reshape(y.shape[0], self.out_dim, n_times, self.Nx, self.Ny).permute(
            0, 2, 3, 4, 1
        )
        return {
            "data": data,
            "constant_scalars": x,
            "constant_fields": None,
        }


def _validate_setup(
    params: dict[str, float],
    shape: tuple[int, int],
    T: float,
    dt_out: float,
    **scales: float | None,
) -> int:
    """Check the simulation inputs and return the number of output intervals."""
    if params.keys() != DEFAULT_PARAMETERS.keys():
        raise ValueError(
            f"Expected parameters {list(DEFAULT_PARAMETERS)}, got {list(params)}."
        )
    if not all(map(math.isfinite, params.values())) or params["s"] < 0:
        msg = "Model parameters must be finite and s must be non-negative."
        raise ValueError(msg)
    for name, size in zip(("Nx", "Ny"), shape, strict=True):
        if isinstance(size, bool) or not isinstance(size, int | np.integer) or size < 1:
            raise ValueError(f"{name} must be a positive integer.")
    scales.update(T=T, dt_out=dt_out)
    for name in ("ell", "tau_g", "L_env"):
        scales[name] = params[name]
    for name, value in scales.items():
        if name == "dt_sub" and value is None:
            continue
        if value is None or not 0 < value < math.inf:
            raise ValueError(f"{name} must be finite and positive.")
    n_out = round(T / dt_out)
    if n_out < 1 or not math.isclose(n_out * dt_out, T):
        raise ValueError(f"T ({T}) must be a whole multiple of dt_out ({dt_out}).")
    return n_out


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
) -> tuple[NumpyLike, ...]:
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
        Tuple of ``(accumulation, rain, phi, envelope)``, each of shape
        ``(T / dt_out, Nx, Ny)``.
        ``accumulation[n]`` is the rain integrated over the interval ending at
        ``(n + 1) * dt_out`` and ``rain[n]`` is the instantaneous rate at that time.
    """
    p = params
    n_out = _validate_setup(p, (Nx, Ny), T, dt_out, Lx=Lx, Ly=Ly, m=m, dt_sub=dt_sub)
    n_sub, h = substep(p, dt_out, m, dt_sub)
    Nt = n_out * n_sub + 1  # includes t = 0 and t = T
    t = np.arange(Nt) * h
    pad = math.ceil(5 * p["tau_g"] / h)
    Nt_pad = Nt + 2 * pad

    w = 2 * np.pi * np.fft.fftfreq(Nt_pad, h)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, Ly / Ny)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)
    spatial_power = np.exp(-(kx**2 + ky[:, None] ** 2) * p["ell"] ** 2 / 2)
    F = np.exp(-((w * p["tau_g"]) ** 2) / 2)
    norm = p["s"] / np.sqrt(spatial_power.mean() * F.mean())

    # A fractional shift rotates both quadratures, including at Nyquist.
    # Unit-variance real and imaginary draws preserve s² after taking the real part.
    spec = np.empty((Nt_pad, Ny, Nx), dtype=np.complex128)
    spec.real = rng.standard_normal(spec.shape)
    spec.imag = rng.standard_normal(spec.shape)
    spec *= np.sqrt(F)[:, None, None]
    spec *= norm * np.sqrt(spatial_power)
    spec = ifft(spec, axis=0, norm="ortho", overwrite_x=True)[pad : pad + Nt]
    phase_x = np.exp(-1j * p["cx"] * t[:, None] * kx)
    phase_y = np.exp(-1j * p["cy"] * t[:, None] * ky)
    spec *= phase_x[:, None, :]
    spec *= phase_y[:, :, None]
    phi = np.asarray(
        ifft2(spec, norm="ortho", overwrite_x=True), dtype=np.complex128
    ).real

    x = np.arange(Nx) * Lx / Nx
    y = np.arange(Ny) * Ly / Ny
    dX = _periodic(x[None, None, :] - p["x0"] - p["cex"] * t[:, None, None], Lx)
    dY = _periodic(y[None, :, None] - p["y0"] - p["cey"] * t[:, None, None], Ly)
    envelope = np.exp(-(dX**2 + dY**2) / (2 * p["L_env"] ** 2))
    rain = np.maximum(phi - (p["theta0"] - p["dtheta"] * envelope), 0.0)

    trapezoids = 0.5 * h * (rain[:-1] + rain[1:])
    accumulation = trapezoids.reshape(n_out, n_sub, Ny, Nx).sum(axis=1)
    ends = slice(n_sub, None, n_sub)
    fields = (accumulation, rain[ends], phi[ends], envelope[ends])

    return tuple(f.transpose(0, 2, 1) for f in fields)


def _periodic(d: NumpyLike, L: float) -> NumpyLike:
    """Map displacements onto [-L/2, L/2) on a periodic domain of length L."""
    return (d + L / 2) % L - L / 2
