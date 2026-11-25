"""Idealised stochastic precipitation simulator."""

import math

import numpy as np
import torch
from scipy.fft import fft2, ifft, ifft2

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
    covariance in space (length :math:`\ell`). Its temporal covariance is
    squared exponential by default, or exponential for the Ornstein-Uhlenbeck
    (OU) model, with timescale :math:`\tau_g`. Rain is
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
        *,
        temporal_model: str = "gp",
        shared_advection_speed_range: tuple[float, float] | None = None,
        shared_advection_heading_deg: float | None = None,
        broad_front: bool = False,
        front_cell_aspect_ratio: float = 4.0,
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
            temporal_model: Either ``"gp"`` (squared-exponential time covariance)
                or ``"ou"`` (Markov evolution of the saved latent field).
            shared_advection_speed_range: Sample a common velocity for the
                field and envelope, replacing the four Cartesian velocity ranges.
            shared_advection_heading_deg: Fix the shared heading in degrees
                counterclockwise from +x. By default the heading is uniform.
            broad_front: Stretch the 3:1 envelope and rain cells across travel.
            front_cell_aspect_ratio: Along-front to cross-front correlation
                length ratio. The cross-front length stays at half of ``ell``.
        """
        if parameters_range is None:
            parameters_range = {k: (v, v) for k, v in DEFAULT_PARAMETERS.items()}
        self.shared_advection_speed_range = shared_advection_speed_range
        self.shared_advection_heading_deg = shared_advection_heading_deg
        self.broad_front = broad_front
        self.front_cell_aspect_ratio = front_cell_aspect_ratio
        if shared_advection_heading_deg is not None and (
            shared_advection_speed_range is None
            or not math.isfinite(shared_advection_heading_deg)
        ):
            msg = "A fixed heading must be finite and use shared_advection_speed_range."
            raise ValueError(msg)
        if shared_advection_speed_range is not None:
            low, high = shared_advection_speed_range
            if not 0 <= low <= high < math.inf:
                msg = (
                    "shared_advection_speed_range must satisfy 0 <= low <= high < inf."
                )
                raise ValueError(msg)
            parameters_range = dict(parameters_range)
            parameters_range.update(
                dict.fromkeys(("cx", "cy", "cex", "cey"), (-high, high))
            )
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
            lower,
            (Nx, Ny),
            T,
            dt_out,
            temporal_model,
            broad_front,
            front_cell_aspect_ratio,
            Lx=Lx,
            Ly=Ly,
            m=m,
            dt_sub=dt_sub,
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
        self.temporal_model = temporal_model

    def sample_inputs(
        self, n_samples: int, random_seed: int | None = None, method: str = "lhs"
    ) -> TensorLike:
        """Draw trajectory parameters and store velocities as signed components."""
        x = super().sample_inputs(n_samples, random_seed, method)
        if self.shared_advection_speed_range is None:
            return x
        low, high = self.shared_advection_speed_range
        if high == 0:
            return x
        cx, cy, cex, cey = (
            self.get_parameter_idx(k) for k in ("cx", "cy", "cex", "cey")
        )
        # Reuse two coordinates of the joint LHS/Sobol design for uniform
        # speed and heading, then store the physical velocities for restarts.
        speed = low + (high - low) * ((x[:, cx] / high + 1) / 2)
        heading = math.pi * (x[:, cy] / high)
        if self.shared_advection_heading_deg is not None:
            heading.fill_(math.radians(self.shared_advection_heading_deg % 360))
        x[:, cx] = x[:, cex] = speed * heading.cos()
        x[:, cy] = x[:, cey] = speed * heading.sin()
        return x

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        fields = self.simulate_trajectory(x[0], rng)
        if not self.return_timeseries:
            fields = tuple(f[-1:] for f in fields)

        concat_array = np.concatenate([f.ravel() for f in fields])
        return torch.tensor(concat_array, dtype=torch.float32).reshape(1, -1)

    def simulate_trajectory(
        self,
        constant_scalars: TensorLike,
        rng: np.random.Generator,
        *,
        n_frames: int | None = None,
        initial_phi: NumpyLike | TensorLike | None = None,
        start_time: float | None = None,
    ) -> tuple[NumpyLike, ...]:
        """Draw one trajectory using a saved sample's parameter row.

        For conditional OU draws, reuse the sample's ``phi`` channel and
        ``constant_scalars`` row with fresh noise from ``rng``. In a full
        timeseries, frame k has ``start_time=(k + 1) * self.dt_out``. A
        final-only frame is at ``self.T``. Each returned channel contains
        ``n_frames`` future intervals (default ``self.n_out``).
        """
        if n_frames is not None and (
            isinstance(n_frames, bool)
            or not isinstance(n_frames, int | np.integer)
            or n_frames < 1
        ):
            msg = "n_frames must be a positive integer."
            raise ValueError(msg)
        if constant_scalars.shape != (self.in_dim,):
            raise ValueError(f"constant_scalars must have shape ({self.in_dim},).")
        if isinstance(initial_phi, torch.Tensor):
            initial_phi = initial_phi.detach().cpu().numpy()
        params = dict(zip(self.param_names, constant_scalars.tolist(), strict=True))
        fields = simulate_simple_precipitation(
            params,
            rng,
            self.Lx,
            self.Ly,
            self.Nx,
            self.Ny,
            self.T if n_frames is None else n_frames * self.dt_out,
            self.dt_out,
            self.m,
            self.dt_sub,
            temporal_model=self.temporal_model,
            broad_front=self.broad_front,
            front_cell_aspect_ratio=self.front_cell_aspect_ratio,
            initial_phi=initial_phi,
            start_time=start_time,
        )
        return tuple(fields[i] for i in self.output_indices)

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
    temporal_model: str,
    broad_front: bool,
    front_cell_aspect_ratio: float,
    **scales: float | None,
) -> int:
    """Check the simulation inputs and return the number of output intervals."""
    if temporal_model not in ("gp", "ou"):
        msg = "temporal_model must be 'gp' or 'ou'."
        raise ValueError(msg)
    if broad_front and not 1 <= front_cell_aspect_ratio < math.inf:
        msg = "front_cell_aspect_ratio must be finite and at least one."
        raise ValueError(msg)
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


def substep(
    params: dict[str, float],
    dt_out: float,
    m: float,
    dt_sub: float | None,
    *,
    broad_front: bool = False,
):
    """Return the number of sub-steps per output interval and their length.

    Args:
        params: Model parameters, as described in :class:`SimplePrecipitation`.
        dt_out: Accumulation period.
        m: Number of sub-steps per effective timescale.
        dt_sub: Optional sub-step overriding the one derived from ``m``.
        broad_front: Resolve the shorter correlation and envelope lengths.

    Returns:
        Tuple of ``(n_sub, h)``, with ``n_sub * h == dt_out``.
    """
    if dt_sub is None:
        ell = params["ell"] / 2 if broad_front else params["ell"]
        width = params["L_env"] / math.sqrt(3) if broad_front else params["L_env"]
        rate2 = (
            params["tau_g"] ** -2
            + (params["cx"] ** 2 + params["cy"] ** 2) / ell**2
            + (params["cex"] ** 2 + params["cey"] ** 2) / width**2
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
    *,
    temporal_model: str = "gp",
    initial_phi: NumpyLike | None = None,
    start_time: float | None = None,
    broad_front: bool = False,
    front_cell_aspect_ratio: float = 4.0,
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
        temporal_model: Either ``"gp"`` or ``"ou"``.
        initial_phi: OU latent field at the forecast origin, shape ``(Nx, Ny)``.
            By default the OU field starts in its stationary distribution.
        start_time: Physical time of the OU forecast origin. Required with
            ``initial_phi`` to align the envelope. A fresh run starts at zero.
        broad_front: Stretch the envelope and cells across envelope travel.
            For a stationary envelope the long axis is y.
        front_cell_aspect_ratio: Along-front to cross-front correlation length
            ratio. The cross-front length stays at half of ``ell``.

    Returns:
        Tuple of ``(accumulation, rain, phi, envelope)``, each of shape
        ``(T / dt_out, Nx, Ny)``.
        ``accumulation[n]`` is the rain integrated over the interval ending at
        ``start_time + (n + 1) * dt_out``. The other fields are sampled then.
    """
    p = params
    n_out = _validate_setup(
        p,
        (Nx, Ny),
        T,
        dt_out,
        temporal_model,
        broad_front,
        front_cell_aspect_ratio,
        Lx=Lx,
        Ly=Ly,
        m=m,
        dt_sub=dt_sub,
    )
    if start_time is None:
        if initial_phi is not None:
            msg = "Supply start_time with initial_phi to align the moving envelope."
            raise ValueError(msg)
        start_time = 0.0
    if not math.isfinite(start_time) or start_time < 0:
        msg = "start_time must be finite and non-negative."
        raise ValueError(msg)
    if temporal_model == "gp" and (initial_phi is not None or start_time != 0):
        msg = "Restarting from initial_phi or start_time requires temporal_model='ou'."
        raise ValueError(msg)
    n_sub, h = substep(p, dt_out, m, dt_sub, broad_front=broad_front)
    Nt = n_out * n_sub + 1  # includes t = 0 and t = T
    t = np.arange(Nt) * h
    ky = 2 * np.pi * np.fft.fftfreq(Ny, Ly / Ny)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)
    if broad_front:
        heading = math.atan2(p["cey"], p["cex"])
        normal = (math.cos(heading), math.sin(heading))
        spatial_power = _front_power(kx, ky, normal, p["ell"], front_cell_aspect_ratio)
    else:
        normal = None
        spatial_power = np.exp(-(kx**2 + ky[:, None] ** 2) * p["ell"] ** 2 / 2)
    if temporal_model == "ou":
        phi = _ou_field(p, rng, Nt, h, kx, ky, spatial_power, initial_phi)
    else:
        phi = _gp_field(p, rng, t, h, kx, ky, spatial_power)

    t += start_time
    x = np.arange(Nx) * Lx / Nx
    y = np.arange(Ny) * Ly / Ny
    dX = _periodic(x[None, None, :] - p["x0"] - p["cex"] * t[:, None, None], Lx)
    dY = _periodic(y[None, :, None] - p["y0"] - p["cey"] * t[:, None, None], Ly)
    if normal is not None:
        envelope = _front_envelope(dX, dY, Lx, Ly, normal, p["L_env"])
    else:
        envelope = np.exp(-(dX**2 + dY**2) / (2 * p["L_env"] ** 2))
    rain = np.maximum(phi - (p["theta0"] - p["dtheta"] * envelope), 0.0)

    trapezoids = 0.5 * h * (rain[:-1] + rain[1:])
    accumulation = trapezoids.reshape(n_out, n_sub, Ny, Nx).sum(axis=1)
    ends = slice(n_sub, None, n_sub)
    fields = (accumulation, rain[ends], phi[ends], envelope[ends])

    return tuple(f.transpose(0, 2, 1) for f in fields)


def _front_power(kx, ky, normal, ell, aspect_ratio):
    """Spatial spectrum of cells elongated along the front."""
    kn = normal[0] * kx + normal[1] * ky[:, None]
    kt = -normal[1] * kx + normal[0] * ky[:, None]
    power = np.exp(-0.5 * (ell / 2) ** 2 * (kn**2 + (aspect_ratio * kt) ** 2))
    # A rotated spectrum needs conjugate partners even on the Nyquist axes.
    partner = power[np.ix_(-np.arange(len(ky)), -np.arange(len(kx)))]
    return (power + partner) / 2


def _front_envelope(dx, dy, Lx, Ly, normal, width):
    """Evaluate a rotated 3:1 Gaussian at its nearest periodic image."""
    nx, ny = normal
    # Squared distance is 3 * normal_offset² + tangent_offset² / 3.
    ax, ay, cross = (1 + 8 * nx**2) / 3, (1 + 8 * ny**2) / 3, 8 * nx * ny / 3
    if ax * Lx**2 > ay * Ly**2:
        dx, dy, Lx, Ly, ax = dy, dx, Ly, Lx, ay
    envelope = np.empty(np.broadcast_shapes(dx.shape, dy.shape))
    for i in range(0, len(dx), 64):
        block = slice(i, i + 64)
        best = envelope[block]
        best.fill(np.inf)
        # Completing the square gives the exact nearest x image for each y.
        # With the shorter metric axis wrapped first and aspect 3:1, only
        # the nearest y image and its two neighbours can minimise the distance.
        for shift in (0, -Ly, Ly):
            y = dy[block] + shift
            x = _periodic(dx[block] + (cross / ax) * y, Lx)
            np.minimum(best, ax * x**2 + y**2 / ax, out=best)
        np.exp(best / (-2 * width**2), out=best)
    return envelope


def _gp_field(p, rng, t, h, kx, ky, spatial_power):
    """Draw the squared-exponential field jointly across output substeps."""
    Nt = len(t)
    pad = math.ceil(5 * p["tau_g"] / h)
    Nt_pad = Nt + 2 * pad
    w = 2 * np.pi * np.fft.fftfreq(Nt_pad, h)
    F = np.exp(-((w * p["tau_g"]) ** 2) / 2)
    norm = p["s"] / np.sqrt(spatial_power.mean() * F.mean())

    # A fractional shift rotates both quadratures, including at Nyquist.
    # Unit-variance real and imaginary draws preserve s² after taking the real part.
    spec = np.empty((Nt_pad, len(ky), len(kx)), dtype=np.complex128)
    spec.real = rng.standard_normal(spec.shape)
    spec.imag = rng.standard_normal(spec.shape)
    spec *= np.sqrt(F)[:, None, None]
    spec *= norm * np.sqrt(spatial_power)
    spec = ifft(spec, axis=0, norm="ortho", overwrite_x=True)[pad : pad + Nt]
    phase_x = np.exp(-1j * p["cx"] * t[:, None] * kx)
    phase_y = np.exp(-1j * p["cy"] * t[:, None] * ky)
    spec *= phase_x[:, None, :]
    spec *= phase_y[:, :, None]
    return np.asarray(
        ifft2(spec, norm="ortho", overwrite_x=True), dtype=np.complex128
    ).real


def _ou_field(p, rng, Nt, h, kx, ky, spatial_power, initial_phi):
    """Advance a stationary spatial GP with exact advected OU transitions."""
    ny, nx = spatial_power.shape
    amplitude = p["s"] * np.sqrt(spatial_power / spatial_power.mean())
    # The even-grid Nyquist cosine has no sine partner in a saved real field.
    # Give that axis zero drift at Nyquist, retaining its full spatial variance.
    wx, wy = p["cx"] * kx, p["cy"] * ky
    if nx % 2 == 0:
        wx[nx // 2] = 0
    if ny % 2 == 0:
        wy[ny // 2] = 0
    drift = np.exp(-h / p["tau_g"] - 1j * h * (wx + wy[:, None]))
    noise_scale = np.sqrt(-np.expm1(-2 * h / p["tau_g"])) * amplitude

    if initial_phi is None:
        state = fft2(rng.standard_normal((ny, nx))) * amplitude
    else:
        initial_phi = np.asarray(initial_phi)
        if (
            initial_phi.shape != (nx, ny)
            or not np.issubdtype(initial_phi.dtype, np.number)
            or not np.isrealobj(initial_phi)
            or not np.isfinite(initial_phi).all()
        ):
            raise ValueError(
                f"initial_phi must be a finite real field of shape ({nx}, {ny})."
            )
        state = fft2(initial_phi.T.astype(np.float64, copy=False))
    phi = np.empty((Nt, ny, nx))
    phi[0] = np.asarray(ifft2(state), dtype=np.complex128).real
    for i in range(1, Nt):
        state = drift * state + noise_scale * fft2(rng.standard_normal((ny, nx)))
        phi[i] = np.asarray(ifft2(state), dtype=np.complex128).real
    return phi


def _periodic(d: NumpyLike, L: float) -> NumpyLike:
    """Map displacements onto [-L/2, L/2) on a periodic domain of length L."""
    return (d + L / 2) % L - L / 2
