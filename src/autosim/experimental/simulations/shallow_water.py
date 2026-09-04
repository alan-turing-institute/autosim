"""Shallow-water equation simulator."""

from __future__ import annotations

import math

import torch

from autosim.experimental.simulations._spectral import (
    expected_filtered_variance,
    gaussian_ring_spectrum,
    sample_filtered_scalar_hat,
    spectral_wavenumbers,
    two_thirds_mask,
)
from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike

# Default param ranges when not overridden (amp required; h_mean, drag, nu optional).
DEFAULT_AMP_RANGE: tuple[float, float] = (0.05, 0.14)
DEFAULT_H_MEAN_RANGE: tuple[float, float] = (0.7, 1.5)
DEFAULT_DRAG_RANGE: tuple[float, float] = (1e-3, 4e-3)
DEFAULT_NU_RANGE: tuple[float, float] = (2e-4, 8e-4)

# IC and solver tuning (used in simulate_swe_2d).
U_SCALE = 0.5  # streamfunction amplitude scale for random component
JET_AMP_FRAC = 0.8  # jet speed ~ amp * JET_AMP_FRAC
PERT_LAT_FRAC = 0.65  # wave-6 perturbation center y/Ly
PERT_WIDTH_FRAC = 0.10  # Gaussian width y/Ly
WAVE_ZONAL_MODE = 6  # zonal wavenumber for mid-lat perturbation
N_JET_MODES = 4  # Fourier modes per column for jet
N_HYPERVISC = 4  # hyperviscosity exponent
K_CUT_FACTOR = 6  # k_cut = k_min * min(nx,ny) // K_CUT_FACTOR
H_MIN_CLIP = 1e-4
H_MAX_CLIP = 100.0
UV_ABS_CLIP = 100.0
SATURATION_THRESHOLD = 0.01  # stop if this fraction of grid hits clip bounds
EPS = 1e-10  # small constant for safe div/norms
MIN_WAVE_SPEED_CFL = 1e-8  # floor for CFL dt; keep conservative to avoid instability
CORIOLIS_MODES = ("f_plane", "periodic_beta")
FORCING_TYPES = ("none", "vortical", "balanced", "momentum")


class ShallowWater2D(SpatioTemporalSimulator):
    r"""Full 2D shallow-water simulator with prognostic :math:`[h, u, v]`.

    The solver evolves fluid height :math:`h` and horizontal velocity
    :math:`(u, v)` using:

    .. math::

        \begin{aligned}
        \partial_t h + \nabla\cdot(h\mathbf{u}) &= 0, \\
        \partial_t u + u\partial_x u + v\partial_y u
            &= f v - g\partial_x h + \nu\nabla^2 u - r u, \\
        \partial_t v + u\partial_x v + v\partial_y v
            &= -f u - g\partial_y h + \nu\nabla^2 v - r v.
        \end{aligned}

    Setting ``g=0`` removes height-gradient feedback from momentum. With
    ``f0=beta=0``, velocity then follows forced, damped 2D vector-Burgers
    dynamics while :math:`h` remains a passive continuity field.

    The stochastic forcing options represent different unresolved processes:

    - ``"vortical"`` injects divergence-free velocity and can represent
      unresolved rotational eddy stirring or wind-stress curl.
    - ``"balanced"`` adds the same rotational velocity together with its
      constant-:math:`f` geostrophic height perturbation. This experimental
      joint perturbation can reduce immediate imbalance, although balance is
      approximate with spatially varying Coriolis parameter.
    - ``"momentum"`` injects unconstrained horizontal velocity and therefore
      includes rotational and divergent components. It is the most direct
      idealization of stochastic wind stress in ocean-atmosphere coupling.
    - ``"none"`` leaves the SWE evolution deterministic after the random
      initial state is fixed.

    Every stochastic mode uses a Gaussian ring in spatial Fourier space. For
    ``"vortical"`` and ``"balanced"`` forcing it filters sampled vorticity;
    for ``"momentum"`` it filters two sampled velocity components directly.
    ``forcing_correlation_time=0`` gives independent white-in-time impulses.
    A positive value evolves a persistent Ornstein-Uhlenbeck (OU) forcing
    tendency with e-folding time :math:`\tau` and integrates that tendency
    exactly over each adaptive step. Larger :math:`\tau` produces more
    persistent, longer-correlated forcing.

    Args:
        parameters_range: Input parameter (min, max) ranges. Supported keys:

            - ``amp`` (required): initial-condition amplitude scale.
            - ``h_mean``: mean layer depth (scalar) around which spatial
              variations are generated (default 1.0 if omitted).
            - ``drag``: linear drag coefficient (default 2e-3).
            - ``nu``: Laplacian viscosity (default 5e-4).
            - ``beta``: central planetary-vorticity gradient.
            - ``f0``: reference Coriolis parameter; zero disables constant
              rotation.
            - ``forcing_energy_rate``: diffusion scale in the linearized SWE
              specific-energy norm.
            - ``forcing_correlation_time``: OU e-folding time; zero selects
              white noise.

            If None, uses ``{"amp": (0.05, 0.14)}`` only.
        output_names, return_timeseries, log_level
            Passed to base. Default outputs: ["h", "u", "v"].
        return_additional_input_fields
            Return the accumulated forcing impulse for each saved state
            transition under the separate ``additional_input_fields`` dataset
            key. Requires ``return_timeseries=True``.
        nx, ny, Lx, Ly, T, dt_save, skip_nt, cfl
            Grid, domain, time and CFL settings.
        g, h_mean, nu, drag, beta, f0
            Physics constants (used when not in parameters_range).
            ``g=0`` removes height feedback and initializes a flat height field.
            ``f0=None`` derives ``sqrt(g * h_mean) / 8``; ``f0=0`` disables
            rotation when ``beta`` is also omitted.
        coriolis_mode
            ``"f_plane"`` or a smooth ``"periodic_beta"`` analogue compatible
            with the doubly periodic Fourier grid.
        dealias
            Whether to apply two-thirds spectral dealiasing.
        forcing_type
            Stochastic forcing geometry: ``"none"``, divergence-free
            ``"vortical"``, geostrophically ``"balanced"``, unconstrained
            ``"momentum"`` forcing.
        forcing_energy_rate
            Diffusion scale in the linearized SWE specific-energy norm. For
            white noise it sets the expected increment energy per unit time;
            for OU forcing it sets the long-time diffusion rate.
        forcing_wavenumber, forcing_bandwidth
            Central angular wavenumber and width of the Gaussian spectral
            ring. A central wavenumber ``k`` corresponds to wavelength
            ``2 * pi / k``.
        forcing_correlation_time
            OU e-folding time in simulation-time units. Zero retains
            independent white-in-time increments.
        dtype
            torch.float32 or torch.float64.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        nx: int = 64,
        ny: int = 128,
        Lx: float = 64.0,
        Ly: float = 128.0,
        T: float = 90.0,
        dt_save: float = 0.2,
        skip_nt: int = 0,
        cfl: float = 0.12,
        g: float = 9.81,
        h_mean: float = 1.0,
        nu: float = 5e-4,
        drag: float = 2e-3,
        dtype: torch.dtype = torch.float64,
        return_additional_input_fields: bool = False,
        beta: float | None = None,
        coriolis_mode: str = "periodic_beta",
        dealias: bool = True,
        forcing_type: str = "none",
        forcing_energy_rate: float = 1e-6,
        forcing_wavenumber: float | None = None,
        forcing_bandwidth: float | None = None,
        forcing_correlation_time: float = 0.0,
        f0: float | None = None,
    ) -> None:
        """Initialize the planar shallow-water simulator."""
        if parameters_range is None:
            parameters_range = {"amp": DEFAULT_AMP_RANGE}
        if output_names is None:
            output_names = ["h", "u", "v"]

        super().__init__(parameters_range, output_names, log_level)
        if skip_nt < 0:
            msg = "skip_nt must be non-negative"
            raise ValueError(msg)
        if return_additional_input_fields and not return_timeseries:
            msg = "return_additional_input_fields requires return_timeseries=True"
            raise ValueError(msg)
        if coriolis_mode not in CORIOLIS_MODES:
            msg = f"coriolis_mode must be one of {CORIOLIS_MODES}"
            raise ValueError(msg)
        if forcing_type not in FORCING_TYPES:
            msg = f"forcing_type must be one of {FORCING_TYPES}"
            raise ValueError(msg)
        self.return_timeseries = return_timeseries
        self.return_additional_input_fields = return_additional_input_fields
        self.additional_input_names = ["forcing_h", "forcing_u", "forcing_v"]
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.dt_save = dt_save
        self.skip_nt = skip_nt
        self.cfl = cfl
        self.g = g
        self.h_mean = h_mean
        self.nu = nu
        self.drag = drag
        self.beta = beta
        self.coriolis_mode = coriolis_mode
        self.dealias = dealias
        self.forcing_type = forcing_type
        self.forcing_energy_rate = forcing_energy_rate
        self.forcing_wavenumber = forcing_wavenumber
        self.forcing_bandwidth = forcing_bandwidth
        self.forcing_correlation_time = forcing_correlation_time
        self.f0 = f0
        self.dtype = dtype

    def _forward(self, x: TensorLike) -> TensorLike:
        if x.shape[0] != 1:
            msg = "Simulator._forward expects a single input (batch size 1)"
            raise ValueError(msg)
        if x.shape[1] != self.in_dim:
            msg = (
                f"Input dim {x.shape[1]} does not match "
                f"parameters_range length {self.in_dim}"
            )
            raise ValueError(msg)
        # Parse by name so parameter order is irrelevant and optional params clear.
        amp = float(x[0, self.get_parameter_idx("amp")].item())
        h_mean = (
            float(x[0, self.get_parameter_idx("h_mean")].item())
            if "h_mean" in self.param_names
            else self.h_mean
        )
        drag = (
            float(x[0, self.get_parameter_idx("drag")].item())
            if "drag" in self.param_names
            else self.drag
        )
        nu = (
            float(x[0, self.get_parameter_idx("nu")].item())
            if "nu" in self.param_names
            else self.nu
        )
        beta = (
            float(x[0, self.get_parameter_idx("beta")].item())
            if "beta" in self.param_names
            else self.beta
        )
        f0 = (
            float(x[0, self.get_parameter_idx("f0")].item())
            if "f0" in self.param_names
            else self.f0
        )
        forcing_energy_rate = (
            float(x[0, self.get_parameter_idx("forcing_energy_rate")].item())
            if "forcing_energy_rate" in self.param_names
            else self.forcing_energy_rate
        )
        forcing_correlation_time = (
            float(x[0, self.get_parameter_idx("forcing_correlation_time")].item())
            if "forcing_correlation_time" in self.param_names
            else self.forcing_correlation_time
        )

        y = simulate_swe_2d(
            amp=amp,
            return_timeseries=self.return_timeseries,
            return_additional_input_fields=self.return_additional_input_fields,
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
            T=self.T,
            dt_save=self.dt_save,
            skip_nt=self.skip_nt,
            cfl=self.cfl,
            g=self.g,
            h_mean=h_mean,
            nu=nu,
            drag=drag,
            beta=beta,
            coriolis_mode=self.coriolis_mode,
            dealias=self.dealias,
            forcing_type=self.forcing_type,
            forcing_energy_rate=forcing_energy_rate,
            forcing_wavenumber=self.forcing_wavenumber,
            forcing_bandwidth=self.forcing_bandwidth,
            dtype=self.dtype,
            forcing_correlation_time=forcing_correlation_time,
            f0=f0,
        )
        return y.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Run sampled trajectories and return `[batch,time,x,y,channels]` data."""
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )
        n_valid = y.shape[0]

        state_channels = 3
        channels = state_channels * (2 if self.return_additional_input_fields else 1)
        features_per_step = self.nx * self.ny * channels

        if self.return_timeseries:
            total = y.shape[1]
            n_time = total // features_per_step
            y = y.reshape(n_valid, n_time, self.nx, self.ny, channels)
        else:
            y = y.reshape(n_valid, 1, self.nx, self.ny, channels)

        additional_input_fields = None
        if self.return_additional_input_fields:
            additional_input_fields = y[..., state_channels:]
            y = y[..., :state_channels]

        return {
            "data": y,
            "additional_input_fields": additional_input_fields,
            "constant_scalars": x,
            "constant_fields": None,
        }


def _save_times(T: float, dt_save: float) -> list[float]:
    """Return regular save times and always include the terminal time."""
    save_times = [i * dt_save for i in range(math.floor(T / dt_save) + 1)]
    if not math.isclose(save_times[-1], T, rel_tol=1e-12, abs_tol=1e-12):
        save_times.append(T)
    else:
        save_times[-1] = T
    return save_times


def _coriolis_grid(
    Y: torch.Tensor,
    *,
    f0: float,
    beta: float,
    Ly: float,
    mode: str,
) -> torch.Tensor:
    """Return a Coriolis field compatible with the configured plane geometry."""
    if mode == "f_plane":
        return torch.full_like(Y, f0)
    if mode == "periodic_beta":
        phase = 2.0 * math.pi * (Y - 0.5 * Ly) / Ly
        return f0 + (beta * Ly / (2.0 * math.pi)) * torch.sin(phase)
    msg = f"mode must be one of {CORIOLIS_MODES}"
    raise ValueError(msg)


def _ou_step_coefficients(
    step_dt: float, correlation_time: float
) -> tuple[float, float, float, float]:
    """Return exact OU endpoint and time-integral coefficients.

    The final two values multiply the sum of the old and new endpoints and an
    independent spatial innovation, respectively, in the exact integral over
    one step.
    """
    ratio = step_dt / correlation_time
    decay = math.exp(-ratio)
    endpoint_innovation_weight = math.sqrt(-math.expm1(-2.0 * ratio))
    integral_endpoint_weight = correlation_time * math.tanh(0.5 * ratio)

    # ratio - 2*tanh(ratio/2) loses precision for well-resolved OU steps.
    if ratio < 1e-3:
        residual_ratio = ratio**3 / 12.0 - ratio**5 / 120.0 + 17.0 * ratio**7 / 20160.0
    else:
        residual_ratio = ratio - 2.0 * math.tanh(0.5 * ratio)
    integral_innovation_variance = correlation_time * max(residual_ratio, 0.0)
    return (
        decay,
        endpoint_innovation_weight,
        integral_endpoint_weight,
        integral_innovation_variance,
    )


def _swe_forcing_expected_unit_energy(
    *,
    forcing_type: str,
    nx: int,
    ny: int,
    g: float,
    h_mean: float,
    f0: float,
    spectrum: torch.Tensor,
    K2_inv: torch.Tensor,
    dKx: torch.Tensor,
    dKy: torch.Tensor,
) -> float:
    """Return expected energy before scaling a unit Gaussian forcing draw."""
    if forcing_type in ("vortical", "balanced"):
        energy_transfer = 0.5 * K2_inv.square() * (dKx.square() + dKy.square())
        if forcing_type == "balanced":
            if g > 0:
                energy_transfer += 0.5 * (g / h_mean) * (f0 / g) ** 2 * K2_inv.square()
            elif f0 != 0:
                msg = "balanced forcing with g=0 requires f0=0"
                raise ValueError(msg)
        expected_energy = expected_filtered_variance(
            nx=nx,
            ny=ny,
            spectrum=spectrum,
            transfer_power=energy_transfer,
        )
    elif forcing_type == "momentum":
        expected_energy = expected_filtered_variance(
            nx=nx,
            ny=ny,
            spectrum=spectrum,
        )
    else:
        msg = f"cannot normalize forcing_type={forcing_type!r}"
        raise ValueError(msg)

    if not math.isfinite(expected_energy) or expected_energy <= 0:
        msg = "stochastic forcing has zero or non-finite expected energy"
        raise RuntimeError(msg)
    return expected_energy


def _sample_swe_forcing_field(
    *,
    forcing_type: str,
    nx: int,
    ny: int,
    target_energy: float,
    g: float,
    h_mean: float,
    f0: float,
    dtype: torch.dtype,
    spectrum: torch.Tensor,
    mask: torch.Tensor,
    K2_inv: torch.Tensor,
    dKx: torch.Tensor,
    dKy: torch.Tensor,
    expected_unit_energy: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Gaussian ``(dh, du, dv)`` with the target expected energy."""
    zero = torch.zeros((nx, ny), dtype=dtype)
    if target_energy < 0:
        msg = "target_energy must be non-negative"
        raise ValueError(msg)
    if target_energy == 0:
        return zero, zero.clone(), zero.clone()

    def to_phys(field_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(field_hat, s=(nx, ny))

    if forcing_type in ("vortical", "balanced"):
        zeta_increment_hat = sample_filtered_scalar_hat(
            nx=nx,
            ny=ny,
            dtype=dtype,
            spectrum=spectrum,
            mask=mask,
        )
        psi_increment_hat = K2_inv * zeta_increment_hat
        du = to_phys(-1j * dKy * psi_increment_hat)
        dv = to_phys(1j * dKx * psi_increment_hat)
        if forcing_type == "vortical":
            dh = zero
        elif g > 0:
            # Constant-f geostrophic balance: g * grad(dh) = f0 * grad(dpsi).
            dh = (f0 / g) * to_phys(psi_increment_hat)
        elif f0 == 0:
            # With no gravity or rotation, balanced forcing reduces to vortical.
            dh = zero
        else:
            msg = "balanced forcing with g=0 requires f0=0"
            raise ValueError(msg)
    elif forcing_type == "momentum":
        du = to_phys(
            sample_filtered_scalar_hat(
                nx=nx,
                ny=ny,
                dtype=dtype,
                spectrum=spectrum,
                mask=mask,
            )
        )
        dv = to_phys(
            sample_filtered_scalar_hat(
                nx=nx,
                ny=ny,
                dtype=dtype,
                spectrum=spectrum,
                mask=mask,
            )
        )
        dh = zero
    else:
        msg = f"cannot sample forcing_type={forcing_type!r}"
        raise ValueError(msg)

    if expected_unit_energy is None:
        expected_unit_energy = _swe_forcing_expected_unit_energy(
            forcing_type=forcing_type,
            nx=nx,
            ny=ny,
            g=g,
            h_mean=h_mean,
            f0=f0,
            spectrum=spectrum,
            K2_inv=K2_inv,
            dKx=dKx,
            dKy=dKy,
        )
    scale = math.sqrt(target_energy / expected_unit_energy)
    return dh * scale, du * scale, dv * scale


def simulate_swe_2d(  # noqa: PLR0912, PLR0915
    amp: float,
    return_timeseries: bool,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    T: float,
    dt_save: float,
    cfl: float,
    g: float,
    h_mean: float,
    nu: float,
    drag: float,
    dtype: torch.dtype = torch.float64,
    skip_nt: int = 0,
    beta: float | None = None,
    coriolis_mode: str = "periodic_beta",
    dealias: bool = True,
    forcing_type: str = "none",
    forcing_energy_rate: float = 1e-6,
    forcing_wavenumber: float | None = None,
    forcing_bandwidth: float | None = None,
    return_additional_input_fields: bool = False,
    forcing_correlation_time: float = 0.0,
    f0: float | None = None,
) -> torch.Tensor:
    """Integrate full shallow-water equations with PDEArena-style random2 ICs.

    Named stochastic forcing modes share a Gaussian spatial spectral ring.
    ``forcing_correlation_time=0`` uses independent white-in-time increments.
    Positive correlation time evolves an OU forcing tendency with exact
    exponential memory and samples its exact time integral at each adaptive
    step. Its stationary linearized expected specific energy is
    ``forcing_energy_rate / (2 * correlation_time)``, so its long-time
    integrated diffusion rate is ``forcing_energy_rate``. The exact integral
    also converges to the white-noise increment as the correlation time tends
    to zero.

    When ``return_additional_input_fields=True``, three forcing-impulse channels
    are appended after ``[h, u, v]``. At saved index ``i`` they contain the sum
    of ``[dh, du, dv]`` increments used to advance state ``i`` to state ``i+1``;
    the final entry is zero because it has no following transition.
    """
    if forcing_type not in FORCING_TYPES:
        msg = f"forcing_type must be one of {FORCING_TYPES}"
        raise ValueError(msg)
    if dtype not in (torch.float32, torch.float64):
        msg = "dtype must be torch.float32 or torch.float64"
        raise ValueError(msg)
    if skip_nt < 0:
        msg = "skip_nt must be non-negative"
        raise ValueError(msg)
    if return_additional_input_fields and not return_timeseries:
        msg = "return_additional_input_fields requires return_timeseries=True"
        raise ValueError(msg)
    if nx < 6 or ny < 6:
        msg = "nx and ny must be at least 6"
        raise ValueError(msg)
    if Lx <= 0 or Ly <= 0:
        msg = "Lx and Ly must be positive"
        raise ValueError(msg)
    if T < 0 or dt_save <= 0 or cfl <= 0:
        msg = "T must be non-negative and dt_save/cfl must be positive"
        raise ValueError(msg)
    if g < 0 or h_mean <= 0:
        msg = "g must be non-negative and h_mean must be positive"
        raise ValueError(msg)
    if nu < 0 or drag < 0:
        msg = "nu and drag must be non-negative"
        raise ValueError(msg)
    if forcing_energy_rate < 0:
        msg = "forcing_energy_rate must be non-negative"
        raise ValueError(msg)
    if forcing_correlation_time < 0:
        msg = "forcing_correlation_time must be non-negative"
        raise ValueError(msg)
    if f0 is not None and not math.isfinite(f0):
        msg = "f0 must be finite or None"
        raise ValueError(msg)
    if coriolis_mode not in CORIOLIS_MODES:
        msg = f"coriolis_mode must be one of {CORIOLIS_MODES}"
        raise ValueError(msg)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    x = torch.linspace(0.0, Lx, nx + 1, dtype=dtype)[:-1]
    y = torch.linspace(0.0, Ly, ny + 1, dtype=dtype)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    dx = Lx / nx
    dy = Ly / ny

    c = math.sqrt(g * h_mean)
    f0 = c / 8.0 if f0 is None else f0
    beta = 0.5 * f0 / Ly if beta is None else beta
    if g == 0 and forcing_type == "balanced" and f0 != 0:
        msg = "balanced forcing with g=0 requires f0=0"
        raise ValueError(msg)
    f_grid = _coriolis_grid(Y, f0=f0, beta=beta, Ly=Ly, mode=coriolis_mode)

    Kx, Ky, dKx, dKy = spectral_wavenumbers(nx, ny, Lx, Ly, dtype)
    K2 = Kx**2 + Ky**2
    K2_inv = torch.where(K2 > 0, -1.0 / K2, torch.zeros_like(K2))
    iKx = 1j * dKx
    iKy = 1j * dKy
    dealias_mask = two_thirds_mask(nx, ny) if dealias else torch.ones_like(K2).bool()
    max_retained_k2 = float(K2[dealias_mask].max())

    forcing_spectrum: torch.Tensor | None = None
    forcing_expected_unit_energy: float | None = None
    if forcing_type != "none":
        fundamental_wavenumber = 2.0 * math.pi / max(Lx, Ly)
        if forcing_wavenumber is None:
            forcing_mode = min(16.0, 0.25 * min(nx, ny))
            forcing_wavenumber = forcing_mode * fundamental_wavenumber
        if forcing_bandwidth is None:
            forcing_bandwidth = 1.5 * fundamental_wavenumber
        forcing_spectrum = gaussian_ring_spectrum(
            K2, forcing_wavenumber, forcing_bandwidth, dealias_mask
        )
        forcing_expected_unit_energy = _swe_forcing_expected_unit_energy(
            forcing_type=forcing_type,
            nx=nx,
            ny=ny,
            g=g,
            h_mean=h_mean,
            f0=f0,
            spectrum=forcing_spectrum,
            K2_inv=K2_inv,
            dKx=dKx,
            dKy=dKy,
        )

    def sample_forcing_field(target_energy: float) -> torch.Tensor:
        if forcing_spectrum is None or forcing_expected_unit_energy is None:
            msg = "stochastic forcing spectrum was not initialized"
            raise RuntimeError(msg)
        return torch.stack(
            _sample_swe_forcing_field(
                forcing_type=forcing_type,
                nx=nx,
                ny=ny,
                target_energy=target_energy,
                g=g,
                h_mean=h_mean,
                f0=f0,
                dtype=dtype,
                spectrum=forcing_spectrum,
                mask=dealias_mask,
                K2_inv=K2_inv,
                dKx=dKx,
                dKy=dKy,
                expected_unit_energy=forcing_expected_unit_energy,
            ),
            dim=-1,
        )

    # Hyperviscosity integrating factor damps grid-scale modes in ~1 time unit
    # while leaving large-scale vortices nearly untouched.
    k_max = math.pi * max(nx / Lx, ny / Ly)
    nu_h = 1.0 / k_max ** (2 * N_HYPERVISC)
    hyp_op = -nu_h * K2**N_HYPERVISC

    def to_spec(field: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(field)

    def to_phys(field_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(field_hat, s=(nx, ny))

    def project(field: torch.Tensor) -> torch.Tensor:
        return to_phys(to_spec(field) * dealias_mask)

    # ------------------------------------------------------------------ #
    # Balanced initial conditions via vorticity → streamfunction         #
    # ------------------------------------------------------------------ #
    # Strategy: specify vorticity ζ (random large-scale + jet shear +
    # wave-6 perturbation), solve ∇²ψ = ζ spectrally, then derive
    #   u = -∂ψ/∂y,  v = ∂ψ/∂x,  h = h_mean + (f0/g)·ψ
    # This satisfies linear f-plane geostrophic balance. Nonlinear acceleration
    # and spatial Coriolis variation leave a small adjustment residual.

    k_min = 2.0 * math.pi / max(Lx, Ly)
    k_cut = k_min * (min(nx, ny) // K_CUT_FACTOR)

    # Component 1: random large-scale streamfunction with k^{-2} weighting
    random_field = torch.randn(nx, ny, dtype=dtype)
    psi_hat_rand = to_spec(random_field).to(dtype=complex_dtype)
    K_mag = torch.sqrt(K2 + k_min**2)
    psi_hat_rand = psi_hat_rand / K_mag**2
    psi_hat_rand = torch.where(
        (k_cut**2 > K2) & dealias_mask,
        psi_hat_rand,
        torch.zeros_like(psi_hat_rand),
    )
    psi_hat_rand[0, 0] = 0.0
    psi_rand_phys = to_phys(psi_hat_rand)
    psi_norm = amp * U_SCALE * min(Lx, Ly) / (float(psi_rand_phys.std()) + EPS)
    psi_hat_rand = psi_hat_rand * psi_norm
    zeta_random = to_phys(-K2 * psi_hat_rand)

    # Component 2: per-column independent random zonal jet (PDEArena :random2 style)
    # Each longitude column i gets its own independent random Fourier coefficients
    # in y — matching PDEArena's truly per-column i.i.d. wind profiles.
    coeff = torch.randn(nx, N_JET_MODES, dtype=dtype)  # i.i.d. per column
    y_frac = Y / Ly  # [nx, ny], values in [0, 1]
    u_jet_field = torch.stack(
        [
            coeff[:, m].unsqueeze(1) * torch.sin((m + 1) * math.pi * y_frac)
            for m in range(N_JET_MODES)
        ],
        dim=0,
    ).sum(dim=0)  # [nx, ny]
    jet_std = float(u_jet_field.std()) + EPS
    u_jet_field = u_jet_field * (amp * JET_AMP_FRAC / jet_std)
    zeta_jet = to_phys(-iKy * to_spec(u_jet_field))

    # Component 3: wave-6 Gaussian perturbation at mid-latitude
    zeta_jet_scale = float(zeta_jet.std())
    A_pert = max(amp * f0 * JET_AMP_FRAC, zeta_jet_scale * 0.25)
    y_center = PERT_LAT_FRAC * Ly
    y_width = PERT_WIDTH_FRAC * Ly
    pert_phase = float(torch.rand(1)) * 2.0 * math.pi
    zeta_pert = (
        A_pert
        * torch.cos(WAVE_ZONAL_MODE * 2.0 * math.pi * X / Lx + pert_phase)
        * torch.exp(-0.5 * ((Y - y_center) / y_width) ** 2)
    )

    # Combine, low-pass filter, solve for ψ, then u, v, h
    zeta = zeta_random + zeta_jet + zeta_pert
    zeta_h = to_spec(zeta)
    zeta_h = torch.where(
        (k_cut**2 > K2) & dealias_mask, zeta_h, torch.zeros_like(zeta_h)
    )
    zeta_h[0, 0] = 0.0

    psi_h = K2_inv * zeta_h  # ∇²ψ = ζ  →  ψ̂ = -ζ̂/K²
    psi0 = to_phys(psi_h)

    u0 = to_phys(-iKy * psi_h)  # u = -∂ψ/∂y
    v0 = to_phys(iKx * psi_h)  # v =  ∂ψ/∂x
    h0 = (
        (h_mean + (f0 / g) * psi0).clamp(min=0.5 * h_mean)
        if g > 0
        else torch.full_like(psi0, h_mean)
    )

    def rhs(
        h: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_safe = h.clamp(min=H_MIN_CLIP)

        # Reuse spectra per field to avoid repeated FFTs in each RHS evaluation.
        u_h = to_spec(u)
        v_h = to_spec(v)
        h_h = to_spec(h)

        du_dx = to_phys(iKx * u_h)
        du_dy = to_phys(iKy * u_h)
        dv_dx = to_phys(iKx * v_h)
        dv_dy = to_phys(iKy * v_h)
        dh_dx = to_phys(iKx * h_h)
        dh_dy_local = to_phys(iKy * h_h)

        lap_u = to_phys(-K2 * u_h)
        lap_v = to_phys(-K2 * v_h)

        hu = h_safe * u
        hv = h_safe * v
        hu_h = to_spec(hu)
        hv_h = to_spec(hv)
        div_hu = to_phys(iKx * hu_h) + to_phys(iKy * hv_h)

        dudt = -(u * du_dx + v * du_dy) + f_grid * v - g * dh_dx + nu * lap_u - drag * u
        dvdt = (
            -(u * dv_dx + v * dv_dy)
            - f_grid * u
            - g * dh_dy_local
            + nu * lap_v
            - drag * v
        )
        dhdt = -div_hu
        return project(dhdt), project(dudt), project(dvdt)

    def rk4_step(
        h: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k1_h, k1_u, k1_v = rhs(h, u, v)
        k2_h, k2_u, k2_v = rhs(
            h + 0.5 * dt * k1_h, u + 0.5 * dt * k1_u, v + 0.5 * dt * k1_v
        )
        k3_h, k3_u, k3_v = rhs(
            h + 0.5 * dt * k2_h, u + 0.5 * dt * k2_u, v + 0.5 * dt * k2_v
        )
        k4_h, k4_u, k4_v = rhs(h + dt * k3_h, u + dt * k3_u, v + dt * k3_v)
        return (
            h + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h),
            u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u),
            v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v),
        )

    def output(h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        h_out = torch.nan_to_num(
            h,
            nan=h_mean,
            posinf=H_MAX_CLIP,
            neginf=H_MIN_CLIP,
        ).clamp(min=H_MIN_CLIP, max=H_MAX_CLIP)
        u_out = torch.nan_to_num(
            u, nan=0.0, posinf=UV_ABS_CLIP, neginf=-UV_ABS_CLIP
        ).clamp(min=-UV_ABS_CLIP, max=UV_ABS_CLIP)
        v_out = torch.nan_to_num(
            v, nan=0.0, posinf=UV_ABS_CLIP, neginf=-UV_ABS_CLIP
        ).clamp(min=-UV_ABS_CLIP, max=UV_ABS_CLIP)
        return torch.stack([h_out.float(), u_out.float(), v_out.float()], dim=-1)

    h = h0
    u = u0
    v = v0

    def _saturation_fraction(
        h_curr: torch.Tensor, u_curr: torch.Tensor, v_curr: torch.Tensor
    ) -> float:
        h_sat = ((h_curr <= H_MIN_CLIP) | (h_curr >= H_MAX_CLIP)).float().mean()
        u_sat = (u_curr.abs() >= UV_ABS_CLIP).float().mean()
        v_sat = (v_curr.abs() >= UV_ABS_CLIP).float().mean()
        return float(torch.maximum(torch.maximum(h_sat, u_sat), v_sat).item())

    if not (
        torch.isfinite(h).all() and torch.isfinite(u).all() and torch.isfinite(v).all()
    ):
        raise RuntimeError(
            "ShallowWater2D simulation failed: "
            f"non-finite initial state at t=0.000000 (amp={amp:.6f})."
        )
    if _saturation_fraction(h, u, v) >= SATURATION_THRESHOLD:
        raise RuntimeError(
            "ShallowWater2D simulation failed: "
            f"initial state saturated at clipping bounds at t=0.000000 "
            f"(amp={amp:.6f})."
        )

    save_times = _save_times(T, dt_save)
    expected_frames = len(save_times)
    if return_timeseries and skip_nt >= expected_frames:
        msg = (
            "skip_nt is too large for the available trajectory length; "
            f"skip_nt={skip_nt}, available_frames={expected_frames}."
        )
        raise ValueError(msg)

    t = 0.0
    last_valid = output(h, u, v)
    history = [last_valid] if return_timeseries else []
    forcing_intervals: list[torch.Tensor] = []
    forcing_since_save = torch.zeros((nx, ny, 3), dtype=dtype)
    forcing_tendency: torch.Tensor | None = None
    if forcing_type != "none" and forcing_correlation_time > 0:
        stationary_energy = forcing_energy_rate / (2.0 * forcing_correlation_time)
        forcing_tendency = sample_forcing_field(stationary_energy)
    next_save_idx = 1
    failure_reason: str | None = None

    while t < T - 1e-12:
        if not (
            torch.isfinite(h).all()
            and torch.isfinite(u).all()
            and torch.isfinite(v).all()
        ):
            failure_reason = "non-finite state encountered"
            break
        if _saturation_fraction(h, u, v) >= SATURATION_THRESHOLD:
            failure_reason = "state saturated at clipping bounds"
            break

        c_now = torch.sqrt(g * h.clamp(min=H_MIN_CLIP))
        speed_x = (u.abs() + c_now).max().item()
        speed_y = (v.abs() + c_now).max().item()
        max_speed = max(speed_x, speed_y, MIN_WAVE_SPEED_CFL)
        if not math.isfinite(max_speed):
            failure_reason = "non-finite wave speed"
            break

        step_dt = cfl * min(dx, dy) / max_speed
        if nu > 0:
            step_dt = min(step_dt, 2.5 / (nu * max_retained_k2))
        step_dt = min(step_dt, T - t)
        if return_timeseries and next_save_idx < expected_frames:
            step_dt = min(step_dt, save_times[next_save_idx] - t)
        if not math.isfinite(step_dt) or step_dt <= 0:
            failure_reason = "non-positive or non-finite timestep"
            break

        h, u, v = rk4_step(h, u, v, step_dt)

        # Apply hyperviscosity integrating factor to all fields (spectral filter).
        hyp_factor = torch.exp(hyp_op * step_dt) * dealias_mask
        u = to_phys(to_spec(u) * hyp_factor)
        v = to_phys(to_spec(v) * hyp_factor)
        h_field_mean = h.mean()
        h_anom = h - h_field_mean
        h_anom = to_phys(to_spec(h_anom) * hyp_factor)
        h = (h_field_mean + h_anom).clamp(min=H_MIN_CLIP)

        if forcing_type != "none":
            if forcing_correlation_time == 0:
                forcing_increment = sample_forcing_field(forcing_energy_rate * step_dt)
            else:
                if forcing_tendency is None:
                    msg = "OU forcing tendency was not initialized"
                    raise RuntimeError(msg)
                (
                    decay,
                    endpoint_innovation_weight,
                    integral_endpoint_weight,
                    integral_innovation_variance,
                ) = _ou_step_coefficients(
                    step_dt=step_dt,
                    correlation_time=forcing_correlation_time,
                )
                stationary_energy = forcing_energy_rate / (
                    2.0 * forcing_correlation_time
                )
                endpoint_innovation = sample_forcing_field(stationary_energy)
                previous_tendency = forcing_tendency
                forcing_tendency = decay * previous_tendency + (
                    endpoint_innovation_weight * endpoint_innovation
                )
                integral_innovation = sample_forcing_field(
                    forcing_energy_rate * integral_innovation_variance
                )
                forcing_increment = (
                    integral_endpoint_weight * (previous_tendency + forcing_tendency)
                    + integral_innovation
                )
            dh, du, dv = forcing_increment.unbind(dim=-1)
            h += dh
            u += du
            v += dv
            if return_additional_input_fields:
                forcing_since_save += forcing_increment
        t += step_dt

        if (
            torch.isfinite(h).all()
            and torch.isfinite(u).all()
            and torch.isfinite(v).all()
        ):
            if _saturation_fraction(h, u, v) >= SATURATION_THRESHOLD:
                failure_reason = "state saturated at clipping bounds after step"
                break
            last_valid = output(h, u, v)
        else:
            failure_reason = "non-finite state after step"
            break

        if (
            return_timeseries
            and next_save_idx < expected_frames
            and t >= save_times[next_save_idx] - 1e-10
        ):
            history.append(last_valid)
            if return_additional_input_fields:
                forcing_intervals.append(forcing_since_save.float().clone())
                forcing_since_save.zero_()
            next_save_idx += 1

    if failure_reason is not None:
        raise RuntimeError(
            "ShallowWater2D simulation failed: "
            f"{failure_reason} at t={t:.6f} (amp={amp:.6f})."
        )

    if return_timeseries:
        state_history = torch.stack(history, dim=0)
        if return_additional_input_fields:
            final_zero = torch.zeros_like(state_history[0])
            additional_inputs = torch.stack([*forcing_intervals, final_zero], dim=0)
            state_history = torch.cat([state_history, additional_inputs], dim=-1)
        return state_history[skip_nt:]
    return output(h, u, v).unsqueeze(0)
