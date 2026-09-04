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

# Default ranges for generated initial conditions and optional physics parameters.
DEFAULT_AMP_RANGE: tuple[float, float] = (0.05, 0.14)
DEFAULT_H_MEAN_RANGE: tuple[float, float] = (0.7, 1.5)
DEFAULT_DRAG_RANGE: tuple[float, float] = (1e-3, 4e-3)
DEFAULT_NU_RANGE: tuple[float, float] = (2e-4, 8e-4)
DEFAULT_INITIAL_RING_MODE = 4.0  # large-scale random-PV seed
DEFAULT_FORCING_RING_MODE_CAP = 16.0  # avoid a resolution-driven tiny scale
DEFAULT_FORCING_RING_GRID_FRACTION = 0.25  # remain below the 2/3 cutoff
DEFAULT_RING_BANDWIDTH_MODES = 1.5  # excite neighbouring Fourier shells

# IC and solver tuning (used in simulate_swe_2d).
U_SCALE = 0.5  # streamfunction amplitude scale for random component
JET_AMP_FRAC = 0.8  # jet speed ~ amp * JET_AMP_FRAC
PERT_LAT_FRAC = 0.65  # wave-6 perturbation center y/Ly
PERT_WIDTH_FRAC = 0.10  # Gaussian width y/Ly
WAVE_ZONAL_MODE = 6  # zonal wavenumber for mid-lat perturbation
N_JET_MODES = 4  # Fourier modes per column for jet
N_HYPERVISC = 4  # hyperviscosity exponent
# The random-IC low-pass uses min(nx, ny) // K_CUT_FACTOR, so this is also the
# minimum supported grid size needed to keep that cutoff nonzero.
K_CUT_FACTOR = 6
H_MIN_CLIP = 1e-4
H_MAX_CLIP = 100.0
UV_ABS_CLIP = 100.0
SATURATION_THRESHOLD = 0.01  # stop if this fraction of grid hits clip bounds
EPS = 1e-10  # small constant for safe div/norms
MIN_WAVE_SPEED_CFL = 1e-8  # floor for CFL dt; keep conservative to avoid instability
CORIOLIS_MODES = ("f_plane", "periodic_beta")
FORCING_TYPES = ("none", "vortical", "balanced", "pv_balanced", "momentum")
INITIAL_CONDITIONS = (
    "random",
    "balanced_random_pv",
    "balanced_double_jet",
    "restart",
)
ENERGY_BUDGET_NAMES = (
    "total_energy",
    "deterministic_energy_change",
    "hyperviscous_energy_change",
    "forcing_energy_change",
    "viscous_dissipation_estimate",
    "drag_dissipation_estimate",
    "effective_forcing_energy_rate",
)


def _max_isotropic_retained_wavenumber(
    *, nx: int, ny: int, Lx: float, Ly: float, dealias: bool
) -> float:
    """Return the largest radial wavenumber retained in every direction."""
    if dealias:
        max_mode_x = (nx - 1) // 3
        max_mode_y = (ny - 1) // 3
    else:
        # Even-grid Nyquist derivatives are zero, so stop at the largest
        # non-Nyquist mode even when spectral dealiasing is disabled.
        max_mode_x = (nx - 1) // 2
        max_mode_y = (ny - 1) // 2
    return 2.0 * math.pi * min(max_mode_x / Lx, max_mode_y / Ly)


def _validate_ring_wavenumber(
    *,
    name: str,
    wavenumber: float,
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    dealias: bool,
) -> None:
    """Reject a ring whose peak is clipped in some Fourier directions."""
    max_wavenumber = _max_isotropic_retained_wavenumber(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dealias=dealias,
    )
    if wavenumber > max_wavenumber:
        msg = (
            f"{name}={wavenumber:.6g} exceeds the maximum isotropically "
            f"retained wavenumber {max_wavenumber:.6g} for this grid"
        )
        raise ValueError(msg)


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
      unresolved rotational eddy stirring or wind-stress curl. It is the
      closest option to spectral stochastic kinetic-energy backscatter.
    - ``"balanced"`` adds the same rotational velocity together with its
      constant-:math:`f` geostrophic height perturbation. This experimental
      joint perturbation can reduce immediate imbalance, although balance is
      approximate with spatially varying Coriolis parameter.
    - ``"pv_balanced"`` samples a potential-vorticity anomaly and applies
      deformation-radius-aware Helmholtz inversion before constructing the
      geostrophic velocity and height perturbation. Its use as repeated
      additive forcing is experimental rather than a standard backscatter
      scheme.
    - ``"momentum"`` injects unconstrained horizontal velocity and therefore
      includes rotational and divergent components. It is the most direct
      idealization of stochastic wind stress in ocean-atmosphere coupling.
    - ``"none"`` leaves the SWE evolution deterministic after the random
      initial state is fixed.

    Every stochastic mode uses a Gaussian ring in spatial Fourier space. It
    filters sampled vorticity for ``"vortical"`` and ``"balanced"``, sampled
    potential vorticity for ``"pv_balanced"``, and two sampled velocity
    components for ``"momentum"``.
    ``forcing_correlation_time=0`` gives independent white-in-time impulses.
    A positive value evolves a persistent Ornstein-Uhlenbeck (OU) forcing
    tendency with e-folding time :math:`\tau` and integrates that tendency
    exactly over each adaptive step, conditional on the incoming tendency and
    that step's effective diffusion rate. Larger :math:`\tau` produces more
    persistent, longer-correlated forcing.

    Initial states can use the original random balanced flow, an isotropic
    spectrally balanced random-PV field, a smooth periodic balanced double
    jet, or a supplied restart tensor. The double jet has zero net zonal
    transport so its geostrophic height field remains periodic.

    Args:
        parameters_range: Input parameter (min, max) ranges. Supported keys:

            - ``amp``: initial-condition amplitude scale. Required for generated
              initial conditions and invalid for ``"restart"`` because a
              supplied state already fixes the amplitude.
            - ``h_mean``: mean layer depth (scalar) around which spatial
              variations are generated (default 1.0 if omitted).
            - ``drag``: linear drag coefficient (default 2e-3).
            - ``nu``: Laplacian viscosity (default 5e-4).
            - ``beta``: central planetary-vorticity gradient.
            - ``f0``: reference Coriolis parameter; zero disables constant
              rotation.
            - ``initial_wavenumber``: central angular wavenumber for a sampled
              ``"balanced_random_pv"`` initial spectrum.
            - ``initial_bandwidth``: Gaussian-ring width for a sampled
              ``"balanced_random_pv"`` initial spectrum.
            - ``forcing_energy_rate``: diffusion scale in the linearized SWE
              specific-energy norm.
            - ``forcing_correlation_time``: OU e-folding time; zero selects
              white noise.
            - ``forcing_wavenumber``: central angular wavenumber for the
              stochastic forcing spectrum.
            - ``forcing_bandwidth``: Gaussian-ring width for the stochastic
              forcing spectrum.
            - ``forcing_backscatter_fraction``: fraction of diagnosed
              viscosity and hyperviscosity losses returned through forcing.

            If None, generated initial conditions use
            ``{"amp": (0.05, 0.14)}``; restart mode uses no sampled scalar
            parameters.
        output_names, return_timeseries, log_level
            Passed to base. Default outputs: ["h", "u", "v"].
        return_additional_input_fields
            Return the accumulated forcing impulse for each saved state
            transition under the separate ``additional_input_fields`` dataset
            key. Requires ``return_timeseries=True``.
        return_energy_budget
            Return total energy and transition-aligned numerical energy-budget
            terms under the separate ``energy_budget`` dataset key. Requires
            ``return_timeseries=True``.
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
            ``"vortical"``, geostrophically ``"balanced"``, deformation-aware
            ``"pv_balanced"``, or unconstrained ``"momentum"`` forcing.
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
        forcing_backscatter_fraction
            Non-negative fraction of diagnosed Laplacian-viscosity and exact
            hyperviscosity energy loss used as an additional forcing diffusion
            rate. Zero preserves fixed-amplitude forcing.
        backscatter_include_drag
            Whether to include diagnosed linear-drag loss in the backscatter
            source. Disabled by default because drag is normally a physical
            large-scale sink rather than unresolved cascade loss.
        initial_condition
            ``"random"``, ``"balanced_random_pv"``,
            ``"balanced_double_jet"``, or ``"restart"``.
        initial_state
            Restart tensor with shape ``[nx, ny, 3]`` in ``[h, u, v]`` order.
            Required only for ``initial_condition="restart"``. This restores
            the physical state, but not a latent correlated-forcing tendency;
            OU forcing is initialized anew for the restarted forecast.
        initial_wavenumber, initial_bandwidth
            Central angular wavenumber and Gaussian-ring width for
            ``"balanced_random_pv"``. Defaults correspond to mode 4 and a
            1.5-mode bandwidth on the longest domain side.
        jet_mode, jet_perturbation_mode, jet_perturbation_fraction
            Meridional double-jet mode, zonal perturbation mode, and relative
            perturbation velocity scale for ``"balanced_double_jet"``.
        dtype
            torch.float32 or torch.float64.
    """

    def __init__(  # noqa: PLR0912, PLR0915
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
        initial_condition: str = "random",
        initial_state: TensorLike | None = None,
        initial_wavenumber: float | None = None,
        initial_bandwidth: float | None = None,
        jet_mode: int = 1,
        jet_perturbation_mode: int = 6,
        jet_perturbation_fraction: float = 0.05,
        return_energy_budget: bool = False,
        forcing_backscatter_fraction: float = 0.0,
        backscatter_include_drag: bool = False,
    ) -> None:
        """Initialize the planar shallow-water simulator."""
        if parameters_range is None:
            parameters_range = (
                {} if initial_condition == "restart" else {"amp": DEFAULT_AMP_RANGE}
            )
        else:
            parameters_range = dict(parameters_range)
        if output_names is None:
            output_names = ["h", "u", "v"]
        if nx < K_CUT_FACTOR or ny < K_CUT_FACTOR:
            msg = f"nx and ny must be at least {K_CUT_FACTOR}"
            raise ValueError(msg)
        if Lx <= 0 or Ly <= 0:
            msg = "Lx and Ly must be positive"
            raise ValueError(msg)

        if initial_condition == "restart":
            if "amp" in parameters_range:
                msg = "amp is not valid for initial_condition='restart'"
                raise ValueError(msg)
        elif "amp" not in parameters_range:
            msg = "amp is required for generated initial conditions"
            raise ValueError(msg)

        for name in (
            "initial_wavenumber",
            "initial_bandwidth",
            "forcing_wavenumber",
            "forcing_bandwidth",
        ):
            if name not in parameters_range:
                continue
            lower, upper = parameters_range[name]
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower <= 0
                or upper <= 0
                or lower > upper
            ):
                msg = f"{name} range must be a finite positive interval"
                raise ValueError(msg)

        if "forcing_backscatter_fraction" in parameters_range:
            lower, upper = parameters_range["forcing_backscatter_fraction"]
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower < 0
                or lower > upper
            ):
                msg = (
                    "forcing_backscatter_fraction range must be a finite "
                    "non-negative interval"
                )
                raise ValueError(msg)

        super().__init__(parameters_range, output_names, log_level)
        if skip_nt < 0:
            msg = "skip_nt must be non-negative"
            raise ValueError(msg)
        if return_additional_input_fields and not return_timeseries:
            msg = "return_additional_input_fields requires return_timeseries=True"
            raise ValueError(msg)
        if return_energy_budget and not return_timeseries:
            msg = "return_energy_budget requires return_timeseries=True"
            raise ValueError(msg)
        if coriolis_mode not in CORIOLIS_MODES:
            msg = f"coriolis_mode must be one of {CORIOLIS_MODES}"
            raise ValueError(msg)
        if forcing_type not in FORCING_TYPES:
            msg = f"forcing_type must be one of {FORCING_TYPES}"
            raise ValueError(msg)
        if initial_condition not in INITIAL_CONDITIONS:
            msg = f"initial_condition must be one of {INITIAL_CONDITIONS}"
            raise ValueError(msg)
        if initial_condition == "restart" and initial_state is None:
            msg = "initial_state is required for initial_condition='restart'"
            raise ValueError(msg)
        if initial_condition != "restart" and initial_state is not None:
            msg = "initial_state is only valid for initial_condition='restart'"
            raise ValueError(msg)
        if initial_wavenumber is not None and (
            not math.isfinite(initial_wavenumber) or initial_wavenumber <= 0
        ):
            msg = "initial_wavenumber must be positive or None"
            raise ValueError(msg)
        if initial_bandwidth is not None and (
            not math.isfinite(initial_bandwidth) or initial_bandwidth <= 0
        ):
            msg = "initial_bandwidth must be positive or None"
            raise ValueError(msg)
        if forcing_wavenumber is not None and (
            not math.isfinite(forcing_wavenumber) or forcing_wavenumber <= 0
        ):
            msg = "forcing_wavenumber must be positive or None"
            raise ValueError(msg)
        if forcing_bandwidth is not None and (
            not math.isfinite(forcing_bandwidth) or forcing_bandwidth <= 0
        ):
            msg = "forcing_bandwidth must be positive or None"
            raise ValueError(msg)
        if jet_mode <= 0 or jet_perturbation_mode <= 0:
            msg = "jet_mode and jet_perturbation_mode must be positive"
            raise ValueError(msg)
        if jet_perturbation_fraction < 0:
            msg = "jet_perturbation_fraction must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(forcing_backscatter_fraction) or (
            forcing_backscatter_fraction < 0
        ):
            msg = "forcing_backscatter_fraction must be non-negative"
            raise ValueError(msg)
        configured_backscatter_upper = (
            parameters_range["forcing_backscatter_fraction"][1]
            if "forcing_backscatter_fraction" in parameters_range
            else forcing_backscatter_fraction
        )
        if forcing_type == "none" and configured_backscatter_upper > 0:
            msg = "forcing_backscatter_fraction requires stochastic forcing"
            raise ValueError(msg)
        fundamental_wavenumber = 2.0 * math.pi / max(Lx, Ly)
        if forcing_type != "none":
            maximum_forcing_wavenumber = (
                parameters_range["forcing_wavenumber"][1]
                if "forcing_wavenumber" in parameters_range
                else (
                    forcing_wavenumber
                    if forcing_wavenumber is not None
                    else min(
                        DEFAULT_FORCING_RING_MODE_CAP,
                        DEFAULT_FORCING_RING_GRID_FRACTION * min(nx, ny),
                    )
                    * fundamental_wavenumber
                )
            )
            _validate_ring_wavenumber(
                name="forcing_wavenumber",
                wavenumber=maximum_forcing_wavenumber,
                nx=nx,
                ny=ny,
                Lx=Lx,
                Ly=Ly,
                dealias=dealias,
            )
        if initial_condition == "balanced_random_pv":
            maximum_initial_wavenumber = (
                parameters_range["initial_wavenumber"][1]
                if "initial_wavenumber" in parameters_range
                else (
                    initial_wavenumber
                    if initial_wavenumber is not None
                    else DEFAULT_INITIAL_RING_MODE * fundamental_wavenumber
                )
            )
            _validate_ring_wavenumber(
                name="initial_wavenumber",
                wavenumber=maximum_initial_wavenumber,
                nx=nx,
                ny=ny,
                Lx=Lx,
                Ly=Ly,
                dealias=dealias,
            )
        self.return_timeseries = return_timeseries
        self.return_additional_input_fields = return_additional_input_fields
        self.additional_input_names = ["forcing_h", "forcing_u", "forcing_v"]
        self.return_energy_budget = return_energy_budget
        self.energy_budget_names = list(ENERGY_BUDGET_NAMES)
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
        self.initial_condition = initial_condition
        self.initial_state = initial_state
        self.initial_wavenumber = initial_wavenumber
        self.initial_bandwidth = initial_bandwidth
        self.jet_mode = jet_mode
        self.jet_perturbation_mode = jet_perturbation_mode
        self.jet_perturbation_fraction = jet_perturbation_fraction
        self.forcing_backscatter_fraction = forcing_backscatter_fraction
        self.backscatter_include_drag = backscatter_include_drag
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
        amp = (
            0.0
            if self.initial_condition == "restart"
            else float(x[0, self.get_parameter_idx("amp")].item())
        )
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
        forcing_wavenumber = (
            float(x[0, self.get_parameter_idx("forcing_wavenumber")].item())
            if "forcing_wavenumber" in self.param_names
            else self.forcing_wavenumber
        )
        forcing_bandwidth = (
            float(x[0, self.get_parameter_idx("forcing_bandwidth")].item())
            if "forcing_bandwidth" in self.param_names
            else self.forcing_bandwidth
        )
        forcing_backscatter_fraction = (
            float(x[0, self.get_parameter_idx("forcing_backscatter_fraction")].item())
            if "forcing_backscatter_fraction" in self.param_names
            else self.forcing_backscatter_fraction
        )
        initial_wavenumber = (
            float(x[0, self.get_parameter_idx("initial_wavenumber")].item())
            if "initial_wavenumber" in self.param_names
            else self.initial_wavenumber
        )
        initial_bandwidth = (
            float(x[0, self.get_parameter_idx("initial_bandwidth")].item())
            if "initial_bandwidth" in self.param_names
            else self.initial_bandwidth
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
            forcing_wavenumber=forcing_wavenumber,
            forcing_bandwidth=forcing_bandwidth,
            dtype=self.dtype,
            forcing_correlation_time=forcing_correlation_time,
            f0=f0,
            initial_condition=self.initial_condition,
            initial_state=self.initial_state,
            initial_wavenumber=initial_wavenumber,
            initial_bandwidth=initial_bandwidth,
            jet_mode=self.jet_mode,
            jet_perturbation_mode=self.jet_perturbation_mode,
            jet_perturbation_fraction=self.jet_perturbation_fraction,
            return_energy_budget=self.return_energy_budget,
            forcing_backscatter_fraction=forcing_backscatter_fraction,
            backscatter_include_drag=self.backscatter_include_drag,
        )
        if self.return_energy_budget:
            if not isinstance(y, tuple):
                msg = "energy-budget simulation did not return diagnostics"
                raise RuntimeError(msg)
            state, energy_budget = y
            return torch.cat((state.flatten(), energy_budget.flatten())).unsqueeze(0)
        if isinstance(y, tuple):
            msg = "unexpected energy-budget diagnostics"
            raise RuntimeError(msg)
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
        state_features_per_step = self.nx * self.ny * channels
        budget_features_per_step = (
            len(self.energy_budget_names) if self.return_energy_budget else 0
        )
        features_per_step = state_features_per_step + budget_features_per_step

        if self.return_timeseries:
            total = y.shape[1]
            n_time = total // features_per_step
            state_total = n_time * state_features_per_step
            state_flat = y[:, :state_total]
            budget_flat = y[:, state_total:]
            y = state_flat.reshape(n_valid, n_time, self.nx, self.ny, channels)
        else:
            n_time = 1
            state_total = state_features_per_step
            state_flat = y[:, :state_total]
            budget_flat = y[:, state_total:]
            y = state_flat.reshape(n_valid, 1, self.nx, self.ny, channels)

        additional_input_fields = None
        if self.return_additional_input_fields:
            additional_input_fields = y[..., state_channels:]
            y = y[..., :state_channels]

        energy_budget = None
        if self.return_energy_budget:
            energy_budget = budget_flat.reshape(
                n_valid, n_time, len(self.energy_budget_names)
            )

        return {
            "data": y,
            "additional_input_fields": additional_input_fields,
            "energy_budget": energy_budget,
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


def _swe_forcing_streamfunction_transfer(
    *,
    forcing_type: str,
    g: float,
    h_mean: float,
    f0: float,
    K2: torch.Tensor,
    K2_inv: torch.Tensor,
) -> torch.Tensor:
    """Return the scalar-to-streamfunction transfer for a forcing geometry."""
    if forcing_type != "pv_balanced":
        return K2_inv

    deformation_wavenumber_squared = f0**2 / (g * h_mean) if g > 0 else 0.0
    return torch.where(
        K2 > 0,
        -h_mean / (K2 + deformation_wavenumber_squared),
        torch.zeros_like(K2_inv),
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
    K2: torch.Tensor,
    K2_inv: torch.Tensor,
    dKx: torch.Tensor,
    dKy: torch.Tensor,
) -> float:
    """Return expected energy before scaling a unit Gaussian forcing draw."""
    if forcing_type in ("vortical", "balanced", "pv_balanced"):
        psi_transfer = _swe_forcing_streamfunction_transfer(
            forcing_type=forcing_type,
            g=g,
            h_mean=h_mean,
            f0=f0,
            K2=K2,
            K2_inv=K2_inv,
        )
        energy_transfer = 0.5 * psi_transfer.square() * (dKx.square() + dKy.square())
        if forcing_type != "vortical":
            if g > 0:
                energy_transfer += (
                    0.5 * (g / h_mean) * (f0 / g) ** 2 * psi_transfer.square()
                )
            elif f0 != 0:
                msg = f"{forcing_type} forcing with g=0 requires f0=0"
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
    K2: torch.Tensor,
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

    if forcing_type in ("vortical", "balanced", "pv_balanced"):
        scalar_increment_hat = sample_filtered_scalar_hat(
            nx=nx,
            ny=ny,
            dtype=dtype,
            spectrum=spectrum,
            mask=mask,
        )
        psi_transfer = _swe_forcing_streamfunction_transfer(
            forcing_type=forcing_type,
            g=g,
            h_mean=h_mean,
            f0=f0,
            K2=K2,
            K2_inv=K2_inv,
        )
        psi_increment_hat = psi_transfer * scalar_increment_hat
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
            msg = f"{forcing_type} forcing with g=0 requires f0=0"
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
            K2=K2,
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
    initial_condition: str = "random",
    initial_state: TensorLike | None = None,
    initial_wavenumber: float | None = None,
    initial_bandwidth: float | None = None,
    jet_mode: int = 1,
    jet_perturbation_mode: int = 6,
    jet_perturbation_fraction: float = 0.05,
    return_energy_budget: bool = False,
    forcing_backscatter_fraction: float = 0.0,
    backscatter_include_drag: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate full shallow-water equations with PDEArena-style random2 ICs.

    Named stochastic forcing modes share a Gaussian spatial spectral ring.
    ``forcing_correlation_time=0`` uses independent white-in-time increments.
    Positive correlation time evolves an OU forcing tendency with exact
    exponential memory and samples its exact time integral at each adaptive
    step, conditional on the incoming tendency and the effective diffusion
    rate held over that step. Its stationary linearized expected specific
    energy is
    ``forcing_energy_rate / (2 * correlation_time)``, so its long-time
    integrated diffusion rate is ``forcing_energy_rate``. The exact integral
    also converges to the white-noise increment as the correlation time tends
    to zero.

    A positive ``forcing_backscatter_fraction`` adds the requested fraction
    of diagnosed Laplacian-viscosity and exact hyperviscosity loss to the
    configured base diffusion rate. Linear-drag loss is included only when
    ``backscatter_include_drag=True``.

    When ``return_additional_input_fields=True``, three forcing-impulse channels
    are appended after ``[h, u, v]``. At saved index ``i`` they contain the sum
    of ``[dh, du, dv]`` increments used to advance state ``i`` to state ``i+1``;
    the final entry is zero because it has no following transition.

    When ``return_energy_budget=True``, the second returned tensor contains
    total energy at every saved state followed by transition-aligned energy
    changes from deterministic RK4 evolution, hyperviscosity, and stochastic
    forcing. Their sum exactly closes the numerical total-energy change. The
    reported Laplacian-viscosity and drag entries are positive loss estimates
    accumulated over each saved transition. They are used to interpret
    dissipation, not as extra terms in that closure. The effective forcing
    energy rate is an interval mean.
    """
    if forcing_type not in FORCING_TYPES:
        msg = f"forcing_type must be one of {FORCING_TYPES}"
        raise ValueError(msg)
    if initial_condition not in INITIAL_CONDITIONS:
        msg = f"initial_condition must be one of {INITIAL_CONDITIONS}"
        raise ValueError(msg)
    if initial_condition == "restart" and initial_state is None:
        msg = "initial_state is required for initial_condition='restart'"
        raise ValueError(msg)
    if initial_condition != "restart" and initial_state is not None:
        msg = "initial_state is only valid for initial_condition='restart'"
        raise ValueError(msg)
    if initial_wavenumber is not None and (
        not math.isfinite(initial_wavenumber) or initial_wavenumber <= 0
    ):
        msg = "initial_wavenumber must be positive or None"
        raise ValueError(msg)
    if initial_bandwidth is not None and (
        not math.isfinite(initial_bandwidth) or initial_bandwidth <= 0
    ):
        msg = "initial_bandwidth must be positive or None"
        raise ValueError(msg)
    if forcing_wavenumber is not None and (
        not math.isfinite(forcing_wavenumber) or forcing_wavenumber <= 0
    ):
        msg = "forcing_wavenumber must be positive or None"
        raise ValueError(msg)
    if forcing_bandwidth is not None and (
        not math.isfinite(forcing_bandwidth) or forcing_bandwidth <= 0
    ):
        msg = "forcing_bandwidth must be positive or None"
        raise ValueError(msg)
    if jet_mode <= 0 or jet_perturbation_mode <= 0:
        msg = "jet_mode and jet_perturbation_mode must be positive"
        raise ValueError(msg)
    if jet_perturbation_fraction < 0:
        msg = "jet_perturbation_fraction must be non-negative"
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
    if return_energy_budget and not return_timeseries:
        msg = "return_energy_budget requires return_timeseries=True"
        raise ValueError(msg)
    if nx < K_CUT_FACTOR or ny < K_CUT_FACTOR:
        msg = f"nx and ny must be at least {K_CUT_FACTOR}"
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
    if not math.isfinite(forcing_backscatter_fraction) or (
        forcing_backscatter_fraction < 0
    ):
        msg = "forcing_backscatter_fraction must be non-negative"
        raise ValueError(msg)
    if forcing_type == "none" and forcing_backscatter_fraction > 0:
        msg = "forcing_backscatter_fraction requires stochastic forcing"
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
    if g == 0 and forcing_type in ("balanced", "pv_balanced") and f0 != 0:
        msg = f"{forcing_type} forcing with g=0 requires f0=0"
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
            forcing_mode = min(
                DEFAULT_FORCING_RING_MODE_CAP,
                DEFAULT_FORCING_RING_GRID_FRACTION * min(nx, ny),
            )
            forcing_wavenumber = forcing_mode * fundamental_wavenumber
        if forcing_bandwidth is None:
            forcing_bandwidth = DEFAULT_RING_BANDWIDTH_MODES * fundamental_wavenumber
        _validate_ring_wavenumber(
            name="forcing_wavenumber",
            wavenumber=forcing_wavenumber,
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            dealias=dealias,
        )
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
            K2=K2,
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
                K2=K2,
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
    # Initial conditions                                                  #
    # ------------------------------------------------------------------ #
    if initial_condition == "restart":
        if initial_state is None:
            msg = "initial_state is required for initial_condition='restart'"
            raise ValueError(msg)
        restart = initial_state.detach().to(dtype=dtype, device=X.device)
        if restart.shape == (1, nx, ny, 3):
            restart = restart[0]
        if restart.shape != (nx, ny, 3):
            msg = f"initial_state must have shape ({nx}, {ny}, 3) or (1, {nx}, {ny}, 3)"
            raise ValueError(msg)
        if not torch.isfinite(restart).all():
            msg = "initial_state must contain only finite values"
            raise ValueError(msg)
        if torch.any(restart[..., 0] <= 0):
            msg = "initial_state height must be strictly positive"
            raise ValueError(msg)
        h0, u0, v0 = restart.unbind(dim=-1)
    elif initial_condition == "balanced_random_pv":
        fundamental_wavenumber = 2.0 * math.pi / max(Lx, Ly)
        pv_wavenumber = (
            DEFAULT_INITIAL_RING_MODE * fundamental_wavenumber
            if initial_wavenumber is None
            else initial_wavenumber
        )
        pv_bandwidth = (
            DEFAULT_RING_BANDWIDTH_MODES * fundamental_wavenumber
            if initial_bandwidth is None
            else initial_bandwidth
        )
        _validate_ring_wavenumber(
            name="initial_wavenumber",
            wavenumber=pv_wavenumber,
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            dealias=dealias,
        )
        pv_spectrum = gaussian_ring_spectrum(
            K2, pv_wavenumber, pv_bandwidth, dealias_mask
        )
        pv_hat = sample_filtered_scalar_hat(
            nx=nx,
            ny=ny,
            dtype=dtype,
            spectrum=pv_spectrum,
            mask=dealias_mask,
        )
        psi_transfer = _swe_forcing_streamfunction_transfer(
            forcing_type="pv_balanced",
            g=g,
            h_mean=h_mean,
            f0=f0,
            K2=K2,
            K2_inv=K2_inv,
        )
        psi_h = psi_transfer * pv_hat
        u0 = to_phys(-iKy * psi_h)
        v0 = to_phys(iKx * psi_h)
        speed_rms = torch.sqrt((u0.square() + v0.square()).mean())
        if not torch.isfinite(speed_rms) or float(speed_rms) <= EPS:
            msg = "balanced_random_pv produced zero or non-finite velocity"
            raise RuntimeError(msg)
        scale = amp / float(speed_rms)
        psi_h *= scale
        u0 *= scale
        v0 *= scale
        psi0 = to_phys(psi_h)
        h0 = h_mean + (f0 / g) * psi0 if g > 0 else torch.full_like(psi0, h_mean)
        if torch.any(h0 <= 0):
            msg = (
                "balanced_random_pv produced non-positive height; "
                "reduce amp or increase initial_wavenumber"
            )
            raise RuntimeError(msg)
    elif initial_condition == "balanced_double_jet":
        meridional_wavenumber = 2.0 * math.pi * jet_mode / Ly
        zonal_wavenumber = 2.0 * math.pi * jet_perturbation_mode / Lx
        peak_speed = math.sqrt(2.0) * amp
        psi_base = (peak_speed / meridional_wavenumber) * torch.cos(
            meridional_wavenumber * Y
        )
        perturbation_wavenumber = math.hypot(zonal_wavenumber, meridional_wavenumber)
        perturbation_streamfunction = (
            jet_perturbation_fraction * peak_speed / perturbation_wavenumber
        )
        perturbation_phase = float(torch.rand(1)) * 2.0 * math.pi
        psi_perturbation = (
            perturbation_streamfunction
            * torch.cos(zonal_wavenumber * X + perturbation_phase)
            * torch.cos(meridional_wavenumber * Y)
        )
        psi_h = to_spec(psi_base + psi_perturbation) * dealias_mask
        psi_h[0, 0] = 0.0
        psi0 = to_phys(psi_h)
        u0 = to_phys(-iKy * psi_h)
        v0 = to_phys(iKx * psi_h)
        h0 = (
            (h_mean + (f0 / g) * psi0).clamp(min=0.5 * h_mean)
            if g > 0
            else torch.full_like(psi0, h_mean)
        )
    else:
        # Specify vorticity, invert ∇²ψ=ζ, then derive balanced u, v, and h.
        k_min = 2.0 * math.pi / max(Lx, Ly)
        k_cut = k_min * (min(nx, ny) // K_CUT_FACTOR)

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

        coeff = torch.randn(nx, N_JET_MODES, dtype=dtype)
        y_frac = Y / Ly
        u_jet_field = torch.stack(
            [
                coeff[:, m].unsqueeze(1) * torch.sin((m + 1) * math.pi * y_frac)
                for m in range(N_JET_MODES)
            ],
            dim=0,
        ).sum(dim=0)
        jet_std = float(u_jet_field.std()) + EPS
        u_jet_field = u_jet_field * (amp * JET_AMP_FRAC / jet_std)
        zeta_jet = to_phys(-iKy * to_spec(u_jet_field))

        zeta_jet_scale = float(zeta_jet.std())
        perturbation_amplitude = max(amp * f0 * JET_AMP_FRAC, zeta_jet_scale * 0.25)
        y_center = PERT_LAT_FRAC * Ly
        y_width = PERT_WIDTH_FRAC * Ly
        perturbation_phase = float(torch.rand(1)) * 2.0 * math.pi
        zeta_perturbation = (
            perturbation_amplitude
            * torch.cos(WAVE_ZONAL_MODE * 2.0 * math.pi * X / Lx + perturbation_phase)
            * torch.exp(-0.5 * ((Y - y_center) / y_width) ** 2)
        )

        zeta_h = to_spec(zeta_random + zeta_jet + zeta_perturbation)
        zeta_h = torch.where(
            (k_cut**2 > K2) & dealias_mask,
            zeta_h,
            torch.zeros_like(zeta_h),
        )
        zeta_h[0, 0] = 0.0
        psi_h = K2_inv * zeta_h
        psi0 = to_phys(psi_h)
        u0 = to_phys(-iKy * psi_h)
        v0 = to_phys(iKx * psi_h)
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

    def total_energy(h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return exact numerical SWE energy relative to the mean layer."""
        kinetic = 0.5 * h * (u.square() + v.square())
        potential = 0.5 * g * (h - h_mean).square()
        return (kinetic + potential).mean()

    def dissipation_rate_estimates(
        h: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive viscosity and drag loss-rate estimates."""
        h_safe = h.clamp(min=H_MIN_CLIP)
        lap_u = to_phys(-K2 * to_spec(u))
        lap_v = to_phys(-K2 * to_spec(v))
        viscous_work = nu * (h_safe * (u * lap_u + v * lap_v)).mean()
        drag_work = -drag * (h_safe * (u.square() + v.square())).mean()
        return (-viscous_work).clamp(min=0.0), (-drag_work).clamp(min=0.0)

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
    energy_history = [total_energy(h, u, v)] if return_energy_budget else []
    energy_intervals: list[torch.Tensor] = []
    energy_since_save = torch.zeros(len(ENERGY_BUDGET_NAMES) - 1, dtype=dtype)
    energy_interval_duration = 0.0
    zero_energy = torch.zeros((), dtype=dtype)
    track_dissipation = return_energy_budget or (
        forcing_type != "none" and forcing_backscatter_fraction > 0
    )
    forcing_tendency: torch.Tensor | None = None
    if forcing_type != "none" and forcing_correlation_time > 0:
        stationary_energy = forcing_energy_rate / (2.0 * forcing_correlation_time)
        forcing_tendency = sample_forcing_field(stationary_energy)
    next_save_idx = 1
    failure_reason: str | None = None

    while t < T - 1e-12:
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

        energy_before_step = zero_energy
        energy_after_deterministic = zero_energy
        energy_after_hyperviscosity = zero_energy
        energy_before_forcing = zero_energy
        viscous_loss_rate = zero_energy
        drag_loss_rate = zero_energy
        if track_dissipation:
            energy_before_step = total_energy(h, u, v)
            viscous_loss_rate, drag_loss_rate = dissipation_rate_estimates(h, u, v)
        h, u, v = rk4_step(h, u, v, step_dt)
        if track_dissipation:
            energy_after_deterministic = total_energy(h, u, v)

        # Apply hyperviscosity integrating factor to all fields (spectral filter).
        hyp_factor = torch.exp(hyp_op * step_dt) * dealias_mask
        u = to_phys(to_spec(u) * hyp_factor)
        v = to_phys(to_spec(v) * hyp_factor)
        h_field_mean = h.mean()
        h_anom = h - h_field_mean
        h_anom = to_phys(to_spec(h_anom) * hyp_factor)
        h = (h_field_mean + h_anom).clamp(min=H_MIN_CLIP)
        if track_dissipation:
            energy_after_hyperviscosity = total_energy(h, u, v)

        effective_forcing_energy_rate = 0.0
        if return_energy_budget:
            energy_before_forcing = energy_after_hyperviscosity

        if forcing_type != "none":
            hyperviscous_loss_rate = (
                energy_after_deterministic - energy_after_hyperviscosity
            ).clamp(min=0.0) / step_dt
            diagnosed_dissipation_rate = viscous_loss_rate + hyperviscous_loss_rate
            if backscatter_include_drag:
                diagnosed_dissipation_rate += drag_loss_rate
            effective_forcing_energy_rate = (
                forcing_energy_rate
                + forcing_backscatter_fraction
                * float(diagnosed_dissipation_rate.item())
            )
            if forcing_correlation_time == 0:
                forcing_increment = sample_forcing_field(
                    effective_forcing_energy_rate * step_dt
                )
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
                stationary_energy = effective_forcing_energy_rate / (
                    2.0 * forcing_correlation_time
                )
                endpoint_innovation = sample_forcing_field(stationary_energy)
                previous_tendency = forcing_tendency
                forcing_tendency = decay * previous_tendency + (
                    endpoint_innovation_weight * endpoint_innovation
                )
                integral_innovation = sample_forcing_field(
                    effective_forcing_energy_rate * integral_innovation_variance
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
        if return_energy_budget:
            energy_after_forcing = total_energy(h, u, v)
            energy_since_save[0] += energy_after_deterministic - energy_before_step
            energy_since_save[1] += (
                energy_after_hyperviscosity - energy_after_deterministic
            )
            energy_since_save[2] += energy_after_forcing - energy_before_forcing
            energy_since_save[3] += viscous_loss_rate * step_dt
            energy_since_save[4] += drag_loss_rate * step_dt
            energy_since_save[5] += effective_forcing_energy_rate * step_dt
            energy_interval_duration += step_dt
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
            if return_energy_budget:
                energy_history.append(total_energy(h, u, v))
                energy_interval = energy_since_save.clone()
                energy_interval[-1] /= energy_interval_duration
                energy_intervals.append(energy_interval)
                energy_since_save.zero_()
                energy_interval_duration = 0.0
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
        state_history = state_history[skip_nt:]
        if return_energy_budget:
            final_interval = torch.zeros(len(ENERGY_BUDGET_NAMES) - 1, dtype=dtype)
            energy_budget = torch.stack(
                [
                    torch.cat((energy.unsqueeze(0), interval))
                    for energy, interval in zip(
                        energy_history[:-1], energy_intervals, strict=True
                    )
                ]
                + [torch.cat((energy_history[-1].unsqueeze(0), final_interval))]
            )
            return state_history, energy_budget[skip_nt:].float()
        return state_history
    return output(h, u, v).unsqueeze(0)
