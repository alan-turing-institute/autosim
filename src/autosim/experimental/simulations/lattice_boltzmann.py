"""
Lattice Boltzmann Method (LBM) Simulator for 2D Fluid Flow.

This module implements a differentiable D2Q9 LBM solver for incompressible
Navier-Stokes equations. It supports complex boundary conditions (like obstacles)
and inflow/outflow profiles, making it suitable for benchmarks like Flow Past Cylinder.
"""

import math

import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class LatticeBoltzmann(SpatioTemporalSimulator):
    r"""Lattice Boltzmann (D2Q9) simulator for channel flow with obstacles.

    Simulates 2D flow past a cylinder using the BGK collision model.
    The simulation domain is a rectangular channel controlled by ``width`` and
    ``height``.

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Bounds on sampled parameters:
        - ``viscosity``: Kinematic viscosity (0.01-0.05 typically).
        - ``u_in``: Maximum inflow velocity (keep < 0.15 for stability).
                - ``oscillation_frequency``: Inlet oscillation frequency in cycles per
                    unit simulation time.
    output_names: list[str], optional
        Names for output channels. Defaults to
        ``["vorticity", "velocity_x", "velocity_y", "rho"]``.
    return_timeseries: bool, default=False
        If True, returns full trajectory; otherwise final frame only.
    use_cylinder: bool, default=True
        If True, include the circular obstacle. If False, run a plain channel.
    oscillatory_inlet: bool | None, default=None
        If True, apply time-dependent inlet modulation (useful for rich dynamics
        in no-cylinder channels). If None, defaults to ``not use_cylinder``.
    width: int, default=128
        Grid width (Nx).
    height: int, default=64
        Grid height (Ny).
    T: float, default=4.0
        Total simulation time (in seconds, approximate).
    dt: float, default=1/250
        Temporal step size in simulation-time units. Smaller values increase
        internal step count and reduce jumpiness in returned trajectories.
    n_saved_frames: int | None, default=None
        Number of saved frames for returned timeseries. If None, save every
        post-warmup LBM step (temporal resolution scales with ``T``). If set,
        the simulator samples exactly that many frames when possible (capped at
        available post-warmup steps).
    skip_nt: int, default=0
        Number of initial saved trajectory frames to drop from returned
        timeseries outputs.
    warmup_steps: int, default=200
        Number of initial LBM steps to run without recording to let the
        inlet flow establish and the simulation stabilize.
    """

    _REQUIRED_PARAMETER_NAMES = ("viscosity", "u_in", "oscillation_frequency")

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        width: int = 128,
        height: int = 64,
        T: float = 4.0,
        dt: float = 1.0 / 250.0,
        use_cylinder: bool = True,
        oscillatory_inlet: bool | None = None,
        n_saved_frames: int | None = None,
        skip_nt: int = 0,
        warmup_steps: int = 200,
    ) -> None:
        if parameters_range is None:
            # Re ~ u_in * D / nu. D=height/5 approx.
            # If u=0.1, D=10, nu=0.02 -> Re=50 (vortex shedding).
            parameters_range = {
                "viscosity": (0.01, 0.05),
                "u_in": (0.04, 0.10),
                "oscillation_frequency": (0.5, 2.5),
            }
        if output_names is None:
            output_names = ["vorticity", "velocity_x", "velocity_y", "rho"]

        missing_params = [
            name
            for name in self._REQUIRED_PARAMETER_NAMES
            if name not in parameters_range
        ]
        if missing_params:
            missing_str = ", ".join(missing_params)
            msg = (
                "parameters_range is missing required keys: "
                f"{missing_str}. Required keys are: {self._REQUIRED_PARAMETER_NAMES}"
            )
            raise ValueError(msg)

        for key in self._REQUIRED_PARAMETER_NAMES:
            lower, upper = parameters_range[key]
            if lower > upper:
                msg = f"Invalid range for {key}: lower bound must be <= upper bound"
                raise ValueError(msg)
        if parameters_range["oscillation_frequency"][0] < 0:
            msg = "oscillation_frequency range must be non-negative"
            raise ValueError(msg)

        if n_saved_frames is not None and n_saved_frames <= 0:
            msg = "n_saved_frames must be positive when provided"
            raise ValueError(msg)
        if dt <= 0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if skip_nt < 0:
            msg = "skip_nt must be non-negative"
            raise ValueError(msg)
        if warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
            raise ValueError(msg)

        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        self.width = width
        self.height = height
        self.T = T
        self.dt = dt
        self.use_cylinder = use_cylinder
        if oscillatory_inlet is None:
            oscillatory_inlet = not use_cylinder
        self.oscillatory_inlet = oscillatory_inlet
        self.n_saved_frames = n_saved_frames
        self.skip_nt = skip_nt
        self.warmup_steps = warmup_steps

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, "Simulator._forward expects a single input"
        values_by_name = {
            name: float(x[0, idx].item()) for idx, name in enumerate(self.param_names)
        }
        sample = torch.tensor(
            [
                values_by_name["viscosity"],
                values_by_name["u_in"],
                values_by_name["oscillation_frequency"],
            ],
            dtype=torch.float32,
            device=x.device,
        )

        out = simulate_lbm_cylinder(
            params=sample,
            return_timeseries=self.return_timeseries,
            width=self.width,
            height=self.height,
            duration=self.T,
            dt=self.dt,
            use_cylinder=self.use_cylinder,
            oscillatory_inlet=self.oscillatory_inlet,
            n_saved_frames=self.n_saved_frames,
            warmup_steps=self.warmup_steps,
        )
        return out.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Run sampled trajectories and return spatiotemporal tensors."""
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )

        # LBM outputs: [B, Steps*Features] or [B, Features]
        channels = len(self.output_names)  # 4
        features_per_frame = self.width * self.height * channels

        if self.return_timeseries:
            total_elements = y.shape[1]
            steps = total_elements // features_per_frame
            y_reshaped = y.reshape(y.shape[0], steps, self.height, self.width, channels)

            if self.skip_nt >= steps:
                raise ValueError(
                    "skip_nt is too large for the available trajectory length; "
                    f"skip_nt={self.skip_nt}, steps={steps}."
                )
            y_reshaped = y_reshaped[:, self.skip_nt :, ...]
        else:
            y_reshaped = y.reshape(y.shape[0], 1, self.height, self.width, channels)

        return {
            "data": y_reshaped,
            "constant_scalars": x,
            "constant_fields": None,
        }


def curl_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute vorticity = dv/dx - du/dy using central differences."""
    dv_dx = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) * 0.5
    du_dy = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) * 0.5
    return dv_dx - du_dy


def simulate_lbm_cylinder(  # noqa: PLR0912, PLR0915
    params: TensorLike,
    return_timeseries: bool,
    width: int,
    height: int,
    duration: float,
    dt: float = 1.0 / 250.0,
    use_cylinder: bool = True,
    oscillatory_inlet: bool = False,
    n_saved_frames: int | None = None,
    warmup_steps: int = 200,
) -> TensorLike:
    """
    Simulate flow past a cylinder using D2Q9 Lattice Boltzmann.

    Args:
        params: Tensor-like with ``[viscosity, u_in, oscillation_frequency]``.
        return_timeseries: Whether to return all saved frames or only the final one.
        width: Grid width.
        height: Grid height.
        duration: Physical duration of the simulated trajectory.
        dt: Simulation timestep.
        use_cylinder: Whether to include a circular obstacle.
        oscillatory_inlet: Whether to modulate inlet velocity over time.
        n_saved_frames: Number of post-warmup frames to keep. If ``None``, save all.
        warmup_steps: Number of initial steps to run without recording.

    Coordinate system: x (width, index 1), y (height, index 0).
    """
    device = params.device

    # 1. Unpack parameters
    viscosity = float(params[0].item())
    u_max = float(params[1].item())
    oscillation_frequency = float(params[2].item())

    # LBM Relaxation
    # nu = (tau - 0.5)/3  => tau = 3*nu + 0.5
    tau = 3.0 * viscosity + 0.5
    omega = 1.0 / tau

    if dt <= 0:
        msg = "dt must be positive"
        raise ValueError(msg)
    if oscillation_frequency < 0:
        msg = "oscillation_frequency must be non-negative"
        raise ValueError(msg)

    total_steps = max(1, int(duration / dt))

    save_all_steps = n_saved_frames is None
    if save_all_steps:
        save_step_set: set[int] | None = None
    else:
        n_target = min(int(n_saved_frames), total_steps)
        if n_target == 1:
            save_indices = torch.tensor([total_steps - 1], dtype=torch.long)
        elif n_target == total_steps:
            save_indices = torch.arange(total_steps, dtype=torch.long)
        else:
            ratio = (total_steps - 1) / (n_target - 1)
            save_indices = torch.round(torch.arange(n_target) * ratio).to(torch.long)
        save_step_set = set(save_indices.tolist())

    # D2Q9 Constants
    # Weights for D2Q9
    w = torch.tensor(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
        device=device,
    )

    # Lattice velocities c_i = [cy, cx]  (row, col) convention
    # 0: (0,0), 1: (0,1) E, 2: (1,0) N, 3: (0,-1) W, 4: (-1,0) S
    # 5: (1,1) NE, 6: (1,-1) NW, 7: (-1,-1) SW, 8: (-1,1) SE
    # Note: cx corresponds to 'width' dim (dim 1), cy to 'height' dimension (dim 0)

    # Correct mapping for standard D2Q9:
    # 0: Rest
    # 1: East  (0, 1)
    # 2: North (1, 0)
    # 3: West  (0, -1)
    # 4: South (-1, 0)
    # 5: NE    (1, 1)
    # 6: NW    (1, -1)
    # 7: SW    (-1, -1)
    # 8: SE    (-1, 1)

    c = torch.tensor(
        [
            [0, 0],  # 0
            [0, 1],  # 1 E
            [1, 0],  # 2 N
            [0, -1],  # 3 W
            [-1, 0],  # 4 S
            [1, 1],  # 5 NE
            [1, -1],  # 6 NW
            [-1, -1],  # 7 SW
            [-1, 1],  # 8 SE
        ],
        device=device,
        dtype=torch.long,
    )  # Shape (9, 2) [cy, cx]

    # Opposite indices for bounce-back
    # 0->0, 1->3, 2->4, 3->1, 4->2, 5->7, 6->8, 7->5, 8->6
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device, dtype=torch.long)

    # 2. Geometry
    # Coordinate grids: y (rows) 0..H-1, x (cols) 0..W-1
    y_coord = torch.arange(height, device=device).view(-1, 1).repeat(1, width)
    x_coord = torch.arange(width, device=device).view(1, -1).repeat(height, 1)

    # Cylinder obstacle (optional)
    if use_cylinder:
        # Geometry from: https://doi.org/10.1098/rspa.2023.0655
        # domain (0, 2.2) x (0, 0.41), cylinder at (0.2, 0.2) with r=0.05
        cy = height * (0.2 / 0.41)
        cx = width * (0.2 / 2.2)
        radius = height * (0.05 / 0.41)
        obstacle_mask = (
            (x_coord.float() - cx) ** 2 + (y_coord.float() - cy) ** 2
        ) <= radius**2
    else:
        obstacle_mask = torch.zeros((height, width), dtype=torch.bool, device=device)

    # 3. Initialization
    rho = torch.ones((height, width), device=device)

    # Parabolic Profile for Inflow (u_x only)
    # y goes from 0 to H-1. Normalized y' = y/(H-1)
    ys = torch.arange(height, device=device).float()
    # Profile u(y) = 4 * u_max * (y/H) * (1 - y/H) roughly
    # Let's align zeros at boundaries exactly
    profile = 4.0 * u_max * (ys / (height - 1)) * (1.0 - (ys / (height - 1)))

    # Initialize velocity everywhere to zero except maybe ramp up?
    # Better start from zero to avoid shocks
    u = torch.zeros((2, height, width), device=device)

    # Prepare inlet profile for later use
    u_inlet = torch.zeros((2, height, 1), device=device)
    u_inlet[0, :, 0] = profile
    # Secondary profile to inject a weak cross-stream oscillation
    lateral_profile = torch.sin(math.pi * ys / (height - 1)).view(height, 1)

    def compute_equilibrium(rho, u):
        # rho: (H, W) or (H, 1)
        # u: (2, H, W) or (2, H, 1)
        # Returns f_eq: (9, H, W)

        # Calculate u.u
        u2 = u[0] ** 2 + u[1] ** 2

        # Calculate c.u
        # c: (9, 2) -> cx, cy
        cy = c[:, 0].view(9, 1, 1).float()
        cx = c[:, 1].view(9, 1, 1).float()

        ux = u[0].unsqueeze(0)  # (1, H, W)
        uy = u[1].unsqueeze(0)

        cu = cx * ux + cy * uy  # (9, H, W)

        # Expansion
        # feq = w * rho * (1 + 3cu + 4.5(cu)^2 - 1.5u^2)
        term1 = 1.0 + 3.0 * cu
        term2 = 4.5 * (cu**2)
        term3 = -1.5 * u2.unsqueeze(0)

        return w.view(9, 1, 1) * rho.unsqueeze(0) * (term1 + term2 + term3)

    # Initialize f to equilibrium at rest (rho=1, u=0)
    f = compute_equilibrium(rho, u)

    # Pre-calculate inlet equilibrium distribution for steady inlet
    # We assume inlet density is fixed at 1.0 (approximated incompressible)
    inlet_rho = torch.ones((height, 1), device=device)
    f_inlet_eq = compute_equilibrium(inlet_rho, u_inlet)

    history = []

    def get_snapshot(u_curr, rho_curr):
        vort = curl_2d(u_curr[0], u_curr[1])
        # Force u=0 on obstacle for visualization cleanly
        u_viz = u_curr.clone()
        u_viz[:, obstacle_mask] = 0
        return torch.stack([vort, u_viz[0], u_viz[1], rho_curr], dim=-1)

    # Main Loop
    # Order: Collision -> Streaming -> Boundary
    # Use double buffering implicitly? No, simplistic single f update with careful
    # boundary overwritting is standard in simple codes
    # Actually, standard is: Stream (propagates) then Collide (locally).
    # Or Collide (locally) then Stream.
    # Let's do: Collision -> Streaming -> BCs.

    full_duration = total_steps + warmup_steps
    for step in range(full_duration):
        # --- 1. Macroscopic ---
        rho = torch.sum(f, dim=0)
        inv_rho = 1.0 / (rho + 1e-6)
        ux = torch.sum(f * c[:, 1].view(9, 1, 1).float(), dim=0) * inv_rho
        uy = torch.sum(f * c[:, 0].view(9, 1, 1).float(), dim=0) * inv_rho
        u = torch.stack([ux, uy], dim=0)

        # --- 2. Collision (BGK) ---
        f_eq = compute_equilibrium(rho, u)
        f_new = f * (1.0 - omega) + f_eq * omega
        if use_cylinder:
            # Skip collision inside the obstacle to preserve bounce-back populations
            f_new[:, obstacle_mask] = f[:, obstacle_mask]
        f = f_new

        # --- 3. Streaming ---
        # "Pull" or "Push"?
        # torch.roll "pushes": correct shift relative to index
        for i in range(9):
            dy = int(c[i, 0].item())
            dx = int(c[i, 1].item())
            if dx == 0 and dy == 0:
                continue

            # Shift +1 means move data to right/down
            f[i] = torch.roll(f[i], shifts=(dy, dx), dims=(0, 1))

        # --- 4. Boundary Conditions ---

        # A. Walls (Top/Bottom, y=0, y=H-1)
        # Bounce-back (No Slip)
        # Reflect populations at top/bottom rows
        f[:, 0, :] = f[opp, 0, :]
        f[:, -1, :] = f[opp, -1, :]

        # B. Inlet (West, x=0)
        # Fixed or oscillatory equilibrium profile (Dirichlet velocity).
        if oscillatory_inlet:
            # Drive richer unsteady dynamics, especially useful when no cylinder is used
            phase = 2.0 * math.pi * oscillation_frequency * (step * dt)
            amp_u = 1.0 + 0.25 * math.sin(phase)
            amp_v = 0.10 * math.sin(phase)

            u_inlet_dynamic = torch.zeros((2, height, 1), device=device)
            u_inlet_dynamic[0, :, 0] = torch.clamp(profile * amp_u, min=0.0, max=0.18)
            u_inlet_dynamic[1, :, 0] = amp_v * lateral_profile[:, 0]

            f_inlet_now = compute_equilibrium(inlet_rho, u_inlet_dynamic)
            f[:, :, 0] = f_inlet_now[:, :, 0]
        else:
            f[:, :, 0] = f_inlet_eq[:, :, 0]

        # C. Outlet (East, x=W-1)
        # Open boundary (Neumann): Copy from neighbor x=W-2
        # Zero-gradient perpendicular to boundary
        f[:, :, -1] = f[:, :, -2]

        # D. Obstacle Bounce-Back (optional)
        if use_cylinder:
            f_obs = f[:, obstacle_mask]
            f[:, obstacle_mask] = f_obs[opp]

        # --- 5. Recording ---
        if step >= warmup_steps and return_timeseries:
            post_warmup_step = step - warmup_steps
            if not save_all_steps:
                if save_step_set is None:
                    continue
                if post_warmup_step not in save_step_set:
                    continue
            # Recalculate macroscopic for saving (since f changed in streaming/BC)
            rho_out = torch.sum(f, dim=0)
            ux_out = torch.sum(f * c[:, 1].view(9, 1, 1).float(), dim=0) / (
                rho_out + 1e-6
            )
            uy_out = torch.sum(f * c[:, 0].view(9, 1, 1).float(), dim=0) / (
                rho_out + 1e-6
            )

            history.append(get_snapshot(torch.stack([ux_out, uy_out]), rho_out))

    if return_timeseries:
        if not history:
            return get_snapshot(u, rho).unsqueeze(0)
        return torch.stack(history, dim=0)
    return get_snapshot(u, rho)
