"""
Lattice Boltzmann Method (LBM) Simulator for 2D Fluid Flow.

This module implements a differentiable D2Q9 LBM solver for incompressible
Navier-Stokes equations. It supports complex boundary conditions (like obstacles)
and inflow/outflow profiles, making it suitable for benchmarks like Flow Past Cylinder.
"""

import math

import torch

from autosim.simulations.base import Simulator
from autosim.types import TensorLike


class LatticeBoltzmann(Simulator):
    r"""Lattice Boltzmann (D2Q9) simulator for channel flow with obstacles.

    Simulates 2D flow past a cylinder using the BGK collision model.
    The simulation domain is a rectangular channel with aspect ratio 4:1 (default).

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Bounds on sampled parameters:
        - ``viscosity``: Kinematic viscosity (0.01-0.05 typically).
        - ``u_in``: Maximum inflow velocity (keep < 0.15 for stability).
    output_names: list[str], optional
        Names for output channels. Defaults to ``["u", "v", "rho", "vorticity"]``.
    return_timeseries: bool, default=False
        If True, returns full trajectory; otherwise final frame only.
    use_cylinder: bool, default=True
        If True, include the circular obstacle. If False, run a plain channel.
    oscillatory_inlet: bool | None, default=None
        If True, apply time-dependent inlet modulation (useful for rich dynamics
        in no-cylinder channels). If None, defaults to ``not use_cylinder``.
    width: int, default=256
        Grid width (Nx).
    height: int, default=64
        Grid height (Ny).
    T: float, default=4.0
        Total simulation time (in seconds, approximate).
        Step count = T * 500.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        width: int = 128,
        height: int = 64,
        T: float = 4.0,
        use_cylinder: bool = True,
        oscillatory_inlet: bool | None = None,
    ) -> None:
        if parameters_range is None:
            # Re ~ u_in * D / nu. D=height/5 approx.
            # If u=0.1, D=10, nu=0.02 -> Re=50 (vortex shedding).
            parameters_range = {
                "viscosity": (0.01, 0.05),
                "u_in": (0.04, 0.10),
            }
        if output_names is None:
            # We will compute vorticity on the fly
            output_names = ["u", "v", "rho", "vorticity"]

        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        self.width = width
        self.height = height
        self.T = T
        self.use_cylinder = use_cylinder
        if oscillatory_inlet is None:
            oscillatory_inlet = not use_cylinder
        self.oscillatory_inlet = oscillatory_inlet

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, "Simulator._forward expects a single input"
        sample = x[0]

        out = simulate_lbm_cylinder(
            params=sample,
            return_timeseries=self.return_timeseries,
            width=self.width,
            height=self.height,
            duration=self.T,
            use_cylinder=self.use_cylinder,
            oscillatory_inlet=self.oscillatory_inlet,
        )
        return out.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(
        self, n: int, random_seed: int | None = None
    ) -> dict:
        """Run sampled trajectories and return spatiotemporal tensors."""
        x = self.sample_inputs(n, random_seed)

        outputs = []
        for i in range(n):
            outputs.append(self._forward(x[i : i + 1]))

        y = torch.cat(outputs, dim=0)

        # LBM outputs: [B, Steps*Features] or [B, Features]
        channels = len(self.output_names)  # 4
        features_per_frame = self.width * self.height * channels

        if self.return_timeseries:
            total_elements = y.shape[1]
            steps = total_elements // features_per_frame
            y_reshaped = y.reshape(n, steps, self.height, self.width, channels)
        else:
            y_reshaped = y.reshape(n, 1, self.height, self.width, channels)

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


def simulate_lbm_cylinder(  # noqa: PLR0915
    params: TensorLike,
    return_timeseries: bool,
    width: int,
    height: int,
    duration: float,
    use_cylinder: bool = True,
    oscillatory_inlet: bool = False,
) -> TensorLike:
    """
    Simulate flow past a cylinder using D2Q9 Lattice Boltzmann.

    Coordinate system: x (width, index 1), y (height, index 0).
    """
    device = params.device

    # 1. Unpack parameters
    viscosity = float(params[0].item())
    u_max = float(params[1].item())

    # LBM Relaxation
    # nu = (tau - 0.5)/3  => tau = 3*nu + 0.5
    tau = 3.0 * viscosity + 0.5
    omega = 1.0 / tau

    # Determine number of steps
    # We define 1 sec = 250 LBM steps arbitrarily to scale 'duration' roughly
    steps_per_sec = 250
    total_steps = int(duration * steps_per_sec)
    warmup_steps = 200

    # Save interval: aim for ~50 frames total per simulation
    save_interval = max(1, total_steps // 50)

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
    # Populations entering from the left (west inlet) and from the right (east outlet)
    inlet_incoming = torch.tensor([1, 5, 8], device=device, dtype=torch.long)
    outlet_incoming = torch.tensor([3, 6, 7], device=device, dtype=torch.long)

    # 2. Geometry
    # Coordinate grids: y (rows) 0..H-1, x (cols) 0..W-1
    y_coord = torch.arange(height, device=device).view(-1, 1).repeat(1, width)
    x_coord = torch.arange(width, device=device).view(1, -1).repeat(height, 1)

    # Cylinder obstacle (optional)
    if use_cylinder:
        cy, cx = height / 2.0, width / 4.0
        radius = height / 9.0
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
        return torch.stack([u_viz[0], u_viz[1], rho_curr, vort], dim=-1)

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
        # Half-way bounce-back to prevent garbage populations from wrapping around
        # and corrupting the macroscopic variables at the boundaries.
        top_hit_N = f[2, 0, :].clone()
        top_hit_NE = f[5, 0, :].clone()
        top_hit_NW = f[6, 0, :].clone()

        bot_hit_S = f[4, -1, :].clone()
        bot_hit_SW = f[7, -1, :].clone()
        bot_hit_SE = f[8, -1, :].clone()

        f[4, -1, :] = top_hit_N
        f[7, -1, :] = top_hit_NE
        f[8, -1, :] = top_hit_NW

        f[2, 0, :] = bot_hit_S
        f[5, 0, :] = bot_hit_SW
        f[6, 0, :] = bot_hit_SE

        # B. Inlet (West, x=0)
        # Fixed or oscillatory equilibrium profile (Dirichlet velocity).
        # To reduce reflection, prescribe only populations entering the domain.
        if oscillatory_inlet:
            # Drive richer unsteady dynamics, especially useful when no cylinder is used
            phase = 2.0 * math.pi * (3.0 * step / max(full_duration, 1))
            amp_u = 1.0 + 0.25 * math.sin(phase)
            amp_v = 0.10 * math.sin(phase)

            u_inlet_dynamic = torch.zeros((2, height, 1), device=device)
            u_inlet_dynamic[0, :, 0] = torch.clamp(profile * amp_u, min=0.0, max=0.18)
            u_inlet_dynamic[1, :, 0] = amp_v * lateral_profile[:, 0]

            f_inlet_now = compute_equilibrium(inlet_rho, u_inlet_dynamic)
            f[inlet_incoming, :, 0] = f_inlet_now[inlet_incoming, :, 0]
        else:
            f[inlet_incoming, :, 0] = f_inlet_eq[inlet_incoming, :, 0]

        # C. Outlet (East, x=W-1)
        # Open boundary (Neumann): copy only populations entering from outlet side.
        # This is less reflective than overwriting all directions.
        f[outlet_incoming, :, -1] = f[outlet_incoming, :, -2]

        # D. Obstacle Bounce-Back (optional)
        if use_cylinder:
            f_obs = f[:, obstacle_mask]
            f[:, obstacle_mask] = f_obs[opp]

        # --- 5. Recording ---
        if (
            step >= warmup_steps
            and return_timeseries
            and ((step - warmup_steps) % save_interval == 0)
        ):
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
