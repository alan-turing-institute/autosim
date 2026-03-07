from __future__ import annotations

import math

import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class ShallowWater2D(SpatioTemporalSimulator):
    """Full 2D shallow-water simulator with prognostic [h, u, v]."""

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
        H: float = 1.0,
        nu: float = 5e-4,
        drag: float = 2e-3,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if parameters_range is None:
            parameters_range = {"amp": (0.05, 0.14)}
        if output_names is None:
            output_names = ["h", "u", "v"]

        super().__init__(parameters_range, output_names, log_level)
        if skip_nt < 0:
            msg = "skip_nt must be non-negative"
            raise ValueError(msg)
        self.return_timeseries = return_timeseries
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.dt_save = dt_save
        self.skip_nt = skip_nt
        self.cfl = cfl
        self.g = g
        self.H = H
        self.nu = nu
        self.drag = drag
        self.dtype = dtype

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, "Simulator._forward expects a single input"
        amp = float(x[0, 0].item())

        y = simulate_swe_2d(
            amp=amp,
            return_timeseries=self.return_timeseries,
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
            T=self.T,
            dt_save=self.dt_save,
            skip_nt=self.skip_nt,
            cfl=self.cfl,
            g=self.g,
            H=self.H,
            nu=self.nu,
            drag=self.drag,
            dtype=self.dtype,
        )
        return y.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(
        self, n: int, random_seed: int | None = None
    ) -> dict:
        """Run sampled trajectories and return `[batch,time,x,y,channels]` data."""
        x = self.sample_inputs(n, random_seed)
        y, x = self.forward_batch(x)

        channels = 3
        features_per_step = self.nx * self.ny * channels

        if self.return_timeseries:
            total = y.shape[1]
            n_time = total // features_per_step
            y = y.reshape(n, n_time, self.nx, self.ny, channels)
        else:
            y = y.reshape(n, 1, self.nx, self.ny, channels)

        return {
            "data": y,
            "constant_scalars": x,
            "constant_fields": None,
        }


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
    H: float,
    nu: float,
    drag: float,
    dtype: torch.dtype = torch.float64,
    skip_nt: int = 0,
) -> torch.Tensor:
    """Integrate full shallow-water equations with PDEArena-style random2 ICs."""
    if dtype not in (torch.float32, torch.float64):
        msg = "dtype must be torch.float32 or torch.float64"
        raise ValueError(msg)
    if skip_nt < 0:
        msg = "skip_nt must be non-negative"
        raise ValueError(msg)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    x = torch.linspace(0.0, Lx, nx + 1, dtype=dtype)[:-1]
    y = torch.linspace(0.0, Ly, ny + 1, dtype=dtype)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    dx = Lx / nx
    dy = Ly / ny

    c = math.sqrt(g * H)
    f0 = c / 8.0
    beta = 0.5 * f0 / Ly
    f_grid = f0 + beta * (Y - 0.5 * Ly)

    kx = (2 * math.pi / Lx) * torch.fft.fftfreq(nx, d=1.0 / nx).to(dtype)
    ky = (2 * math.pi / Ly) * torch.fft.rfftfreq(ny, d=1.0 / ny).to(dtype)
    Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")
    K2 = Kx**2 + Ky**2
    K2_inv = torch.where(K2 > 0, -1.0 / K2, torch.zeros_like(K2))
    iKx = 1j * Kx
    iKy = 1j * Ky

    # Hyperviscosity integrating-factor operator (same approach as barotropic solver).
    # Damps grid-scale modes in ~1 time unit; leaves large-scale vortices untouched.
    n_hyp = 4
    k_max = math.pi * max(nx / Lx, ny / Ly)
    nu_h = 1.0 / k_max ** (2 * n_hyp)
    hyp_op = -nu_h * K2**n_hyp

    def to_spec(field: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(field)

    def to_phys(field_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(field_hat, s=(nx, ny))

    def grad_x(field: torch.Tensor) -> torch.Tensor:
        return to_phys(1j * Kx * to_spec(field))

    def grad_y(field: torch.Tensor) -> torch.Tensor:
        return to_phys(1j * Ky * to_spec(field))

    def laplacian(field: torch.Tensor) -> torch.Tensor:
        return to_phys(-K2 * to_spec(field))

    # ------------------------------------------------------------------ #
    # Balanced initial conditions via vorticity → streamfunction         #
    # ------------------------------------------------------------------ #
    # Strategy: specify vorticity ζ (random large-scale + jet shear +
    # wave-6 perturbation), solve ∇²ψ = ζ spectrally, then derive
    #   u = -∂ψ/∂y,  v = ∂ψ/∂x,  h = H + (f0/g)·ψ
    # This guarantees exact geostrophic balance at t=0 so no spurious
    # gravity-wave transients are excited.

    k_min = 2.0 * math.pi / max(Lx, Ly)
    k_cut = k_min * min(nx, ny) // 6

    # Component 1: random large-scale streamfunction (k^{-2} → E(k)~k^{-3})
    rand_re = torch.randn(nx, ny // 2 + 1, dtype=dtype)
    rand_im = torch.randn(nx, ny // 2 + 1, dtype=dtype)
    psi_hat_rand = (rand_re + 1j * rand_im).to(dtype=complex_dtype)
    K_mag = torch.sqrt(K2 + k_min**2)
    psi_hat_rand = psi_hat_rand / K_mag**2
    psi_hat_rand = torch.where(
        k_cut**2 > K2, psi_hat_rand, torch.zeros_like(psi_hat_rand)
    )
    psi_hat_rand[0, 0] = 0.0
    psi_rand_phys = to_phys(psi_hat_rand)
    U_scale = 0.5
    psi_norm = amp * U_scale * min(Lx, Ly) / (float(psi_rand_phys.std()) + 1e-10)
    psi_hat_rand = psi_hat_rand * psi_norm
    zeta_random = to_phys(-K2 * psi_hat_rand)

    # Component 2: per-column independent random zonal jet (PDEArena :random2 style)
    # Each longitude column i gets its own independent random Fourier coefficients
    # in y — matching PDEArena's truly per-column i.i.d. wind profiles.
    n_modes = 4
    coeff = torch.randn(nx, n_modes, dtype=dtype)  # [nx, n_modes], i.i.d. per column
    y_frac = Y / Ly  # [nx, ny], values in [0, 1]
    u_jet_field = torch.stack(
        [
            coeff[:, m].unsqueeze(1) * torch.sin((m + 1) * math.pi * y_frac)
            for m in range(n_modes)
        ],
        dim=0,
    ).sum(dim=0)  # [nx, ny]
    # Normalise so |u_jet| ~ amp * 0.8 regardless of random draw
    jet_std = float(u_jet_field.std()) + 1e-10
    u_jet_field = u_jet_field * (amp * 0.8 / jet_std)
    zeta_jet = to_phys(-1j * Ky * to_spec(u_jet_field))

    # Component 3: wave-6 Gaussian perturbation at mid-latitude
    zeta_jet_scale = float(zeta_jet.std()) + 1e-10
    A_pert = max(amp * f0 * 0.8, zeta_jet_scale * 0.25)
    m_wave = 6
    y_center = 0.65 * Ly
    y_width = 0.10 * Ly
    pert_phase = float(torch.rand(1)) * 2.0 * math.pi
    zeta_pert = (
        A_pert
        * torch.cos(m_wave * 2.0 * math.pi * X / Lx + pert_phase)
        * torch.exp(-0.5 * ((Y - y_center) / y_width) ** 2)
    )

    # Combine, low-pass filter, solve for ψ, then u, v, h
    zeta = zeta_random + zeta_jet + zeta_pert
    zeta_h = to_spec(zeta)
    zeta_h = torch.where(k_cut**2 > K2, zeta_h, torch.zeros_like(zeta_h))
    zeta_h[0, 0] = 0.0

    psi_h = K2_inv * zeta_h  # ∇²ψ = ζ  →  ψ̂ = -ζ̂/K²
    psi0 = to_phys(psi_h)

    u0 = to_phys(-1j * Ky * psi_h)  # u = -∂ψ/∂y
    v0 = to_phys(1j * Kx * psi_h)  # v =  ∂ψ/∂x
    h0 = (H + (f0 / g) * psi0).clamp(min=0.5 * H)  # geostrophic balance

    def rhs(
        h: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_safe = h.clamp(min=1e-4)

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
        return dhdt, dudt, dvdt

    def output(h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        h_out = torch.nan_to_num(h, nan=H, posinf=100.0, neginf=1e-4).clamp(
            min=1e-4, max=100.0
        )
        u_out = torch.nan_to_num(u, nan=0.0, posinf=100.0, neginf=-100.0).clamp(
            min=-100.0, max=100.0
        )
        v_out = torch.nan_to_num(v, nan=0.0, posinf=100.0, neginf=-100.0).clamp(
            min=-100.0, max=100.0
        )
        return torch.stack([h_out.float(), u_out.float(), v_out.float()], dim=-1)

    h = h0
    u = u0
    v = v0

    history: list[torch.Tensor] = []
    expected_frames = int(T / dt_save) + 1
    t = 0.0
    next_save = 0.0
    last_valid = output(h, u, v)

    while t <= T + 1e-10:
        if not (
            torch.isfinite(h).all()
            and torch.isfinite(u).all()
            and torch.isfinite(v).all()
        ):
            break

        if return_timeseries and t >= next_save - 1e-10:
            last_valid = output(h, u, v)
            history.append(last_valid)
            next_save += dt_save

        if t >= T - 1e-10:
            break

        c_now = torch.sqrt(g * h.clamp(min=1e-4))
        speed_x = (u.abs() + c_now).max().item()
        speed_y = (v.abs() + c_now).max().item()
        max_speed = max(speed_x, speed_y, 1e-8)
        if not math.isfinite(max_speed):
            break

        dt = cfl * min(dx, dy) / max_speed
        dt = min(dt, next_save - t, T - t)
        dt = min(dt, 0.5 * dt_save)
        dt = max(dt, 1e-10)

        k1_h, k1_u, k1_v = rhs(h, u, v)
        k2_h, k2_u, k2_v = rhs(
            h + 0.5 * dt * k1_h,
            u + 0.5 * dt * k1_u,
            v + 0.5 * dt * k1_v,
        )
        k3_h, k3_u, k3_v = rhs(
            h + 0.5 * dt * k2_h,
            u + 0.5 * dt * k2_u,
            v + 0.5 * dt * k2_v,
        )
        k4_h, k4_u, k4_v = rhs(
            h + dt * k3_h,
            u + dt * k3_u,
            v + dt * k3_v,
        )

        h = h + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
        u = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        v = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # Apply hyperviscosity integrating factor to all fields (spectral filter).
        hyp_factor = torch.exp(hyp_op * dt)  # real, shape [nx, ny//2+1]
        u = to_phys(to_spec(u) * hyp_factor)
        v = to_phys(to_spec(v) * hyp_factor)
        h_mean = h.mean()
        h_anom = h - h_mean
        h_anom = to_phys(to_spec(h_anom) * hyp_factor)
        h = (h_mean + h_anom).clamp(min=1e-4)
        if (
            torch.isfinite(h).all()
            and torch.isfinite(u).all()
            and torch.isfinite(v).all()
        ):
            last_valid = output(h, u, v)
        else:
            break
        t += dt

    if return_timeseries:
        if skip_nt >= expected_frames:
            msg = (
                "skip_nt is too large for the available trajectory length; "
                f"skip_nt={skip_nt}, available_frames={expected_frames}."
            )
            raise ValueError(msg)
        while len(history) < expected_frames:
            history.append(last_valid)
        if len(history) > expected_frames:
            history = history[:expected_frames]
        return torch.stack(history[skip_nt:], dim=0)
    return output(h, u, v).unsqueeze(0)
