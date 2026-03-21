import math
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def generate_complex_potential(
    X: torch.Tensor,
    Y: torch.Tensor,
    config: dict[str, Any],
    t: float = 0.0,
    static_disorder: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a highly controllable 2D potential landscape.

    Args:
        X: Meshgrid X coordinates.
        Y: Meshgrid Y coordinates.
        config: Dictionary containing all complexity parameters.
        t: Current simulation time.
        static_disorder: Optional precomputed stationary disorder field.
        rng: Random number generator for spatial disorder.
    """
    # 1. Base Geometry (Anisotropic Trap)
    # wx, wy control the "squish", box_param adds steep walls.
    wx = config.get("wx", 1.0)
    wy = config.get("wy", 1.0)

    V_base = 0.5 * (wx**2 * X**2 + wy**2 * Y**2)

    box_param = float(config.get("box_param", 0.0))
    if box_param > 0:
        box_power = float(config.get("box_power", 4.0))
        box_anisotropy = float(config.get("box_anisotropy", 1.0))
        box_type = str(config.get("box_type", "power"))

        R_wall = (1.0 / box_param) ** 0.25

        if box_type == "power":
            term_x = (X / R_wall) ** box_power
            term_y = (Y / (R_wall * box_anisotropy)) ** box_power
            V_base += term_x + term_y
        elif box_type == "woods_saxon":
            ws_a = float(config.get("ws_a", 0.1))
            ws_V0 = float(config.get("ws_V0", 100.0))
            r_eff = torch.sqrt(X**2 + (Y / box_anisotropy) ** 2)
            V_base += ws_V0 / (1.0 + torch.exp(-(r_eff - R_wall) / ws_a))

    # 2. Add Spatial Disorder (Optical Speckle)
    V_disorder = torch.zeros_like(X)
    disorder_strength = config.get("disorder_strength", 0.0)
    disorder_time_dependent = bool(config.get("disorder_time_dependent", False))

    if disorder_strength > 0:
        if static_disorder is not None and not disorder_time_dependent:
            V_disorder = static_disorder
        else:
            V_disorder = _generate_speckle_field(
                X,
                Y,
                wx,
                wy,
                disorder_strength,
                disorder_radius=float(config.get("disorder_radius", 0.0)),
                rng=rng,
            )

    # 3. Add a Dynamic "Spoon" (if calculating at a specific time t)
    V_spoon = torch.zeros_like(X)
    spoon_strength = config.get("spoon_strength", 0.0)

    if spoon_strength > 0:
        # Spoon moves in a circle: x = R*cos(wt), y = R*sin(wt)
        R = config.get("spoon_radius", 2.0)
        speed = config.get("spoon_speed", 1.0)
        spoon_width = config.get("spoon_width", 0.5)

        xs = R * math.cos(speed * t)
        ys = R * math.sin(speed * t)

        # Gaussian obstacle
        r2 = (X - xs) ** 2 + (Y - ys) ** 2
        V_spoon = spoon_strength * torch.exp(-r2 / (spoon_width**2))

    # Combine everything
    V_total = V_base + V_disorder + V_spoon

    return V_total


def _generate_speckle_field(
    X: torch.Tensor,
    Y: torch.Tensor,
    wx: float,
    wy: float,
    disorder_strength: float,
    disorder_radius: float = 0.0,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a spatially windowed speckle disorder field.

    Sums random-phase sine waves at multiple spatial frequencies then windows
    the result with a Gaussian envelope centred on the trap.
    """
    freqs = [1.5, 3.1, 5.8]
    V = torch.zeros_like(X)
    for f in freqs:
        phase_x = torch.rand(1, generator=rng, device=X.device) * 2 * math.pi
        phase_y = torch.rand(1, generator=rng, device=Y.device) * 2 * math.pi
        V += torch.sin(f * X + phase_x) * torch.cos(f * Y + phase_y)
    V = disorder_strength * (V / len(freqs))
    if disorder_radius <= 0.0:
        disorder_radius = 2.0 / math.sqrt(wx * wy)
    r2 = X**2 + Y**2
    return V * torch.exp(-r2 / (2.0 * disorder_radius**2))


def generate_static_disorder(
    X: torch.Tensor,
    Y: torch.Tensor,
    config: dict[str, Any],
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a stationary speckle-like potential for one trajectory."""
    disorder_strength = float(config.get("disorder_strength", 0.0))
    if disorder_strength <= 0.0:
        return torch.zeros_like(X)
    wx = float(config.get("wx", 1.0))
    wy = float(config.get("wy", 1.0))
    return _generate_speckle_field(
        X,
        Y,
        wx,
        wy,
        disorder_strength,
        disorder_radius=float(config.get("disorder_radius", 0.0)),
        rng=rng,
    )


class GPESimulator2D:
    """Internal Simulator class for the Gross-Pitaevskii Equation."""

    def __init__(self, N=128, L=10.0, dt=0.005, device=None):
        """Initialize the 2D Quantum Simulator.

        N: Grid resolution (N x N)
        L: Physical size of the simulation box
        dt: Time step size
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.N = N
        self.L = L
        self.dt = dt
        self.dx = L / N

        # 1. Setup Spatial Grid
        x = (torch.arange(N, device=self.device) - N // 2) * self.dx
        y = (torch.arange(N, device=self.device) - N // 2) * self.dx
        self.X, self.Y = torch.meshgrid(x, y, indexing="ij")

        # 2. Setup Momentum (k) Grid for the Kinetic Energy step
        kx = torch.fft.fftfreq(N, d=self.dx, device=self.device) * 2 * math.pi
        ky = torch.fft.fftfreq(N, d=self.dx, device=self.device) * 2 * math.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")

        # Kinetic Energy Operator in momentum space
        kinetic_energy = 0.5 * (KX**2 + KY**2)
        self.exp_K = (
            torch.exp(-1j * kinetic_energy * self.dt)
            .to(self.device)
            .to(torch.complex64)
        )
        # exp_K_it: imaginary-time kinetic propagator exp(-k²/2 * dt)
        self.exp_K_it = (
            torch.exp(-kinetic_energy * self.dt).to(self.device).to(torch.complex64)
        )

        # Store momentum grids for spectral Lz computation
        self.KX = KX.to(torch.complex64)
        self.KY = KY.to(torch.complex64)

        # Cache for the rotation sampling grid (recomputed only when angle changes)
        self._cached_rot_angle: float | None = None
        self._cached_rot_grid: torch.Tensor | None = None

    def _apply_rotation(self, psi: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate the wavefunction by `angle` radians via bicubic interpolation.

        Implements exp(i * angle * Lz) as a coordinate rotation
        psi'(r) = psi(R_{-angle} r).
        Unlike a forward-Euler Lz step this is exactly norm-preserving.  The sampling
        grid depends only on `angle` and the fixed spatial grid, so it is cached and
        recomputed only when the angle changes.
        """
        if self._cached_rot_angle != angle:
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            # Pullback: source coordinates under R_{-angle}
            xs_src = cos_a * self.X + sin_a * self.Y
            ys_src = -sin_a * self.X + cos_a * self.Y

            # Normalize to grid_sample coordinates.
            # With align_corners=False pixel i -> (2i+1)/N - 1,
            # so physical x -> 2x/L + 1/N.
            # This avoids the 1-pixel translation bias on even-sized grids.
            inv_L = 2.0 / self.L
            offset = 1.0 / self.N
            xs_norm = (xs_src * inv_L + offset).clamp(-1.0, 1.0)
            ys_norm = (ys_src * inv_L + offset).clamp(-1.0, 1.0)

            # grid_sample: grid[..., 0] = W (col = Y), grid[..., 1] = H (row = X)
            self._cached_rot_grid = torch.stack([ys_norm, xs_norm], dim=-1).unsqueeze(0)
            self._cached_rot_angle = angle

        assert self._cached_rot_grid is not None
        grid = self._cached_rot_grid
        psi_r = F.grid_sample(
            psi.real.unsqueeze(0).unsqueeze(0).float(),
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze()
        psi_i = F.grid_sample(
            psi.imag.unsqueeze(0).unsqueeze(0).float(),
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze()
        return torch.complex(psi_r, psi_i).to(torch.complex64)

    def _apply_V_half(
        self, psi: torch.Tensor, V: torch.Tensor, g: float, half_dt: complex
    ) -> torch.Tensor:
        """Apply the potential + nonlinear half-step.

        This is psi *= exp(-i*(V + g|psi|²)*half_dt)
        """
        density = torch.abs(psi) ** 2
        return psi * torch.exp(-1j * (V + g * density) * half_dt)

    def _apply_Lz_exp(self, psi: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply exp(alpha * Lz) for imaginary-time rotation steps.

        For imaginary time the correct sub-step is exp(Ω·dτ·Lz), which is a
        *non-unitary* operator that amplifies positive-angular-momentum modes and
        thereby drives vortex nucleation.  A unitary coordinate rotation (used for
        real time) leaves the density unchanged and cannot form a vortex lattice.

        Lz = -i(x ∂/∂y - y ∂/∂x) is evaluated spectrally:
            ∂ψ/∂x = IFFT(i kx ψ̂)   ∂ψ/∂y = IFFT(i ky ψ̂)

        exp(alpha * Lz) is approximated to first order (I + alpha*Lz); with the
        small sub-step alpha = Ω·dt/2 ≈ 5x10⁻⁴ the truncation error is negligible
        and the caller's renormalisation keeps the norm bounded.
        """
        psi_hat = torch.fft.fftn(psi)
        dpsi_dx = torch.fft.ifftn(1j * self.KX * psi_hat)
        dpsi_dy = torch.fft.ifftn(1j * self.KY * psi_hat)
        Lz_psi = -1j * (self.X * dpsi_dy - self.Y * dpsi_dx)
        return (psi + alpha * Lz_psi).to(torch.complex64)

    def initialize_state(self, x0=0.0, y0=0.0, width=1.0, kx0=0.0, ky0=0.0):
        """Create a meaningful initial condition: a moving Gaussian wavepacket."""
        if width <= 0:
            msg = "width must be positive"
            raise ValueError(msg)

        r2 = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        amplitude = torch.exp(-r2 / (2 * width**2))

        # Add a phase gradient for initial momentum
        phase = torch.exp(1j * (kx0 * self.X + ky0 * self.Y))

        psi = amplitude * phase

        # Normalize the wave function so total probability = 1
        norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2) * self.dx**2)
        return (psi / norm).to(torch.complex64)

    def step(self, psi, V, g, Omega=0.0, imaginary_time=False):
        """Evolve the system forward by one time step using Strang splitting.

        Sequence: [V-half] [rot-half] [K-full] [rot-half] [V-half] [renorm]

        Real-time uses half_dt = dt/2 (phase evolution).
        Imaginary-time uses half_dt = -i*dt/2 (amplitude decay toward ground state).
        In both cases the rotation is the same real coordinate rotation exp(i*angle*Lz).
        """
        # Effective half time-step
        half_dt: complex = (-1j if imaginary_time else 1.0) * self.dt / 2.0
        exp_K = self.exp_K_it if imaginary_time else self.exp_K

        # 1. Potential + nonlinear half-step
        psi = self._apply_V_half(psi, V, g, half_dt)

        # 2. Rotation half-step
        # Real time:      exp(i Ω dt/2 Lz) — unitary coordinate rotation
        # Imaginary time: exp(Ω dτ/2 Lz)  — non-unitary amplifier, required for
        #                 vortex nucleation (a coordinate rotation only shifts phases
        #                 and cannot change the density toward a vortex lattice).
        if Omega != 0.0:
            rot_angle = Omega * self.dt / 2.0
            if imaginary_time:
                psi = self._apply_Lz_exp(psi, rot_angle)
            else:
                psi = self._apply_rotation(psi, rot_angle)

        # 3. Kinetic full-step in momentum space
        psi = torch.fft.ifftn(torch.fft.fftn(psi) * exp_K)

        # 4. Rotation half-step (symmetric)
        if Omega != 0.0:
            if imaginary_time:
                psi = self._apply_Lz_exp(psi, rot_angle)  # pyright: ignore[reportPossiblyUnboundVariable] since set above in block with same condition
            else:
                psi = self._apply_rotation(psi, rot_angle)  # pyright: ignore[reportPossiblyUnboundVariable] since set above in block with same condition

        # 5. Potential + nonlinear half-step
        psi = self._apply_V_half(psi, V, g, half_dt)

        # 6. Renormalise: preserves the physical norm invariant and suppresses
        # floating-point drift; for imaginary-time this is required for convergence.
        norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2) * self.dx**2)
        return psi / norm


def simulate_gpe_2d(  # noqa: PLR0912, PLR0915
    config: dict[str, Any],
    *,
    return_timeseries: bool = False,
    n: int = 128,
    L: float = 10.0,
    T: float = 5.0,
    dt: float = 0.005,
    snapshot_dt: float | None = None,
    random_seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Simulate the Gross-Pitaevskii Equation in 2D."""
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    if L <= 0:
        msg = "L must be positive"
        raise ValueError(msg)
    if T <= 0:
        msg = "T must be positive"
        raise ValueError(msg)
    if dt <= 0:
        msg = "dt must be positive"
        raise ValueError(msg)
    if snapshot_dt is not None and snapshot_dt <= 0:
        msg = "snapshot_dt must be positive when provided"
        raise ValueError(msg)

    if snapshot_dt is None:
        snapshot_dt = dt

    if device is None:
        device = torch.device("cpu")

    if random_seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        seed = int(random_seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    simulator = GPESimulator2D(N=n, L=L, dt=dt, device=device)

    psi = simulator.initialize_state(
        x0=config.get("x0", 0.0),
        y0=config.get("y0", 0.0),
        width=config.get("width", 1.0),
        kx0=config.get("kx0", 0.0),
        ky0=config.get("ky0", 0.0),
    )

    imaginary_time = config.get("imaginary_time", False)
    if imaginary_time or config.get("imaginary_time_steps", 0) > 0:
        # Seed initial state with tiny noise to break symmetry, which is required
        # for vortex lattices to nucleate during imaginary time evolution.
        noise_amp = config.get("initial_noise", 0.05)
        if noise_amp > 0:
            noise_real = torch.randn_like(psi.real, generator=rng)
            noise_imag = torch.randn_like(psi.imag, generator=rng)
            psi = psi + noise_amp * torch.complex(noise_real, noise_imag)
            psi = psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2) * simulator.dx**2)

    g = config.get("g", 10.0)

    # Pre-calculate spatial disorder once (stationary speckle) unless explicitly
    # requesting time-dependent disorder.
    static_disorder = None
    if float(config.get("disorder_strength", 0.0)) > 0.0 and not bool(
        config.get("disorder_time_dependent", False)
    ):
        static_disorder = generate_static_disorder(
            simulator.X, simulator.Y, config, rng
        )

    history_density = []
    history_real = []
    history_imag = []

    def _snapshot(p) -> torch.Tensor:
        density = torch.abs(p) ** 2
        real = p.real
        imag = p.imag
        return torch.stack([density, real, imag], dim=-1)

    with torch.no_grad():
        next_snapshot_t = float(snapshot_dt)
        last_snapshot_t = 0.0

        if return_timeseries:
            history_density.append(torch.abs(psi) ** 2)
            history_real.append(psi.real)
            history_imag.append(psi.imag)

        t = 0.0

        Omega = float(config.get("Omega", 0.0))
        imaginary_time = bool(config.get("imaginary_time", False))
        imaginary_time_steps = int(config.get("imaginary_time_steps", 0))

        if imaginary_time_steps > 0:
            for _ in range(imaginary_time_steps):
                V_init = generate_complex_potential(
                    simulator.X,
                    simulator.Y,
                    config,
                    0.0,
                    static_disorder=static_disorder,
                    rng=rng,
                )
                psi = simulator.step(psi, V_init, g, Omega=Omega, imaginary_time=True)

        while t < T - 1e-12:
            t_mid = t + 0.5 * dt
            V = generate_complex_potential(
                simulator.X,
                simulator.Y,
                config,
                t_mid,
                static_disorder=static_disorder,
                rng=rng,
            )
            psi = simulator.step(psi, V, g, Omega=Omega, imaginary_time=imaginary_time)
            t += dt

            if return_timeseries:
                while next_snapshot_t <= t + 1e-12:
                    history_density.append(torch.abs(psi) ** 2)
                    history_real.append(psi.real)
                    history_imag.append(psi.imag)
                    last_snapshot_t = next_snapshot_t
                    next_snapshot_t += snapshot_dt

        if return_timeseries and T - last_snapshot_t > 1e-9:
            history_density.append(torch.abs(psi) ** 2)
            history_real.append(psi.real)
            history_imag.append(psi.imag)

    if return_timeseries:
        densities = torch.stack(history_density, dim=0)
        real_parts = torch.stack(history_real, dim=0)
        imag_parts = torch.stack(history_imag, dim=0)
        # return shape: (T, X, Y, Channels)
        return torch.stack([densities, real_parts, imag_parts], dim=-1)

    return _snapshot(psi)


class GrossPitaevskiiEquation2D(SpatioTemporalSimulator):
    """Gross-Pitaevskii Equation simulator for quantum fluids."""

    _DEFAULT_SIM_PARAMS: ClassVar[dict[str, Any]] = {
        "wx": 1.0,
        "wy": 1.0,
        "box_param": 0.0,
        "box_power": 4.0,
        "box_anisotropy": 1.0,
        "ws_a": 0.1,
        "ws_V0": 100.0,
        "disorder_strength": 0.0,
        "disorder_time_dependent": False,
        "disorder_radius": 0.0,  # 0.0 -> auto (2/sqrt(wx*wy))
        "spoon_strength": 0.0,
        "spoon_radius": 2.0,
        "spoon_speed": 1.0,
        "spoon_width": 0.5,
        "g": 10.0,
        "x0": 0.0,
        "y0": 0.0,
        "width": 1.0,
        "kx0": 0.0,
        "ky0": 0.0,
        "Omega": 0.0,
        "imaginary_time": False,
        "imaginary_time_steps": 0,
        "initial_noise": 0.05,
    }

    _ALLOWED_PARAMETER_NAMES: ClassVar[tuple[str, ...]] = tuple(
        _DEFAULT_SIM_PARAMS.keys()
    )

    # Per-parameter post-processors applied after float sampling.
    # Parameters not listed here are kept as plain floats.
    _PARAM_CONVERTERS: ClassVar[dict[str, Callable[[float], Any]]] = {
        "imaginary_time": lambda v: bool(round(v)),
        "imaginary_time_steps": lambda v: round(v),
    }

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 128,
        L: float = 10.0,
        T: float = 5.0,
        dt: float = 0.005,
        snapshot_dt: float | None = None,
        skip_nt: int = 0,
        random_seed: int | None = None,
        box_type: str = "woods_saxon",
    ) -> None:
        if parameters_range is None:
            # Provide some sensible defaults for complexity knobs
            parameters_range = {
                "g": (0.0, 50.0),
                "disorder_strength": (0.0, 2.0),
                "spoon_strength": (0.0, 5.0),
                "wx": (0.5, 2.0),
                "wy": (0.5, 2.0),
            }
        if output_names is None:
            output_names = ["density", "real", "imag"]

        unknown_parameters = set(parameters_range) - set(self._ALLOWED_PARAMETER_NAMES)
        if unknown_parameters:
            unknown = ", ".join(sorted(unknown_parameters))
            msg = (
                f"Unsupported parameters in parameters_range: {unknown}. "
                f"Allowed names are: {sorted(self._ALLOWED_PARAMETER_NAMES)}"
            )
            raise ValueError(msg)

        super().__init__(parameters_range, output_names, log_level)

        if n <= 0:
            msg = "n must be positive"
            raise ValueError(msg)
        if L <= 0:
            msg = "L must be positive"
            raise ValueError(msg)
        if T <= 0:
            msg = "T must be positive"
            raise ValueError(msg)
        if dt <= 0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if skip_nt < 0:
            msg = "skip_nt must be non-negative"
            raise ValueError(msg)
        if snapshot_dt is not None and snapshot_dt <= 0:
            msg = "snapshot_dt must be positive when provided"
            raise ValueError(msg)

        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt
        self.snapshot_dt = snapshot_dt
        self.skip_nt = skip_nt
        self.random_seed = random_seed
        self.box_type = box_type
        self._rng = (
            torch.Generator().manual_seed(random_seed)
            if random_seed is not None
            else None
        )

    def _extract_run_parameters(self, x: torch.Tensor) -> dict[str, Any]:
        param_values = {}
        for idx, name in enumerate(self.param_names):
            raw = float(x[0, idx].item())
            converter = self._PARAM_CONVERTERS.get(name)
            param_values[name] = converter(raw) if converter is not None else raw

        config = self._DEFAULT_SIM_PARAMS.copy()
        config.update(param_values)
        config["box_type"] = self.box_type
        return config

    def _forward(self, x: TensorLike) -> TensorLike:
        if x.shape[0] != 1:
            msg = "GrossPitaevskiiEquation2D expects a single input sample"
            raise ValueError(msg)

        x = torch.as_tensor(x, dtype=torch.float32)
        config = self._extract_run_parameters(x)

        seed = None
        if self._rng is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())

        sol = simulate_gpe_2d(
            config=config,
            return_timeseries=self.return_timeseries,
            n=self.n,
            L=self.L,
            T=self.T,
            dt=self.dt,
            snapshot_dt=self.snapshot_dt,
            random_seed=seed,
            device=x.device,  # use same device as input tensor
        )
        return sol.flatten().unsqueeze(0)

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

        channels = 3  # density, real, imag
        features_per_step = self.n * self.n * channels

        if self.return_timeseries:
            total_features = y.shape[1]
            if total_features % features_per_step != 0:
                raise RuntimeError(
                    "Returned tensor does not align with n*n*channels; "
                    f"received {total_features}, expected multiples of "
                    f"{features_per_step}."
                )
            n_time = total_features // features_per_step
            y_reshaped = y.reshape(y.shape[0], n_time, self.n, self.n, channels)

            start_idx = 1 + self.skip_nt
            if start_idx >= n_time:
                raise ValueError(
                    "skip_nt is too large for the available trajectory length; "
                    f"computed start index {start_idx} for n_time={n_time}."
                )
            y_reshaped = y_reshaped[:, start_idx:, ...]
        else:
            if y.shape[1] != features_per_step:
                raise RuntimeError(
                    "Unexpected flattened size for single snapshot; "
                    f"received {y.shape[1]}, expected {features_per_step}."
                )
            y_reshaped = y.reshape(y.shape[0], 1, self.n, self.n, channels)

        return {
            "data": y_reshaped,
            "constant_scalars": x,
            "constant_fields": None,
        }
