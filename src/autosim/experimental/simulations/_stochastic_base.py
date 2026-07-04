"""Shared base classes for the stochastic toy simulators.

These private bases hold the trajectory-generation scaffolding that would
otherwise be copy-pasted across the scalar-SDE testbeds (Ornstein-Uhlenbeck,
Cox-Ingersoll-Ross, double-well) and the correlated-field testbeds (1-D/2-D
correlated diffusion, correlated Lorenz-96). Concrete simulators supply only the
parts that genuinely differ -- the per-step update, the initial state and the
closed-form oracle -- so the numerics live in exactly one place.
"""

from __future__ import annotations

import abc
from abc import abstractmethod

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class ScalarSDESimulator(SpatioTemporalSimulator, abc.ABC):
    """Base class for 0-D scalar SDE testbeds integrated step-by-step.

    A subclass implements ``_step`` (one integration step of the scalar state)
    and sets ``self.n_steps`` in its constructor. This base provides the three
    trajectory methods (``_forward``, ``forward_samples_spatiotemporal``,
    ``mc_reference``) so the concrete simulators differ only in their ``_step``
    update and their module-level closed-form oracle.

    The scalar process has no spatial extent, so the spatiotemporal payload uses
    singleton spatial axes: ``data`` has shape ``(batch, n_steps, 1, 1, 1)``.
    """

    n_steps: int

    @abstractmethod
    def _step(self, x: float, rng: np.random.Generator) -> float:
        """Advance the scalar state by one integration step.

        Args:
            x: Current state.
            rng: Random generator supplying the noise increment.

        Returns:
            The updated state after one step.
        """

    def _integrate(
        self, x0: float, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Integrate one trajectory of ``n_steps`` from ``x0``.

        Args:
            x0: Initial state.
            n_steps: Number of steps to record.
            rng: Random generator supplying the process noise.

        Returns:
            Float32 trajectory of shape ``(n_steps,)``.
        """
        traj = np.empty(n_steps, dtype=np.float32)
        xt = x0
        for t in range(n_steps):
            xt = self._step(xt, rng)
            traj[t] = xt
        return traj

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single trajectory from the initial condition ``x0``.

        Args:
            x: Input tensor of shape ``(1, 1)`` containing ``x0``.

        Returns:
            Flattened trajectory tensor of shape ``(1, n_steps)``.
        """
        if x.shape[0] != 1:
            msg = (
                f"{type(self).__name__}._forward expects a single input, "
                f"got {x.shape[0]}"
            )
            raise ValueError(msg)

        rng = np.random.default_rng()  # fresh process-noise path per call
        x0 = float(torch.as_tensor(x).detach().cpu().numpy()[0, 0])
        traj = self._integrate(x0, self.n_steps, rng)
        return torch.from_numpy(traj).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002 -- generation is always exact
    ) -> dict:
        """Produce trajectories along with the sampled initial conditions.

        Both the sampled initial conditions and the process noise are seeded from
        ``random_seed``, so a given seed reproduces the full batch. These scalar
        integrators never fail, so ``ensure_exact_n`` is always satisfied and the
        base retry machinery is not needed.

        Args:
            n: Number of trajectories to sample.
            random_seed: Seed for reproducible initial conditions and process
                noise.
            ensure_exact_n: Accepted for API parity; the batch already contains
                exactly ``n`` trajectories.

        Returns:
            A dict with ``data`` (float32 ``(batch, n_steps, 1, 1, 1)``, singleton
            spatial axes), ``constant_scalars`` (sampled ``x0``, ``(batch, 1)``)
            and ``constant_fields`` (``None``, API parity).
        """
        x = self.sample_inputs(n, random_seed)
        rng = np.random.default_rng(random_seed)
        traj = np.stack(
            [self._integrate(float(x[i, 0]), self.n_steps, rng) for i in range(n)]
        )
        data = torch.from_numpy(traj).reshape(n, self.n_steps, 1, 1, 1)
        return {
            "data": data,
            "constant_scalars": x,
            "constant_fields": None,
        }

    def mc_reference(
        self,
        x_state: torch.Tensor,
        n_draws: int,
        n_steps: int,
        random_seed: int | None = None,
    ) -> torch.Tensor:
        """Draw Monte Carlo trajectories from a fixed initial state.

        All ``n_draws`` trajectories start from the same ``x_state`` but receive
        independent noise, giving an empirical predictive distribution to check
        against the closed-form oracle.

        Args:
            x_state: Initial state; any shape -- the first scalar value is used
                as ``x0``.
            n_draws: Number of independent Monte Carlo trajectories.
            n_steps: Number of steps per trajectory.
            random_seed: Seed for reproducible draws.

        Returns:
            Float32 tensor of shape ``(n_draws, n_steps, 1, 1, 1)``.
        """
        rng = np.random.default_rng(random_seed)
        x0 = float(torch.as_tensor(x_state).reshape(-1)[0])
        out = np.stack([self._integrate(x0, n_steps, rng) for _ in range(n_draws)])
        return torch.from_numpy(out).reshape(n_draws, n_steps, 1, 1, 1)


class CorrelatedFieldSimulator(SpatioTemporalSimulator, abc.ABC):
    """Base for field testbeds generated by a seeded burn-in then recorded rollout.

    A subclass implements ``_step`` (one field update) and ``_initial_state``
    (the pre-burn-in field), and sets ``self.n_steps``, ``self.burn``,
    ``self.n_sites`` and ``self._field_shape`` (the per-sample spatial shape,
    e.g. ``(n_sites,)`` or ``(n_side, n_side)``). This base provides the seeded
    ``_generate`` rollout and all the ``Simulator`` plumbing, so the whole batch
    is reproducible from ``random_seed``.

    The recorded field is embedded in the 5-D spatiotemporal layout
    ``(batch, time, space_0, space_1, channels)`` by padding ``_field_shape`` to
    three spatial axes with singletons.
    """

    n_steps: int
    burn: int
    n_sites: int
    _field_shape: tuple[int, ...]

    @abstractmethod
    def _step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Advance a batch of fields by one recorded step.

        Args:
            x: Field batch of shape ``(batch, *_field_shape)``.
            rng: Random generator supplying the process noise.

        Returns:
            The updated field batch, same shape as ``x``.
        """

    @abstractmethod
    def _initial_state(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the ``n`` pre-burn-in fields of shape ``(n, *_field_shape)``.

        Args:
            n: Number of trajectories.
            rng: Random generator supplying the initial perturbation.

        Returns:
            Float array of shape ``(n, *_field_shape)``.
        """

    @property
    def _spatial_shape(self) -> tuple[int, int, int]:
        """The field shape padded with singletons to three spatial axes."""
        pad = (1,) * (3 - len(self._field_shape))
        return (*self._field_shape, *pad)  # type: ignore[return-value]

    def _generate(self, n: int, seed: int | None) -> np.ndarray:
        """Generate ``n`` trajectories through a seeded burn-in then rollout.

        Args:
            n: Number of trajectories.
            seed: Seed for the whole trajectory (initial state and process
                noise).

        Returns:
            Float32 trajectory array of shape ``(n, n_steps, *_field_shape)``.

        Raises:
            RuntimeError: If the generated trajectory contains non-finite values
                (the process diverged).
        """
        rng = np.random.default_rng(seed)
        x = self._initial_state(n, rng)
        for _ in range(self.burn):
            x = self._step(x, rng)
        traj = np.empty((n, self.n_steps, *self._field_shape), dtype=np.float32)
        for t in range(self.n_steps):
            x = self._step(x, rng)
            traj[:, t] = x
        if not np.isfinite(traj).all():
            msg = f"{type(self).__name__} generation diverged"
            raise RuntimeError(msg)
        return traj

    def sample_state(self, seed: int | None = None) -> torch.Tensor:
        """Draw one plausible field state (flattened to ``(n_sites,)``).

        Args:
            seed: Seed for the burn-in.

        Returns:
            Float32 tensor of shape ``(n_sites,)``.
        """
        traj = self._generate(1, seed)
        return torch.from_numpy(traj[0, -1].reshape(-1)).float()

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single trajectory (one fresh process-noise path).

        Args:
            x: Input tensor of shape ``(1, ...)``; its value is unused, only the
                batch dimension is validated.

        Returns:
            Flattened trajectory tensor of shape ``(1, n_steps * n_sites)``.
        """
        if x.shape[0] != 1:
            msg = (
                f"{type(self).__name__}._forward expects a single input, "
                f"got {x.shape[0]}"
            )
            raise ValueError(msg)
        traj = self._generate(1, None)  # (1, n_steps, *_field_shape)
        return torch.from_numpy(traj.reshape(1, -1))

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002 -- generation is always exact
    ) -> dict:
        """Produce ``n`` field trajectories in the spatiotemporal data layout.

        The batch is generated in one seeded pass, so ``random_seed`` reproduces
        it exactly; ``_generate`` returns exactly ``n`` trajectories or raises, so
        ``ensure_exact_n`` needs no retries.

        Args:
            n: Number of trajectories to sample.
            random_seed: Seed for reproducible draws (initial perturbation and
                process noise).
            ensure_exact_n: Accepted for API parity; the batch already contains
                exactly ``n`` trajectories.

        Returns:
            A dict with ``data`` (float32 ``(n, n_steps, *spatial)``),
            ``constant_scalars`` (``zeros(n, 1)`` -- the field exposes no scalar,
            kept for API parity) and ``constant_fields`` (``None``).
        """
        traj = self._generate(n, random_seed)
        data = torch.from_numpy(traj).reshape(n, self.n_steps, *self._spatial_shape)
        return {
            "data": data.float(),
            "constant_scalars": torch.zeros(n, 1),
            "constant_fields": None,
        }

    def mc_reference(
        self,
        x_state: torch.Tensor,
        n_draws: int,
        n_steps: int,
        random_seed: int | None = None,
    ) -> torch.Tensor:
        """Sample-feed Monte-Carlo rollout from a fixed field (the DGP oracle).

        Each of ``n_draws`` members advances its own field with fresh per-step
        forcing evaluated at that member's own current state, so the across-member
        spread is a faithful predictive oracle.

        Args:
            x_state: Initial state; any shape flattening to ``(n_sites,)``.
            n_draws: Number of independent ensemble members.
            n_steps: Number of recorded steps (leads) per member.
            random_seed: Seed for reproducible draws.

        Returns:
            Float32 tensor of shape ``(n_draws, n_steps, *spatial)``.
        """
        rng = np.random.default_rng(random_seed)
        x0 = (
            torch.as_tensor(x_state)
            .detach()
            .cpu()
            .numpy()
            .reshape(self._field_shape)
            .astype(np.float64)
        )
        xm = np.repeat(x0[None], n_draws, axis=0)  # (n_draws, *_field_shape)
        out = np.empty((n_draws, n_steps, *self._field_shape), dtype=np.float32)
        for t in range(n_steps):
            xm = self._step(xm, rng)
            out[:, t] = xm
        return torch.from_numpy(out).reshape(n_draws, n_steps, *self._spatial_shape)


class LinearDiffusionSimulator(CorrelatedFieldSimulator, abc.ABC):
    """Base for the stable linear-diffusion field toys (1-D ring, 2-D torus).

    The conditional mean is the contractive linear operator
    ``A = (1 - dt gamma) I + dt kappa Lap`` and the forcing is
    ``eps ~ N(0, Sigma(x))`` with ``Sigma(x) = G diag(sigma2(x)) G^T + delta^2 I``
    and ``sigma2 = sigma0^2 (1 + beta |grad x|)``. The Fourier operators
    ``self.a_hat`` / ``self.g_hat`` and the axis set ``self._field_axes`` are
    built by the subclass constructor; the only genuinely dimension-specific piece
    left to the subclass is ``_grad_mag``.

    Subclasses must set (in ``__init__``, before ``super().__init__``): ``a_hat``,
    ``g_hat``, ``_field_axes``; and (as usual) ``_field_shape``, ``n_sites``,
    ``n_steps``, ``burn``, ``dt``, ``gamma``, ``kappa``, ``ell``, ``sigma0``,
    ``beta``, ``delta``.
    """

    a_hat: np.ndarray
    g_hat: np.ndarray
    _field_axes: tuple[int, ...]
    sigma0: float
    beta: float
    delta: float

    def _validate_dynamics(self) -> None:
        """Fail-early on a non-contractive (unstable) mean operator.

        Raises:
            ValueError: If the mean operator's spectral radius is ``>= 1``.
        """
        sr = float(np.max(np.abs(self.a_hat)))
        if sr >= 1.0:
            msg = (
                f"{type(self).__name__} requires a contractive mean operator "
                f"(spectral radius |A| < 1) for a stable process; got |A| = {sr:.4f}."
            )
            raise ValueError(msg)

    def A_spectral_radius(self) -> float:
        """Return the spectral radius of ``A`` (``< 1`` when stable)."""
        return float(np.max(np.abs(self.a_hat)))

    @abstractmethod
    def _grad_mag(self, x: np.ndarray) -> np.ndarray:
        """Magnitude of the central-difference gradient of the field.

        Args:
            x: Field array with the spatial axes last.

        Returns:
            Gradient magnitude, same shape as ``x``.
        """

    def _apply_A(self, x: np.ndarray) -> np.ndarray:
        """Apply the damped-diffusion mean operator ``A`` (Fourier)."""
        return np.real(
            np.fft.ifftn(
                self.a_hat * np.fft.fftn(x, axes=self._field_axes),
                axes=self._field_axes,
            )
        )

    def _apply_G(self, v: np.ndarray) -> np.ndarray:
        """Apply the Gaussian low-pass smoother ``G`` (Fourier)."""
        return np.real(
            np.fft.ifftn(
                self.g_hat * np.fft.fftn(v, axes=self._field_axes),
                axes=self._field_axes,
            )
        )

    def _sigma2(self, x: np.ndarray) -> np.ndarray:
        """State-dependent per-site forcing variance (front-concentrated)."""
        return self.sigma0**2 * (1.0 + self.beta * self._grad_mag(x))

    def _step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One DGP step ``A x + G(sqrt(sigma2) eta1) + delta eta2`` (raw space).

        The two standard-normal draws are taken in the same order as the
        reference generator so trajectories are byte-identical.

        Args:
            x: Field batch of shape ``(batch, *_field_shape)``.
            rng: Random generator supplying the two forcing draws.

        Returns:
            The updated field batch, same shape as ``x``.
        """
        mean = self._apply_A(x)
        std = np.sqrt(self._sigma2(x))
        eta1 = rng.standard_normal(x.shape)
        eta2 = rng.standard_normal(x.shape)
        eps = self._apply_G(std * eta1) + self.delta * eta2
        return mean + eps

    def _initial_state(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Small Gaussian field, ``0.1 * N(0, I)`` of shape ``(n, *_field_shape)``."""
        return 0.1 * rng.standard_normal((n, *self._field_shape))

    def _smoother_matrix(self) -> np.ndarray:
        """Build the explicit symmetric smoother matrix ``G`` from ``g_hat``.

        Applies ``_apply_G`` to each unit impulse on the flattened field, then
        symmetrises (the kernel is symmetric up to floating point).

        Returns:
            The ``(n_sites, n_sites)`` smoother matrix (float64).
        """
        g = np.zeros((self.n_sites, self.n_sites))
        for p in range(self.n_sites):
            e = np.zeros(self.n_sites)
            e[p] = 1.0
            g[:, p] = self._apply_G(e.reshape(self._field_shape)).reshape(-1)
        return 0.5 * (g + g.T)

    def closed_form_cov(self, x: TensorLike) -> torch.Tensor:
        """Exact one-step forcing covariance ``Sigma(x)`` for THIS instance.

        Evaluates ``G diag(sigma2(x)) G^T + delta^2 I`` using the instance's own
        DGP parameters, so a non-default simulator and its oracle cannot desync.

        Args:
            x: A single field state, any shape flattening to ``(n_sites,)`` in
                the field's row-major order.

        Returns:
            The ``(n_sites, n_sites)`` covariance, float32.
        """
        x_np = (
            torch.as_tensor(x)
            .detach()
            .cpu()
            .numpy()
            .reshape(self._field_shape)
            .astype(np.float64)
        )
        g_mat = self._smoother_matrix()
        s2 = self._sigma2(x_np).reshape(-1)
        cov = g_mat @ (s2[:, None] * g_mat.T) + self.delta**2 * np.eye(self.n_sites)
        return torch.from_numpy(cov).float()

    def sample_noise(
        self, x: TensorLike, n: int, seed: int | None = None
    ) -> torch.Tensor:
        """Draw ``n`` forcing samples ``eps ~ N(0, Sigma(x))`` via the DGP.

        Uses the same ``G(sqrt(sigma2(x)) eta1) + delta eta2`` forcing as the
        dynamics (no Cholesky), so the empirical covariance matches
        :meth:`closed_form_cov`.

        Args:
            x: A single field state flattening to ``(n_sites,)``.
            n: Number of forcing samples.
            seed: Seed for reproducible draws.

        Returns:
            Float32 tensor of shape ``(n, n_sites)`` (row-major).
        """
        x_np = torch.as_tensor(x).detach().cpu().numpy().reshape(self._field_shape)
        rng = np.random.default_rng(seed)
        std = np.sqrt(self._sigma2(x_np))
        eta1 = rng.standard_normal((n, *self._field_shape))
        eta2 = rng.standard_normal((n, *self._field_shape))
        eps = self._apply_G(std[None] * eta1) + self.delta * eta2
        return torch.from_numpy(eps.reshape(n, self.n_sites)).float()
