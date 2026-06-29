"""Tests for toy stochastic simulators: Ornstein-Uhlenbeck, double-well,
Lorenz-96, stochastic Gray-Scott, correlated-diffusion ring/torus."""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Imports that will fail until the module is created (RED phase).
# ---------------------------------------------------------------------------
from autosim.experimental.simulations.correlated_diffusion_1d import (
    CorrelatedDiffusion1D,
    diff1d_closed_form_cov,
)
from autosim.experimental.simulations.correlated_diffusion_2d import (
    CorrelatedDiffusion2D,
    diff2d_closed_form_cov,
)
from autosim.experimental.simulations.cox_ingersoll_ross import (
    CoxIngersollRoss,
    cir_closed_form_mean,
    cir_closed_form_var,
)
from autosim.experimental.simulations.double_well import DoubleWell
from autosim.experimental.simulations.gray_scott_stochastic import GrayScottStochastic
from autosim.experimental.simulations.lorenz96 import Lorenz96
from autosim.experimental.simulations.multivariate_ou_lens import (
    MultivariateOULens,
    latent_var,
)
from autosim.experimental.simulations.ornstein_uhlenbeck import (
    OrnsteinUhlenbeck,
    ou_closed_form_var,
)


def test_forward_samples_shape_and_keys():
    """Output dict has the right keys and the data tensor has the expected shape."""
    sim = OrnsteinUhlenbeck(n_steps=20, kappa=1.0, c=0.5, m=0.0, dt=0.05)
    out = sim.forward_samples_spatiotemporal(8, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    assert out["data"].shape == (8, 20, 1, 1, 1)
    assert out["constant_fields"] is None


def test_mc_reference_matches_closed_form_variance():
    """The MC predictive variance must trace the OU closed-form curve."""
    sim = OrnsteinUhlenbeck(n_steps=30, kappa=1.0, c=0.5, m=0.0, dt=0.05)
    x0 = torch.zeros(1, 1, 1)
    # shape: (4000, 30, 1, 1, 1)
    mc = sim.mc_reference(x0, n_draws=4000, n_steps=30, random_seed=1)
    emp_var = mc.var(dim=0).reshape(30)
    leads = torch.arange(1, 31)
    closed = ou_closed_form_var(leads, kappa=1.0, c=0.5, dt=0.05)
    max_err = (emp_var - closed).abs().max().item()
    # Discretisation bias O(dt)≈0.003 + MC noise at 4000 draws ≈ 0.003; allow 0.02
    assert max_err < 0.02, f"max|emp-closed| = {max_err:.4f} exceeds tolerance 0.02"


def test_closed_form_var_monotone_and_saturates():
    """ou_closed_form_var is monotonically increasing and saturates at c^2/(2*kappa)."""
    kappa, c, dt = 1.0, 0.5, 0.05
    leads = torch.arange(1, 201)
    var = ou_closed_form_var(leads, kappa=kappa, c=c, dt=dt)

    # Monotonically non-decreasing
    assert (var[1:] >= var[:-1]).all(), "closed-form variance is not monotone"

    # Saturation: large-lead value should be close to c^2 / (2*kappa)
    asymptote = c**2 / (2.0 * kappa)
    assert abs(var[-1].item() - asymptote) < 1e-3, (
        f"Saturation value {var[-1].item():.6f} far from asymptote {asymptote:.6f}"
    )


def test_mc_reference_shape():
    """mc_reference returns shape (n_draws, n_steps, 1, 1, 1)."""
    sim = OrnsteinUhlenbeck(n_steps=10, kappa=1.0, c=0.5, m=0.0, dt=0.05)
    x0 = torch.zeros(1, 1, 1)
    mc = sim.mc_reference(x0, n_draws=16, n_steps=10, random_seed=42)
    assert mc.shape == (16, 10, 1, 1, 1), f"Unexpected shape: {mc.shape}"


# ---------------------------------------------------------------------------
# Cox-Ingersoll-Ross tests (the nonlinear, state-dependent-spread value toy:
# OU's square-root sibling with a closed-form mean+variance oracle).
# ---------------------------------------------------------------------------


def test_cir_forward_samples_shape_and_keys():
    """Output dict has the right keys and the data tensor has the expected shape."""
    sim = CoxIngersollRoss(n_steps=20, kappa=1.0, theta=1.0, sigma=0.3, dt=0.05)
    out = sim.forward_samples_spatiotemporal(8, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    assert out["data"].shape == (8, 20, 1, 1, 1)
    assert out["constant_fields"] is None
    assert torch.isfinite(out["data"]).all()


def test_cir_default_feller_is_high():
    """The defaults sit well off the zero boundary (Feller number >> 1)."""
    sim = CoxIngersollRoss(kappa=1.0, theta=1.0, sigma=0.3)
    assert sim.feller > 10.0  # 2*1*1/0.09 ~= 22


def test_cir_mc_reference_matches_closed_form_mean_and_var():
    """THE oracle gate: the MC predictive mean AND variance trace the closed
    forms (state-dependent), within O(dt) Euler bias + MC noise."""
    sim = CoxIngersollRoss(n_steps=30, kappa=1.0, theta=1.0, sigma=0.3, dt=0.05)
    x0 = 1.5  # off theta=1 so the state-dependent transient term is active
    x_state = torch.full((1, 1, 1), x0)
    # shape: (40000, 30, 1, 1, 1)
    mc = sim.mc_reference(x_state, n_draws=40000, n_steps=30, random_seed=1)
    emp_mean = mc.mean(dim=0).reshape(30)
    emp_var = mc.var(dim=0).reshape(30)
    leads = torch.arange(1, 31)
    cf_mean = cir_closed_form_mean(leads, kappa=1.0, theta=1.0, x0=x0, dt=0.05)
    cf_var = cir_closed_form_var(leads, kappa=1.0, theta=1.0, sigma=0.3, x0=x0, dt=0.05)
    mean_err = (emp_mean - cf_mean).abs().max().item()
    var_err = (emp_var - cf_var).abs().max().item()
    # mean ~ O(1), nearly unbiased; var ~ 0.045 stationary, O(dt) bias + MC noise.
    assert mean_err < 0.01, f"max|emp_mean - cf_mean| = {mean_err:.5f}"
    assert var_err < 5e-3, f"max|emp_var - cf_var| = {var_err:.5f}"


def test_cir_variance_is_state_dependent():
    """The property OU lacks: the predictive variance depends on x0 (the
    diffusion sqrt(X) scales the noise with the state)."""
    leads = torch.arange(1, 11)
    var_lo = cir_closed_form_var(
        leads, kappa=1.0, theta=1.0, sigma=0.3, x0=0.5, dt=0.05
    )
    var_hi = cir_closed_form_var(
        leads, kappa=1.0, theta=1.0, sigma=0.3, x0=3.0, dt=0.05
    )
    # a higher initial state injects more noise early -> larger short-lead variance
    assert var_hi[0] > var_lo[0] + 1e-4
    # both saturate at the same x0-independent stationary variance
    asymptote = 1.0 * 0.3**2 / (2.0 * 1.0)  # theta * sigma^2 / (2 kappa)
    long = cir_closed_form_var(
        torch.arange(1, 401), kappa=1.0, theta=1.0, sigma=0.3, x0=3.0, dt=0.05
    )
    assert abs(long[-1].item() - asymptote) < 1e-3


def test_cir_closed_form_mean_reverts_to_theta():
    """The conditional mean decays from x0 toward theta (mean reversion)."""
    leads = torch.arange(1, 401)
    mean = cir_closed_form_mean(leads, kappa=1.0, theta=1.0, x0=2.5, dt=0.05)
    assert (mean[1:] <= mean[:-1] + 1e-6).all()  # monotone decreasing from x0>theta
    assert abs(mean[-1].item() - 1.0) < 1e-3  # reverts to theta


def test_cir_high_feller_stays_off_zero():
    """With a high Feller number the simulated paths stay essentially positive."""
    sim = CoxIngersollRoss(n_steps=64, kappa=1.0, theta=1.0, sigma=0.3, dt=0.05)
    x_state = torch.ones(1, 1, 1)
    mc = sim.mc_reference(x_state, n_draws=2000, n_steps=64, random_seed=0)
    # a high-Feller CIR almost never visits zero; allow a vanishing fraction.
    frac_nonpos = (mc <= 0.0).float().mean().item()
    assert frac_nonpos < 1e-3, f"too many non-positive states: {frac_nonpos:.4f}"


def test_cir_mc_reference_shape():
    """mc_reference returns shape (n_draws, n_steps, 1, 1, 1)."""
    sim = CoxIngersollRoss(n_steps=10, kappa=1.0, theta=1.0, sigma=0.3, dt=0.05)
    x_state = torch.ones(1, 1, 1)
    mc = sim.mc_reference(x_state, n_draws=16, n_steps=10, random_seed=42)
    assert mc.shape == (16, 10, 1, 1, 1), f"Unexpected shape: {mc.shape}"


# ---------------------------------------------------------------------------
# Double-well Langevin tests (research sec. 3.6, rung 2: bimodal predictive)
# ---------------------------------------------------------------------------


def test_double_well_forward_samples_shape_and_keys():
    """Output dict has the right keys and the data tensor has the expected shape."""
    sim = DoubleWell(n_steps=20, c=0.5, dt=0.05)
    out = sim.forward_samples_spatiotemporal(8, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    assert out["data"].shape == (8, 20, 1, 1, 1)
    assert out["constant_fields"] is None


def test_double_well_mc_reference_shape():
    """mc_reference returns shape (n_draws, n_steps, 1, 1, 1)."""
    sim = DoubleWell(n_steps=10, c=0.5, dt=0.05)
    x0 = torch.zeros(1, 1, 1)
    mc = sim.mc_reference(x0, n_draws=16, n_steps=10, random_seed=42)
    assert mc.shape == (16, 10, 1, 1, 1), f"Unexpected shape: {mc.shape}"


def test_double_well_mc_reference_is_bimodal_at_intermediate_lead():
    """From x0~0, the MC predictive should populate BOTH wells (~±1) at an
    intermediate lead: a dip in density near 0 between two peaks."""
    sim = DoubleWell(n_steps=80, c=0.5, dt=0.05)
    x0 = torch.zeros(1, 1, 1)
    # shape: (4000, 80, 1, 1, 1)
    mc = sim.mc_reference(x0, n_draws=4000, n_steps=80, random_seed=0)
    # lead=40 (t=2.0): enough time for trajectories to escape the x=0 saddle and
    # settle into either well, while the bimodal structure is still clear before
    # mixing reduces the signal at very long leads.
    lead = 40
    vals = mc[:, lead].reshape(-1)
    frac_left = (vals < -0.5).float().mean()
    frac_right = (vals > 0.5).float().mean()
    frac_mid = ((vals > -0.25) & (vals < 0.25)).float().mean()
    # both wells populated, and the middle is a density dip relative to the wells
    assert frac_left > 0.2, f"Left-well fraction too low: {frac_left:.3f}"
    assert frac_right > 0.2, f"Right-well fraction too low: {frac_right:.3f}"
    assert frac_mid < min(frac_left, frac_right), (
        f"Bimodal dip absent: frac_mid={frac_mid:.3f}, "
        f"frac_left={frac_left:.3f}, frac_right={frac_right:.3f}"
    )


# ---------------------------------------------------------------------------
# Lorenz-96 ring tests (research sec. 3.6, rung 3: chaotic growing spread)
# ---------------------------------------------------------------------------


def test_lorenz96_forward_samples_shape_and_keys():
    sim = Lorenz96(n_steps=32, n_sites=40, forcing=8.0, c=0.5, dt=0.01)
    out = sim.forward_samples_spatiotemporal(6, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    assert out["data"].shape == (6, 32, 40, 1, 1)
    assert out["constant_fields"] is None
    assert torch.isfinite(out["data"]).all()  # bounded / no blow-up


def test_lorenz96_mc_spread_grows_with_lead():
    """Chaotic divergence + process noise => conditional spread grows with lead."""
    sim = Lorenz96(n_steps=64, n_sites=40, forcing=8.0, c=0.5, dt=0.01)
    # a fixed plausible current state (small perturbation off the F-fixed-point)
    torch.manual_seed(0)
    x_state = 8.0 + 0.1 * torch.randn(1, 40, 1)
    # shape: (300, 64, 40, 1, 1)
    mc = sim.mc_reference(x_state, n_draws=300, n_steps=64, random_seed=1)
    assert torch.isfinite(mc).all()
    # total spread per lead = mean over sites of the across-draw variance
    spread = mc.var(dim=0).reshape(64, 40).mean(dim=1)  # (64,)
    early = spread[:8].mean()
    late = spread[-8:].mean()
    assert late > 3.0 * early  # spread grows substantially with lead


# ---------------------------------------------------------------------------
# Stochastic Gray-Scott tests (research sec. 3.6, rung 4: spatial, sustained
# non-contractive spread)
# ---------------------------------------------------------------------------


def test_gray_scott_forward_samples_shape_and_keys():
    sim = GrayScottStochastic(n_steps=32, grid_size=8)
    out = sim.forward_samples_spatiotemporal(5, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    assert out["data"].shape == (5, 32, 8, 8, 2)
    assert out["constant_fields"] is None
    assert torch.isfinite(out["data"]).all()  # bounded / no blow-up


def test_gray_scott_mc_reference_shape():
    """mc_reference returns shape (n_draws, n_steps, 8, 8, 2)."""
    sim = GrayScottStochastic(n_steps=10, grid_size=8)
    x_state = torch.zeros(8, 8, 2)
    mc = sim.mc_reference(x_state, n_draws=16, n_steps=10, random_seed=42)
    assert mc.shape == (16, 10, 8, 8, 2), f"Unexpected shape: {mc.shape}"


def test_gray_scott_mc_spread_grows_with_lead():
    """Reaction-sustained dynamics + process noise => conditional spread is
    nonzero and grows with lead (the non-contractive signature), in contrast to
    the contractive advection-diffusion control whose spread decays toward zero."""
    sim = GrayScottStochastic(n_steps=96, grid_size=8)
    # a fixed plausible current state: the active central-seed initial condition
    u0, v0 = sim._initial_condition(0.5)
    x_state = torch.from_numpy(np.stack([u0, v0], axis=-1))  # (8, 8, 2)
    # shape: (300, 96, 8, 8, 2)
    mc = sim.mc_reference(x_state, n_draws=300, n_steps=96, random_seed=1)
    assert torch.isfinite(mc).all()
    # spread per lead = mean over (grid, grid, channel) of the across-draw variance
    spread = mc.var(dim=0).reshape(96, -1).mean(dim=1)  # (96,)
    early = spread[:8].mean()
    late = spread[-8:].mean()
    assert early > 1e-4, f"Spread is degenerate/near-zero at early lead: {early:.2e}"
    # ratio runs ~7-12x across seeds at these defaults; 2.5x is a safe,
    # decisively non-contractive margin.
    assert late > 2.5 * early, (
        f"Spread does not grow with lead: early={early:.5f}, late={late:.5f}"
    )


# ---------------------------------------------------------------------------
# MultivariateOULens: latent multivariate-OU through a fixed nonlinear lens.
# ---------------------------------------------------------------------------


def test_mvou_lens_forward_samples_shape_and_keys():
    """Default d_z=4/d_x=16 reshapes to a 4x4x1 field; latent z0 is exposed."""
    sim = MultivariateOULens(n_steps=12)
    out = sim.forward_samples_spatiotemporal(8, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields", "latent_states"}
    assert out["data"].shape == (8, 12, 4, 4, 1)
    assert out["constant_fields"] is None
    assert out["latent_states"].shape == (8, 4)


def test_mvou_lens_latent_var_matches_monte_carlo():
    """The Lyapunov recursion equals the empirical k-step latent covariance."""
    sim = MultivariateOULens()
    a = sim.A
    chol = np.linalg.cholesky(sim.Sigma_z)
    leads = [1, 3, 10]

    rng = np.random.default_rng(0)
    n = 200_000
    z0 = np.array([0.5, -0.3, 0.1, 0.2])
    z = np.tile(z0, (n, 1))
    emp = {}
    for k in range(1, max(leads) + 1):
        eps = rng.standard_normal((n, sim.d_z)) @ chol.T
        z = z @ a.T + eps
        if k in leads:
            emp[k] = np.cov(z.T, bias=False)

    vk = latent_var(
        torch.tensor(leads), torch.as_tensor(sim.A), torch.as_tensor(sim.Sigma_z)
    ).numpy()
    for i, k in enumerate(leads):
        max_err = np.abs(emp[k] - vk[i]).max()
        assert max_err < 0.02, f"lead {k}: max|cov_MC - V_k| = {max_err:.4f}"


def test_mvou_lens_latent_var_recursion_and_saturation():
    """V_k satisfies V_k = A V_{k-1} Aᵀ + Sigma_z and saturates at the fixed point."""
    sim = MultivariateOULens()
    a, sigma = sim.A, sim.Sigma_z
    vk = latent_var(
        torch.arange(1, 41), torch.as_tensor(a), torch.as_tensor(sigma)
    ).numpy()
    # one-step recursion holds between consecutive leads
    for k in range(1, vk.shape[0]):
        recur = a @ vk[k - 1] @ a.T + sigma
        assert np.abs(vk[k] - recur).max() < 1e-9
    # saturation: V_inf is a fixed point of the Lyapunov map
    v_inf = vk[-1]
    resid = np.abs(a @ v_inf @ a.T + sigma - v_inf).max()
    assert resid < 1e-3, f"saturation residual {resid:.2e} too large"


def test_mvou_lens_marginals_are_heavy_tailed():
    """The sinh marginals have kurtosis ~7.7 at the default tail_alpha=0.7."""
    sim = MultivariateOULens(tail_alpha=0.7)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2_000_000, sim.d_z))  # unit-variance latent
    x = sim.lens(z)  # (N, d_x)
    # first d_z ambient coords are the pure sinh marginals
    col = x[:, 0]
    col = col - col.mean()
    kurt = (col**4).mean() / (col**2).mean() ** 2
    assert 6.5 < kurt < 9.0, f"sinh-marginal kurtosis {kurt:.2f} not heavy-tailed"


def test_mvou_lens_lens_is_deterministic_and_maps_dims():
    sim = MultivariateOULens()
    z = np.random.default_rng(1).standard_normal((5, sim.d_z))
    x1 = sim.lens(z)
    x2 = sim.lens(z)
    assert x1.shape == (5, sim.obs_dim)
    assert np.array_equal(x1, x2)


def test_mvou_lens_noise_covariance_matches_sigma_z():
    """Both gaussian and student-t latent noise have second moment Sigma_z."""
    for noise in ("gaussian", "student_t"):
        sim = MultivariateOULens(latent_noise=noise, student_t_dof=5.0)
        rng = np.random.default_rng(0)
        draws = np.stack([sim._draw_noise(rng) for _ in range(200_000)])
        emp_cov = np.cov(draws.T, bias=False)
        assert np.abs(emp_cov - sim.Sigma_z).max() < 0.03, noise


def test_mvou_lens_student_t_is_heavier_tailed_negative_control():
    """The heavy-tailed-latent control: student-t eps has higher kurtosis than
    gaussian eps at matched covariance."""
    rng = np.random.default_rng(0)
    g_sim = MultivariateOULens(latent_noise="gaussian")
    t_sim = MultivariateOULens(latent_noise="student_t", student_t_dof=4.0)
    g = np.stack([g_sim._draw_noise(rng) for _ in range(400_000)])[:, 0]
    t = np.stack([t_sim._draw_noise(rng) for _ in range(400_000)])[:, 0]

    def kurt(v):
        v = v - v.mean()
        return (v**4).mean() / (v**2).mean() ** 2

    assert kurt(t) > kurt(g) + 1.0


def test_mvou_lens_mc_reference_shape():
    sim = MultivariateOULens(n_steps=8)
    z0 = torch.zeros(4)
    mc = sim.mc_reference(z0, n_draws=64, n_steps=10, random_seed=0)
    assert mc.shape == (64, 10, 4, 4, 1)
    assert torch.isfinite(mc).all()


def test_mvou_lens_rejects_non_contractive_transition():
    a = np.eye(4)  # spectral radius 1.0
    with pytest.raises(ValueError, match="contractive"):
        MultivariateOULens(A=torch.as_tensor(a))


def test_mvou_lens_rejects_obs_dim_not_greater_than_latent():
    with pytest.raises(ValueError, match="obs_dim > d_z"):
        MultivariateOULens(obs_dim=4)  # default d_z=4


def test_mvou_lens_rejects_obs_shape_mismatch():
    with pytest.raises(ValueError, match="must multiply to"):
        MultivariateOULens(obs_dim=16, obs_shape=(3, 3, 1))


def test_mvou_lens_rejects_unknown_latent_noise():
    with pytest.raises(ValueError, match="latent_noise"):
        MultivariateOULens(latent_noise="cauchy")


def test_mvou_lens_rejects_student_t_dof_not_greater_than_two():
    with pytest.raises(ValueError, match="student_t_dof > 2"):
        MultivariateOULens(latent_noise="student_t", student_t_dof=2.0)


def test_mvou_lens_rejects_singular_noise_covariance():
    # zero D_z entries not covered by the rank-1 U_z direction -> only PSD
    with pytest.raises(ValueError, match="positive-definite"):
        MultivariateOULens(
            A=torch.as_tensor(np.diag([0.9, 0.8])),
            D_z=torch.tensor([0.0, 0.0]),
            U_z=torch.tensor([[0.3], [0.4]]),
            obs_dim=4,
        )


def test_latent_var_rejects_non_positive_leads():
    sim = MultivariateOULens()
    with pytest.raises(ValueError, match="1-based positive"):
        latent_var(
            torch.tensor([0, 1]), torch.as_tensor(sim.A), torch.as_tensor(sim.Sigma_z)
        )


# ---------------------------------------------------------------------------
# Correlated-diffusion ring (diff1d): stable linear diffusion with a known
# state-dependent low-rank-plus-diagonal forcing covariance (the
# correlation-structure preservation toy; closed-form Sigma(x) oracle).
# ---------------------------------------------------------------------------


def test_diff1d_forward_samples_shape_and_keys():
    """Output dict has the right keys and the data tensor has the expected shape."""
    sim = CorrelatedDiffusion1D(n_steps=12)
    out = sim.forward_samples_spatiotemporal(6, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    # Ring of N=40 sites, single channel; gen_diff1d.py layout (n, T, N, 1, 1).
    assert out["data"].shape == (6, 12, 40, 1, 1)
    assert out["constant_fields"] is None
    assert torch.isfinite(out["data"]).all()


def test_diff1d_closed_form_cov_matches_mc():
    """The closed-form one-step covariance traces the Monte-Carlo covariance."""
    sim = CorrelatedDiffusion1D()  # default ring params (N=40)
    x = sim.sample_state(seed=0)  # a single state vector (N,)
    cov_cf = diff1d_closed_form_cov(x)  # (N, N): G diag(sigma2(x)) Gᵀ + delta² I
    eps = sim.sample_noise(x, n=200_000, seed=1)  # (n, N) draws from N(0, Sigma(x))
    cov_mc = torch.cov(eps.T)
    assert torch.allclose(cov_cf, cov_mc, atol=2e-2, rtol=5e-2)


def test_diff1d_forcing_is_genuinely_correlated():
    """The smoothed forcing carries real off-diagonal correlation (so the toy
    actually exercises collection coverage; gen_diff1d.py verification gate 3)."""
    sim = CorrelatedDiffusion1D()
    x = sim.sample_state(seed=0)
    cov = diff1d_closed_form_cov(x)
    d = torch.sqrt(torch.diag(cov))
    corr = cov / torch.outer(d, d)
    nn = torch.tensor([corr[i, (i + 1) % sim.n_sites] for i in range(sim.n_sites)])
    assert nn.mean() > 0.1  # nearest-neighbour forcing correlation present


def test_diff1d_spectral_radius_below_one():
    """Stable / contractive mean operator (fail-early invariant)."""
    sim = CorrelatedDiffusion1D()
    assert sim.A_spectral_radius() < 1.0


def test_diff1d_rejects_non_contractive_mean():
    """gamma/kappa that push the mean operator past the unit circle are rejected."""
    with pytest.raises(ValueError, match="contractive"):
        CorrelatedDiffusion1D(gamma=0.0, kappa=2.0)


# ---------------------------------------------------------------------------
# Correlated-diffusion torus (diff2d): the 2-D version of the ring toy (8x8
# field, 5-point Laplacian). Sharper correlation-structure showcase.
# ---------------------------------------------------------------------------


def test_diff2d_forward_samples_shape_and_keys():
    """Output dict has the right keys and the data tensor has the expected shape."""
    sim = CorrelatedDiffusion2D(n_steps=12)
    out = sim.forward_samples_spatiotemporal(5, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    # 8x8 torus, single channel; gen_diff2d.py layout (n, T, NS, NS, 1).
    assert out["data"].shape == (5, 12, 8, 8, 1)
    assert out["constant_fields"] is None
    assert torch.isfinite(out["data"]).all()


def test_diff2d_closed_form_cov_matches_mc():
    """The closed-form one-step covariance traces the Monte-Carlo covariance."""
    sim = CorrelatedDiffusion2D()  # default torus params (8x8 = 64 sites)
    x = sim.sample_state(seed=0)  # a single field, flattened to (64,)
    cov_cf = diff2d_closed_form_cov(x)  # (64, 64): G diag(sigma2(x)) Gᵀ + delta² I
    eps = sim.sample_noise(x, n=200_000, seed=1)  # (n, 64) draws from N(0, Sigma(x))
    cov_mc = torch.cov(eps.T)
    assert torch.allclose(cov_cf, cov_mc, atol=2e-2, rtol=5e-2)


def test_diff2d_forcing_is_genuinely_correlated():
    """The smoothed forcing carries real off-diagonal correlation on the torus
    (gen_diff2d.py verification gate 3)."""
    sim = CorrelatedDiffusion2D()
    ns = sim.n_side
    x = sim.sample_state(seed=0)
    cov = diff2d_closed_form_cov(x)
    d = torch.sqrt(torch.diag(cov))
    corr = cov / torch.outer(d, d)
    idx = torch.arange(ns * ns).reshape(ns, ns)
    nn = torch.tensor(
        [corr[idx[i, j], idx[i, (j + 1) % ns]] for i in range(ns) for j in range(ns)]
    )
    assert nn.mean() > 0.1


def test_diff2d_spectral_radius_below_one():
    """Stable / contractive mean operator (fail-early invariant)."""
    sim = CorrelatedDiffusion2D()
    assert sim.A_spectral_radius() < 1.0


def test_diff2d_rejects_non_contractive_mean():
    """gamma/kappa that push the mean operator past the unit circle are rejected."""
    with pytest.raises(ValueError, match="contractive"):
        CorrelatedDiffusion2D(gamma=0.0, kappa=2.0)
