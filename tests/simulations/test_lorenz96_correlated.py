"""Tests for the chaotic Lorenz-96 toy with known spatially-correlated forcing
(l96c): a smooth ring-correlated process noise instead of the white forcing of
:class:`Lorenz96`.

Mirrors the Lorenz-96 (chaos / growing-spread) and correlated-diffusion-ring
(correlated-forcing / oracle) test patterns in ``test_toy_simulators.py``.
"""

import torch

# ---------------------------------------------------------------------------
# Imports that will fail until the module is created (RED phase).
# ---------------------------------------------------------------------------
from autosim.experimental.simulations.lorenz96_correlated import (
    Lorenz96Correlated,
    l96c_forcing_cov,
)

# ---------------------------------------------------------------------------
# Shape / dtype / keys (mirrors the lorenz96 + diff1d shape tests).
# ---------------------------------------------------------------------------


def test_l96c_forward_samples_shape_and_keys():
    sim = Lorenz96Correlated(n_steps=32, n_sites=40, forcing=8.0, c=0.5, dt=0.01)
    out = sim.forward_samples_spatiotemporal(6, random_seed=0)
    assert set(out) == {"data", "constant_scalars", "constant_fields"}
    # Ring of N=40 sites, single channel; gen_l96c.py layout (n, T, N, 1, 1).
    assert out["data"].shape == (6, 32, 40, 1, 1)
    assert out["data"].dtype == torch.float32
    assert out["constant_scalars"].shape == (6, 1)
    assert out["constant_fields"] is None
    assert torch.isfinite(out["data"]).all()  # bounded / no blow-up


def test_l96c_forward_samples_is_deterministic_under_seed():
    """Same seed -> identical trajectories (RNG discipline)."""
    sim = Lorenz96Correlated(n_steps=16, n_sites=40)
    a = sim.forward_samples_spatiotemporal(4, random_seed=7)["data"]
    b = sim.forward_samples_spatiotemporal(4, random_seed=7)["data"]
    assert torch.equal(a, b)
    c = sim.forward_samples_spatiotemporal(4, random_seed=8)["data"]
    assert not torch.equal(a, c)


# ---------------------------------------------------------------------------
# Chaos: infinitesimally-perturbed ICs diverge (positive finite-time Lyapunov
# signature) -- the rung-3 defining property, like the white-forcing l96.
# ---------------------------------------------------------------------------


def test_l96c_mc_spread_grows_with_lead():
    """Chaotic divergence + correlated process noise => conditional spread grows
    with lead. Mirrors ``test_lorenz96_mc_spread_grows_with_lead``."""
    sim = Lorenz96Correlated(n_steps=64, n_sites=40, forcing=8.0, c=0.5, dt=0.01)
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


def test_l96c_chaotic_divergence_of_perturbed_ics():
    """Two trajectories from an infinitesimally-perturbed deterministic IC diverge
    with a POSITIVE finite-time Lyapunov exponent -- the rung-3 chaos signature.

    Run with c=0 (no process noise) so the divergence is purely the chaotic
    sensitivity to initial conditions, not noise-driven spread, starting from a
    burned-in attractor state so the orbit is on the chaotic attractor. Note the
    growth is exponential-in-the-mean but NOT monotone window-to-window: finite-
    time local Lyapunov exponents alternate expansion/contraction along the orbit,
    so we assert the overall growth and a positive log-separation slope rather
    than per-window monotonicity.
    """
    sim = Lorenz96Correlated(n_steps=140, n_sites=40, forcing=8.0, c=0.0, dt=0.01)
    x0 = sim.sample_state(seed=0)  # burned-in onto the chaotic attractor (N,)
    # deterministic (c=0) rollout: a single draw is the trajectory.
    base = sim.mc_reference(x0, n_draws=1, n_steps=140, random_seed=0)
    pert = sim.mc_reference(x0 + 1e-6, n_draws=1, n_steps=140, random_seed=0)
    sep = (base - pert).reshape(140, 40).pow(2).sum(dim=1).sqrt()  # (140,)
    assert torch.isfinite(sep).all()
    # initial separation ~ sqrt(40)*1e-6 (a tiny seed perturbation).
    assert sep[0] < 1e-4
    # it grows materially by mid-rollout (lowest ~3.8x across seeds at lead 120;
    # 2x is a safe positive-Lyapunov margin -- a damped/contractive system would
    # SHRINK the separation toward 0 instead).
    assert sep[120] > 2.0 * sep[0]
    # positive finite-time Lyapunov exponent: least-squares slope of log(sep) vs
    # lead is > 0 over the rollout (exponential divergence, not decay).
    leads = torch.arange(140, dtype=torch.float64)
    log_sep = torch.log(sep.double())
    leads_c = leads - leads.mean()
    slope = (leads_c * (log_sep - log_sep.mean())).sum() / (leads_c**2).sum()
    assert slope.item() > 0.0, f"log-separation slope {slope.item():.4f} not positive"


# ---------------------------------------------------------------------------
# Forcing is genuinely correlated (NOT white), with unit/constant per-site
# variance equal to white l96. Mirrors ``test_diff1d_forcing_is_genuinely_
# correlated`` + the diff1d closed-form-vs-MC check.
# ---------------------------------------------------------------------------


def test_l96c_forcing_correlation_matrix_is_unit_diagonal_smooth_ring():
    """The constructed correlation C has unit diagonal and a smooth, materially
    non-zero nearest-neighbour off-diagonal (the added cross-site structure)."""
    sim = Lorenz96Correlated(n_sites=40, ell=2.0)
    c_mat = sim.correlation_matrix()
    assert c_mat.shape == (40, 40)
    diag = torch.diag(c_mat)
    assert torch.allclose(diag, torch.ones(40), atol=1e-6)  # UNIT diagonal
    nn = torch.tensor([c_mat[i, (i + 1) % 40] for i in range(40)])
    assert nn.mean() > 0.5  # ell=2 -> nn corr ~0.88: clearly correlated
    # smooth decay: next-nearest neighbour weaker than nearest.
    nnn = torch.tensor([c_mat[i, (i + 2) % 40] for i in range(40)])
    assert (nnn.mean() < nn.mean()).item()


def test_l96c_forcing_variance_matches_white_l96():
    """Per-site one-step forcing VARIANCE equals white l96 (c^2 * dt): only the
    cross-site CORRELATION is added. Constant across sites (translation-invariant)."""
    c, dt = 0.5, 0.01
    sim = Lorenz96Correlated(n_sites=40, c=c, dt=dt, ell=2.0)
    fcov = l96c_forcing_cov(n_sites=40, c=c, dt=dt, ell=2.0)
    # the simulator's own forcing-covariance accessor delegates to the helper.
    assert torch.equal(sim.forcing_covariance(), fcov)
    assert fcov.shape == (40, 40)
    var = torch.diag(fcov)
    expected = c**2 * dt  # white-l96 per-site forcing variance
    assert torch.allclose(var, torch.full((40,), expected), atol=1e-8)


def test_l96c_forcing_is_correlated_not_white():
    """The constructed one-step forcing covariance and the empirically-sampled
    forcing both carry real off-diagonal correlation (nn corr > 0) while the
    diagonal is constant -- the defining property vs. white l96."""
    sim = Lorenz96Correlated(n_sites=40, c=0.5, dt=0.01, ell=2.0)
    cov_cf = l96c_forcing_cov(n_sites=40, c=0.5, dt=0.01, ell=2.0)
    eps = sim.sample_noise(n=200_000, seed=1)  # (n, N) draws of the per-lead forcing
    assert eps.shape == (200_000, 40)
    cov_mc = torch.cov(eps.T)
    # closed-form one-step forcing covariance traces the MC covariance.
    assert torch.allclose(cov_cf, cov_mc, atol=2e-4, rtol=5e-2)
    # off-diagonal is materially non-zero (correlated), diagonal ~constant.
    d = torch.sqrt(torch.diag(cov_cf))
    corr = cov_cf / torch.outer(d, d)
    nn = torch.tensor([corr[i, (i + 1) % 40] for i in range(40)])
    assert nn.mean() > 0.5  # NOT white
    var = torch.diag(cov_cf)
    assert var.std() < 1e-6 * var.mean()  # constant per-site variance


# ---------------------------------------------------------------------------
# MC-oracle sanity: shape, reproducibility, finite mean, growing spread.
# ---------------------------------------------------------------------------


def test_l96c_mc_reference_shape():
    """mc_reference returns shape (n_draws, n_steps, n_sites, 1, 1)."""
    sim = Lorenz96Correlated(n_steps=10, n_sites=40)
    x_state = 8.0 + 0.1 * torch.zeros(1, 40, 1)
    mc = sim.mc_reference(x_state, n_draws=16, n_steps=10, random_seed=42)
    assert mc.shape == (16, 10, 40, 1, 1)
    assert mc.dtype == torch.float32


def test_l96c_mc_reference_is_reproducible_and_finite_mean():
    """Fixed seed -> identical oracle ensemble; ensemble mean is finite."""
    sim = Lorenz96Correlated(n_steps=12, n_sites=40)
    x_state = 8.0 + 0.1 * torch.zeros(40)
    a = sim.mc_reference(x_state, n_draws=32, n_steps=12, random_seed=5)
    b = sim.mc_reference(x_state, n_draws=32, n_steps=12, random_seed=5)
    assert torch.equal(a, b)
    c = sim.mc_reference(x_state, n_draws=32, n_steps=12, random_seed=6)
    assert not torch.equal(a, c)
    assert torch.isfinite(a.mean(dim=0)).all()


def test_l96c_mc_oracle_rollout_alias_matches_mc_reference():
    """The task-named ``mc_oracle_rollout`` is an alias of ``mc_reference``
    (same draws under a shared seed, documented (B, n_leads, n_sites, 1, M) shape)."""
    sim = Lorenz96Correlated(n_steps=8, n_sites=40)
    x0 = 8.0 + 0.1 * torch.zeros(40)
    ref = sim.mc_reference(x0, n_draws=24, n_steps=10, random_seed=11)
    alias = sim.mc_oracle_rollout(x0, n_leads=10, n_members=24, seed=11)
    # alias documents a member-last layout (B=1, n_leads, n_sites, 1, M); it must
    # carry the same ensemble as the member-first mc_reference.
    assert alias.shape == (1, 10, 40, 1, 24)
    assert torch.equal(alias[0].permute(3, 0, 1, 2).unsqueeze(-1), ref)


def test_l96c_oracle_spread_grows_with_correlated_forcing_only():
    """With the deterministic part frozen (forcing makes it chaotic anyway), the
    oracle spread is non-degenerate at early lead and grows -- and the early
    cross-site structure reflects the correlated forcing (positive mean
    nearest-neighbour spatial correlation of the increments)."""
    sim = Lorenz96Correlated(n_steps=64, n_sites=40, c=0.5, dt=0.01, ell=2.0)
    x_state = 8.0 + 0.1 * torch.zeros(40)
    mc = sim.mc_reference(x_state, n_draws=400, n_steps=64, random_seed=2)
    spread = mc.var(dim=0).reshape(64, 40).mean(dim=1)
    assert spread[0] > 0.0
    assert spread[-8:].mean() > spread[:8].mean()
    # one-step increments across members are spatially correlated (the forcing
    # signature visible before chaos scrambles the structure).
    inc = mc[:, 0].reshape(400, 40)  # member field at lead 1 (~= forcing draw)
    inc = inc - inc.mean(dim=0, keepdim=True)
    corr = torch.corrcoef(inc.T)
    nn = torch.tensor([corr[i, (i + 1) % 40] for i in range(40)])
    assert nn.mean() > 0.1
