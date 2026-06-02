import numpy as np
import pytest

import autosim.simulations.reaction_diffusion as legacy_reaction_diffusion
from autosim.simulations.spatiotemporal import reaction_diffusion


def test_reaction_diffusion_canonical_matches_legacy_small_case() -> None:
    x = np.array([1.3, 0.1])
    kwargs = {
        "return_timeseries": True,
        "n": 8,
        "L": 20,
        "T": 1.0,
        "dt": 0.25,
    }

    with pytest.warns(DeprecationWarning, match="simulate_reaction_diffusion"):
        legacy_u, legacy_v = legacy_reaction_diffusion.simulate_reaction_diffusion(
            x, **kwargs
        )
    canonical_u, canonical_v = reaction_diffusion.simulate_reaction_diffusion(
        x, **kwargs
    )

    assert legacy_u.shape == canonical_u.shape
    assert legacy_v.shape == canonical_v.shape
    np.testing.assert_allclose(canonical_u, legacy_u, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(canonical_v, legacy_v, atol=1e-4, rtol=1e-4)
