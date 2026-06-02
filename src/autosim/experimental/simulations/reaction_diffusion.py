"""Deprecated compatibility wrapper for the promoted reaction-diffusion simulator."""

import warnings
from typing import Any

from autosim.simulations.spatiotemporal.reaction_diffusion import (
    ReactionDiffusion as _ReactionDiffusion,
)
from autosim.simulations.spatiotemporal.reaction_diffusion import (
    integrator_keywords,
)
from autosim.simulations.spatiotemporal.reaction_diffusion import (
    reaction_diffusion as _reaction_diffusion,
)
from autosim.simulations.spatiotemporal.reaction_diffusion import (
    simulate_reaction_diffusion as _simulate_reaction_diffusion,
)


class ReactionDiffusion(_ReactionDiffusion):
    """Deprecated experimental alias for the canonical reaction-diffusion simulator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the deprecated experimental wrapper."""
        warnings.warn(
            "autosim.experimental.simulations.ReactionDiffusion is deprecated. "
            "Use autosim.simulations.spatiotemporal.ReactionDiffusion instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def reaction_diffusion(*args: Any, **kwargs: Any) -> Any:
    """Deprecated experimental alias for the canonical RHS helper."""
    warnings.warn(
        "autosim.experimental.simulations.reaction_diffusion.reaction_diffusion "
        "is deprecated. Use autosim.simulations.spatiotemporal.reaction_diffusion."
        "reaction_diffusion instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _reaction_diffusion(*args, **kwargs)


def simulate_reaction_diffusion(*args: Any, **kwargs: Any) -> Any:
    """Deprecated experimental alias for the canonical simulation helper."""
    warnings.warn(
        "autosim.experimental.simulations.reaction_diffusion."
        "simulate_reaction_diffusion is deprecated. Use "
        "autosim.simulations.spatiotemporal.reaction_diffusion."
        "simulate_reaction_diffusion instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _simulate_reaction_diffusion(*args, **kwargs)


__all__ = [
    "ReactionDiffusion",
    "integrator_keywords",
    "reaction_diffusion",
    "simulate_reaction_diffusion",
]
