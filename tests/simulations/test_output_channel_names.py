from autosim.experimental.simulations import GrayScott
from autosim.simulations import AdvectionDiffusion, ReactionDiffusion


def test_default_output_channel_names_are_explicit() -> None:
    advection = AdvectionDiffusion()
    reaction = ReactionDiffusion()
    gray_scott = GrayScott()

    assert advection.output_names == ["vorticity"]
    assert reaction.output_names == ["u", "v"]
    assert gray_scott.output_names == ["u", "v"]
