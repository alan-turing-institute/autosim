from autosim.experimental.simulations import (
    CompressibleFluid2D,
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
    ReactionDiffusion,
    ShallowWater2D,
)
from autosim.simulations import (
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    Epidemic,
    FlowProblem,
    Projectile,
    ProjectileMultioutput,
    SEIRSimulator,
)


def test_all_simulators_have_explicit_output_names() -> None:
    simulators = [
        AdvectionDiffusion(),
        AdvectionDiffusionMultichannel(),
        ReactionDiffusion(),
        Epidemic(),
        SEIRSimulator(),
        FlowProblem(),
        Projectile(),
        ProjectileMultioutput(),
        GrayScott(),
        LatticeBoltzmann(),
        ConditionedNavierStokes2D(),
        Hydrodynamics2D(),
        CompressibleFluid2D(),
        ShallowWater2D(),
        GrossPitaevskiiEquation2D(),
    ]

    for sim in simulators:
        names = sim.output_names
        assert isinstance(names, list)
        assert names
        assert all(isinstance(name, str) and name.strip() for name in names)
        assert names != ["solution"]
