from autosim.simulations import GrayScott


def test_pattern_overrides_default_ranges_only() -> None:
    sim = GrayScott(pattern="gliders", fixed_parameters_given_pattern=False)
    assert sim.parameters_range["F"] == (0.013, 0.015)
    assert sim.parameters_range["k"] == (0.053, 0.055)


def test_pattern_does_not_override_user_ranges() -> None:
    custom_range = {
        "F": (0.02, 0.03),
        "k": (0.055, 0.06),
        "delta_u": (2.0e-5, 2.0e-5),
        "delta_v": (1.0e-5, 1.0e-5),
    }

    sim = GrayScott(
        pattern="gliders",
        parameters_range=custom_range,
        fixed_parameters_given_pattern=False,
    )

    assert sim.parameters_range["F"] == custom_range["F"]
    assert sim.parameters_range["k"] == custom_range["k"]


def test_pattern_fixed_params_mode() -> None:
    sim = GrayScott(pattern="gliders", fixed_parameters_given_pattern=True)

    assert sim.parameters_range["F"] == (0.014, 0.014)
    assert sim.parameters_range["k"] == (0.054, 0.054)


def test_defaults_used_when_pattern_and_range_missing() -> None:
    sim = GrayScott()

    assert sim.parameters_range["F"] == (0.014, 0.1)
    assert sim.parameters_range["k"] == (0.051, 0.065)
