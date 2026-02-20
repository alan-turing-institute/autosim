"""Shared fixtures and helpers for autosim tests."""

import pytest
import torch

from autosim.simulations.base import Simulator
from autosim.types import TensorLike


class ConcreteSimulator(Simulator):
    """Minimal concrete Simulator for testing the abstract base class.

    _forward returns the first `out_dim` input columns, making outputs
    deterministic and easy to assert on.
    """

    def _forward(self, x: TensorLike) -> TensorLike:
        # x expected shape: (1, in_dim); output shape: (1, out_dim)
        return x[:, : self.out_dim].float()


@pytest.fixture
def simple_sim():
    """2-input, 1-output simulator with varying bounds."""
    return ConcreteSimulator(
        parameters_range={"a": (0.0, 1.0), "b": (0.0, 2.0)},
        output_names=["output"],
        log_level="error",
    )


@pytest.fixture
def two_output_sim():
    """2-input, 2-output simulator."""
    return ConcreteSimulator(
        parameters_range={"a": (0.0, 1.0), "b": (0.0, 2.0)},
        output_names=["out1", "out2"],
        log_level="error",
    )


@pytest.fixture
def constant_sim():
    """Simulator where the first parameter is a constant (min == max)."""
    return ConcreteSimulator(
        parameters_range={"fixed": (1.0, 1.0), "varied": (0.0, 2.0)},
        output_names=["output"],
        log_level="error",
    )


@pytest.fixture
def all_constant_sim():
    """Simulator where all parameters are constants."""
    return ConcreteSimulator(
        parameters_range={"a": (3.0, 3.0), "b": (7.0, 7.0)},
        output_names=["output"],
        log_level="error",
    )
