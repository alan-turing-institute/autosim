from typing import Any, cast

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from autosim.utils import plot_spatiotemporal_1d, plot_spatiotemporal_video


def test_plot_1d_orients_time_and_space_and_saves(tmp_path) -> None:
    true = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8, 1, 1)
    output = tmp_path / "trajectory.png"
    figure = plot_spatiotemporal_1d(
        true,
        x=torch.arange(8) * 0.1,
        times=torch.tensor([0.0, 0.2, 0.4]),
        channel_names=["u"],
        save_path=str(output),
    )
    axis = figure.axes[0]
    values = axis.collections[0].get_array()
    assert values is not None
    np.testing.assert_array_equal(values.reshape(3, 8), true[0, :, :, 0, 0].numpy())
    assert axis.get_xlabel() == "x"
    assert axis.get_ylabel() == "Time"
    assert axis.get_title() == "Ground Truth: u"
    assert output.exists()
    plt.close(figure)


def test_plot_1d_comparison_shares_scales_and_supports_compact_fields() -> None:
    true = torch.zeros(1, 3, 8, 2)
    pred = torch.ones_like(true)
    figure = plot_spatiotemporal_1d(
        true, pred, torch.full_like(true, 0.1), channel_names=["h"]
    )
    axes = [axis for axis in figure.axes if axis.get_xlabel() == "Grid index"]
    assert len(axes) == 8
    assert axes[0].collections[0].norm is axes[2].collections[0].norm
    assert "Difference" in axes[4].get_title()
    plt.close(figure)


def test_plot_1d_rejects_2d_fields_and_wrong_coordinates() -> None:
    with pytest.raises(ValueError, match="singleton"):
        plot_spatiotemporal_1d(torch.zeros(1, 3, 8, 8, 1))
    with pytest.raises(ValueError, match="Coordinates"):
        plot_spatiotemporal_1d(torch.zeros(1, 3, 8, 1), times=np.array([0, 0]))
    with pytest.raises(ValueError, match="match"):
        plot_spatiotemporal_1d(torch.zeros(1, 3, 8, 1), pred=torch.zeros(1, 2, 8, 1))


def test_plot_video_accepts_short_channel_names_and_preserve_aspect() -> None:
    true = torch.rand(1, 3, 8, 16, 3)

    anim = plot_spatiotemporal_video(
        true=true,
        batch_idx=0,
        channel_names=["h"],
        preserve_aspect=True,
    )

    assert anim is not None


def test_plot_video_accepts_physical_row_labels() -> None:
    true = torch.rand(1, 2, 4, 4, 1)
    pred = torch.rand_like(true)

    anim = plot_spatiotemporal_video(
        true=true,
        pred=pred,
        true_label="Forced",
        pred_label="Control",
    )

    figure = cast(Any, anim)._fig
    row_labels = [axis.get_ylabel() for axis in figure.axes]
    assert "Forced" in row_labels
    assert "Control" in row_labels
    assert "Difference (Forced - Control)" in row_labels


def test_plot_video_preserves_original_positional_arguments() -> None:
    true = torch.rand(1, 2, 4, 8, 1)
    anim = plot_spatiotemporal_video(
        true,
        torch.rand_like(true),
        torch.rand_like(true),
        0,
        5,
        None,
        None,
        "viridis",
        None,
        "Comparison",
        "Uncertainty",
        "column",
        "row",
        ["Depth"],
        True,
    )
    figure = cast(Any, anim)._fig
    assert "Uncertainty" in [axis.get_ylabel() for axis in figure.axes]
    assert "Depth" in [axis.get_title() for axis in figure.axes]
