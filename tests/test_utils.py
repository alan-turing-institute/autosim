from typing import Any, cast

import torch

from autosim.utils import plot_spatiotemporal_video


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
