from typing import Any, cast

import pytest
import torch

from autosim.utils import plot_spatiotemporal_sphere_video, plot_spatiotemporal_video


def _draw_animation_frame(anim, frame: int) -> None:
    cast(Any, anim)._draw_next_frame(frame, blit=False)


def test_plot_video_accepts_short_channel_names_and_preserve_aspect() -> None:
    true = torch.rand(1, 3, 8, 16, 3)

    anim = plot_spatiotemporal_video(
        true=true,
        batch_idx=0,
        channel_names=["h"],
        preserve_aspect=True,
    )

    _draw_animation_frame(anim, 0)
    assert anim is not None


def test_plot_video_can_render_sphere_projection() -> None:
    true = torch.rand(1, 3, 8, 4, 2)

    anim = plot_spatiotemporal_video(
        true=true,
        batch_idx=0,
        channel_names=["h"],
        projection="sphere",
    )

    _draw_animation_frame(anim, 0)
    assert anim is not None


def test_plot_sphere_video_accepts_prediction_rows() -> None:
    true = torch.rand(1, 2, 8, 4, 1)
    pred = true + 0.1
    pred_uq = torch.rand_like(true)

    anim = plot_spatiotemporal_sphere_video(
        true=true,
        pred=pred,
        pred_uq=pred_uq,
        batch_idx=0,
        channel_names=["h"],
        colorbar_mode="column",
        colorbar_mode_uq="row",
    )

    _draw_animation_frame(anim, 0)
    assert anim is not None


def test_plot_video_rejects_unknown_projection() -> None:
    true = torch.rand(1, 3, 8, 16, 3)

    with pytest.raises(ValueError, match="Invalid projection"):
        plot_spatiotemporal_video(true=true, projection="globe")  # type: ignore[arg-type]
