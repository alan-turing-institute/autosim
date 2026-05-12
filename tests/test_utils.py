from typing import Any, cast

import numpy as np
import pytest
import torch

from autosim.utils import (
    _apply_video_downsampling,
    plot_spatiotemporal_sphere_video,
    plot_spatiotemporal_video,
)


def _draw_animation_frame(anim, frame: int) -> None:
    cast(Any, anim)._draw_next_frame(frame, blit=False)


@pytest.mark.parametrize(
    ("transpose_spatial", "expected"),
    [
        (False, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)),
        (True, np.array([[0, 3], [1, 4], [2, 5]], dtype=np.float32)),
    ],
)
def test_plot_video_can_optionally_transpose_spatial_axes(
    transpose_spatial: bool, expected: np.ndarray
) -> None:
    true = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3, 1)

    anim = plot_spatiotemporal_video(
        true=true,
        batch_idx=0,
        transpose_spatial=transpose_spatial,
    )

    _draw_animation_frame(anim, 0)

    image = cast(Any, anim)._fig.axes[0].images[0]
    np.testing.assert_array_equal(np.asarray(image.get_array()), expected)


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


def test_apply_video_downsampling_strides_time_and_space() -> None:
    true = torch.rand(2, 8, 32, 16, 3)
    pred = true + 0.1

    true_ds, pred_ds, uq_ds = _apply_video_downsampling(
        true, pred, None, time_stride=2, spatial_stride=(2, 4)
    )

    assert true_ds.shape == (2, 4, 16, 4, 3)
    assert pred_ds is not None
    assert pred_ds.shape == (2, 4, 16, 4, 3)
    assert uq_ds is None
    torch.testing.assert_close(true_ds, true[:, ::2, ::2, ::4, :])

    # scalar stride applies to every spatial dim; identity strides are a no-op
    same_true, _, _ = _apply_video_downsampling(true, None, None, 1, 1)
    assert same_true is true
    int_ds, _, _ = _apply_video_downsampling(true, None, None, 1, 2)
    assert int_ds.shape == (2, 8, 16, 8, 3)


@pytest.mark.parametrize(
    ("time_stride", "spatial_stride", "match"),
    [
        (0, 1, "time_stride must be a positive integer"),
        (1, -1, "must be a positive integer"),
        (1, (2,), "spatial_stride must have 2 entries"),
    ],
)
def test_apply_video_downsampling_rejects_bad_strides(
    time_stride: int, spatial_stride: Any, match: str
) -> None:
    true = torch.rand(1, 4, 8, 8, 1)
    with pytest.raises(ValueError, match=match):
        _apply_video_downsampling(true, None, None, time_stride, spatial_stride)


def test_plot_video_downsampling_reduces_rendered_grid() -> None:
    true = torch.rand(1, 6, 8, 16, 2)

    anim = plot_spatiotemporal_video(
        true=true, batch_idx=0, time_stride=3, spatial_stride=2
    )

    _draw_animation_frame(anim, 0)
    image = cast(Any, anim)._fig.axes[0].images[0]
    assert np.asarray(image.get_array()).shape == (4, 8)  # 8/2, 16/2


def test_plot_sphere_video_accepts_downsampling() -> None:
    true = torch.rand(1, 6, 16, 8, 1)

    anim = plot_spatiotemporal_video(
        true=true, batch_idx=0, projection="sphere", time_stride=2, spatial_stride=2
    )

    _draw_animation_frame(anim, 0)
    assert anim is not None
