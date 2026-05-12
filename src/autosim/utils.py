from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from einops import rearrange
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from torch import Tensor

from autosim.simulations.base import SpatioTemporalSimulator


def generate_output_data(
    sim: SpatioTemporalSimulator,
    n_train: int = 200,
    n_valid: int = 20,
    n_test: int = 20,
):
    """Run simulations and save outputs in a dictionary."""
    train = sim.forward_samples_spatiotemporal(n=n_train, random_seed=None)
    valid = sim.forward_samples_spatiotemporal(n=n_valid, random_seed=None)
    test = sim.forward_samples_spatiotemporal(n=n_test, random_seed=None)
    return {"train": train, "valid": valid, "test": test}


@dataclass
class _VideoPlotData:
    rows_to_plot: list[tuple[np.ndarray, str, str]]
    primary_rows: list[np.ndarray]
    norms: list[list[Normalize]]
    diff_norm: Normalize | None
    pred_uq_batch: np.ndarray | None
    channel_names: list[str]
    n_primary_rows: int
    n_time: int
    spatial: tuple[int, ...]
    n_channels: int


def _validate_mode(name: str, value: str, valid_modes: set[str]) -> str:
    value_str = value.lower()
    if value_str not in valid_modes:
        raise ValueError(
            f"Invalid {name} '{value}'. Expected one of {sorted(valid_modes)}."
        )
    return value_str


def _resolve_channel_names(
    n_channels: int, channel_names: list[str] | None
) -> list[str]:
    resolved_channel_names = [f"Channel {ch}" for ch in range(n_channels)]
    if channel_names is None:
        return resolved_channel_names
    for idx, name in enumerate(channel_names[:n_channels]):
        resolved_channel_names[idx] = str(name)
    return resolved_channel_names


def _range_from_arrays(
    arrays: list[np.ndarray],
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    min_val = vmin if vmin is not None else min(float(arr.min()) for arr in arrays)
    max_val = vmax if vmax is not None else max(float(arr.max()) for arr in arrays)
    return min_val, max_val


def _require_resolved_norms(
    norms: list[list[Normalize | None]],
) -> list[list[Normalize]]:
    resolved_norms: list[list[Normalize]] = []
    for row in norms:
        resolved_row: list[Normalize] = []
        for norm in row:
            if norm is None:
                msg = "Color normalization could not be resolved."
                raise ValueError(msg)
            resolved_row.append(norm)
        resolved_norms.append(resolved_row)
    return resolved_norms


def _resolve_primary_norms(
    primary_rows: list[np.ndarray],
    n_channels: int,
    colorbar_mode: str,
    vmin: float | None,
    vmax: float | None,
) -> list[list[Normalize]]:
    n_primary_rows = len(primary_rows)
    norms: list[list[Normalize | None]] = [
        [None] * n_channels for _ in range(n_primary_rows)
    ]

    if colorbar_mode == "column":
        for ch in range(n_channels):
            channel_arrays = [row[:, :, :, ch] for row in primary_rows]
            min_val, max_val = _range_from_arrays(channel_arrays, vmin, vmax)
            norm = Normalize(vmin=min_val, vmax=max_val)
            for row_idx in range(n_primary_rows):
                norms[row_idx][ch] = norm
    elif colorbar_mode == "row":
        for row_idx, row in enumerate(primary_rows):
            min_val, max_val = _range_from_arrays([row], vmin, vmax)
            norm = Normalize(vmin=min_val, vmax=max_val)
            for ch in range(n_channels):
                norms[row_idx][ch] = norm
    elif colorbar_mode == "all":
        min_val, max_val = _range_from_arrays(primary_rows, vmin, vmax)
        norm = Normalize(vmin=min_val, vmax=max_val)
        for row_idx in range(n_primary_rows):
            for ch in range(n_channels):
                norms[row_idx][ch] = norm
    else:  # "none"
        for row_idx, row in enumerate(primary_rows):
            for ch in range(n_channels):
                min_val, max_val = _range_from_arrays([row[:, :, :, ch]], vmin, vmax)
                norms[row_idx][ch] = Normalize(vmin=min_val, vmax=max_val)

    return _require_resolved_norms(norms)


def _resolve_diff_norm(diff_batch: np.ndarray | None) -> Normalize | None:
    if diff_batch is None:
        return None
    diff_max = float(np.abs(diff_batch).max())
    diff_span = diff_max if diff_max > 0 else 1e-9
    return TwoSlopeNorm(vmin=-diff_span, vcenter=0, vmax=diff_span)


def _prepare_video_plot_data(
    true: Tensor,
    pred: Tensor | None,
    pred_uq: Tensor | None,
    batch_idx: int,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
    pred_uq_label: str,
    colorbar_mode: str,
    channel_names: list[str] | None,
) -> _VideoPlotData:
    true_tensor = true[batch_idx]
    n_time = int(true_tensor.shape[0])
    spatial = tuple(int(dim) for dim in true_tensor.shape[1:-1])
    n_channels = int(true_tensor.shape[-1])

    true_batch = true_tensor.detach().cpu().numpy()
    pred_batch = pred[batch_idx].detach().cpu().numpy() if pred is not None else None
    pred_uq_batch = (
        pred_uq[batch_idx].detach().cpu().numpy() if pred_uq is not None else None
    )

    primary_rows = [true_batch]
    rows_to_plot: list[tuple[np.ndarray, str, str]] = [
        (true_batch, "Ground Truth", cmap),
    ]
    diff_batch = None
    if pred_batch is not None:
        diff_batch = true_batch - pred_batch
        primary_rows.append(pred_batch)
        rows_to_plot.append((pred_batch, "Prediction", cmap))
        rows_to_plot.append((diff_batch, "Difference (True - Pred)", "RdBu"))
    if pred_uq_batch is not None:
        rows_to_plot.append((pred_uq_batch, pred_uq_label, "inferno"))

    return _VideoPlotData(
        rows_to_plot=rows_to_plot,
        primary_rows=primary_rows,
        norms=_resolve_primary_norms(
            primary_rows=primary_rows,
            n_channels=n_channels,
            colorbar_mode=colorbar_mode,
            vmin=vmin,
            vmax=vmax,
        ),
        diff_norm=_resolve_diff_norm(diff_batch),
        pred_uq_batch=pred_uq_batch,
        channel_names=_resolve_channel_names(n_channels, channel_names),
        n_primary_rows=len(primary_rows),
        n_time=n_time,
        spatial=spatial,
        n_channels=n_channels,
    )


def _resolve_row_norm(
    plot_data: _VideoPlotData,
    row_idx: int,
    ch: int,
    colorbar_mode_uq: str,
) -> Normalize:
    if row_idx < plot_data.n_primary_rows:
        return plot_data.norms[row_idx][ch]
    if (
        row_idx == len(plot_data.rows_to_plot) - 1
        and plot_data.pred_uq_batch is not None
    ):
        if colorbar_mode_uq == "none":
            uq_min = float(plot_data.pred_uq_batch[..., ch].min())
            uq_max = float(plot_data.pred_uq_batch[..., ch].max())
        else:
            uq_min = float(plot_data.pred_uq_batch.min())
            uq_max = float(plot_data.pred_uq_batch.max())
        return Normalize(vmin=uq_min, vmax=uq_max)
    if plot_data.diff_norm is None:
        msg = "Difference normalization could not be resolved."
        raise ValueError(msg)
    return plot_data.diff_norm


def _save_animation_if_requested(
    anim: animation.FuncAnimation,
    save_path: str | None,
    fps: int,
) -> None:
    if not save_path:
        return
    Writer = (
        animation.writers["pillow"]
        if save_path.endswith(".gif")
        else animation.writers["ffmpeg"]
    )
    writer = Writer(fps=fps, metadata={"artist": "autoemulate"}, bitrate=1800)
    anim.save(save_path, writer=writer)
    print(f"Video saved to {save_path}")


def plot_spatiotemporal_video(  # noqa: PLR0915
    true: Tensor,
    pred: Tensor | None = None,
    pred_uq: Tensor | None = None,
    batch_idx: int = 0,
    fps: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    save_path: str | None = None,
    title: str = "Ground Truth vs Prediction",
    pred_uq_label: str = "Prediction UQ",
    colorbar_mode: Literal["none", "row", "column", "all"] = "none",
    colorbar_mode_uq: Literal["none", "row"] = "none",
    channel_names: list[str] | None = None,
    preserve_aspect: bool = False,
    projection: Literal["plane", "sphere"] = "plane",
):
    """Create a video comparing ground truth and predicted spatiotemporal time series.

    Parameters
    ----------
    true: array_like (B, T, W, H, C)
        Ground-truth tensor.
    pred: array_like
        Optional predicted tensor of shape (B, T, W, H, C).
    batch_idx: int
        Which batch index to visualize (default: 0).
    fps: int, optional
        Frames per second for the video (default: 5).
    vmin: float, optional
        Minimum value for color scale (default: auto from data).
    vmax: float, optional
        Maximum value for color scale (default: auto from data).
    cmap: str, optional
        Colormap to use (default: "viridis").
    save_path: str, optional
        Optional path to save the video (e.g., "output.mp4").
    title: str, optional
        Title for the video (default: "Ground Truth vs Prediction").
    colorbar_mode: {"none", "row", "column", "all"}
        Select how colorbars (and underlying color scales) are shared for the
        first two rows (true vs prediction):
        - "none": every subplot gets its own colorbar (default).
        - "row": a single colorbar per row (first two rows only).
        - "column": a single colorbar per column (true/pred share per channel).
        - "all": one colorbar shared across the first two rows.
    channel_names: list[str] | None
        Optional list of channel names for titles.
    preserve_aspect: bool
        If True, resize each subplot panel to match the spatial WxH ratio of the
        data so the image fills the panel without distortion. If False (default),
        panels are square and the image is stretched to fill via ``aspect='auto'``.
    projection: {"plane", "sphere"}
        Render the spatial grid as a planar image or as a sphere. Spherical
        rendering assumes the spatial dimensions are ``[longitude, colatitude]``.

    Returns
    -------
    animation.FuncAnimation
        Animation object that can be displayed in notebooks.
    """
    projection_str = _validate_mode("projection", projection, {"plane", "sphere"})
    if projection_str == "sphere":
        return plot_spatiotemporal_sphere_video(
            true=true,
            pred=pred,
            pred_uq=pred_uq,
            batch_idx=batch_idx,
            fps=fps,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            save_path=save_path,
            title=title,
            pred_uq_label=pred_uq_label,
            colorbar_mode=colorbar_mode,
            colorbar_mode_uq=colorbar_mode_uq,
            channel_names=channel_names,
        )

    colorbar_mode_str = _validate_mode(
        "colorbar_mode", colorbar_mode, {"none", "row", "column", "all"}
    )
    colorbar_mode_uq_str = _validate_mode(
        "colorbar_mode_uq", colorbar_mode_uq, {"none", "row"}
    )
    plot_data = _prepare_video_plot_data(
        true=true,
        pred=pred,
        pred_uq=pred_uq,
        batch_idx=batch_idx,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        pred_uq_label=pred_uq_label,
        colorbar_mode=colorbar_mode_str,
        channel_names=channel_names,
    )
    T = plot_data.n_time
    C = plot_data.n_channels
    total_rows = len(plot_data.rows_to_plot)

    _base = 4.0
    if preserve_aspect and len(plot_data.spatial) == 2:
        W, H = plot_data.spatial
        # _to_imshow_frame does NOT transpose by default, so imshow receives (W, H):
        # rows = W (figure height), cols = H (figure width).
        # Scale the smaller base dimension and cap to avoid excessively large figures.
        if H > 0 and W > 0:
            ratio = W / H  # rows / cols
            if ratio >= 1:  # taller than wide
                panel_width = _base
                panel_height = min(_base * ratio, 3 * _base)
            else:  # wider than tall
                panel_height = _base
                panel_width = min(_base / ratio, 3 * _base)
        else:
            panel_width = _base
            panel_height = _base
    else:
        panel_width = _base
        panel_height = _base

    fig = plt.figure(figsize=(C * panel_width, total_rows * panel_height))
    gs = GridSpec(total_rows, C, figure=fig, hspace=0.3, wspace=0.3)

    axes = []
    images = []

    def _to_imshow_frame(
        frame: np.ndarray | Tensor, transpose: bool = False
    ) -> np.ndarray:
        frame = np.asarray(frame)
        if transpose:
            frame = np.asarray(rearrange(frame, "s1 s2 -> s2 s1"))
        return frame

    for row_idx, (data, row_label, row_cmap) in enumerate(plot_data.rows_to_plot):
        row_axes = []
        row_images = []

        for ch in range(C):
            ax = fig.add_subplot(gs[row_idx, ch])

            frame0 = _to_imshow_frame(data[0, :, :, ch])

            norm = _resolve_row_norm(plot_data, row_idx, ch, colorbar_mode_uq_str)
            aspect = "equal" if preserve_aspect else "auto"
            im = ax.imshow(frame0, cmap=row_cmap, aspect=aspect, norm=norm)

            if row_idx == 0:
                ax.set_title(plot_data.channel_names[ch])
            if ch == 0:
                ax.set_ylabel(row_label)

            row_axes.append(ax)
            row_images.append(im)

        axes.append(row_axes)
        images.append(row_images)

    def _attach_colorbars():
        for row_idx, row_axes in enumerate(axes):
            for ch_idx, ax in enumerate(row_axes):
                fig.colorbar(
                    images[row_idx][ch_idx],
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                )

    _attach_colorbars()

    suptitle_text = fig.suptitle("", fontsize=14, fontweight="bold")

    def update(frame):
        for row_idx, (data, _row_label, _row_cmap) in enumerate(plot_data.rows_to_plot):
            for ch in range(C):
                images[row_idx][ch].set_array(_to_imshow_frame(data[frame, :, :, ch]))
        suptitle_text.set_text(
            f"{title} - Batch {batch_idx} - Time Step: {frame}/{T - 1}"
        )
        return [img for row in images for img in row] + [suptitle_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False, repeat=True
    )

    _save_animation_if_requested(anim, save_path, fps)

    plt.close()
    return anim


def plot_spatiotemporal_sphere_video(  # noqa: PLR0915
    true: Tensor,
    pred: Tensor | None = None,
    pred_uq: Tensor | None = None,
    batch_idx: int = 0,
    fps: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    save_path: str | None = None,
    title: str = "Ground Truth vs Prediction",
    pred_uq_label: str = "Prediction UQ",
    colorbar_mode: Literal["none", "row", "column", "all"] = "none",
    colorbar_mode_uq: Literal["none", "row"] = "none",
    channel_names: list[str] | None = None,
    radius: float = 1.0,
):
    """Create a spherical video for data on a longitude/colatitude grid.

    Parameters
    ----------
    true: array_like (B, T, longitude, colatitude, C)
        Ground-truth tensor.
    pred: array_like
        Optional predicted tensor of shape (B, T, longitude, colatitude, C).
    pred_uq: array_like
        Optional uncertainty tensor of shape (B, T, longitude, colatitude, C).
    batch_idx: int
        Which batch index to visualize (default: 0).
    fps: int, optional
        Frames per second for the video (default: 5).
    vmin: float, optional
        Minimum value for color scale (default: auto from data).
    vmax: float, optional
        Maximum value for color scale (default: auto from data).
    cmap: str, optional
        Colormap to use for true/prediction rows (default: "viridis").
    save_path: str, optional
        Optional path to save the video (e.g., "output.mp4").
    title: str, optional
        Title for the video.
    pred_uq_label: str, optional
        Row label for uncertainty plots.
    colorbar_mode: {"none", "row", "column", "all"}
        Select how color scales are shared for the true/prediction rows.
    colorbar_mode_uq: {"none", "row"}
        Select how uncertainty color scales are shared.
    channel_names: list[str] | None
        Optional list of channel names for titles.
    radius: float
        Sphere radius used for rendering.

    Returns
    -------
    animation.FuncAnimation
        Animation object that can be displayed in notebooks.
    """
    colorbar_mode_str = _validate_mode(
        "colorbar_mode", colorbar_mode, {"none", "row", "column", "all"}
    )
    colorbar_mode_uq_str = _validate_mode(
        "colorbar_mode_uq", colorbar_mode_uq, {"none", "row"}
    )
    plot_data = _prepare_video_plot_data(
        true=true,
        pred=pred,
        pred_uq=pred_uq,
        batch_idx=batch_idx,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        pred_uq_label=pred_uq_label,
        colorbar_mode=colorbar_mode_str,
        channel_names=channel_names,
    )
    if len(plot_data.spatial) != 2:
        msg = "Spherical plotting expects two spatial dimensions."
        raise ValueError(msg)
    T = plot_data.n_time
    nlon, ncolat = plot_data.spatial
    C = plot_data.n_channels
    total_rows = len(plot_data.rows_to_plot)

    # Dedalus colatitude nodes run south -> north (theta from ~pi down to ~0) and
    # never touch the poles. Approximate them with uniform midpoints in that same
    # order so ``data[:, k]`` lines up with colatitude row ``k``, then bracket the
    # grid with explicit pole vertices so the polar quads collapse to a single
    # (longitude-averaged) colour rather than a multi-coloured pinwheel.
    colat_mid = ((np.arange(ncolat) + 0.5) * np.pi / ncolat)[::-1]  # south -> north
    theta = np.concatenate(([np.pi], colat_mid, [0.0]))  # + south/north pole caps
    phi = np.linspace(0.0, 2.0 * np.pi, nlon + 1)  # closes the seam at 2*pi
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
    x_sphere = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y_sphere = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z_sphere = radius * np.cos(theta_grid)

    fig = plt.figure(figsize=(C * 4.0, total_rows * 4.0))
    gs = GridSpec(total_rows, C, figure=fig, hspace=0.3, wspace=0.15)

    axes = []
    surfaces = []
    surface_norms = []

    def _sphere_scalar(frame: np.ndarray | Tensor) -> np.ndarray:
        # frame: (nlon, ncolat), colatitude ordered south -> north.
        frame = np.asarray(frame)
        south_cap = np.full((frame.shape[0], 1), float(frame[:, 0].mean()))
        north_cap = np.full((frame.shape[0], 1), float(frame[:, -1].mean()))
        field = np.concatenate([south_cap, frame, north_cap], axis=1)  # add pole rows
        return np.concatenate([field, field[:1, :]], axis=0)  # wrap longitude

    def _surface_colors(
        frame: np.ndarray | Tensor,
        row_cmap: str,
        norm: Normalize,
    ) -> np.ndarray:
        return plt.get_cmap(row_cmap)(norm(_sphere_scalar(frame)))

    for row_idx, (data, row_label, row_cmap) in enumerate(plot_data.rows_to_plot):
        row_axes = []
        row_surfaces = []
        row_surface_norms = []

        for ch in range(C):
            ax = fig.add_subplot(gs[row_idx, ch], projection="3d")

            frame0 = data[0, :, :, ch]

            norm = _resolve_row_norm(plot_data, row_idx, ch, colorbar_mode_uq_str)

            surface = ax.plot_surface(
                x_sphere,
                y_sphere,
                z_sphere,
                facecolors=_surface_colors(frame0, row_cmap, norm),
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            ax.set_axis_off()
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.view_init(elev=25.0, azim=-60.0)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(-radius, radius)

            if row_idx == 0:
                ax.set_title(plot_data.channel_names[ch])
            if ch == 0:
                ax.text2D(
                    -0.08,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                )

            colorbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=row_cmap)
            colorbar_mappable.set_array([])
            fig.colorbar(colorbar_mappable, ax=ax, fraction=0.046, pad=0.04)

            row_axes.append(ax)
            row_surfaces.append(surface)
            row_surface_norms.append(norm)

        axes.append(row_axes)
        surfaces.append(row_surfaces)
        surface_norms.append(row_surface_norms)

    suptitle_text = fig.suptitle("", fontsize=14, fontweight="bold")

    def update(frame):
        for row_idx, (data, _row_label, row_cmap) in enumerate(plot_data.rows_to_plot):
            for ch in range(C):
                surfaces[row_idx][ch].remove()
                norm = surface_norms[row_idx][ch]

                surfaces[row_idx][ch] = axes[row_idx][ch].plot_surface(
                    x_sphere,
                    y_sphere,
                    z_sphere,
                    facecolors=_surface_colors(data[frame, :, :, ch], row_cmap, norm),
                    rstride=1,
                    cstride=1,
                    linewidth=0,
                    antialiased=False,
                    shade=False,
                )

        suptitle_text.set_text(
            f"{title} - Batch {batch_idx} - Time Step: {frame}/{T - 1}"
        )
        return [surface for row in surfaces for surface in row] + [suptitle_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False, repeat=True
    )

    _save_animation_if_requested(anim, save_path, fps)

    plt.close()
    return anim
