import random
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from torch import Tensor

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import OutputLike, TensorLike, TorchScalarDType


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


class ValidationMixin:
    """
    Mixin class for validation methods.

    This class provides static methods for checking the types and shapes of
    input and output data, as well as validating specific tensor shapes.
    """

    @staticmethod
    def _check(x: TensorLike, y: TensorLike | None):
        """
        Check the types and shape are correct for the input data.

        Checks are equivalent to sklearn's check_array.
        """
        if not isinstance(x, TensorLike):
            raise ValueError(f"Expected x to be TensorLike, got {type(x)}")

        if y is not None and not isinstance(y, TensorLike):
            raise ValueError(f"Expected y to be TensorLike, got {type(y)}")

        # Check x
        if not torch.isfinite(x).all():
            msg = "Input tensor x contains non-finite values"
            raise ValueError(msg)
        if x.dtype not in TorchScalarDType:
            msg = (
                f"Input tensor x has unsupported dtype {x.dtype}. "
                "Expected float32, float64, int32, or int64."
            )
            raise ValueError(msg)

        # Check y if not None
        if y is not None:
            if not torch.isfinite(y).all():
                msg = "Input tensor y contains non-finite values"
                raise ValueError(msg)
            if y.dtype not in TorchScalarDType:
                msg = (
                    f"Input tensor y has unsupported dtype {y.dtype}. "
                    "Expected float32, float64, int32, or int64."
                )
                raise ValueError(msg)

        return x, y

    @staticmethod
    def _check_output(output: OutputLike):
        """Check the types and shape are correct for the output data."""
        if not isinstance(output, OutputLike):
            raise ValueError(f"Expected OutputLike, got {type(output)}")

    @staticmethod
    def check_vector(x: TensorLike) -> TensorLike:
        """
        Validate that the input is a 1D TensorLike.

        Parameters
        ----------
        x: TensorLike
            Input tensor to validate.

        Returns
        -------
        TensorLike
            Validated 1D tensor.

        Raises
        ------
        ValueError
            If x is not a TensorLike or is not 1-dimensional.
        """
        if not isinstance(x, TensorLike):
            raise ValueError(f"Expected TensorLike, got {type(x)}")
        if x.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got {x.ndim}D")
        return x

    @staticmethod
    def check_tensor_is_2d(x: TensorLike) -> TensorLike:
        """
        Validate that the input is a 2D TensorLike.

        Parameters
        ----------
        x: TensorLike
            Input tensor to validate.

        Returns
        -------
        TensorLike
            Validated 2D tensor.

        Raises
        ------
        ValueError
            If x is not a TensorLike or is not 2-dimensional.
        """
        if not isinstance(x, TensorLike):
            raise ValueError(f"Expected TensorLike, got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {x.ndim}D")
        return x

    @staticmethod
    def check_pair(x: TensorLike, y: TensorLike) -> tuple[TensorLike, TensorLike]:
        """
        Validate that two tensors have the same number of rows.

        Parameters
        ----------
        x: TensorLike
            First tensor.
        y: TensorLike
            Second tensor.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            The validated pair of tensors.

        Raises
        ------
        ValueError
            If x and y do not have the same number of rows.
        """
        if x.shape[0] != y.shape[0]:
            msg = "x and y must have the same number of rows"
            raise ValueError(msg)
        return x, y

    @staticmethod
    def check_covariance(y: TensorLike, Sigma: TensorLike) -> TensorLike:
        """
        Validate and return the covariance matrix.

        Parameters
        ----------
        y: TensorLike
            Output tensor.
        Sigma: TensorLike
            Covariance matrix, which may be full, diagonal, or a scalar per sample.

        Returns
        -------
        TensorLike
            Validated covariance matrix.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape relative to y.
        """
        if (
            Sigma.shape == (y.shape[0], y.shape[1], y.shape[1])
            or Sigma.shape == (y.shape[0], y.shape[1])
            or Sigma.shape == (y.shape[0],)
        ):
            return Sigma
        msg = "Invalid covariance matrix shape"
        raise ValueError(msg)

    @staticmethod
    def trace(Sigma: TensorLike, d: int) -> TensorLike:
        """
        Compute the trace of the covariance matrix (A-optimal design criterion).

        Parameters
        ----------
        Sigma: TensorLike
            Covariance matrix (full, diagonal, or scalar).
        d: int
            Dimension of the output.

        Returns
        -------
        TensorLike
            The computed trace value.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if Sigma.dim() == 3 and Sigma.shape[1:] == (d, d):
            return torch.diagonal(Sigma, dim1=1, dim2=2).sum(dim=1).mean()
        if Sigma.dim() == 2 and Sigma.shape[1] == d:
            return Sigma.sum(dim=1).mean()
        if Sigma.dim() == 1:
            return d * Sigma.mean()
        raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")

    @staticmethod
    def logdet(Sigma: TensorLike, dim: int) -> TensorLike:
        """
        Return the log-determinant of the covariance matrix.

        Compute the log-determinant of the covariance matrix (D-optimal design
        criterion).

        Parameters
        ----------
        Sigma: TensorLike
            Covariance matrix (full, diagonal, or scalar).
        dim: int
            Dimension of the output.

        Returns
        -------
        TensorLike
            The computed log-determinant value.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if len(Sigma.shape) == 3 and Sigma.shape[1:] == (dim, dim):
            return torch.logdet(Sigma).mean()
        if len(Sigma.shape) == 2 and Sigma.shape[1] == dim:
            return torch.sum(torch.log(Sigma), dim=1).mean()
        if len(Sigma.shape) == 1:
            return dim * torch.log(Sigma).mean()
        raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")

    @staticmethod
    def max_eigval(Sigma: TensorLike) -> TensorLike:
        """
        Return the maximum eigenvalue of the covariance matrix.

        Compute the maximum eigenvalue of the covariance matrix (E-optimal design
        criterion).

        Parameters
        ----------
        Sigma: TensorLike
            Covariance matrix (full, diagonal, or scalar).

        Returns
        -------
        TensorLike
            The average maximum eigenvalue.

        Raises
        ------
        ValueError
            If Sigma does not have a valid shape.
        """
        if Sigma.dim() == 3 and Sigma.shape[1:] == (Sigma.shape[1], Sigma.shape[1]):
            eigvals = torch.linalg.eigvalsh(Sigma)
            return eigvals[:, -1].mean()  # Eigenvalues are sorted in ascending order
        if Sigma.dim() == 2:
            return Sigma.max(dim=1).values.mean()
        if Sigma.dim() == 1:
            return Sigma.mean()
        raise ValueError(f"Invalid covariance matrix shape: {Sigma.shape}")


def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed: int
        The random seed to use.
    deterministic: bool
        Use "deterministic" algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def plot_spatiotemporal_video(  # noqa: PLR0915, PLR0912
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

    Returns
    -------
    animation.FuncAnimation
        Animation object that can be displayed in notebooks.
    """
    colorbar_mode_str = colorbar_mode.lower()
    valid_modes = {"none", "row", "column", "all"}
    if colorbar_mode_str not in valid_modes:
        raise ValueError(
            "Invalid colorbar_mode "
            f"'{colorbar_mode}'. Expected one of {sorted(valid_modes)}."
        )

    true_batch = true[batch_idx]
    pred_batch = pred[batch_idx] if pred is not None else None
    pred_uq_batch = pred_uq[batch_idx] if pred_uq is not None else None

    # Extract dims and move to CPU
    T, *spatial, C = true_batch.shape
    true_batch = true_batch.detach().cpu().numpy()
    if pred_batch is not None:
        pred_batch = pred_batch.detach().cpu().numpy()
    if pred_uq_batch is not None:
        pred_uq_batch = pred_uq_batch.detach().cpu().numpy()

    primary_rows = [true_batch]

    # Calculate difference
    diff_batch = None
    if pred_batch is not None:
        diff_batch = true_batch - pred_batch
        primary_rows.append(pred_batch)

    # Set-up rows
    n_primary_rows = len(primary_rows)

    def _range_from_arrays(arrays):
        min_val = vmin if vmin is not None else min(float(arr.min()) for arr in arrays)
        max_val = vmax if vmax is not None else max(float(arr.max()) for arr in arrays)
        return min_val, max_val

    norms: list[list[Normalize | None]] = [[None] * C for _ in range(n_primary_rows)]

    if colorbar_mode_str == "column":
        for ch in range(C):
            channel_arrays = [row[:, :, :, ch] for row in primary_rows]
            min_val, max_val = _range_from_arrays(channel_arrays)
            norm = Normalize(vmin=min_val, vmax=max_val)
            for row_idx in range(n_primary_rows):
                norms[row_idx][ch] = norm
    elif colorbar_mode_str == "row":
        for row_idx, row in enumerate(primary_rows):
            min_val, max_val = _range_from_arrays([row])
            norm = Normalize(vmin=min_val, vmax=max_val)
            for ch in range(C):
                norms[row_idx][ch] = norm
    elif colorbar_mode_str == "all":
        min_val, max_val = _range_from_arrays(primary_rows)
        norm = Normalize(vmin=min_val, vmax=max_val)
        for row_idx in range(n_primary_rows):
            for ch in range(C):
                norms[row_idx][ch] = norm
    else:  # "none"
        for row_idx, row in enumerate(primary_rows):
            for ch in range(C):
                min_val, max_val = _range_from_arrays([row[:, :, :, ch]])
                norms[row_idx][ch] = Normalize(vmin=min_val, vmax=max_val)

    diff_norm = None
    if diff_batch is not None:
        diff_max = float(np.abs(diff_batch).max())
        diff_span = diff_max if diff_max > 0 else 1e-9
        diff_norm = TwoSlopeNorm(vmin=-diff_span, vcenter=0, vmax=diff_span)

    rows_to_plot: list[tuple[np.ndarray | Tensor | None, str, str]] = [
        (true_batch, "Ground Truth", cmap),
    ]
    if pred is not None:
        rows_to_plot.append((pred_batch, "Prediction", cmap))
        rows_to_plot.append((diff_batch, "Difference (True - Pred)", "RdBu"))
    if pred_uq is not None:
        rows_to_plot.append((pred_uq_batch, pred_uq_label, "inferno"))
    total_rows = len(rows_to_plot)

    _base = 4.0
    if preserve_aspect and len(spatial) == 2:
        W, H = spatial
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

    for row_idx, (data, row_label, row_cmap) in enumerate(rows_to_plot):
        row_axes = []
        row_images = []

        for ch in range(C):
            ax = fig.add_subplot(gs[row_idx, ch])

            if data is None:
                msg = "Data for plotting cannot be None."
                raise ValueError(msg)
            frame0 = _to_imshow_frame(data[0, :, :, ch])

            if row_idx < n_primary_rows:
                norm = norms[row_idx][ch]
            elif row_idx == len(rows_to_plot) - 1 and pred_uq_batch is not None:
                uq_min = (
                    float(pred_uq_batch[..., ch].min())
                    if colorbar_mode_uq == "none"
                    else float(pred_uq_batch.min())
                )
                uq_max = (
                    float(pred_uq_batch[..., ch].max())
                    if colorbar_mode_uq == "none"
                    else float(pred_uq_batch.max())
                )
                uq_norm = Normalize(vmin=uq_min, vmax=uq_max)
                norm = uq_norm
            else:
                norm = diff_norm
            im = ax.imshow(frame0, cmap=row_cmap, aspect="auto", norm=norm)

            if row_idx == 0:
                (
                    ax.set_title(f"Channel {ch}")
                    if channel_names is None
                    else ax.set_title(f"{channel_names[ch]}")
                )
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
        for ch in range(C):
            images[0][ch].set_array(_to_imshow_frame(true_batch[frame, :, :, ch]))
            if pred_batch is not None:
                images[1][ch].set_array(_to_imshow_frame(pred_batch[frame, :, :, ch]))
            if diff_batch is not None:
                images[2][ch].set_array(_to_imshow_frame(diff_batch[frame, :, :, ch]))
            if pred_uq_batch is not None:
                images[3][ch].set_array(
                    _to_imshow_frame(pred_uq_batch[frame, :, :, ch])
                )
        suptitle_text.set_text(
            f"{title} - Batch {batch_idx} - Time Step: {frame}/{T - 1}"
        )
        return [img for row in images for img in row] + [suptitle_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False, repeat=True
    )

    if save_path:
        Writer = (
            animation.writers["pillow"]
            if save_path.endswith(".gif")
            else animation.writers["ffmpeg"]
        )
        writer = Writer(fps=fps, metadata={"artist": "autoemulate"}, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Video saved to {save_path}")

    plt.close()
    return anim
