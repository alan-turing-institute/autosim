"""Seeded Gaussian fields for shape-matched performance workloads."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from einops import rearrange

from autosim.simulations.base import SpatioTemporalSimulator


class GaussianField(SpatioTemporalSimulator):
    """Generate Gaussian fields with optional spatial, channel and time correlation.

    These fields have unit marginal variance and no physical interpretation.
    With ``rho=0`` all frames are independent. Otherwise each frame follows
    ``x[t] = rho * x[t-1] + sqrt(1-rho**2) * epsilon[t]``, starting in its
    stationary distribution. Optional Gaussian smoothing is periodic along both
    spatial axes and normalized to preserve unit marginal variance. Optional
    channel correlation uses a shared Gaussian component, giving every distinct
    channel pair the same positive correlation.

    Generation uses a private CPU random-number generator. The output limit
    applies to one returned data tensor, or one chunk when iterating, not total
    process memory. The eager CLI retains all splits and allocates additional
    tensors for statistics.

    Args:
        height: Number of points along the first spatial axis.
        width: Number of points along the second spatial axis.
        n_channels: Number of scalar fields.
        n_steps: Number of saved frames per trajectory, including the initial one.
        rho: Temporal correlation between successive frames; strictly within (-1, 1).
        spatial_length_scale: Gaussian smoothing width in grid cells. Zero gives
            independent spatial positions; both axes have periodic boundaries.
        channel_correlation: Correlation of distinct channels, in [0, 1).
        dtype: Stored field precision, either ``float32`` or ``float64``.
        max_output_bytes: Maximum bytes for one returned batch or chunk tensor.
        output_names: Optional unique channel labels; defaults to ``field_0``, etc.
        log_level: Logging verbosity inherited from the simulator interface.
    """

    def __init__(
        self,
        height: int = 16,
        width: int = 32,
        n_channels: int = 4,
        n_steps: int = 4,
        rho: float = 0.0,
        spatial_length_scale: float = 0.0,
        channel_correlation: float = 0.0,
        dtype: str = "float32",
        max_output_bytes: int = 512 * 1024**2,
        output_names: list[str] | None = None,
        log_level: str = "warning",
    ) -> None:
        """Configure the field shape, correlations and per-output memory limit."""
        for name, value in (
            ("height", height),
            ("width", width),
            ("n_channels", n_channels),
            ("n_steps", n_steps),
            ("max_output_bytes", max_output_bytes),
        ):
            self._check_positive_int(name, value)
        if isinstance(rho, bool) or not math.isfinite(rho) or not -1 < rho < 1:
            msg = "rho must be finite and strictly between -1 and 1."
            raise ValueError(msg)
        if (
            isinstance(spatial_length_scale, bool)
            or not math.isfinite(spatial_length_scale)
            or spatial_length_scale < 0
        ):
            msg = "spatial_length_scale must be finite and non-negative."
            raise ValueError(msg)
        if (
            isinstance(channel_correlation, bool)
            or not math.isfinite(channel_correlation)
            or not 0 <= channel_correlation < 1
        ):
            msg = "channel_correlation must be finite and in [0, 1)."
            raise ValueError(msg)
        if dtype not in ("float32", "float64"):
            msg = "dtype must be 'float32' or 'float64'."
            raise ValueError(msg)
        names = (
            [f"field_{idx}" for idx in range(n_channels)]
            if output_names is None
            else list(output_names)
        )
        if (
            len(names) != n_channels
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            msg = "output_names must contain one unique, non-empty name per channel."
            raise ValueError(msg)

        super().__init__({}, names, log_level)
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.n_steps = n_steps
        self.rho = float(rho)
        self.spatial_length_scale = float(spatial_length_scale)
        self.channel_correlation = float(channel_correlation)
        self.dtype = getattr(torch, dtype)
        self.max_output_bytes = max_output_bytes

    def _smoothing_filter(self) -> torch.Tensor | None:
        """Construct a periodic Gaussian filter with unit output marginal variance."""
        if self.spatial_length_scale == 0:
            return None
        frequencies = [
            torch.fft.fftfreq(size, dtype=torch.float64, device="cpu")
            for size in (self.height, self.width)
        ]
        # Multiply in float64 before squaring, keeping the zero mode finite
        # even when a very large width sends the other exponents to infinity.
        filters = []
        for frequency in frequencies:
            exponent = (frequency * self.spatial_length_scale).square()
            filters.append(torch.exp(-2 * math.pi**2 * exponent).to(self.dtype))
        height_filter, width_filter = filters
        rms_gain = (height_filter.square().mean() * width_filter.square().mean()).sqrt()
        return (
            height_filter[:, None] * width_filter[None, : self.width // 2 + 1]
        ) / rms_gain

    def _draw_frame(
        self, generator: torch.Generator, smoothing_filter: torch.Tensor | None
    ) -> torch.Tensor:
        """Draw one stationary spatial/channel field without a trajectory-sized copy."""
        correlated = self.channel_correlation > 0 and self.n_channels > 1
        noise = torch.randn(
            (self.height, self.width, self.n_channels + int(correlated)),
            dtype=self.dtype,
            device="cpu",
            generator=generator,
        )
        if smoothing_filter is not None:
            spectrum = torch.fft.rfft2(noise, dim=(0, 1))
            spectrum.mul_(smoothing_filter[..., None])
            noise = torch.fft.irfft2(spectrum, s=(self.height, self.width), dim=(0, 1))
        if correlated:
            independent, shared = noise[..., :-1], noise[..., -1:]
            independent.mul_(math.sqrt(1 - self.channel_correlation))
            independent.add_(shared, alpha=math.sqrt(self.channel_correlation))
            return independent
        return noise

    @staticmethod
    def _check_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer."
            raise ValueError(msg)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Support the base forward API with no physical input parameters."""
        if x.ndim != 2 or x.shape[1] != 0:
            msg = "GaussianField has no input parameters; expected shape (n, 0)."
            raise ValueError(msg)
        # The base API seeds the global CPU RNG for each independent sample.
        seed = int(torch.randint(2**63 - 1, (), device="cpu"))
        payload = self.forward_samples_spatiotemporal(n=x.shape[0], random_seed=seed)
        return rearrange(payload["data"], "b t h w c -> b (t h w c)")

    def _check_output_size(self, shape: tuple[int, ...]) -> None:
        item_size = torch.empty((), dtype=self.dtype, device="cpu").element_size()
        output_bytes = math.prod(shape) * item_size
        if output_bytes > self.max_output_bytes:
            msg = (
                f"Requested data tensor needs {output_bytes:,} bytes, exceeding "
                f"max_output_bytes={self.max_output_bytes:,}. Reduce trajectories "
                "or frames per output, or explicitly increase the limit. Other "
                "splits and normalization statistics need additional memory."
            )
            raise ValueError(msg)

    @staticmethod
    def _make_generator(random_seed: int | None) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        if random_seed is None:
            generator.seed()
        else:
            generator.manual_seed(random_seed)
        return generator

    def _trajectory_frames(
        self, generator: torch.Generator, smoothing_filter: torch.Tensor | None
    ) -> Iterator[torch.Tensor]:
        """Keep one previous frame for a continuous stationary trajectory."""
        previous = None
        innovation_scale = math.sqrt(1 - self.rho**2)
        for _ in range(self.n_steps):
            frame = self._draw_frame(generator, smoothing_filter)
            if previous is not None and self.rho != 0:
                frame.mul_(innovation_scale).add_(previous, alpha=self.rho)
            yield frame
            previous = frame

    def _payload(self, data: torch.Tensor, seed: int) -> dict:
        return {
            "data": data,
            "constant_scalars": None,
            "constant_fields": None,
            "provenance": {
                "kind": "synthetic",
                "generator": "GaussianField",
                "random_seed": seed,
                "rho": self.rho,
                "spatial_length_scale": self.spatial_length_scale,
                "spatial_boundaries": (
                    "periodic" if self.spatial_length_scale > 0 else None
                ),
                "channel_correlation": self.channel_correlation,
            },
        }

    def iter_samples_spatiotemporal(
        self,
        n: int,
        chunk_steps: int,
        random_seed: int | None = None,
    ) -> Iterator[dict]:
        """Yield trajectory chunks while preserving the eager sampling sequence.

        Args:
            n: Positive trajectory count.
            chunk_steps: Maximum frames in each returned chunk. The final chunk
                of each trajectory may be shorter. The output byte limit applies
                to each chunk, independently of the full dataset size.
            random_seed: Seed for all trajectories, with the same interpretation
                as in ``forward_samples_spatiotemporal``.

        Yields:
            Standard payloads with data shaped ``[1, time, height, width,
            channels]``, plus zero-based ``trajectory_index`` and ``start_step``.
            Chunks are in trajectory/time order and are not reused or mutated.
            Random draws and temporal correlations continue across chunk
            boundaries; trajectories start independently. Discard consumed chunks
            to keep memory bounded by the chunk and per-frame working buffers.
        """
        self._check_positive_int("n", n)
        self._check_positive_int("chunk_steps", chunk_steps)
        shape = (
            1,
            min(chunk_steps, self.n_steps),
            self.height,
            self.width,
            self.n_channels,
        )
        self._check_output_size(shape)
        generator = self._make_generator(random_seed)
        smoothing_filter = self._smoothing_filter()
        seed = generator.initial_seed()
        for trajectory in range(n):
            frames = self._trajectory_frames(generator, smoothing_filter)
            for start in range(0, self.n_steps, chunk_steps):
                length = min(chunk_steps, self.n_steps - start)
                data = torch.empty(
                    (1, length, self.height, self.width, self.n_channels),
                    dtype=self.dtype,
                    device="cpu",
                )
                for offset in range(length):
                    data[0, offset].copy_(next(frames))
                yield {
                    **self._payload(data, seed),
                    "trajectory_index": trajectory,
                    "start_step": start,
                }

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Return exactly ``n`` trajectories in the standard AutoSim format.

        Args:
            n: Positive trajectory count.
            random_seed: Seed for this batch. When omitted, a fresh seed is drawn
                from system entropy and recorded in the returned provenance.
            ensure_exact_n: Accepted for interface compatibility; generation
                always returns exactly ``n`` trajectories or raises an error.

        Returns:
            Data shaped ``[n, time, height, width, channels]``, empty physical
            constants and explicit synthetic-data provenance. Padding belongs
            to the consuming model, so the requested spatial shape is preserved.
        """
        del ensure_exact_n
        self._check_positive_int("n", n)
        shape = (n, self.n_steps, self.height, self.width, self.n_channels)
        self._check_output_size(shape)
        generator = self._make_generator(random_seed)
        smoothing_filter = self._smoothing_filter()
        data = torch.empty(shape, dtype=self.dtype, device="cpu")
        for trajectory in range(n):
            for step, frame in enumerate(
                self._trajectory_frames(generator, smoothing_filter)
            ):
                data[trajectory, step].copy_(frame)
        return self._payload(data, generator.initial_seed())
