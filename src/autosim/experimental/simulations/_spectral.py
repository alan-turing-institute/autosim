from __future__ import annotations

import math

import torch


def spectral_wavenumbers(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Laplacian and real-derivative wavenumbers for ``rfft2`` fields."""
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=Lx / nx, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.rfftfreq(ny, d=Ly / ny, dtype=dtype)
    Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")

    # A first derivative of an even-grid Nyquist mode has no real-valued
    # collocation-grid representation. Zero it, while retaining the mode in
    # K² for Laplacians and dissipation.
    dKx = Kx.clone()
    dKy = Ky.clone()
    if nx % 2 == 0:
        dKx[nx // 2, :] = 0.0
    if ny % 2 == 0:
        dKy[:, -1] = 0.0
    return Kx, Ky, dKx, dKy


def two_thirds_mask(nx: int, ny: int) -> torch.Tensor:
    """Return the standard rectangular two-thirds dealiasing mask."""
    mode_x = torch.fft.fftfreq(nx, d=1.0 / nx)
    mode_y = torch.fft.rfftfreq(ny, d=1.0 / ny)
    mx, my = torch.meshgrid(mode_x, mode_y, indexing="ij")
    return (mx.abs() < nx / 3.0) & (my < ny / 3.0)


def gaussian_ring_spectrum(
    K2: torch.Tensor,
    forcing_wavenumber: float,
    forcing_bandwidth: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Construct a zero-mean Gaussian ring spectrum on retained modes."""
    if forcing_wavenumber <= 0:
        msg = "forcing_wavenumber must be positive"
        raise ValueError(msg)
    if forcing_bandwidth <= 0:
        msg = "forcing_bandwidth must be positive"
        raise ValueError(msg)

    K = torch.sqrt(K2)
    spectrum = torch.exp(
        -((K - forcing_wavenumber) ** 2) / (2.0 * forcing_bandwidth**2)
    )
    spectrum = torch.where(mask, spectrum, torch.zeros_like(spectrum))
    spectrum[0, 0] = 0.0

    forcing_band = (
        mask & (K2 > 0) & ((K - forcing_wavenumber).abs() <= 3.0 * forcing_bandwidth)
    )
    if not bool(forcing_band.any()):
        msg = (
            "forcing ring does not intersect any retained Fourier modes; "
            "reduce forcing_wavenumber or increase resolution/bandwidth"
        )
        raise ValueError(msg)
    return spectrum


def sample_filtered_scalar_hat(
    *,
    nx: int,
    ny: int,
    dtype: torch.dtype,
    spectrum: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Draw a real-compatible, zero-mean scalar field in a target spectrum."""
    noise = torch.randn((nx, ny), dtype=dtype)
    increment_hat = torch.fft.rfft2(noise) * torch.sqrt(spectrum)
    increment_hat = increment_hat * mask
    increment_hat[0, 0] = 0.0
    return increment_hat


def expected_filtered_variance(
    *,
    nx: int,
    ny: int,
    spectrum: torch.Tensor,
    transfer_power: torch.Tensor | None = None,
) -> float:
    """Return the expected mean square of a filtered unit Gaussian field.

    ``transfer_power`` is the squared magnitude of an optional spectral
    transfer function. The half-spectrum weights account for the omitted
    negative-y frequencies in ``rfft2``.
    """
    weights = torch.full(
        (ny // 2 + 1,),
        2.0,
        dtype=spectrum.dtype,
        device=spectrum.device,
    )
    weights[0] = 1.0
    if ny % 2 == 0:
        weights[-1] = 1.0

    power = spectrum if transfer_power is None else spectrum * transfer_power
    return float((power * weights.unsqueeze(0)).sum() / (nx * ny))
