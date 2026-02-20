import random

import numpy as np
import torch

from autosim.types import (
    OutputLike,
    TensorLike,
    TorchScalarDType,
)


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
