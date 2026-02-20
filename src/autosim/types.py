from typing import TypeAlias

import numpy as np
import torch
import torch.utils
import torch.utils.data

NumpyLike: TypeAlias = np.ndarray
TensorLike: TypeAlias = torch.Tensor
DeviceLike: TypeAlias = str | torch.device
