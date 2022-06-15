from torch import Tensor
from numpy.typing import ArrayLike


def one_hot_embedding(labels, num_classes, dim: int = 1) -> Tensor:
    ...


def one_hot_lookup(data: ArrayLike, indices: ArrayLike) -> ArrayLike:
    ...
