from typing import Tuple
import torch
from numpy import ndarray
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator


class SlidingWindow(ub.NiceRepr):
    stride: Incomplete
    overlap: Incomplete
    window: Incomplete
    input_shape: Incomplete
    undershot_shape: Incomplete
    basis_slices: Incomplete
    basis_shape: Incomplete
    n_total: Incomplete

    def __init__(self,
                 shape,
                 window,
                 overlap: Incomplete | None = ...,
                 stride: Incomplete | None = ...,
                 keepbound: bool = ...,
                 allow_overshoot: bool = ...) -> None:
        ...

    def __nice__(self):
        ...

    def __len__(self):
        ...

    def __iter__(self):
        ...

    def __getitem__(self, index):
        ...

    @property
    def grid(self) -> Generator[Tuple[int, ...], None, None]:
        ...

    @property
    def slices(self):
        ...

    @property
    def centers(self) -> Generator[Tuple[float, ...], None, None]:
        ...


__devnote__: str


class Stitcher(ub.NiceRepr):
    nan_policy: str
    shape: tuple
    device: str | int | torch.device
    sums: Incomplete
    weights: Incomplete
    sumview: Incomplete
    weightview: Incomplete

    def __init__(self,
                 shape: tuple,
                 device: str | int | torch.device = 'numpy',
                 dtype: str = 'float32',
                 nan_policy: str = 'propogate') -> None:
        ...

    def __nice__(self):
        ...

    def add(self,
            indices: slice | tuple | None,
            patch: ndarray,
            weight: float | ndarray | None = None) -> None:
        ...

    def __getitem__(self, indices):
        ...

    def average(self) -> ndarray:
        ...

    def finalize(self, indices: None | slice | tuple = None) -> ndarray:
        ...
