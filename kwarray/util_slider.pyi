from typing import Tuple
from typing import Union
from numpy import ndarray
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


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
    def grid(self) -> Generator[Any, None, None]:
        ...

    @property
    def slices(self):
        ...

    @property
    def centers(self) -> Generator[Tuple[float, ...], None, None]:
        ...


class Stitcher(ub.NiceRepr):

    def __init__(stitcher, shape, device: str = ...) -> None:
        ...

    def __nice__(stitcher):
        ...

    def add(stitcher,
            indices: Union[slice, tuple],
            patch: ndarray,
            weight: Union[float, ndarray] = None) -> None:
        ...

    def average(stitcher) -> ndarray:
        ...

    def finalize(stitcher,
                 indices: Union[None, slice, tuple] = None) -> ndarray:
        ...
