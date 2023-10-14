from torch import Tensor
from typing import Union
from typing import Tuple
from numpy.typing import ArrayLike
from numpy import ndarray
import numpy
import torch
from _typeshed import Incomplete

from torch import Tensor
import numpy.typing
from numbers import Number

Numeric = Union[Number, ArrayLike, Tensor]
__docstubs__: str


class _ImplRegistry:
    registered: Incomplete

    def __init__(self) -> None:
        ...


class TorchImpls:
    is_tensor: bool
    is_numpy: bool

    @staticmethod
    def result_type(*arrays_and_dtypes):
        ...

    @staticmethod
    def cat(datas, axis: int = ...):
        ...

    @staticmethod
    def hstack(datas):
        ...

    @staticmethod
    def vstack(datas):
        ...

    @staticmethod
    def atleast_nd(arr, n, front: bool = ...):
        ...

    @staticmethod
    def view(data, *shape):
        ...

    @staticmethod
    def take(data, indices, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def compress(data, flags, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def tile(data, reps):
        ...

    @staticmethod
    def repeat(data, repeats, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def T(data):
        ...

    @staticmethod
    def transpose(data, axes):
        ...

    @staticmethod
    def numel(data):
        ...

    @staticmethod
    def full_like(data, fill_value, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def empty_like(data, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def zeros_like(data, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def ones_like(data, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def full(shape, fill_value, dtype=...):
        ...

    @staticmethod
    def empty(shape, dtype=...):
        ...

    @staticmethod
    def zeros(shape, dtype=...):
        ...

    @staticmethod
    def ones(shape, dtype=...):
        ...

    @staticmethod
    def argmax(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def argmin(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def argsort(data, axis: int = ..., descending: bool = ...):
        ...

    @staticmethod
    def max(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def min(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def max_argmax(data: Tensor,
                   axis: None | int = None) -> Tuple[Numeric, Numeric]:
        ...

    @staticmethod
    def min_argmin(data: Tensor,
                   axis: None | int = None) -> Tuple[Numeric, Numeric]:
        ...

    @staticmethod
    def maximum(data1, data2, out: Incomplete | None = ...):
        ...

    @staticmethod
    def minimum(data1, data2, out: Incomplete | None = ...):
        ...

    @staticmethod
    def array_equal(data1, data2, equal_nan: bool = ...) -> bool:
        ...

    @staticmethod
    def matmul(data1, data2, out: Incomplete | None = ...):
        ...

    @staticmethod
    def sum(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def nan_to_num(x, copy: bool = ...):
        ...

    @staticmethod
    def copy(data):
        ...

    @staticmethod
    def log(data):
        ...

    @staticmethod
    def log2(data):
        ...

    @staticmethod
    def any(data):
        ...

    @staticmethod
    def all(data):
        ...

    @staticmethod
    def nonzero(data):
        ...

    @staticmethod
    def astype(data, dtype, copy: bool = ...):
        ...

    @staticmethod
    def tensor(data, device=...):
        ...

    @staticmethod
    def numpy(data):
        ...

    @staticmethod
    def tolist(data):
        ...

    @staticmethod
    def contiguous(data):
        ...

    @staticmethod
    def pad(data, pad_width, mode: str = ...):
        ...

    @staticmethod
    def asarray(data, dtype: Incomplete | None = ...):
        ...

    ensure = asarray

    @staticmethod
    def dtype_kind(data):
        ...

    @staticmethod
    def floor(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def ceil(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def ifloor(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def iceil(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def round(data, decimals: int = ..., out: Incomplete | None = ...):
        ...

    @staticmethod
    def iround(data, out: Incomplete | None = ..., dtype=...):
        ...

    @staticmethod
    def clip(data,
             a_min: Incomplete | None = ...,
             a_max: Incomplete | None = ...,
             out: Incomplete | None = ...):
        ...

    @staticmethod
    def softmax(data, axis: Incomplete | None = ...):
        ...


class NumpyImpls:
    is_tensor: bool
    is_numpy: bool
    hstack: Incomplete
    vstack: Incomplete

    @staticmethod
    def result_type(*arrays_and_dtypes):
        ...

    @staticmethod
    def cat(datas, axis: int = ...):
        ...

    @staticmethod
    def atleast_nd(arr, n, front: bool = ...):
        ...

    @staticmethod
    def view(data, *shape):
        ...

    @staticmethod
    def take(data, indices, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def compress(data, flags, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def repeat(data, repeats, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def tile(data, reps):
        ...

    @staticmethod
    def T(data):
        ...

    @staticmethod
    def transpose(data, axes):
        ...

    @staticmethod
    def numel(data):
        ...

    @staticmethod
    def empty_like(data, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def full_like(data, fill_value, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def zeros_like(data, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def ones_like(data, dtype: Incomplete | None = ...):
        ...

    @staticmethod
    def full(shape, fill_value, dtype=...):
        ...

    @staticmethod
    def empty(shape, dtype=...):
        ...

    @staticmethod
    def zeros(shape, dtype=...):
        ...

    @staticmethod
    def ones(shape, dtype=...):
        ...

    @staticmethod
    def argmax(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def argmin(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def argsort(data, axis: int = ..., descending: bool = ...):
        ...

    @staticmethod
    def max(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def min(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def max_argmax(data: ArrayLike,
                   axis: None | int = None) -> Tuple[Numeric, Numeric]:
        ...

    @staticmethod
    def min_argmin(data: ArrayLike,
                   axis: None | int = None) -> Tuple[Numeric, Numeric]:
        ...

    @staticmethod
    def sum(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def maximum(data1, data2, out: Incomplete | None = ...):
        ...

    @staticmethod
    def minimum(data1, data2, out: Incomplete | None = ...):
        ...

    matmul: Incomplete
    nan_to_num: Incomplete
    array_equal: Incomplete
    log: Incomplete
    log2: Incomplete
    any: Incomplete
    all: Incomplete
    copy: Incomplete
    nonzero: Incomplete

    @staticmethod
    def astype(data, dtype, copy: bool = ...):
        ...

    @staticmethod
    def tensor(data, device=...):
        ...

    @staticmethod
    def numpy(data):
        ...

    @staticmethod
    def tolist(data):
        ...

    @staticmethod
    def contiguous(data):
        ...

    @staticmethod
    def pad(data, pad_width, mode: str = ...):
        ...

    @staticmethod
    def asarray(data, dtype: Incomplete | None = ...):
        ...

    ensure = asarray

    @staticmethod
    def dtype_kind(data):
        ...

    @staticmethod
    def floor(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def ceil(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def ifloor(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def iceil(data, out: Incomplete | None = ...):
        ...

    @staticmethod
    def round(data, decimals: int = ..., out: Incomplete | None = ...):
        ...

    @staticmethod
    def iround(data, out: Incomplete | None = ..., dtype=...):
        ...

    clip: Incomplete

    @staticmethod
    def softmax(data, axis: Incomplete | None = ...):
        ...

    @staticmethod
    def kron(a: ndarray, b: ndarray) -> ndarray:
        ...


class ArrayAPI:

    @staticmethod
    def impl(data: ndarray | Tensor):
        ...

    @staticmethod
    def coerce(data):
        ...

    @staticmethod
    def result_type(*arrays_and_dtypes):
        ...

    @staticmethod
    def cat(datas, *args, **kwargs):
        ...

    @staticmethod
    def hstack(datas, *args, **kwargs):
        ...

    @staticmethod
    def vstack(datas, *args, **kwargs):
        ...

    take: Incomplete
    compress: Incomplete
    repeat: Incomplete
    tile: Incomplete
    view: Incomplete
    numel: Incomplete
    atleast_nd: Incomplete
    full_like: Incomplete
    ones_like: Incomplete
    zeros_like: Incomplete
    empty_like: Incomplete
    sum: Incomplete
    argsort: Incomplete
    argmax: Incomplete
    argmin: Incomplete
    max: Incomplete
    min: Incomplete
    max_argmax: Incomplete
    min_argmin: Incomplete
    maximum: Incomplete
    minimum: Incomplete
    matmul: Incomplete
    astype: Incomplete
    nonzero: Incomplete
    nan_to_num: Incomplete
    tensor: Incomplete
    numpy: Incomplete
    tolist: Incomplete
    asarray: Incomplete
    T: Incomplete
    transpose: Incomplete
    contiguous: Incomplete
    pad: Incomplete
    dtype_kind: Incomplete
    any: Incomplete
    all: Incomplete
    array_equal: Incomplete
    log2: Incomplete
    log: Incomplete
    copy: Incomplete
    iceil: Incomplete
    ifloor: Incomplete
    floor: Incomplete
    ceil: Incomplete
    round: Incomplete
    iround: Incomplete
    clip: Incomplete
    softmax: Incomplete


TorchNumpyCompat = ArrayAPI


def dtype_info(
        dtype: type) -> numpy.iinfo | numpy.finfo | torch.iinfo | torch.finfo:
    ...
