from typing import Union
from numpy import ndarray
from torch import Tensor
import numpy
import torch
from _typeshed import Incomplete


class _ImplRegistry:
    registered: Incomplete

    def __init__(self) -> None:
        ...


class TorchImpls:
    is_tensor: bool
    is_numpy: bool

    def result_type(*arrays_and_dtypes):
        ...

    def cat(datas, axis: int = ...):
        ...

    def hstack(datas):
        ...

    def vstack(datas):
        ...

    def atleast_nd(arr, n, front: bool = ...):
        ...

    def view(data, *shape):
        ...

    def take(data, indices, axis: Incomplete | None = ...):
        ...

    def compress(data, flags, axis: Incomplete | None = ...):
        ...

    def tile(data, reps):
        ...

    def repeat(data, repeats, axis: Incomplete | None = ...):
        ...

    def T(data):
        ...

    def transpose(data, axes):
        ...

    def numel(data):
        ...

    def full_like(data, fill_value, dtype: Incomplete | None = ...):
        ...

    def empty_like(data, dtype: Incomplete | None = ...):
        ...

    def zeros_like(data, dtype: Incomplete | None = ...):
        ...

    def ones_like(data, dtype: Incomplete | None = ...):
        ...

    def full(shape, fill_value, dtype=...):
        ...

    def empty(shape, dtype=...):
        ...

    def zeros(shape, dtype=...):
        ...

    def ones(shape, dtype=...):
        ...

    def argmax(data, axis: Incomplete | None = ...):
        ...

    def argsort(data, axis: int = ..., descending: bool = ...):
        ...

    def max(data, axis: Incomplete | None = ...):
        ...

    def max_argmax(data, axis: Incomplete | None = ...):
        ...

    def maximum(data1, data2, out: Incomplete | None = ...):
        ...

    def minimum(data1, data2, out: Incomplete | None = ...):
        ...

    matmul: Incomplete

    def sum(data, axis: Incomplete | None = ...):
        ...

    def nan_to_num(x, copy: bool = ...):
        ...

    def copy(data):
        ...

    log: Incomplete
    log2: Incomplete
    any: Incomplete
    all: Incomplete

    def nonzero(data):
        ...

    def astype(data, dtype, copy: bool = ...):
        ...

    def tensor(data, device=...):
        ...

    def numpy(data):
        ...

    def tolist(data):
        ...

    def contiguous(data):
        ...

    def pad(data, pad_width, mode: str = ...):
        ...

    def asarray(data, dtype: Incomplete | None = ...):
        ...

    ensure: Incomplete

    def dtype_kind(data):
        ...

    def floor(data, out: Incomplete | None = ...):
        ...

    def ceil(data, out: Incomplete | None = ...):
        ...

    def ifloor(data, out: Incomplete | None = ...):
        ...

    def iceil(data, out: Incomplete | None = ...):
        ...

    def round(data, decimals: int = ..., out: Incomplete | None = ...):
        ...

    def iround(data, out: Incomplete | None = ..., dtype=...):
        ...

    def clip(data,
             a_min: Incomplete | None = ...,
             a_max: Incomplete | None = ...,
             out: Incomplete | None = ...):
        ...

    def softmax(data, axis: Incomplete | None = ...):
        ...


class NumpyImpls:
    is_tensor: bool
    is_numpy: bool
    hstack: Incomplete
    vstack: Incomplete

    def result_type(*arrays_and_dtypes):
        ...

    def cat(datas, axis: int = ...):
        ...

    def atleast_nd(arr, n, front: bool = ...):
        ...

    def view(data, *shape):
        ...

    def take(data, indices, axis: Incomplete | None = ...):
        ...

    def compress(data, flags, axis: Incomplete | None = ...):
        ...

    def repeat(data, repeats, axis: Incomplete | None = ...):
        ...

    def tile(data, reps):
        ...

    def T(data):
        ...

    def transpose(data, axes):
        ...

    def numel(data):
        ...

    def empty_like(data, dtype: Incomplete | None = ...):
        ...

    def full_like(data, fill_value, dtype: Incomplete | None = ...):
        ...

    def zeros_like(data, dtype: Incomplete | None = ...):
        ...

    def ones_like(data, dtype: Incomplete | None = ...):
        ...

    def full(shape, fill_value, dtype=...):
        ...

    def empty(shape, dtype=...):
        ...

    def zeros(shape, dtype=...):
        ...

    def ones(shape, dtype=...):
        ...

    def argmax(data, axis: Incomplete | None = ...):
        ...

    def argsort(data, axis: int = ..., descending: bool = ...):
        ...

    def max(data, axis: Incomplete | None = ...):
        ...

    def max_argmax(data, axis: Incomplete | None = ...):
        ...

    def sum(data, axis: Incomplete | None = ...):
        ...

    def maximum(data1, data2, out: Incomplete | None = ...):
        ...

    def minimum(data1, data2, out: Incomplete | None = ...):
        ...

    matmul: Incomplete
    nan_to_num: Incomplete
    log: Incomplete
    log2: Incomplete
    any: Incomplete
    all: Incomplete
    copy: Incomplete
    nonzero: Incomplete

    def astype(data, dtype, copy: bool = ...):
        ...

    def tensor(data, device=...):
        ...

    def numpy(data):
        ...

    def tolist(data):
        ...

    def contiguous(data):
        ...

    def pad(data, pad_width, mode: str = ...):
        ...

    def asarray(data, dtype: Incomplete | None = ...):
        ...

    ensure: Incomplete

    def dtype_kind(data):
        ...

    def floor(data, out: Incomplete | None = ...):
        ...

    def ceil(data, out: Incomplete | None = ...):
        ...

    def ifloor(data, out: Incomplete | None = ...):
        ...

    def iceil(data, out: Incomplete | None = ...):
        ...

    def round(data, decimals: int = ..., out: Incomplete | None = ...):
        ...

    def iround(data, out: Incomplete | None = ..., dtype=...):
        ...

    clip: Incomplete

    def softmax(data, axis: Incomplete | None = ...):
        ...


class ArrayAPI:

    @staticmethod
    def impl(data: Union[ndarray, Tensor]):
        ...

    @staticmethod
    def coerce(data):
        ...

    def result_type(*arrays_and_dtypes):
        ...

    def cat(datas, *args, **kwargs):
        ...

    def hstack(datas, *args, **kwargs):
        ...

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
    argmax: Incomplete
    argsort: Incomplete
    max: Incomplete
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
    max_argmax: Incomplete
    any: Incomplete
    all: Incomplete
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
