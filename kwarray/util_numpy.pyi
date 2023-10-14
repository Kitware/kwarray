from nptyping import NDArray
from typing import Any
from nptyping import Int
from typing import Callable
from typing import Iterator
from typing import Iterable
from numpy.typing import ArrayLike
from _typeshed import Incomplete


def boolmask(indices: NDArray,
             shape: int | tuple | None = None) -> NDArray[Any, Int]:
    ...


def iter_reduce_ufunc(ufunc: Callable,
                      arrs: Iterator[NDArray],
                      out: Incomplete | None = ...,
                      default: object | None = None) -> NDArray:
    ...


def isect_flags(arr: NDArray, other: Iterable) -> NDArray:
    ...


def atleast_nd(arr: ArrayLike, n: int, front: bool = False) -> NDArray:
    ...


def argmaxima(arr: NDArray,
              num: int,
              axis: int | None = None,
              ordered: bool = True) -> NDArray:
    ...


def argminima(arr: NDArray,
              num: int,
              axis: int | None = None,
              ordered: bool = True):
    ...


def unique_rows(arr: NDArray, ordered: bool = False, return_index: bool = ...):
    ...


def arglexmax(keys: tuple, multi: bool = False) -> int | NDArray[Any, Int]:
    ...


def generalized_logistic(x: NDArray,
                         floor: float = 0,
                         capacity: float = 1,
                         C: float = 1,
                         y_intercept: float | None = None,
                         Q: float | None = None,
                         growth: float = 1,
                         v: float = 1) -> NDArray:
    ...


def equal_with_nan(a1: ArrayLike, a2: ArrayLike):
    ...
