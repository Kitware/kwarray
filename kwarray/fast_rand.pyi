from typing import Tuple
import numpy
from numpy import ndarray
from nptyping import Float32
from typing import Any
from nptyping import Float64


def uniform(low: float = 0.0,
            high: float = 1.0,
            size: int | Tuple[int, ...] | None = None,
            dtype: type = ...,
            rng: numpy.random.RandomState = ...) -> ndarray:
    ...


def standard_normal(size: int | Tuple[int, ...],
                    mean: float = 0,
                    std: float = 1,
                    dtype: type = float,
                    rng: numpy.random.RandomState = ...) -> ndarray:
    ...


def standard_normal32(
        size: int | Tuple[int, ...],
        mean: float = 0,
        std: float = 1,
        rng: numpy.random.RandomState = ...) -> ndarray[Any, Float32]:
    ...


def standard_normal64(
        size: int | Tuple[int, ...],
        mean: float = 0,
        std: float = 1,
        rng: numpy.random.RandomState = ...) -> ndarray[Any, Float64]:
    ...


def uniform32(low: float = 0.0,
              high: float = 1.0,
              size: int | Tuple[int, ...] | None = None,
              rng=...) -> ndarray[Any, Float32]:
    ...
