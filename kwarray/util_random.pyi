from typing import Union
from numpy import ndarray
import random
import numpy
from typing import List
from typing import Tuple
from typing import Sequence
import random
from collections.abc import Generator


def seed_global(seed: int, offset: int = 0) -> None:
    ...


def shuffle(
    items: Union[list, ndarray],
    rng: Union[int, float, None, numpy.random.RandomState,
               random.Random] = None
) -> list:
    ...


def random_combinations(
    items: List,
    size: int,
    num: Union[int, None] = None,
    rng: Union[int, float, None, numpy.random.RandomState,
               random.Random] = None
) -> Generator[Tuple, None, None]:
    ...


def random_product(
    items: List[Sequence],
    num: Union[int, None] = None,
    rng: Union[int, float, None, numpy.random.RandomState,
               random.Random] = None
) -> Generator[Tuple, None, None]:
    ...


def ensure_rng(
        rng: Union[int, float, None, numpy.random.RandomState, random.Random],
        api: str = 'numpy') -> (numpy.random.RandomState | random.Random):
    ...
