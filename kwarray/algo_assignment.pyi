import numpy as np
from numpy import ndarray
from typing import Tuple
import numpy as np


def mindist_assignment(vecs1: np.ndarray,
                       vecs2: np.ndarray,
                       p: float = 2) -> Tuple[list, float]:
    ...


def mincost_assignment(cost: ndarray) -> Tuple[list, float]:
    ...


def maxvalue_assignment(value: ndarray) -> Tuple[list, float]:
    ...
