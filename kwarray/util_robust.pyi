from numpy import ndarray
from typing import Union
from typing import Dict
from _typeshed import Incomplete


def find_robust_normalizers(
        data: ndarray,
        params: Union[str, dict] = 'auto') -> Dict[str, str | float]:
    ...


def robust_normalize(imdata: ndarray,
                     return_info: bool = False,
                     nodata: Incomplete | None = ...,
                     axis: Incomplete | None = ...,
                     dtype=...,
                     params: Union[str, dict] = 'auto') -> ndarray:
    ...
