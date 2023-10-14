from numpy import ndarray
from typing import Dict
from typing import Tuple
from typing import Any
from nptyping import NDArray
from _typeshed import Incomplete


def find_robust_normalizers(data: ndarray,
                            params: str | dict = 'auto'
                            ) -> Dict[str, str | float]:
    ...


def robust_normalize(
        imdata: ndarray,
        return_info: bool = False,
        nodata: None | int = None,
        axis: None | int = None,
        dtype: type = ...,
        params: str | dict = 'auto',
        mask: ndarray | None = None) -> ndarray | Tuple[ndarray, Any]:
    ...


def normalize(arr: NDArray,
              mode: str = 'linear',
              alpha: float | None = None,
              beta: float | None = None,
              out: NDArray | None = None,
              min_val: Incomplete | None = ...,
              max_val: Incomplete | None = ...):
    ...
