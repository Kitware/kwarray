from nptyping import NDArray
from typing import Union
from typing import Dict
from typing import Any
from typing import List
from typing import Tuple
from nptyping import Int

__TODO__: str


def group_items(item_list: NDArray,
                groupid_list: NDArray,
                assume_sorted: bool = False,
                axis: Union[int, None] = None) -> Dict[Any, NDArray]:
    ...


def group_indices(
        idx_to_groupid: NDArray,
        assume_sorted: bool = False) -> Tuple[NDArray, List[NDArray]]:
    ...


def apply_grouping(items: NDArray,
                   groupxs: List[NDArray[None, Int]],
                   axis: int = ...) -> List[NDArray]:
    ...


def group_consecutive(arr: NDArray, offset: float = 1) -> List[NDArray]:
    ...


def group_consecutive_indices(arr: NDArray,
                              offset: float = 1) -> List[NDArray]:
    ...
