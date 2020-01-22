# -*- coding: utf-8 -*-
"""
Functions for partitioning numpy arrays into groups.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import ubelt as ub


def group_items(item_list, groupid_list, assume_sorted=False, axis=None):
    """
    Groups a list of items by group id.

    Works like :func:`ubelt.group_items`, but with numpy optimizations.
    This can be quite a bit faster than using :func:`itertools.groupby` [1]_
    [2]_.

    In cases where there are many lists of items to group (think column-major
    data), consider using :func:`group_indices` and :func:`apply_grouping`
    instead.

    Args:
        item_list (ndarray[T1]):
            The input array of items to group.

        groupid_list (ndarray[T2]):
            Each item is an id corresponding to the item at the same position
            in ``item_list``.  For the fastest runtime, the input array must be
            numeric (ideally with integer types). This list must be
            1-dimensional.

        assume_sorted (bool, default=False):
            If the input array is sorted, then setting this to True will avoid
            an unnecessary sorting operation and improve efficiency.

        axis (int | None):
            group along a particular axis in ``items`` if it is n-dimensional

    Returns:
        Dict[T2, ndarray[T1]]: mapping from groupids to corresponding items

    References:
        .. [1] http://stackoverflow.com/questions/4651683/
        .. [2] numpy-grouping-using-itertools-groupby-performance

    Example:
        >>> from kwarray.util_groups import *  # NOQA
        >>> items = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        >>> keys = np.array( [2, 2, 1, 1, 0, 1, 0, 1])
        >>> grouped = group_items(items, keys)
        >>> print(ub.repr2(grouped, nl=1, with_dtype=False))
        {
            0: np.array([4, 6]),
            1: np.array([2, 3, 5, 7]),
            2: np.array([0, 1]),
        }
    """
    keys, groupxs = group_indices(groupid_list, assume_sorted=assume_sorted)
    grouped_values = apply_grouping(item_list, groupxs, axis=axis)
    grouped = dict(zip(keys, grouped_values))
    return grouped


def group_indices(idx_to_groupid, assume_sorted=False):
    """
    Find unique items and the indices at which they appear in an array.

    A common use case of this function is when you have a list of objects
    (often numeric but sometimes not) and an array of "group-ids" corresponding
    to that list of objects.

    Using this function will return a list of indices that can be used in
    conjunction with :func:`apply_grouping` to group the elements.  This is
    most useful when you have many lists (think column-major data)
    corresponding to the group-ids.

    In cases where there is only one list of objects or knowing the indices
    doesn't matter, then consider using func:`group_items` instead.

    Args:
        idx_to_groupid (ndarray):
            The input array, where each item is interpreted as a group id.
            For the fastest runtime, the input array must be numeric (ideally
            with integer types).  If the type is non-numeric then the less
            efficient :func:`ubelt.group_items` is used.

        assume_sorted (bool, default=False):
            If the input array is sorted, then setting this to True will avoid
            an unnecessary sorting operation and improve efficiency.

    Returns:
        Tuple[ndarray, List[ndarrays]]: (keys, groupxs) -
            keys (ndarray):
                The unique elements of the input array in order
            groupxs (List[ndarray]):
                Corresponding list of indexes.  The i-th item is an array
                indicating the indices where the item ``key[i]`` appeared in
                the input array.

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> import ubelt as ub
        >>> idx_to_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> (keys, groupxs) = group_indices(idx_to_groupid)
        >>> print(ub.repr2(keys, with_dtype=False))
        >>> print(ub.repr2(groupxs, with_dtype=False))
        np.array([1, 2, 3])
        [
            np.array([1, 3, 5]),
            np.array([0, 2, 4, 6]),
            np.array([ 7,  8,  9, 10]),
        ]

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> import ubelt as ub
        >>> idx_to_groupid = np.array([[  24], [ 129], [ 659], [ 659], [ 24],
        ...       [659], [ 659], [ 822], [ 659], [ 659], [24]])
        >>> # 2d arrays must be flattened before coming into this function so
        >>> # information is on the last axis
        >>> (keys, groupxs) = group_indices(idx_to_groupid.T[0])
        >>> print(ub.repr2(keys, with_dtype=False))
        >>> print(ub.repr2(groupxs, with_dtype=False))
        np.array([ 24, 129, 659, 822])
        [
            np.array([ 0,  4, 10]),
            np.array([1]),
            np.array([2, 3, 5, 6, 8, 9]),
            np.array([7]),
        ]

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> import ubelt as ub
        >>> idx_to_groupid = np.array([True, True, False, True, False, False, True])
        >>> (keys, groupxs) = group_indices(idx_to_groupid)
        >>> print(ub.repr2(keys, with_dtype=False))
        >>> print(ub.repr2(groupxs, with_dtype=False))
        np.array([False,  True])
        [
            np.array([2, 4, 5]),
            np.array([0, 1, 3, 6]),
        ]

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> import ubelt as ub
        >>> idx_to_groupid = [('a', 'b'),  ('d', 'b'), ('a', 'b'), ('a', 'b')]
        >>> (keys, groupxs) = group_indices(idx_to_groupid)
        >>> print(ub.repr2(keys, with_dtype=False))
        >>> print(ub.repr2(groupxs, with_dtype=False))
        [
            ('a', 'b'),
            ('d', 'b'),
        ]
        [
            np.array([0, 2, 3]),
            np.array([1]),
        ]
    """
    _idx_to_groupid_orig = idx_to_groupid
    idx_to_groupid = np.array(idx_to_groupid, copy=False)
    _n_item = idx_to_groupid.size
    _dtype = idx_to_groupid.dtype
    _kind = _dtype.kind
    if _kind == 'U' or _kind == 'O':
        # fallback to slower algorithm for non-numeric data
        group = ub.group_items(range(_n_item), _idx_to_groupid_orig)
        try:
            # attempt to return values in a consistant order
            sortx = ub.argsort(list(group.keys()))
            keys = list(ub.take(list(group.keys()), sortx))
            groupxs = list(ub.take(list(map(np.array, group.values())), sortx))
        except Exception:
            keys = list(group.keys())
            groupxs = list(map(np.array, group.values()))
        return keys, groupxs

    # Sort items and idx_to_groupid by groupid
    if assume_sorted:
        sortx = np.arange(len(idx_to_groupid))
        groupids_sorted = idx_to_groupid
    else:
        sortx = idx_to_groupid.argsort()
        groupids_sorted = idx_to_groupid.take(sortx)

    if _kind == 'b':
        # Ensure bools are internally cast to integers
        # However, be sure that the groups are returned as the original dtype
        _groupids = groupids_sorted.astype(np.int8)
    else:
        _groupids = groupids_sorted

    # Find the boundaries between groups
    diff = np.ones(_n_item + 1, _groupids.dtype)
    np.subtract(_groupids[1:], _groupids[:-1], out=diff[1:_n_item])
    idxs = np.flatnonzero(diff)
    # Groups are between bounding indexes
    groupxs = [sortx[lx:rx] for lx, rx in zip(idxs, idxs[1:])]  # 34.5%
    # Unique group keys
    keys = groupids_sorted[idxs[:-1]]
    return keys, groupxs


def apply_grouping(items, groupxs, axis=0):
    """
    Applies grouping from group_indicies.

    Typically used in conjunction with :func:`group_indices`.

    Args:
        items (ndarray): items to group

        groupxs (List[ndarrays[int]]): groups of indices

        axis (None|int, default=0) axis along which to group

    Returns:
        List[ndarray]: grouped items


    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> idx_to_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> items          = np.array([1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9])
        >>> (keys, groupxs) = group_indices(idx_to_groupid)
        >>> grouped_items = apply_grouping(items, groupxs)
        >>> result = str(grouped_items)
        >>> print(result)
        [array([8, 5, 6]), array([1, 5, 8, 7]), array([5, 3, 0, 9])]
    """
    # Should there be a contiguous check here?
    # items_ = np.ascontiguousarray(items)
    return [items.take(xs, axis=axis) for xs in groupxs]


def group_consecutive(arr, offset=1):
    """
    Returns lists of consecutive values. Implementation inspired by [3]_.

    Args:
        arr (ndarray):
            array of ordered values

        offset (float, default=1):
            any two values separated by this offset are grouped.  In the
            default case, when offset=1, this groups increasing values like: 0,
            1, 2. When offset is 0 it groups consecutive values thta are the
            same, e.g.: 4, 4, 4.

    Returns:
        List[ndarray]: a list of arrays that are the groups from the input

    Notes:
        This is equivalent (and faster) to using:
        apply_grouping(data, group_consecutive_indices(data))

    References:
        .. [3] http://stackoverflow.com/questions/7352684/groups-consecutive-elements

    Example:
        >>> arr = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 99, 100, 101])
        >>> groups = group_consecutive(arr)
        >>> print('groups = {}'.format(list(map(list, groups))))
        groups = [[1, 2, 3], [5, 6, 7, 8, 9, 10], [15], [99, 100, 101]]
        >>> arr = np.array([0, 0, 3, 0, 0, 7, 2, 3, 4, 4, 4, 1, 1])
        >>> groups = group_consecutive(arr, offset=1)
        >>> print('groups = {}'.format(list(map(list, groups))))
        groups = [[0], [0], [3], [0], [0], [7], [2, 3, 4], [4], [4], [1], [1]]
        >>> groups = group_consecutive(arr, offset=0)
        >>> print('groups = {}'.format(list(map(list, groups))))
        groups = [[0, 0], [3], [0, 0], [7], [2], [3], [4, 4, 4], [1, 1]]
    """
    split_indicies = np.nonzero(np.diff(arr) != offset)[0] + 1
    groups = np.array_split(arr, split_indicies)
    return groups


def group_consecutive_indices(arr, offset=1):
    """
    Returns lists of indices pointing to consecutive values

    Args:
        arr (ndarray):
            array of ordered values

        offset (float, default=1):
            any two values separated by this offset are grouped.

    Returns:
        List[ndarray]: groupxs: a list of indices

    SeeAlso:

        :func:`group_consecutive`

        :func:`apply_grouping`

    Example:
        >>> arr = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 99, 100, 101])
        >>> groupxs = group_consecutive_indices(arr)
        >>> print('groupxs = {}'.format(list(map(list, groupxs))))
        groupxs = [[0, 1, 2], [3, 4, 5, 6, 7, 8], [9], [10, 11, 12]]
        >>> assert all(np.array_equal(a, b) for a, b in zip(group_consecutive(arr, 1), apply_grouping(arr, groupxs)))
        >>> arr = np.array([0, 0, 3, 0, 0, 7, 2, 3, 4, 4, 4, 1, 1])
        >>> groupxs = group_consecutive_indices(arr, offset=1)
        >>> print('groupxs = {}'.format(list(map(list, groupxs))))
        groupxs = [[0], [1], [2], [3], [4], [5], [6, 7, 8], [9], [10], [11], [12]]
        >>> assert all(np.array_equal(a, b) for a, b in zip(group_consecutive(arr, 1), apply_grouping(arr, groupxs)))
        >>> groupxs = group_consecutive_indices(arr, offset=0)
        >>> print('groupxs = {}'.format(list(map(list, groupxs))))
        groupxs = [[0, 1], [2], [3, 4], [5], [6], [7], [8, 9, 10], [11, 12]]
        >>> assert all(np.array_equal(a, b) for a, b in zip(group_consecutive(arr, 0), apply_grouping(arr, groupxs)))
    """
    split_indicies = np.nonzero(np.diff(arr) != offset)[0] + 1
    groupxs = np.array_split(np.arange(len(arr)), split_indicies)
    return groupxs


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwarray.util_groups
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
