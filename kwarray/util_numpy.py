# -*- coding: utf-8 -*-
"""
Numpy specific extensions
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def boolmask(indices, shape=None):
    """
    Constructs an array of booleans where an item is True if its position is in
    ``indices`` otherwise it is False. This can be viewed as the inverse of
    :func:`numpy.where`.

    Args:
        indices (ndarray): list of integer indices

        shape (int | tuple): length of the returned list. If not specified
            the minimal possible shape to incoporate all the indices is used.
            In general, it is best practice to always specify this argument.

    Returns:
        ndarray[int]: mask: mask[idx] is True if idx in indices

    Example:
        >>> indices = [0, 1, 4]
        >>> mask = boolmask(indices, shape=6)
        >>> assert np.all(mask == [True, True, False, False, True, False])
        >>> mask = boolmask(indices)
        >>> assert np.all(mask == [True, True, False, False, True])

    Example:
        >>> indices = np.array([(0, 0), (1, 1), (2, 1)])
        >>> shape = (3, 3)
        >>> mask = boolmask(indices, shape)
        >>> import ubelt as ub  # NOQA
        >>> result = ub.repr2(mask)
        >>> print(result)
        np.array([[ True, False, False],
                  [False,  True, False],
                  [False,  True, False]], dtype=np.bool)
    """
    indices = np.asanyarray(indices)
    if indices.dtype.kind not in {'i', 'u'}:
        indices = indices.astype(np.int)
    if shape is None:
        shape = indices.max() + 1
    mask = np.zeros(shape, dtype=np.bool)
    if mask.ndim > 1:
        mask[tuple(indices.T)] = True
    else:
        mask[indices] = True
    return mask


def iter_reduce_ufunc(ufunc, arrs, out=None, default=None):
    """
    constant memory iteration and reduction

    applys ufunc from left to right over the input arrays

    Args:
        ufunc (Callable): called on each pair of consecutive ndarrays
        arrs (Iterator[ndarray]): iterator of ndarrays
        default (object): return value when iterator is empty

    Returns:
        ndarray:
            if len(arrs) == 0, returns ``default``
            if len(arrs) == 1, returns arrs[0],
            if len(arrs) >= 2, returns
                ufunc(...ufunc(ufunc(arrs[0], arrs[1]), arrs[2]),...arrs[n-1])

    Example:
        >>> arr_list = [
        ...     np.array([0, 1, 2, 3, 8, 9]),
        ...     np.array([4, 1, 2, 3, 4, 5]),
        ...     np.array([0, 5, 2, 3, 4, 5]),
        ...     np.array([1, 1, 6, 3, 4, 5]),
        ...     np.array([0, 1, 2, 7, 4, 5])
        ... ]
        >>> memory = np.array([9, 9, 9, 9, 9, 9])
        >>> gen_memory = memory.copy()
        >>> def arr_gen(arr_list, gen_memory):
        ...     for arr in arr_list:
        ...         gen_memory[:] = arr
        ...         yield gen_memory
        >>> print('memory = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> ufunc = np.maximum
        >>> res1 = iter_reduce_ufunc(ufunc, iter(arr_list), out=None)
        >>> res2 = iter_reduce_ufunc(ufunc, iter(arr_list), out=memory)
        >>> res3 = iter_reduce_ufunc(ufunc, arr_gen(arr_list, gen_memory), out=memory)
        >>> print('res1       = %r' % (res1,))
        >>> print('res2       = %r' % (res2,))
        >>> print('res3       = %r' % (res3,))
        >>> print('memory     = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> assert np.all(res1 == res2)
        >>> assert np.all(res2 == res3)
    """
    # Get first item in iterator
    try:
        initial = next(arrs)
    except StopIteration:
        return default
    # Populate the outvariable if specified otherwise make a copy of the first
    # item to be the output memory
    if out is not None:
        out[:] = initial
    else:
        out = initial.copy()
    # Iterate and reduce
    for arr in arrs:
        ufunc(out, arr, out=out)
    return out


def isect_flags(arr, other):
    """
    Check which items in an array intersect with another set of items

    Args:
        arr (ndarray): items to check
        other (Iterable): items to check if they exist in arr

    Returns:
        ndarray: booleans corresponding to arr indicating if that item is
            also contained in other.

    Example:
        >>> arr = np.array([
        >>>     [1, 2, 3, 4],
        >>>     [5, 6, 3, 4],
        >>>     [1, 1, 3, 4],
        >>> ])
        >>> other = np.array([1, 4, 6])
        >>> mask = isect_flags(arr, other)
        >>> print(mask)
        [[ True False False  True]
         [False  True False  True]
         [ True  True False  True]]
    """
    flags = iter_reduce_ufunc(np.logical_or, (arr == item for item in other))
    if flags is None:
        flags = np.zeros(arr.size, dtype=np.bool)
    return flags


def atleast_nd(arr, n, front=False):
    r"""
    View inputs as arrays with at least n dimensions.

    Args:
        arr (array_like):
            An array-like object.  Non-array inputs are converted to arrays.
            Arrays that already have n or more dimensions are preserved.

        n (int):
            number of dimensions to ensure

        front (bool, default=False):
            if True new dimensions are added to the front of the array.
            otherwise they are added to the back.

    Returns:
        ndarray :
            An array with ``a.ndim >= n``.  Copies are avoided where possible,
            and views with three or more dimensions are returned.  For example,
            a 1-D array of shape ``(N,)`` becomes a view of shape
            ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a view
            of shape ``(M, N, 1)``.

    See Also:
        numpy.atleast_1d, numpy.atleast_2d, numpy.atleast_3d

    Example:
        >>> n = 2
        >>> arr = np.array([1, 1, 1])
        >>> arr_ = atleast_nd(arr, n)
        >>> import ubelt as ub  # NOQA
        >>> result = ub.repr2(arr_.tolist(), nl=0)
        >>> print(result)
        [[1], [1], [1]]

    Example:
        >>> n = 4
        >>> arr1 = [1, 1, 1]
        >>> arr2 = np.array(0)
        >>> arr3 = np.array([[[[[1]]]]])
        >>> arr1_ = atleast_nd(arr1, n)
        >>> arr2_ = atleast_nd(arr2, n)
        >>> arr3_ = atleast_nd(arr3, n)
        >>> import ubelt as ub  # NOQA
        >>> result1 = ub.repr2(arr1_.tolist(), nl=0)
        >>> result2 = ub.repr2(arr2_.tolist(), nl=0)
        >>> result3 = ub.repr2(arr3_.tolist(), nl=0)
        >>> result = '\n'.join([result1, result2, result3])
        >>> print(result)
        [[[[1]]], [[[1]]], [[[1]]]]
        [[[[0]]]]
        [[[[[1]]]]]

    Ignore:
        # Hmm, mine is actually faster
        %timeit atleast_nd(arr, 3)
        %timeit np.atleast_3d(arr)

    Benchmark:

        import ubelt
        N = 100

        t1 = ubelt.Timerit(N, label='mine')
        for timer in t1:
            arr = np.empty((10, 10))
            with timer:
                atleast_nd(arr, 3)

        t2 = ubelt.Timerit(N, label='baseline')
        for timer in t2:
            arr = np.empty((10, 10))
            with timer:
                np.atleast_3d(arr)
    """
    arr_ = np.asanyarray(arr)
    ndims = len(arr_.shape)
    if n is not None and ndims <  n:
        # append the required number of dimensions to the front or back
        if front:
            expander = (None,) * (n - ndims) + (Ellipsis,)
        else:
            expander = (Ellipsis,) + (None,) * (n - ndims)
        arr_ = arr_[expander]
    return arr_


def argmaxima(arr, num, axis=None, ordered=True):
    """
    Returns the top ``num`` maximum indicies.

    This can be significantly faster than using argsort.

    Args:
        arr (ndarray): input array
        num (int): number of maximum indices to return
        axis (int|None): axis to find maxima over. If None this is equivalent
            to using arr.ravel().
        ordered (bool): if False, returns the maximum elements in an arbitrary
            order, otherwise they are in decending order. (Setting this to
            false is a bit faster).

    TODO:
        - [ ] if num is None, return arg for all values equal to the maximum

    Returns:
        ndarray

    Example:
        >>> # Test cases with axis=None
        >>> arr = (np.random.rand(100) * 100).astype(np.int)
        >>> for num in range(0, len(arr) + 1):
        >>>     idxs = argmaxima(arr, num)
        >>>     idxs2 = argmaxima(arr, num, ordered=False)
        >>>     assert np.all(arr[idxs] == np.array(sorted(arr)[::-1][:len(idxs)])), 'ordered=True must return in order'
        >>>     assert sorted(idxs2) == sorted(idxs), 'ordered=False must return the right idxs, but in any order'

    Example:
        >>> # Test cases with axis
        >>> arr = (np.random.rand(3, 5, 7) * 100).astype(np.int)
        >>> for axis in range(len(arr.shape)):
        >>>     for num in range(0, len(arr) + 1):
        >>>         idxs = argmaxima(arr, num, axis=axis)
        >>>         idxs2 = argmaxima(arr, num, ordered=False, axis=axis)
        >>>         assert idxs.shape[axis] == num
        >>>         assert idxs2.shape[axis] == num
    """
    # if axis is not None:
    #     raise NotImplementedError('axis must be None for now')
    # Gets top N maximum or minimum indices
    if num < 0:
        raise IndexError
    if axis is None:
        axis_size = arr.size
        if num == 0:
            return np.empty(0, dtype=np.int)
        elif num == 1:
            # very fast
            idxs = np.array([arr.argmax(axis=axis)])
        elif num < axis_size:
            # argpartition is almost what we want, and its faster than argsort
            kth = axis_size - num
            part_idxs = np.argpartition(arr, kth=kth, axis=axis)
            idxs = part_idxs[kth:]
            if ordered:
                sortx = arr.take(idxs, axis=axis).argsort()[::-1]
                idxs = idxs.take(sortx)
        else:
            # sort all the indices
            if ordered:
                idxs = arr.argsort(axis=axis)
                idxs = idxs[::-1][0:num]
            else:
                # Arbitrary order is allowed, so cheat
                idxs = np.arange(arr.size)
    else:
        axis_size = arr.shape[axis]
        if num == 0:
            new_shape = list(arr.shape)
            new_shape[axis] = 0
            return np.empty(new_shape, dtype=np.int)
        elif num == 1:
            newshape = list(arr.shape)
            newshape[axis] = 1
            idxs = arr.argmax(axis=axis).reshape(*newshape)
        elif num < axis_size:
            # TODO: is there a better implementation for this case?
            kth = axis_size - num
            part_idxs = np.argpartition(arr, kth=kth, axis=axis)
            fancy_index = [slice(None)] * (axis + 1)
            fancy_index[axis] = slice(kth, None)
            idxs = part_idxs[tuple(fancy_index)]
            if ordered:
                # move the axis of interest to the back
                idxs_swap = idxs.swapaxes(-1, axis)
                idxs2d = idxs_swap.reshape(-1, num)
                arr2d = arr.swapaxes(-1, axis).reshape(-1, arr.shape[axis])
                # now ensure each row is in order
                new_idxs2d = []
                for a, i in zip(arr2d, idxs2d):
                    sortx = a[i].argsort()[::-1]
                    new_idxs2d.append(i[sortx])
                new_idxs2d = np.array(new_idxs2d)
                # transform back to original shape
                idxs = new_idxs2d.reshape(*idxs_swap.shape).swapaxes(-1, axis)
        else:
            # sort all the indices
            if ordered:
                idxs = arr.argsort(axis=axis)
                fancy_index = [slice(None)] * (axis + 1)
                fancy_index[axis] = slice(None, None, -1)
                idxs = idxs[tuple(fancy_index)]
            else:
                # Arbitrary order is allowed, so cheat
                idxs = np.arange(arr.shape[axis])
                newshape = [1] * len(arr.shape)
                newshape[axis] = arr.shape[axis]
                repeats = list(arr.shape)
                repeats[axis] = 1
                idxs = np.tile(idxs.reshape(*newshape), repeats)
    return idxs


def argminima(arr, num, axis=None, ordered=True):
    """
    Returns the top ``num`` minimum indicies.

    This can be significantly faster than using argsort.

    Args:
        arr (ndarray): input array
        num (int): number of minimum indices to return
        axis (int|None): must be None for now
        ordered (bool): if False, returns the minimum elements in an arbitrary
            order, otherwise they are in ascending order. (Setting this to
            false is a bit faster).

    Example:
        >>> arr = (np.random.rand(100) * 100).astype(np.int)
        >>> for num in range(0, len(arr) + 1):
        >>>     idxs = argminima(arr, num)
        >>>     assert np.all(arr[idxs] == np.array(sorted(arr)[:len(idxs)])), 'ordered=True must return in order'
        >>>     idxs2 = argminima(arr, num, ordered=False)
        >>>     assert sorted(idxs2) == sorted(idxs), 'ordered=False must return the right idxs, but in any order'

    Example:
        >>> arr = (np.random.rand(32, 32) * 100).astype(np.int)
        >>> argminima(arr, 10)
    """
    if axis is not None:
        raise NotImplementedError('axis must be None for now')
    # Gets top N maximum or minimum indices
    if num < 0:
        raise IndexError
    elif num == 0:
        return np.empty(0, dtype=np.int)
    elif num == 1:
        # very fast
        idxs = np.array([arr.argmin(axis=axis)])
    elif num < len(arr):
        # argpartition is almost what we want, and its faster than argsort
        kth = num
        part_idxs = np.argpartition(arr, kth=kth, axis=axis)
        idxs = part_idxs[:kth]
        if ordered:
            sortx = arr.take(idxs, axis=axis).argsort()
            idxs = idxs.take(sortx)
    else:
        # sort all the indices
        if ordered:
            idxs = arr.argsort(axis=axis)[0:num]
        else:
            # Arbitrary order is allowed, so cheat
            idxs = np.arange(arr.size)
    return idxs


def arglexmax(keys, multi=False):
    """
    Find the index of the maximum element in a sequence of keys.

    Args:
        keys (tuple): a k-tuple of k N-dimensional arrays.
            Like np.lexsort the last key in the sequence is used for the
            primary sort order, the second-to-last key for the secondary sort
            order, and so on.

        multi (bool): if True, returns all indices that share the max value

    Returns:
        int | ndarray[int] : either the index or list of indices

    Example:
        >>> k, N = 100, 100
        >>> rng = np.random.RandomState(0)
        >>> keys = [(rng.rand(N) * N).astype(np.int) for _ in range(k)]
        >>> multi_idx = arglexmax(keys, multi=True)
        >>> idxs = np.lexsort(keys)
        >>> assert sorted(idxs[::-1][:len(multi_idx)]) == sorted(multi_idx)

    Benchark:
        >>> import ubelt as ub
        >>> k, N = 100, 100
        >>> rng = np.random
        >>> keys = [(rng.rand(N) * N).astype(np.int) for _ in range(k)]
        >>> for timer in ub.Timerit(100, bestof=10, label='arglexmax'):
        >>>     with timer:
        >>>         arglexmax(keys)
        >>> for timer in ub.Timerit(100, bestof=10, label='lexsort'):
        >>>     with timer:
        >>>         np.lexsort(keys)[-1]
    """
    # Handle keys in reverse order to be consistent with np.lexsort
    reverse_keys = keys[::-1]
    arr = reverse_keys[0]
    breakers = reverse_keys[1:]
    # Look for the maximum value in the first array, and continue using new
    # arrays until a unique maximum index is found.
    _cand_idxs = np.where(arr == arr.max())[0]
    if len(_cand_idxs) > 1:
        for breaker in breakers:
            vals = breaker[_cand_idxs]
            _cand_idxs = _cand_idxs[vals == vals.max()]
            if len(_cand_idxs) == 1:
                break
    # If multiple maximum values are found then either
    # return them all or return an arbitrary one.
    return _cand_idxs if multi else _cand_idxs[0]
