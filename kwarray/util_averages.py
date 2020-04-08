# -*- coding: utf-8 -*-
"""
Currently just defines "stats_dict", which is a nice way to gather multiple
numeric statistics (e.g. max, min, median, mode, arithmetic-mean,
geometric-mean, standard-deviation, etc...) about data in an array.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import numpy as np
import torch


def stats_dict(inputs, axis=None, nan=False, sum=False, extreme=True,
               n_extreme=False, median=False, shape=True, size=False):
    """
    Describe statistics about an input array

    Args:
        inputs (ArrayLike): set of values to get statistics of
        axis (int): if ``inputs`` is ndarray then this specifies the axis
        nan (bool): report number of nan items
        sum (bool): report sum of values
        extreme (bool): report min and max values
        n_extreme (bool): report extreme value frequencies
        median (bool): report median
        size (bool): report array size
        shape (bool): report array shape

    Returns:
        collections.OrderedDict: stats: dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    SeeAlso:
        scipy.stats.describe

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> from kwarray.util_averages import *  # NOQA
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> inputs = rng.rand(10, 2).astype(np.float32)
        >>> stats = stats_dict(inputs, axis=axis, nan=False, median=True)
        >>> import ubelt as ub  # NOQA
        >>> result = str(ub.repr2(stats, nl=1, precision=4, with_dtype=True))
        >>> print(result)
        {
            'mean': np.array([ 0.5206,  0.6425], dtype=np.float32),
            'std': np.array([ 0.2854,  0.2517], dtype=np.float32),
            'min': np.array([ 0.0202,  0.0871], dtype=np.float32),
            'max': np.array([ 0.9637,  0.9256], dtype=np.float32),
            'med': np.array([0.5584, 0.6805], dtype=np.float32),
            'shape': (10, 2),
        }

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> inputs = rng.randint(0, 42, size=100).astype(np.float32)
        >>> inputs[4] = np.nan
        >>> stats = stats_dict(inputs, axis=axis, nan=True)
        >>> import ubelt as ub  # NOQA
        >>> result = str(ub.repr2(stats, nl=0, precision=1, strkeys=True))
        >>> print(result)
        {mean: 20.0, std: 13.2, min: 0.0, max: 41.0, num_nan: 1, shape: (100,)}
    """
    stats = collections.OrderedDict([])

    # Ensure input is in numpy format
    if isinstance(inputs, np.ndarray):
        nparr = inputs
    elif isinstance(inputs, list):
        nparr = np.array(inputs)
    elif isinstance(inputs, torch.Tensor):
        nparr = inputs.data.cpu().numpy()
    else:
        nparr = np.array(list(inputs))
    # Check to make sure stats are feasible
    if len(nparr) == 0:
        stats['empty_list'] = True
        if size:
            stats['size'] = 0
    else:
        if nan:
            min_val = np.nanmin(nparr, axis=axis)
            max_val = np.nanmax(nparr, axis=axis)
            mean_ = np.nanmean(nparr, axis=axis)
            std_  = np.nanstd(nparr, axis=axis)
        else:
            min_val = nparr.min(axis=axis)
            max_val = nparr.max(axis=axis)
            mean_ = nparr.mean(axis=axis)
            std_  = nparr.std(axis=axis)
        # number of entries with min/max val
        nMin = np.sum(nparr == min_val, axis=axis)
        nMax = np.sum(nparr == max_val, axis=axis)

        # Notes:
        # the first central moment is 0
        # the first raw moment is the mean
        # the second central moment is the variance
        # the third central moment is the skweness * var ** 3
        # the fourth central moment is the kurtosis * var ** 4
        # the third standardized moment is the skweness
        # the fourth standardized moment is the kurtosis

        if True:
            stats['mean'] = np.float32(mean_)
            stats['std'] = np.float32(std_)
        if extreme:
            stats['min'] = np.float32(min_val)
            stats['max'] = np.float32(max_val)
        if n_extreme:
            stats['nMin'] = np.int32(nMin)
            stats['nMax'] = np.int32(nMax)
        if median:
            stats['med'] = np.nanmedian(nparr, axis=axis)
        if nan:
            stats['num_nan'] = np.isnan(nparr).sum()
        if sum:
            sumfunc = np.nansum if nan else np.sum
            stats['sum'] = sumfunc(nparr, axis=axis)
        if size:
            stats['size'] = nparr.size
        if shape:
            stats['shape'] = nparr.shape
    return stats


def _gmean(a, axis=0, dtype=None, clobber=False):
    """
    Compute the geometric mean along the specified axis.

    Modification of the scikit-learn method to be more memory efficient

    Example
        >>> rng = np.random.RandomState(0)
        >>> C, H, W = 8, 32, 32
        >>> axis = 0
        >>> a = [rng.rand(C, H, W).astype(np.float16),
        >>>      rng.rand(C, H, W).astype(np.float16)]

    """
    if isinstance(a, np.ndarray):
        if clobber:
            # NOTE: we reuse (a), we clobber the input array!
            log_a = np.log(a, out=a)
        else:
            log_a = np.log(a)
    else:
        if dtype is None:
            # if not an ndarray object attempt to convert it
            log_a = np.log(np.array(a, dtype=dtype))
        else:
            # Must change the default dtype allowing array type
            # Note: that this will use memory, but there isn't anything we can
            # do here.
            if isinstance(a, np.ma.MaskedArray):
                a_ = np.ma.asarray(a, dtype=dtype)
            else:
                a_ = np.asarray(a, dtype=dtype)
            # We can reuse `a_` because it was a temp var
            log_a = np.log(a_, out=a_)

    # attempt to reuse memory when computing mean
    mem = log_a[axis]
    mean_log_a = log_a.mean(axis=axis, out=mem)

    # And reuse memory again when computing the final result
    result = np.exp(mean_log_a, out=mean_log_a)
    return result
