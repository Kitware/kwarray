# -*- coding: utf-8 -*-
"""
Currently just defines "stats_dict", which is a nice way to gather multiple
numeric statistics (e.g. max, min, median, mode, arithmetic-mean,
geometric-mean, standard-deviation, etc...) about data in an array.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import numpy as np
import ubelt as ub


try:
    import torch
except Exception:
    torch = None


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
    elif torch is not None and isinstance(inputs, torch.Tensor):
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


class RunningStats(ub.NiceRepr):
    """
    Dynamically records per-element array statistics and can summarized them
    per-element, across channels, or globally.

    TODO:
        - [ ] This may need a few API tweaks and good documentation

    Example:
        >>> run = RunningStats()
        >>> ch1 = np.array([[0, 1], [3, 4]])
        >>> ch2 = np.zeros((2, 2))
        >>> img = np.dstack([ch1, ch2])
        >>> run.update(np.dstack([ch1, ch2]))
        >>> run.update(np.dstack([ch1 + 1, ch2]))
        >>> run.update(np.dstack([ch1 + 2, ch2]))
        >>> # No marginalization
        >>> print('current-ave = ' + ub.repr2(run.summarize(axis=ub.NoParam), nl=2, precision=3))
        >>> # Average over channels (keeps spatial dims separate)
        >>> print('chann-ave(k=1) = ' + ub.repr2(run.summarize(axis=0), nl=2, precision=3))
        >>> print('chann-ave(k=0) = ' + ub.repr2(run.summarize(axis=0, keepdims=0), nl=2, precision=3))
        >>> # Average over spatial dims (keeps channels separate)
        >>> print('spatial-ave(k=1) = ' + ub.repr2(run.summarize(axis=(1, 2)), nl=2, precision=3))
        >>> print('spatial-ave(k=0) = ' + ub.repr2(run.summarize(axis=(1, 2), keepdims=0), nl=2, precision=3))
        >>> # Average over all dims
        >>> print('alldim-ave(k=1) = ' + ub.repr2(run.summarize(axis=None), nl=2, precision=3))
        >>> print('alldim-ave(k=0) = ' + ub.repr2(run.summarize(axis=None, keepdims=0), nl=2, precision=3))
        """

    def __init__(run):
        run.raw_max = -np.inf
        run.raw_min = np.inf
        run.raw_total = 0
        run.raw_squares = 0
        run.n = 0

    def __nice__(self):
        return '{}'.format(self.shape)

    @property
    def shape(run):
        try:
            return run.raw_total.shape
        except Exception:
            return None

    def update(run, data):
        """
        Updates statistics across all data dimensions on a per-element basis
        """
        run.n += 1
        run.raw_max = np.maximum(run.raw_max, data)
        run.raw_min = np.minimum(run.raw_min, data)
        run.raw_total += data
        run.raw_squares += data ** 2

    def _sumsq_std(run, total, squares, n):
        """
        Sum of squares method to compute standard deviation
        """
        numer = (n * squares - total ** 2)
        denom = (n * (n - 1.0))
        std = np.sqrt(numer / denom)
        return std

    def summarize(run, axis=None, keepdims=True):
        """
        Compute summary statistics across a one or more dimension

        Args:
            axis (int | List[int] | None | ub.NoParam):
                axis or axes to summarize over,
                if None, all axes are summarized.
                if ub.NoParam, no axes are summarized the current result is
                    returned.

            keepdims (bool, default=True):
                if False removes the dimensions that are summarized over

        Returns:
            Dict: containing minimum, maximum, mean, std, etc..
        """
        if axis is ub.NoParam:
            total = run.raw_total
            squares = run.raw_squares
            maxi = run.raw_max
            mini = run.raw_min
            n = run.n
            info = ub.odict([
                ('n', n),
                ('max', maxi),
                ('min', mini),
                ('total', total),
                ('squares', squares),
                ('mean', total / n),
                ('std', run._sumsq_std(total, squares, n)),
            ])
            return info
        else:
            if run.n <= 0:
                raise RuntimeError('No statistics have been accumulated')
            total   = run.raw_total.sum(axis=axis, keepdims=keepdims)
            squares = run.raw_squares.sum(axis=axis, keepdims=keepdims)
            maxi    = run.raw_max.max(axis=axis, keepdims=keepdims)
            mini    = run.raw_min.min(axis=axis, keepdims=keepdims)
            if not hasattr(run.raw_total, 'shape'):
                n = run.n
            elif axis is None:
                n = run.n * np.prod(run.raw_total.shape)
            else:
                n = run.n * np.prod(np.take(run.raw_total.shape, axis))

            info = ub.odict([
                ('n', n),
                ('max', maxi),
                ('min', mini),
                ('total', total),
                ('squares', squares),
                ('mean', total / n),
                ('std', run._sumsq_std(total, squares, n)),
            ])
            return info

    def current(run):
        """
        Returns current staticis on a per-element basis
        (not summarized over any axis)

        TODO:
            - [X] I want this method and summarize to be unified somehow.
                I don't know how to paramatarize it because axis=None usually
                means summarize over everything, and I need to way to encode,
                summarize over nothing but the "sequence" dimension (which was
                given incrementally by the update function), which is what
                this function does.
        """
        info = run.summarize(axis=ub.NoParam)
        return info

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwarray/kwarray/util_averages.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
