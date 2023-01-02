# -*- coding: utf-8 -*-
"""
Currently just defines "stats_dict", which is a nice way to gather multiple
numeric statistics (e.g. max, min, median, mode, arithmetic-mean,
geometric-mean, standard-deviation, etc...) about data in an array.
"""
import warnings
import collections
import numpy as np
import ubelt as ub


try:
    import torch
except Exception:
    torch = None


# Maybe name or alias or use in kwarray.describe?
def stats_dict(inputs, axis=None, nan=False, sum=False, extreme=True,
               n_extreme=False, median=False, shape=True, size=False,
               quantile='auto'):
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
        collections.OrderedDict:
            dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    SeeAlso:
        :func:`scipy.stats.describe`
        :func:`pandas.DataFrame.describe`

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
            'mean': np.array([[0.5206, 0.6425]], dtype=np.float32),
            'std': np.array([[0.2854, 0.2517]], dtype=np.float32),
            'min': np.array([[0.0202, 0.0871]], dtype=np.float32),
            'max': np.array([[0.9637, 0.9256]], dtype=np.float32),
            'q_0.25': np.array([0.4271, 0.5329], dtype=np.float64),
            'q_0.50': np.array([0.5584, 0.6805], dtype=np.float64),
            'q_0.75': np.array([0.7343, 0.8607], dtype=np.float64),
            'med': np.array([0.5584, 0.6805], dtype=np.float32),
            'shape': (10, 2),
        }

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> from kwarray.util_averages import *  # NOQA
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> inputs = rng.randint(0, 42, size=100).astype(np.float32)
        >>> inputs[4] = np.nan
        >>> stats = stats_dict(inputs, axis=axis, nan=True, quantile='auto')
        >>> import ubelt as ub  # NOQA
        >>> result = str(ub.repr2(stats, nl=0, precision=1, strkeys=True))
        >>> print(result)

    Example:
        >>> import kwarray
        >>> import ubelt as ub
        >>> rng = kwarray.ensure_rng(0)
        >>> orig_inputs = rng.rand(1, 1, 2, 3)
        >>> param_grid = ub.named_product({
        >>>     #'axis': (None, 0, (0, 1), -1),
        >>>     'axis': [None],
        >>>     'percent_nan': [0, 0.5, 1.0],
        >>>     'nan': [True, False],
        >>>     'sum': [1],
        >>>     'extreme': [True],
        >>>     'n_extreme': [True],
        >>>     'median': [1],
        >>>     'size': [1],
        >>>     'shape': [1],
        >>>     'quantile': ['auto'],
        >>> })
        >>> for params in param_grid:
        >>>     kwargs = params.copy()
        >>>     percent_nan = kwargs.pop('percent_nan', 0)
        >>>     if percent_nan:
        >>>         inputs = orig_inputs.copy()
        >>>         inputs[rng.rand(*inputs.shape) < percent_nan] = np.nan
        >>>     else:
        >>>         inputs = orig_inputs
        >>>     stats = kwarray.stats_dict(inputs, **kwargs)
        >>>     print('---')
        >>>     print('params = {}'.format(ub.repr2(params, nl=1)))
        >>>     print('stats = {}'.format(ub.repr2(stats, nl=1)))

    Ignore:
        import kwarray
        inputs = np.random.rand(3, 2, 1)
        stats = kwarray.stats_dict(inputs, axis=2, nan=True, quantile='auto')
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
        keepdims = (axis is not None)
        if nan:
            # invalid_mask = np.isnan(nparr)
            # valid_mask = ~invalid_mask
            # valid_values = nparr[~valid_mask]
            min_val = np.nanmin(nparr, axis=axis, keepdims=keepdims)
            max_val = np.nanmax(nparr, axis=axis, keepdims=keepdims)
            mean_ = np.nanmean(nparr, axis=axis, keepdims=keepdims)
            std_  = np.nanstd(nparr, axis=axis, keepdims=keepdims)
        else:
            min_val = nparr.min(axis=axis, keepdims=keepdims)
            max_val = nparr.max(axis=axis, keepdims=keepdims)
            mean_ = nparr.mean(axis=axis, keepdims=keepdims)
            std_  = nparr.std(axis=axis, keepdims=keepdims)

        # Note:
        # the first central moment is 0
        # the first raw moment is the mean
        # the second central moment is the variance
        # the third central moment is the skweness * var ** 3
        # the fourth central moment is the kurtosis * var ** 4
        # the third standardized moment is the skweness
        # the fourth standardized moment is the kurtosis

        if quantile:
            # if not ub.iterable(quantile):
            if quantile is True:
                quantile == 'auto'

            if quantile == 'auto':
                quantile = [0.25, 0.50, 0.75]

        if True:
            stats['mean'] = np.float32(mean_)
            stats['std'] = np.float32(std_)
        if extreme:
            stats['min'] = np.float32(min_val)
            stats['max'] = np.float32(max_val)
        if quantile:
            if quantile == 'auto':
                quantile = [0.25, 0.50, 0.75]
            quant_values = np.quantile(nparr, quantile, axis=axis)
            quant_keys = ['q_{:0.2f}'.format(q) for q in quantile]
            for k, v in zip(quant_keys, quant_values):
                stats[k] = v
        if n_extreme:
            # number of entries with min/max val
            nMin = np.sum(nparr == min_val, axis=axis, keepdims=keepdims)
            nMax = np.sum(nparr == max_val, axis=axis, keepdims=keepdims)
            nMin = nMin.astype(int)
            nMax = nMax.astype(int)
            if nMax.size == 1:
                nMax = nMax.ravel()[0]
            if nMin.size == 1:
                nMin = nMin.ravel()[0]
            stats['nMin'] = nMin
            stats['nMax'] = nMax
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


class NoSupportError(RuntimeError):
    ...


class RunningStats(ub.NiceRepr):
    """
    Track mean, std, min, and max values over time with constant memory.

    Dynamically records per-element array statistics and can summarized them
    per-element, across channels, or globally.

    TODO:
        - [ ] This may need a few API tweaks and good documentation

    Example:
        >>> import kwarray
        >>> run = kwarray.RunningStats()
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

    def __init__(run, nan_behavior='ignore', check_weights=True):
        """
        Args:
            nan_behavior (str): indicates how we will handle nan values
               * if "ignore" - set weights of nan items to zero.
               * if "propogate" - propogate nans (old behavior).

           check_weights (bool):
               if True, we check the weights for zeros (which can also
               implicitly occur when data has nans). Disabling this check will
               result in faster computation, but it is your responsibility to
               ensure all data passed to update is valid.

        """
        run.raw_max = -np.inf
        run.raw_min = np.inf
        run.raw_total = 0
        run.raw_squares = 0
        run.n = 0
        run.nan_behavior = nan_behavior
        run.check_weights = check_weights
        if run.nan_behavior not in {'ignore', 'propogate'}:
            if run.nan_behavior == 'ignore':
                if run.check_weights:
                    raise ValueError(
                        'To prevent foot-shooting, check weights must be '
                        'initialized to True when nan behavior is ignore'
                    )
            raise KeyError(run.nan_behavior)

    def __nice__(self):
        return '{}'.format(self.shape)

    @property
    def shape(run):
        try:
            return run.raw_total.shape
        except Exception:
            return None

    def update_many(run, data, weights=1):
        """
        Assumes first data axis represents multiple observations

        Example:
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> run = kwarray.RunningStats()
            >>> data = rng.randn(1, 2, 3)
            >>> run.update_many(data)
            >>> print(run.current())
            >>> data = rng.randn(2, 2, 3)
            >>> run.update_many(data)
            >>> print(run.current())
            >>> data = rng.randn(3, 2, 3)
            >>> run.update_many(data)
            >>> print(run.current())
            >>> run.update_many(1000)
            >>> print(run.current())
            >>> assert np.all(run.current()['n'] == 7)

        Example:
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> run = kwarray.RunningStats()
            >>> data = rng.randn(1, 2, 3)
            >>> run.update_many(data.ravel())
            >>> print(run.current())
            >>> data = rng.randn(2, 2, 3)
            >>> run.update_many(data.ravel())
            >>> print(run.current())
            >>> data = rng.randn(3, 2, 3)
            >>> run.update_many(data.ravel())
            >>> print(run.current())
            >>> run.update_many(1000)
            >>> print(run.current())
            >>> assert np.all(run.current()['n'] == 37)
        """
        data = np.asarray(data)
        if run.nan_behavior == 'ignore':
            weights = weights * (~np.isnan(data)).astype(float)
        elif run.nan_behavior == 'propogate':
            ...
        else:
            raise AssertionError('should not be here')
        has_ignore_items = False
        if ub.iterable(weights):
            ignore_flags = (weights == 0)
            has_ignore_items = np.any(ignore_flags)

        if has_ignore_items:
            data = data.copy()
            # Replace the bad value with somehting sensible for each operation.
            data[ignore_flags] = 0

            # Multiply data by weights
            w_data = data * weights

            run.raw_total += w_data.sum(axis=0)
            run.raw_squares += (w_data ** 2).sum(axis=0)
            data[ignore_flags] = -np.inf
            run.raw_max = np.maximum(run.raw_max, data.max(axis=0))
            data[ignore_flags] = +np.inf
            run.raw_min = np.minimum(run.raw_min, data.min(axis=0))
        else:
            w_data = data * weights
            run.raw_total += w_data.sum(axis=0)
            run.raw_squares += (w_data ** 2).sum(axis=0)
            run.raw_max = np.maximum(run.raw_max, data.max(axis=0))
            run.raw_min = np.minimum(run.raw_min, data.min(axis=0))
        run.n += weights.sum(axis=0)

    def update(run, data, weights=1):
        """
        Updates statistics across all data dimensions on a per-element basis

        Example:
            >>> import kwarray
            >>> data = np.full((7, 5), fill_value=1.3)
            >>> weights = np.ones((7, 5), dtype=np.float32)
            >>> run = kwarray.RunningStats()
            >>> run.update(data, weights=1)
            >>> run.update(data, weights=weights)
            >>> rng = np.random
            >>> weights[rng.rand(*weights.shape) > 0.5] = 0
            >>> run.update(data, weights=weights)

        Example:
            >>> import kwarray
            >>> run = kwarray.RunningStats()
            >>> data = np.array([[1, np.nan, np.nan], [0, np.nan, 1.]])
            >>> run.update(data)
            >>> print('current = {}'.format(ub.repr2(run.current(), nl=1)))
            >>> print('summary(axis=None) = {}'.format(ub.repr2(run.summarize(), nl=1)))
            >>> print('summary(axis=1) = {}'.format(ub.repr2(run.summarize(axis=1), nl=1)))
            >>> print('summary(axis=0) = {}'.format(ub.repr2(run.summarize(axis=0), nl=1)))
            >>> data = np.array([[2, 0, 1], [0, 1, np.nan]])
            >>> run.update(data)
            >>> data = np.array([[3, 1, 1], [0, 1, np.nan]])
            >>> run.update(data)
            >>> data = np.array([[4, 1, 1], [0, 1, 1.]])
            >>> run.update(data)
            >>> print('----')
            >>> print('current = {}'.format(ub.repr2(run.current(), nl=1)))
            >>> print('summary(axis=None) = {}'.format(ub.repr2(run.summarize(), nl=1)))
            >>> print('summary(axis=1) = {}'.format(ub.repr2(run.summarize(axis=1), nl=1)))
            >>> print('summary(axis=0) = {}'.format(ub.repr2(run.summarize(axis=0), nl=1)))
        """
        if run.nan_behavior == 'ignore':
            weights = weights * (~np.isnan(data)).astype(float)
        elif run.nan_behavior == 'propogate':
            ...
        else:
            raise AssertionError('should not be here')

        has_ignore_items = False
        if ub.iterable(weights):
            ignore_flags = (weights == 0)
            has_ignore_items = np.any(ignore_flags)

        if has_ignore_items:
            data = data.copy()
            # Replace the bad value with somehting sensible for each operation.
            data[ignore_flags] = 0

            # Multiply data by weights
            w_data = data * weights

            run.raw_total += w_data
            run.raw_squares += w_data ** 2
            data[ignore_flags] = -np.inf
            run.raw_max = np.maximum(run.raw_max, data)
            data[ignore_flags] = +np.inf
            run.raw_min = np.minimum(run.raw_min, data)
        else:
            w_data = data * weights
            run.raw_total += w_data
            run.raw_squares += w_data ** 2
            run.raw_max = np.maximum(run.raw_max, data)
            run.raw_min = np.minimum(run.raw_min, data)
        run.n += weights

    def _sumsq_std(run, total, squares, n):
        """
        Sum of squares method to compute standard deviation
        """
        numer = (n * squares - total ** 2)
        # FIXME: this isn't exactly correct when we have fractional weights.
        # Integer weights should be ok. I suppose it really is
        # what "type" of weights they are (see numpy weighted quantile
        # discussion)
        denom = (n * (n - 1.0))
        std = np.sqrt(numer / denom)
        return std

    def summarize(run, axis=None, keepdims=True):
        """
        Compute summary statistics across a one or more dimension

        Args:
            axis (int | List[int] | None | NoParamType):
                axis or axes to summarize over,
                if None, all axes are summarized.
                if ub.NoParam, no axes are summarized the current result is
                returned.

            keepdims (bool, default=True):
                if False removes the dimensions that are summarized over

        Returns:
            Dict: containing minimum, maximum, mean, std, etc..

        Raises:
            NoSupportError : if update was never called with valid data

        Example:
            >>> # Test to make sure summarize works across different shapes
            >>> base = np.array([1, 1, 1, 1, 0, 0, 0, 1])
            >>> run0 = RunningStats()
            >>> for _ in range(3):
            >>>     run0.update(base.reshape(8, 1))
            >>> run1 = RunningStats()
            >>> for _ in range(3):
            >>>     run1.update(base.reshape(4, 2))
            >>> run2 = RunningStats()
            >>> for _ in range(3):
            >>>     run2.update(base.reshape(2, 2, 2))
            >>> #
            >>> # Summarizing over everything should be exactly the same
            >>> s0N = run0.summarize(axis=None, keepdims=0)
            >>> s1N = run1.summarize(axis=None, keepdims=0)
            >>> s2N = run2.summarize(axis=None, keepdims=0)
            >>> #assert ub.util_indexable.indexable_allclose(s0N, s1N, rel_tol=0.0, abs_tol=0.0)
            >>> #assert ub.util_indexable.indexable_allclose(s1N, s2N, rel_tol=0.0, abs_tol=0.0)
            >>> assert s0N['mean'] == 0.625
        """
        with np.errstate(divide='ignore'):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'invalid value encountered', category=RuntimeWarning)
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
                    if np.all(run.n <= 0):
                        raise NoSupportError('No statistics have been accumulated')
                    total   = run.raw_total.sum(axis=axis, keepdims=keepdims)
                    squares = run.raw_squares.sum(axis=axis, keepdims=keepdims)
                    maxi    = run.raw_max.max(axis=axis, keepdims=keepdims)
                    mini    = run.raw_min.min(axis=axis, keepdims=keepdims)
                    if not hasattr(run.raw_total, 'shape'):
                        n = run.n
                    elif axis is None:
                        n = (run.n * np.ones_like(run.raw_total)).sum(axis=axis, keepdims=keepdims)
                        # n = run.n * np.prod(run.raw_total.shape)
                    else:
                        n = (run.n * np.ones_like(run.raw_total)).sum(axis=axis, keepdims=keepdims)
                        # n = run.n * np.prod(np.take(run.raw_total.shape, axis))

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
