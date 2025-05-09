"""
Currently just defines "stats_dict", which is a nice way to gather multiple
numeric statistics (e.g. max, min, median, mode, arithmetic-mean,
geometric-mean, standard-deviation, etc...) about data in an array.
"""
import warnings
import collections
import numpy as np
import ubelt as ub
import sys


# Maybe name or alias or use in kwarray.describe?
def stats_dict(inputs, axis=None, nan=False, sum=False, extreme=True,
               n_extreme=False, median=False, shape=True, size=False,
               quantile='auto'):
    """
    Describe statistics about an input array

    Args:
        inputs (ArrayLike): set of values to get statistics of
        axis (int): if ``inputs`` is ndarray then this specifies the axis
        nan (bool): report number of nan items (TODO: rename to skipna)
        sum (bool): report sum of values
        extreme (bool): report min and max values
        n_extreme (bool): report extreme value frequencies
        median (bool): report median
        size (bool): report array size
        shape (bool): report array shape
        quantile (str | bool | List[float]):
            defaults to 'auto'. Can also be a list of quantiles to compute.
            if truthy computes quantiles.

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
        >>> result = str(ub.urepr(stats, nl=1, precision=4, with_dtype=True))
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
        >>> result = str(ub.urepr(stats, nl=0, precision=1, strkeys=True))
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
        >>>     print('params = {}'.format(ub.urepr(params, nl=1)))
        >>>     print('stats = {}'.format(ub.urepr(stats, nl=1)))

    Ignore:
        import kwarray
        inputs = np.random.rand(3, 2, 1)
        stats = kwarray.stats_dict(inputs, axis=2, nan=True, quantile='auto')
    """
    stats = collections.OrderedDict([])
    # only use torch if its already in use
    torch = sys.modules.get('torch', None)

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
            # FIXME allow nans
            if nan:
                _quantile = nanquantile
            else:
                # Prefer numpy implementation if available
                _quantile = getattr(np, 'nanquantile', nanquantile)
            quant_values = _quantile(nparr, quantile, axis=axis)
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


def nanquantile(a, q, axis=None, interpolation='linear'):
    """
    Compute the q-th quantile of the data along the specified axis,
    ignoring NaN values.

    Args:
        a : array_like
            Input array or object that can be converted to an array.
        q : float or array-like of float
            Quantile(s) to compute, which must be between 0 and 1 inclusive.
        axis : int or None, optional
            Axis along which the quantiles are computed. The default is None,
            in which case the array is flattened.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional
            This optional parameter specifies the interpolation method to use when the
            desired quantile lies between two data points `i` and `j`:
              * 'linear': i + (j - i) * fraction, where fraction is the fractional part of the index.
              * 'lower': i.
              * 'higher': j.
              * 'nearest': i or j, whichever is nearest.
              * 'midpoint': (i + j) / 2.
            The default is 'linear'.

    Returns:
        quantile : float or ndarray
            If `q` is a single quantile and `axis=None`, then the result is a float.
            Otherwise, the result is an array with the same shape as `q` (if `axis` is None)
            or with the axis along which the quantiles were computed replaced by the shape of `q`.

    References:
        https://chatgpt.com/c/67ef1ed7-e9a0-8002-ba8e-abac0ab390b4

    Example:
        >>> from kwarray.util_averages import *  # NOQA
        >>> data = np.array([[1, 2, np.nan, 4],
        >>>                  [5, np.nan, 7, 8],
        >>>                  [9, 10, 11, 12]])
        >>> # Compute the 50th percentile (median) ignoring NaNs, along axis=0.
        >>> medians = nanquantile(data, 0.5, axis=0)
        >>> print("Medians along axis 0:", medians)
        >>> # Compute the 25th and 75th percentiles along axis=1.
        >>> percentiles = nanquantile(data, [0.25, 0.75], axis=1)
        >>> print("25th and 75th percentiles along axis 1:\n", percentiles)
        >>> # Compute the median ignoring NaNs over both axes (flattened) using a tuple.
        >>> median_all = nanquantile(data, 0.5, axis=(0, 1))
        >>> print("Median over all elements:", median_all)
        >>> # Test generic axes
        >>> nanquantile(np.random.rand(5,3,2,3), 0.5, axis=(0, 1))
    """
    a = np.asarray(a)

    # Helper function for computing quantile on a 1D array while ignoring NaNs.
    def quantile_ignore_nan(arr):
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            if np.isscalar(q):
                return np.nan
            else:
                return np.full(np.shape(q), np.nan)
        return np.quantile(valid, q, interpolation=interpolation)

    # Case 1: When axis is None, flatten the array.
    if axis is None:
        valid = a[~np.isnan(a)]
        if valid.size == 0:
            return np.nan if np.isscalar(q) else np.full(np.shape(q), np.nan)
        return np.quantile(valid, q, interpolation=interpolation)

    # Case 2: When axis is an integer.
    if isinstance(axis, int):
        return np.apply_along_axis(quantile_ignore_nan, axis, a)

    # Case 3: When axis is a tuple.
    # Normalize the tuple to ensure sorted order.
    axes = tuple(sorted(axis))
    # Determine axes that are NOT being reduced.
    remaining_axes = tuple(i for i in range(a.ndim) if i not in axes)
    # Move the axes to reduce to the front.
    new_order = axes + remaining_axes
    a_moved = np.moveaxis(a, new_order, range(a.ndim))

    # Collapse the axes to be reduced into one axis.
    n_reduce = np.prod(a_moved.shape[:len(axes)])
    new_shape = (n_reduce,) + a_moved.shape[len(axes):]
    a_collapsed = a_moved.reshape(new_shape)

    # Apply the 1D quantile function along the first axis (which is the combined axis).
    result = np.apply_along_axis(quantile_ignore_nan, 0, a_collapsed)

    # The shape of result depends on whether q is scalar or array:
    # - If q is scalar, result's shape is the same as the "remaining" dimensions.
    # - If q is an array, the new axis for quantiles is prepended.

    # Apply the function along the specified axis.
    return result


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
        >>> print('current-ave = ' + ub.urepr(run.summarize(axis=ub.NoParam), nl=2, precision=3))
        >>> # Average over channels (keeps spatial dims separate)
        >>> print('chann-ave(k=1) = ' + ub.urepr(run.summarize(axis=0), nl=2, precision=3))
        >>> print('chann-ave(k=0) = ' + ub.urepr(run.summarize(axis=0, keepdims=0), nl=2, precision=3))
        >>> # Average over spatial dims (keeps channels separate)
        >>> print('spatial-ave(k=1) = ' + ub.urepr(run.summarize(axis=(1, 2)), nl=2, precision=3))
        >>> print('spatial-ave(k=0) = ' + ub.urepr(run.summarize(axis=(1, 2), keepdims=0), nl=2, precision=3))
        >>> # Average over all dims
        >>> print('alldim-ave(k=1) = ' + ub.urepr(run.summarize(axis=None), nl=2, precision=3))
        >>> print('alldim-ave(k=0) = ' + ub.urepr(run.summarize(axis=None, keepdims=0), nl=2, precision=3))
        """

    def __init__(run, nan_policy='omit', check_weights=True, **kwargs):
        """
        Args:
            nan_policy (str): indicates how we will handle nan values
               * if "omit" - set weights of nan items to zero.
               * if "propagate" - propagate nans.
               * if "raise" - then raise a ValueError if nans are given.

           check_weights (bool):
               if True, we check the weights for zeros (which can also
               implicitly occur when data has nans). Disabling this check will
               result in faster computation, but it is your responsibility to
               ensure all data passed to update is valid.
        """
        if len(kwargs):
            if 'nan_behavior' in kwargs:
                nan_behavior = kwargs.pop('nan_behavior', None)
                if nan_behavior == 'ignore':
                    nan_behavior = 'omit'
                nan_policy = nan_behavior
                ub.schedule_deprecation(
                    'kwarray', 'nan_policy', 'use nan_policy instead',
                    deprecate='0.6.10', error='0.7.0', remove='0.8.0')
            if len(kwargs):
                raise ValueError(f'Unsupported arguments: {list(kwargs.keys())}')

        run.raw_max = -np.inf
        run.raw_min = np.inf
        run.raw_total = 0
        run.raw_squares = 0
        run.n = 0
        run.nan_policy = nan_policy
        run.check_weights = check_weights
        if run.nan_policy not in {'omit', 'propogate'}:
            if run.nan_policy == 'omit':
                if run.check_weights:
                    raise ValueError(
                        'To prevent foot-shooting, check weights must be '
                        'initialized to True when nan_policy is omit'
                    )
            raise KeyError(run.nan_policy)

    def __nice__(self):
        return '{}'.format(self.shape)

    def copy(self):
        """
        Return a deep copy
        """
        from copy import deepcopy
        return deepcopy(self)

    @property
    def shape(run):
        try:
            return run.raw_total.shape
        except Exception:
            return None

    def _update_from_other(run, other):
        """
        Combine this runner with another one.
        """
        run.raw_max = np.maximum(run.raw_max, other.raw_max)
        run.raw_min = np.minimum(run.raw_min, other.raw_min)
        run.raw_total = run.raw_total + other.raw_total
        run.raw_squares = run.raw_squares + other.raw_squares
        run.n = run.n + other.n

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

        Example:
            >>> # Case where data are integers
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> run = kwarray.RunningStats()
            >>> data = rng.randint(0, 10, 32)
            >>> weight = rng.randint(0, 10, 32)
            >>> run.update_many(data, weight)
        """
        data = np.asarray(data)
        if run.nan_policy == 'omit':
            weights = weights * (~np.isnan(data)).astype(float)
        elif run.nan_policy == 'propogate':
            ...
        elif run.nan_policy == 'raise':
            if np.any(np.isnan(data)):
                raise ValueError('nan policy is raise')
        else:
            raise AssertionError('should not be here')
        has_ignore_items = False
        if ub.iterable(weights):
            ignore_flags = (weights == 0)
            has_ignore_items = np.any(ignore_flags)

        if has_ignore_items:
            from kwarray.arrayapi import _numpy_dtype_info
            dtype_info = _numpy_dtype_info(data.dtype)

            data = data.copy()
            # Replace the bad value with something sensible for each operation.
            data[ignore_flags] = 0

            # Multiply data by weights
            w_data = data * weights

            run.raw_total += w_data.sum(axis=0)
            run.raw_squares += (w_data ** 2).sum(axis=0)
            data[ignore_flags] = dtype_info.min
            run.raw_max = np.maximum(run.raw_max, data.max(axis=0))
            data[ignore_flags] = dtype_info.max
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
            >>> print('current = {}'.format(ub.urepr(run.current(), nl=1)))
            >>> print('summary(axis=None) = {}'.format(ub.urepr(run.summarize(), nl=1)))
            >>> print('summary(axis=1) = {}'.format(ub.urepr(run.summarize(axis=1), nl=1)))
            >>> print('summary(axis=0) = {}'.format(ub.urepr(run.summarize(axis=0), nl=1)))
            >>> data = np.array([[2, 0, 1], [0, 1, np.nan]])
            >>> run.update(data)
            >>> data = np.array([[3, 1, 1], [0, 1, np.nan]])
            >>> run.update(data)
            >>> data = np.array([[4, 1, 1], [0, 1, 1.]])
            >>> run.update(data)
            >>> print('----')
            >>> print('current = {}'.format(ub.urepr(run.current(), nl=1)))
            >>> print('summary(axis=None) = {}'.format(ub.urepr(run.summarize(), nl=1)))
            >>> print('summary(axis=1) = {}'.format(ub.urepr(run.summarize(axis=1), nl=1)))
            >>> print('summary(axis=0) = {}'.format(ub.urepr(run.summarize(axis=0), nl=1)))
        """
        if run.nan_policy == 'omit':
            weights = weights * (~np.isnan(data)).astype(float)
        elif run.nan_policy == 'propogate':
            ...
        elif run.nan_policy == 'raise':
            if np.any(np.isnan(data)):
                raise ValueError('nan policy is raise')
        else:
            raise AssertionError('should not be here')

        has_ignore_items = False
        if ub.iterable(weights):
            ignore_flags = (weights == 0)
            has_ignore_items = np.any(ignore_flags)

        if has_ignore_items:
            data = data.copy()
            # Replace the bad value with something sensible for each operation.
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
                    else:
                        n = (run.n * np.ones_like(run.raw_total)).sum(axis=axis, keepdims=keepdims)

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

    def collapse(run, axis=None, keepdims=True):
        """
        Reduce the shape of the accumulator array.

        Example:
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> run = kwarray.RunningStats()
            >>> run.update(rng.rand(3, 5))
            >>> new = run.collapse()
        """
        new = RunningStats(nan_policy=run.nan_policy,
                           check_weights=run.check_weights)
        new.raw_total   = run.raw_total.sum(axis=axis, keepdims=keepdims)
        new.raw_squares = run.raw_squares.sum(axis=axis, keepdims=keepdims)
        new.raw_max     = run.raw_max.max(axis=axis, keepdims=keepdims)
        new.raw_min     = run.raw_min.min(axis=axis, keepdims=keepdims)
        if not hasattr(run.raw_total, 'shape'):
            new.n = run.n
        else:
            new.n = (run.n * np.ones_like(run.raw_total)).sum(axis=axis, keepdims=keepdims)
        return new

    def _update_other(run, other):
        """
        Update this RunningStats object with statistics from another one.

        Args:
            other (RunningStats): another RunningStats object to combine with.

        Example:
            >>> import kwarray
            >>> rs1 = kwarray.RunningStats()
            >>> rs2 = kwarray.RunningStats()
            >>> rs1.update(np.array([1, 2, 3]))
            >>> rs2.update(np.array([4, 5, 6]))
            >>> rs3 = rs1.copy()
            >>> rs3._update_other(rs2)
            >>> rs4 = kwarray.RunningStats()
            >>> rs4.update(np.array([1, 2, 3]))
            >>> rs4.update(np.array([4, 5, 6]))
            >>> summary3 = rs3.summarize()
            >>> summary4 = rs4.summarize()
            >>> assert summary4 == summary3
            >>> print('mean:', np.round(summary3['mean'], 2))
            >>> print('std:', np.round(summary3['std'], 2))
            >>> print('n:', summary3['n'])
            mean: [3.5]
            std: [1.87]
            n: [6.]

        Example:
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> rs1 = kwarray.RunningStats()
            >>> rs2 = kwarray.RunningStats()
            >>> rs1.update(rng.rand(3, 5))
            >>> rs1.update(rng.rand(3, 5))
            >>> rs2.update(rng.rand(7, 9))
            >>> rs2.update(rng.rand(7, 9))
            >>> rs3 = rs1.copy()
            >>> rs3 = rs3.collapse()
            >>> rs3._update_other(rs2)
            >>> summary3 = rs3.summarize()
            >>> print('mean:', np.round(summary3['mean'], 2))
            >>> print('std:', np.round(summary3['std'], 2))
            >>> print('n:', summary3['n'])
        """
        if run.shape != other.shape:
            raise ValueError(f'could not be broadcast together with shapes {run.shape} and {other.shape}')
        np.maximum(run.raw_max, other.raw_max, out=run.raw_max)
        np.minimum(run.raw_min, other.raw_min, out=run.raw_min)
        np.add(run.n, other.n, out=run.n)
        np.add(run.raw_total, other.raw_total, out=run.raw_total)
        np.add(run.raw_squares, other.raw_squares, out=run.raw_squares)


def _combine_mean_stds(means, stds, nums=None, axis=None, keepdims=False,
                       bessel=True):
    r"""
    Args:
        means (array): means[i] is the mean of the ith entry to combine

        stds (array): stds[i] is the std of the ith entry to combine

        nums (array | None):
            nums[i] is the number of samples in the ith entry to combine.
            if None, assumes sample sizes are infinite.

        axis (int | Tuple[int] | None):
            axis to combine the statistics over

        keepdims (bool):
            if True return arrays with the same number of dimensions they were
            given in.

        bessel (int):
            Set to 1 to enables bessel correction to unbias the combined std
            estimate.  Only disable if you have the true population means, or
            you think you know what you are doing.

    References:
        https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation

    Sympy:
        >>> # xdoctest: +REQUIRES(env:SHOW_SYMPY)
        >>> # What about the case where we don't know population size of the
        >>> # estimates. We could treat it as a fixed number, or perhaps take the
        >>> # limit as n -> infinity.
        >>> import sympy
        >>> import sympy as sym
        >>> from sympy import symbols, sqrt, limit, IndexedBase, summation
        >>> from sympy import Indexed, Idx, symbols
        >>> means = IndexedBase('m')
        >>> stds = IndexedBase('s')
        >>> nums = IndexedBase('n')
        >>> i = symbols('i', cls=Idx)
        >>> k = symbols('k', cls=Idx)
        >>> #
        >>> combo_mean = symbols('C')
        >>> #
        >>> bessel = 1
        >>> total = summation(nums[i], (i, 1, k))
        >>> combo_mean_expr = summation(nums[i] * means[i], (i, 1, k)) / total
        >>> p1 = summation((nums[i] - bessel) * stds[i], (i, 1, k))
        >>> p2 = summation(nums[i] * ((means[i] - combo_mean) ** 2), (i, 1, k))
        >>> #
        >>> combo_std_expr = sqrt((p1 + p2) / (total - bessel))
        >>> print('------------------------------------')
        >>> print('General Combined Mean / Std Formulas')
        >>> print('C = combined mean')
        >>> print('S = combined std')
        >>> print('------------------------------------')
        >>> print(ub.hzcat(['C = ', sym.pretty(combo_mean_expr, use_unicode=True, use_unicode_sqrt_char=True)]))
        >>> print(ub.hzcat(['S = ', sym.pretty(combo_std_expr, use_unicode=True, use_unicode_sqrt_char=True)]))
        >>> print('')
        >>> print('---------')
        >>> print('Now assuming all sample sizes are the same constant value N')
        >>> print('---------')
        >>> # Now assume all n[i] = N (i.e. a constant value)
        >>> N = symbols('N')
        >>> combo_mean_const_n_expr = combo_mean_expr.copy().xreplace({nums[i]: N})
        >>> combo_std_const_n_expr = combo_std_expr.copy().xreplace({nums[i]: N})
        >>> p1_const_n = p1.copy().xreplace({nums[i]: N})
        >>> p2_const_n = p2.copy().xreplace({nums[i]: N})
        >>> total_const_n = total.copy().xreplace({nums[i]: N})
        >>> #
        >>> print(ub.hzcat(['C = ', sym.pretty(combo_mean_const_n_expr, use_unicode=True, use_unicode_sqrt_char=True)]))
        >>> print(ub.hzcat(['S = ', sym.pretty(combo_std_const_n_expr, use_unicode=True, use_unicode_sqrt_char=True)]))
        >>> #
        >>> print('')
        >>> print('---------')
        >>> print('Take the limit as N -> infinity')
        >>> print('---------')
        >>> #
        >>> # Limit doesnt directly but we can break it into parts
        >>> lim_C = limit(combo_mean_const_n_expr, N, float('inf'))
        >>> lim_p1 = limit(p1_const_n / (total_const_n - bessel), N, float('inf'))
        >>> lim_p2 = limit(p2_const_n / (total_const_n - bessel), N, float('inf'))
        >>> lim_expr = sym.sqrt(lim_p1 + lim_p2)
        >>> print(ub.hzcat(['lim(C, N->inf) = ', sym.pretty(lim_C)]))
        >>> print(ub.hzcat(['lim(S, N->inf) = ', sym.pretty(lim_expr)]))

    Ignore:
        # lim0_p1 = limit(p1 / (total - 1), n, 0)
        # lim0_p2 = limit(p2 / (total - 1), n, 0)
        # lim0_expr = sym.sqrt(lim0_p1 + lim0_p2)
        # print(sym.pretty(lim0_expr))

        kcase = combo_std_expr.subs({k: 2})
        print(sym.pretty(kcase))
        print(sym.pretty(sympy.simplify(kcase)))
        limit(kcase, N, float('inf'))
        limit(combo_std_expr, N, float('inf'))

    Example:
        >>> from kwarray.util_averages import *  # NOQA
        >>> from kwarray.util_averages import _combine_mean_stds
        >>> means = np.array([1.2, 3.2, 4.1])
        >>> stds = np.array([4.2, 0.2, 2.1])
        >>> nums = np.array([10, 100, 10])
        >>> _combine_mean_stds(means, stds, nums)
        >>> means = np.array([1, 2, 3])
        >>> stds = np.array([1, 2, 3])
        >>> #
        >>> nums = np.array([1, 1, 1]) / 3
        >>> print(_combine_mean_stds(means, stds, nums, bessel=True), '- .3 B')
        >>> print(_combine_mean_stds(means, stds, nums, bessel=False), '- .3')
        >>> nums = np.array([1, 1, 1])
        >>> print(_combine_mean_stds(means, stds, nums, bessel=True), '- 1 B')
        >>> print(_combine_mean_stds(means, stds, nums, bessel=False), '- 1')
        >>> nums = np.array([10, 10, 10])
        >>> print(_combine_mean_stds(means, stds, nums, bessel=True), '- 10 B')
        >>> print(_combine_mean_stds(means, stds, nums, bessel=False), '- 10')
        >>> nums = np.array([1000, 1000, 1000])
        >>> print(_combine_mean_stds(means, stds, nums, bessel=True), '- 1000 B')
        >>> print(_combine_mean_stds(means, stds, nums, bessel=False), '- 1000')
        >>> #
        >>> nums = None
        >>> print(_combine_mean_stds(means, stds, nums, bessel=True), '- inf B')
        >>> print(_combine_mean_stds(means, stds, nums, bessel=False), '- inf')

    Example:
        >>> from kwarray.util_averages import *  # NOQA
        >>> from kwarray.util_averages import _combine_mean_stds
        >>> means = np.stack([np.array([1.2, 3.2, 4.1])] * 100, axis=0)
        >>> stds = np.stack([np.array([4.2, 0.2, 2.1])] * 100, axis=0)
        >>> nums = np.stack([np.array([10, 100, 10])] * 100, axis=0)
        >>> cm1, cs1, _ = _combine_mean_stds(means, stds, nums, axis=None)
        >>> print('combo_mean = {}'.format(ub.urepr(cm1, nl=1)))
        >>> print('combo_std  = {}'.format(ub.urepr(cs1, nl=1)))
        >>> means = np.stack([np.array([1.2, 3.2, 4.1])] * 1, axis=0)
        >>> stds = np.stack([np.array([4.2, 0.2, 2.1])] * 1, axis=0)
        >>> nums = np.stack([np.array([10, 100, 10])] * 1, axis=0)
        >>> cm2, cs2, _ = _combine_mean_stds(means, stds, nums, axis=None)
        >>> print('combo_mean = {}'.format(ub.urepr(cm2, nl=1)))
        >>> print('combo_std  = {}'.format(ub.urepr(cs2, nl=1)))
        >>> means = np.stack([np.array([1.2, 3.2, 4.1])] * 5, axis=0)
        >>> stds = np.stack([np.array([4.2, 0.2, 2.1])] * 5, axis=0)
        >>> nums = np.stack([np.array([10, 100, 10])] * 5, axis=0)
        >>> cm3, cs3, combo_num = _combine_mean_stds(means, stds, nums, axis=1)
        >>> print('combo_mean = {}'.format(ub.urepr(cm3, nl=1)))
        >>> print('combo_std  = {}'.format(ub.urepr(cs3, nl=1)))
        >>> assert np.allclose(cm1, cm2) and np.allclose(cm2,  cm3)
        >>> assert not np.allclose(cs1, cs2)
        >>> assert np.allclose(cs2, cs3)

    Example:
        >>> from kwarray.util_averages import *  # NOQA
        >>> from kwarray.util_averages import _combine_mean_stds
        >>> means = np.random.rand(2, 3, 5, 7)
        >>> stds = np.random.rand(2, 3, 5, 7)
        >>> nums = (np.random.rand(2, 3, 5, 7) * 10) + 1
        >>> cm, cs, cn = _combine_mean_stds(means, stds, nums, axis=1, keepdims=1)
        >>> assert cm.shape == cs.shape == cn.shape
        >>> print(f'cm.shape={cm.shape}')
        >>> cm, cs, cn = _combine_mean_stds(means, stds, nums, axis=(0, 2), keepdims=1)
        >>> assert cm.shape == cs.shape == cn.shape
        >>> print(f'cm.shape={cm.shape}')
        >>> cm, cs, cn = _combine_mean_stds(means, stds, nums, axis=(1, 3), keepdims=1)
        >>> assert cm.shape == cs.shape == cn.shape
        >>> print(f'cm.shape={cm.shape}')
        >>> cm, cs, cn = _combine_mean_stds(means, stds, nums, axis=None)
        >>> assert cm.shape == cs.shape == cn.shape
        >>> print(f'cm.shape={cm.shape}')
        cm.shape=(2, 1, 5, 7)
        cm.shape=(1, 3, 1, 7)
        cm.shape=(2, 1, 5, 1)
        cm.shape=()
    """
    if nums is None:
        # Assume the limit as nums -> infinite
        combo_num = None
        combo_mean = np.average(means, weights=None, axis=axis)
        combo_mean = _postprocess_keepdims(means, combo_mean, axis)
        numer_p1 = stds.sum(axis=axis, keepdims=1)
        numer_p2 = (((means - combo_mean) ** 2)).sum(axis=axis, keepdims=1)
        numer = numer_p1 + numer_p2
        denom = len(stds)
        combo_std = np.sqrt(numer / denom)
    else:
        combo_num = nums.sum(axis=axis, keepdims=1)
        weights = nums / combo_num
        combo_mean = np.average(means, weights=weights, axis=axis)
        combo_mean = _postprocess_keepdims(means, combo_mean, axis)
        numer_p1 = (np.maximum(nums - bessel, 0) * stds).sum(axis=axis, keepdims=1)
        numer_p2 = (nums * ((means - combo_mean) ** 2)).sum(axis=axis, keepdims=1)
        numer = numer_p1 + numer_p2
        denom = np.maximum(combo_num - bessel, 0)
        combo_std = np.sqrt(numer / denom)

    if not keepdims:
        indexer = _no_keepdim_indexer(combo_mean, axis)
        combo_mean = combo_mean[indexer]
        combo_std = combo_std[indexer]
        if combo_num is not None:
            combo_num = combo_num[indexer]

    return combo_mean, combo_std, combo_num


def _no_keepdim_indexer(result, axis):
    """
    Computes an indexer to postprocess a result with keepdims=True
    that will modify the result as if keepdims=False
    """
    if axis is None:
        indexer = [0] * len(result.shape)
    else:
        indexer = [slice(None)] * len(result.shape)
        if isinstance(axis, (list, tuple)):
            for a in axis:
                indexer[a] = 0
        else:
            indexer[axis] = 0
    indexer = tuple(indexer)
    return indexer


def _postprocess_keepdims(original, result, axis):
    """
    Can update the result of a function that does not support keepdims to look
    as if keepdims was supported.
    """
    # Newer versions of numpy have keepdims on more functions
    if axis is not None:
        expander = [slice(None)] * len(original.shape)
        if isinstance(axis, (list, tuple)):
            for a in axis:
                expander[a] = None
        else:
            expander[axis] = None
        result = result[tuple(expander)]
    else:
        expander = [None] * len(original.shape)
        result = np.array(result)[tuple(expander)]
    return result


class RunningHistogram:
    """
    Running histogram accumulator for streaming data.

    NOTE:
        This class is still experimental and subject to change.

    TODO:
        - [ ] We are going to combine implementations for continuous and
              discrete cases and make them interchangeable. The continuous case
              is more general and can subsume the discrete case, but the
              discrete case should be more efficient and we should use that
              implementation if we can, until we can't and then we convert to
              continuous.

        - [ ] Add a method to group bins and combine the counts within them
              to downsample the histogram resolution.

        - [ ] Dynamic binning.

    Efficiently bins incoming data into fixed bins and maintains a cumulative count.
    Supports NaN-safe policies.

    Args:
        edges (array-like): 1D array of monotonically increasing bin edges of length M+1
            defining M bins.
        nan_policy (str): How to handle NaN or infinite values in data:
            - 'omit' (default): ignore NaNs/infs (they are not counted).
            - 'raise': raise ValueError if NaN/inf encountered.

    References:
        https://chatgpt.com/c/681bf8a1-5f24-8002-8590-f3fd969159a4

    Attributes:
        edges (ndarray): The edges provided.
        accum (ndarray): 1D array of length M with counts per bin.
        nan_policy (str): Policy for NaN/infinite values.

    Example:
        >>> # Basic usage
        >>> from kwarray.util_averages import *  # NOQA
        >>> import kwarray
        >>> edges = [0, 1, 2, 3]
        >>> self = rh = kwarray.RunningHistogram(edges)
        >>> data = np.array([0.1, 1.5, 2.5, 0.9])
        >>> rh.update(data)
        >>> rh.accum
        array([2, 1, 1])

        >>> # Multiple updates accumulate
        >>> rh.update(np.array([1.1, 2.9, 2.0]))
        >>> rh.accum
        array([2, 2, 3])

        >>> # NaN omission
        >>> rh2 = RunningHistogram([0, 1, 2], nan_policy='omit')
        >>> rh2.update([0.5, np.nan, 1.5])
        >>> rh2.accum  # ignores nan
        array([1, 1])

        >>> # Raise on NaN
        >>> import pytest
        >>> rh4 = RunningHistogram([0, 1, 2], nan_policy='raise')
        >>> try:
        ...     rh4.update([0.1, np.nan])
        ... except ValueError:
        ...     print('raised')
        raised

    Example:
        >>> # Image-like ndarray of shape (32, 32, 3)
        >>> # Create a small synthetic RGB image with values in [0, 255]
        >>> from kwarray.util_averages import *  # NOQA
        >>> img = np.zeros((32, 32, 3), dtype=np.uint8)
        >>> img[..., 0] = 50   # Red channel constant 50
        >>> img[..., 1] = 150  # Green channel constant 150
        >>> img[..., 2] = 250  # Blue channel constant 250
        >>> # Bin edges for intensity: [0, 100, 200, 300]
        >>> edges = [0, 100, 200, 300]
        >>> rh_img = RunningHistogram(edges)
        >>> # Updating with the multi-channel image flattens all values
        >>> rh_img.update(img)
        >>> # There are 32*32 pixels per channel => 1024 each
        >>> # Red=50 falls in bin 0, Green=150 in bin 1, Blue=250 in bin 2
        >>> rh_img.accum  # counts for each bin across all channels
        array([1024, 1024, 1024])
    """
    def __init__(self, edges, nan_policy='omit'):
        self.edges = np.asarray(edges, dtype=float)
        if self.edges.ndim != 1 or self.edges.size < 2:
            raise ValueError("edges must be a 1D array of length >=2")
        if not np.all(np.diff(self.edges) > 0):
            raise ValueError("edges must be strictly increasing")
        self.accum = np.zeros(self.edges.size - 1, dtype=np.int64)
        self.nan_policy = nan_policy
        if self.nan_policy not in {'omit', 'raise'}:
            raise ValueError(f"Unknown nan_policy: {nan_policy}")
        self.n_updates = 0
        self.n_total = 0
        self.n_underflow = 0
        self.n_overflow = 0
        self.n_nan = 0

    def update(self, data):  # noqa: C901
        """
        Bin the input data and update the running histogram.

        Args:
            data (array-like): Values to bin.
        """
        arr = np.asarray(data).ravel()
        self.n_updates += 1
        self.n_total += arr.size

        # nan/inf handling
        mask = ~np.isnan(arr)
        if self.nan_policy == 'omit':
            if not mask.all():
                arr = arr[mask]
                self.n_nan += int((~mask).sum())
        elif self.nan_policy == 'raise':
            if not mask.all():
                raise ValueError("Non-finite data encountered with nan_policy='raise'")

        # TODO: make extra computation optional
        HANDLE_OVERFLOW = True
        if HANDLE_OVERFLOW:
            # underflow/overflow
            under = arr < self.edges[0]
            over = arr >= self.edges[-1]
            self.n_underflow += under.sum()
            self.n_overflow += over.sum()
            mid = arr[~(under | over)]
        else:
            mid = arr
        # histogram
        counts, _ = np.histogram(mid, bins=self.edges)
        self.accum += counts

    def _update_other(self, other):
        if not np.allclose(self.edges, other.edges):
            raise ValueError("edges must match to merge")
        if self.nan_policy != other.nan_policy:
            raise ValueError("nan_policy must match to merge")
        self.accum += other.counts
        self.n_updates += other.n_updates
        self.n_total += other.n_total
        self.n_nan += other.n_nan
        self.n_underflow += other.n_underflow
        self.n_overflow += other.n_overflow
        return self

    def __iadd__(self, other):
        return self._update_other(other)


class RunningDiscreteHistogram:
    """
    Discrete histogram using pure NumPy arrays for vectorized updates.

    Users may preallocate a domain of possible values via `bins`.
    New values not in the domain will be automatically appended.

    Tracks:
        bins (ndarray): sorted unique values seen so far
        counts (ndarray): integer counts corresponding to each value
        n_total (int): total items processed
        n_nan (int): count of NaN or infinite values

    Args:
        bins (array-like, optional): initial set of values to preallocate bins.
            Will be sorted and uniqued.
        dtype (data-type, default=float): data type for storing values.

    Methods:
        update(data): vectorized count of new observations
        __iadd__(other): merge two histograms (must have same dtype)
        most_common(n=None): return top-n frequent values

    Example:
        >>> # Preallocate bins for values [0,1,2]
        >>> from kwarray.util_averages import *  # NOQA
        >>> rd = RunningDiscreteHistogram([0,1,2], dtype=int)
        >>> rd.update([0,2,2,3,1,3,3])
        >>> rd.bins.tolist(), rd.counts.tolist()
        ([0, 1, 2, 3], [1, 1, 2, 3])
        >>> rd.n_total, rd.n_nan
        (7, 0)

    Example:
        >>> from kwarray.util_averages import *  # NOQA
        >>> # Test that new bins are added to the histogram as needed.
        >>> self = RunningDiscreteHistogram()
        >>> self.update([1, 2, 1, 1, 3])
        >>> assert np.all(self.bins == [1, 2, 3])
        >>> self.update([9])
        >>> assert np.all(self.bins == [1, 2, 3, 9])
        >>> self.update([9001, 7])
        >>> assert np.all(self.bins == [1, 2, 3, 7, 9, 9001])
    """
    def __init__(self, bins=None, dtype=float):
        self.dtype = dtype
        if bins is not None:
            arr = np.asarray(bins, dtype=self.dtype)
            uniq = np.unique(arr)
            self.bins = uniq
            self.counts = np.zeros(uniq.shape, dtype=int)
        else:
            self.bins = np.zeros((0,), dtype=self.dtype)
            self.counts = np.zeros((0,), dtype=int)
        self.n_total = 0
        self.n_nan = 0

    def update(self, data):
        """
        Vectorized update: count each unique in data and merge into bins.

        Args:
            data (array-like): numeric array of observations.
        """
        arr = np.asarray(data, dtype=self.dtype).ravel()
        self.n_total += arr.size
        mask = np.isfinite(arr)
        self.n_nan += int((~mask).sum())
        good = arr[mask]
        if good.size == 0:
            return
        uniq, cnts = np.unique(good, return_counts=True)
        if self.bins.size > 0:
            # determine which uniq are existing
            isin = np.isin(uniq, self.bins)
            # update existing bins
            if np.any(isin):
                exist_vals = uniq[isin]
                exist_cnts = cnts[isin]
                idxs = np.searchsorted(self.bins, exist_vals)
                self.counts[idxs] += exist_cnts
            # append new bins
            if not np.all(isin):
                new_vals = uniq[~isin]
                new_cnts = cnts[~isin]
                pos = np.searchsorted(self.bins, new_vals)
                self.bins = np.insert(self.bins, pos, new_vals)
                self.counts = np.insert(self.counts, pos, new_cnts)
        else:
            # first data
            self.bins = uniq
            self.counts = cnts.astype(int)

    def __iadd__(self, other):
        """
        Merge another histogram into this one.
        """
        if not isinstance(other, RunningDiscreteHistogram):
            raise ValueError("Can only merge with another RunningDiscreteHistogram")
        if self.dtype != other.dtype:
            raise ValueError("dtype mismatch when merging histograms")
        # merge other's data by updating from its values repeated by counts
        if other.values.size:
            # vectorized: repeat values and call update
            self.update(np.repeat(other.bins, other.counts))
        self.n_total += other.n_total
        self.n_nan += other.n_nan
        return self
