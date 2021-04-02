# -* -coding: utf-8 -*-
"""
Fast 32-bit random functions for numpy as of 2018. (More recent versions of
numpy may have these natively supported).
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def uniform(low=0.0, high=1.0, size=None, dtype=np.float32, rng=np.random):
    """
    Draws float32 samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).

    Args:
        low (float, default=0.0):
            Lower boundary of the output interval.  All values generated will
            be greater than or equal to low.

        high (float, default=1.0):
            Upper boundary of the output interval.  All values generated will
            be less than high.

        size (int | Tuple[int], default=None):
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``low`` and ``high`` are both scalars.
            Otherwise, ``np.broadcast(low, high).size`` samples are drawn.

        dtype (type): either np.float32 or np.float64

        rng (numpy.random.RandomState): underlying random state

    Returns:
        ndarray[dtype]: normally distributed random numbers with chosen dtype

    Benchmark:
        >>> from timerit import Timerit
        >>> import kwarray
        >>> size = (300, 300, 3)
        >>> for timer in Timerit(100, bestof=10, label='dtype=np.float32'):
        >>>     rng = kwarray.ensure_rng(0)
        >>>     with timer:
        >>>         ours = standard_normal(size, rng=rng, dtype=np.float32)
        >>> # Timed best=4.705 ms, mean=4.75 ± 0.085 ms for dtype=np.float32
        >>> for timer in Timerit(100, bestof=10, label='dtype=np.float64'):
        >>>     rng = kwarray.ensure_rng(0)
        >>>     with timer:
        >>>         theirs = standard_normal(size, rng=rng, dtype=np.float64)
        >>> # Timed best=9.327 ms, mean=9.794 ± 0.4 ms for rng.np.float64
    """
    if dtype is np.float32:
        return uniform32(low, high, size, rng)
    elif dtype is np.float64:
        return rng.uniform(low, high, size)
    else:
        raise ValueError('dtype = {!r}'.format(dtype))


def standard_normal(size, mean=0, std=1, dtype=float, rng=np.random):
    """
    Draw samples from a standard Normal distribution with a specified mean and
    standard deviation.

    Args:
        size (int | Tuple[int, *int]) : shape of the returned ndarray
        mean (float, default=0): mean of the normal distribution
        std (float, default=1): standard deviation of the normal distribution
        dtype (type): either np.float32 or np.float64
        rng (numpy.random.RandomState): underlying random state

    Returns:
        ndarray[dtype]: normally distributed random numbers with chosen dtype

    Benchmark:
        >>> from timerit import Timerit
        >>> import kwarray
        >>> size = (300, 300, 3)
        >>> for timer in Timerit(100, bestof=10, label='dtype=np.float32'):
        >>>     rng = kwarray.ensure_rng(0)
        >>>     with timer:
        >>>         ours = standard_normal(size, rng=rng, dtype=np.float32)
        >>> # Timed best=4.705 ms, mean=4.75 ± 0.085 ms for dtype=np.float32
        >>> for timer in Timerit(100, bestof=10, label='dtype=np.float64'):
        >>>     rng = kwarray.ensure_rng(0)
        >>>     with timer:
        >>>         theirs = standard_normal(size, rng=rng, dtype=np.float64)
        >>> # Timed best=9.327 ms, mean=9.794 ± 0.4 ms for rng.np.float64
    """
    if dtype is np.float32:
        return standard_normal32(size, mean, std, rng)
    elif dtype is np.float64:
        return standard_normal64(size, mean, std, rng)
    else:
        raise ValueError('dtype = {!r}'.format(dtype))


def standard_normal32(size, mean=0, std=1, rng=np.random):
    """
    Fast normally distributed random variables using the Box–Muller transform

    The difference between this function and
    :func:`numpy.random.standard_normal` is that we use float32 arrays in the
    backend instead of float64.  Halving the amount of bits that need to be
    manipulated can significantly reduce the execution time, and 32-bit
    precision is often good enough.

    Args:
        size (int | Tuple[int, *int]) : shape of the returned ndarray
        mean (float, default=0): mean of the normal distribution
        std (float, default=1): standard deviation of the normal distribution
        rng (numpy.random.RandomState): underlying random state

    Returns:
        ndarray[float32]: normally distributed random numbers

    References:
        https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

    SeeAlso:
        * standard_normal
        * standard_normal64

    Example:
        >>> import scipy
        >>> import scipy.stats
        >>> pts = 1000
        >>> # Our numbers are normally distributed with high probability
        >>> rng = np.random.RandomState(28041990)
        >>> ours_a = standard_normal32(pts, rng=rng)
        >>> ours_b = standard_normal32(pts, rng=rng) + 2
        >>> ours = np.concatenate((ours_a, ours_b))  # numerical stability?
        >>> p = scipy.stats.normaltest(ours)[1]
        >>> print('Probability our data is non-normal is: {:.4g}'.format(p))
        Probability our data is non-normal is: 1.573e-14
        >>> rng = np.random.RandomState(28041990)
        >>> theirs_a = rng.standard_normal(pts)
        >>> theirs_b = rng.standard_normal(pts) + 2
        >>> theirs = np.concatenate((theirs_a, theirs_b))
        >>> p = scipy.stats.normaltest(theirs)[1]
        >>> print('Probability their data is non-normal is: {:.4g}'.format(p))
        Probability their data is non-normal is: 3.272e-11

    Example:
        >>> pts = 1000
        >>> rng = np.random.RandomState(28041990)
        >>> ours = standard_normal32(pts, mean=10, std=3, rng=rng)
        >>> assert np.abs(ours.std() - 3.0) < 0.1
        >>> assert np.abs(ours.mean() - 10.0) < 0.1

    Example:
        >>> # Test an even and odd numbers of points
        >>> assert standard_normal32(3).shape == (3,)
        >>> assert standard_normal32(2).shape == (2,)
        >>> assert standard_normal32(1).shape == (1,)
        >>> assert standard_normal32(0).shape == (0,)
        >>> assert standard_normal32((3, 1)).shape == (3, 1)
        >>> assert standard_normal32((3, 0)).shape == (3, 0)
    """
    # return rng.standard_normal(size) * std + mean
    # The integer and float dtype must have the same number of bits
    dtype = np.float32
    int_dtype = np.uint32
    MAX_INT = np.core.getlimits.iinfo(int_dtype).max

    # Preallocate output
    out = np.empty(size, dtype=dtype)
    total = out.size
    n = int(np.ceil(total / 2))

    # Generate uniform-01 random numbers that we will transform
    u12 = rng.randint(1, MAX_INT - 1, size=n * 2, dtype=int_dtype).astype(dtype)
    np.divide(u12, dtype(MAX_INT), out=u12)

    u1 = u12[:n]
    u2 = u12[n:]
    # Box–Muller transform transforms two uniformly distributed values
    # into normally distributed random values. Because of the way it works
    # we generate two sets of uniform random values, and then
    flat_out = out.ravel()

    # The following logic is equivalent to:
    #   R = std * np.sqrt(-2 * np.log(u1))
    #   Theta = 2 * np.pi * u2
    #   flat_out[0:n] = R * np.cos(Theta) + mean
    #   flat_out[-n:] = R * np.sin(Theta) + mean
    np.log(u1, out=u1)
    np.multiply(-2 * (std ** 2), u1, out=u1)
    R = np.sqrt(u1, out=u1)

    Theta = np.multiply(2 * np.pi, u2, out=u2)

    out1 = flat_out[0:n]
    out2 = flat_out[-n:]
    np.cos(Theta, out=out1)
    np.sin(Theta, out=out2)

    np.multiply(R, out1, out=out1)
    np.multiply(R, out2, out=out2)

    if mean != 0:
        np.add(out, mean, out=out)
    return out


def standard_normal64(size, mean=0, std=1, rng=np.random):
    """
    Simple wrapper around rng.standard_normal to make an API compatible with
    :func:`standard_normal32`.

    Args:
        size (int | Tuple[int, *int]) : shape of the returned ndarray
        mean (float, default=0): mean of the normal distribution
        std (float, default=1): standard deviation of the normal distribution
        rng (numpy.random.RandomState): underlying random state

    Returns:
        ndarray[float64]: normally distributed random numbers

    SeeAlso:
        * standard_normal
        * standard_normal32

    Example:
        >>> pts = 1000
        >>> rng = np.random.RandomState(28041994)
        >>> out = standard_normal64(pts, mean=10, std=3, rng=rng)
        >>> assert np.abs(out.std() - 3.0) < 0.1
        >>> assert np.abs(out.mean() - 10.0) < 0.1
    """
    out = rng.standard_normal(size)
    if std != 1:
        np.multiply(out, std, out=out)
    if mean != 0:
        np.add(out, mean, out=out)
    return out


def uniform32(low=0.0, high=1.0, size=None, rng=np.random):
    """
    Draws float32 samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).

    Args:
        low (float, default=0.0):
            Lower boundary of the output interval.  All values generated will
            be greater than or equal to low.

        high (float, default=1.0):
            Upper boundary of the output interval.  All values generated will
            be less than high.

        size (int | Tuple[int], default=None):
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``low`` and ``high`` are both scalars.
            Otherwise, ``np.broadcast(low, high).size`` samples are drawn.

    Example:
        >>> rng = np.random.RandomState(0)
        >>> uniform32(low=0.0, high=1.0, size=None, rng=rng)
        0.5488...
        >>> uniform32(low=0.0, high=1.0, size=2000, rng=rng).sum()
        1004.94...
        >>> uniform32(low=-10, high=10.0, size=2000, rng=rng).sum()
        202.44...

    Benchmark:
        >>> from timerit import Timerit
        >>> import kwarray
        >>> size = 512 * 512
        >>> for timer in Timerit(100, bestof=10, label='theirs: dtype=np.float64'):
        >>>     rng = kwarray.ensure_rng(0)
        >>>     with timer:
        >>>         theirs = rng.uniform(size=size)
        >>> for timer in Timerit(100, bestof=10, label='theirs: dtype=np.float32'):
        >>>     rng = kwarray.ensure_rng(0)
        >>>     with timer:
        >>>         theirs = rng.rand(size).astype(np.float32)
        >>> for timer in Timerit(100, bestof=10, label='ours: dtype=np.float32'):
        >>>     rng = kwarray.ensure_rng(0)
        >>>     with timer:
        >>>         ours = uniform32(size=size)
    """
    if size is None:
        out = np.float32(rng.uniform(low, high))
    else:
        total = size if isinstance(size, int) else np.prod(size)
        if total < 1764:
            # For smaller sizes simply casting to float32 is faster
            out = rng.uniform(low, high, size).astype(np.float32)
        else:
            # This is only faster for large sizes.
            # The cuttoff size seems to be (42 * 42).
            dtype = np.float32
            int_dtype = np.uint32
            MAX_INT = np.core.getlimits.iinfo(int_dtype).max
            # Generate uniform-01 random numbers that we will transform
            out = rng.randint(0, MAX_INT - 1, size=size, dtype=int_dtype).astype(dtype)
            if low == 0 and high == 1.0:
                np.divide(out, dtype(MAX_INT), out=out)
            else:
                extent = dtype(MAX_INT) / (high - low)
                np.divide(out, extent, out=out)
                np.add(out, low, out=out)
    return out
