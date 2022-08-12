"""
Utilities related to slicing

References:
    https://stackoverflow.com/questions/41153803/zero-padding-slice-past-end-of-array-in-numpy

TODO:
    - [ ] Could have a kwarray function to expose this inverse slice
          functionality. Also having a top-level call to apply an embedded slice
          would be good.
"""
import ubelt as ub
import numpy as np


def padded_slice(data, slices, pad=None, padkw=None, return_info=False):
    """
    Allows slices with out-of-bound coordinates. Any out of bounds coordinate
    will be sampled via padding.

    Args:
        data (Sliceable): data to slice into. Any channels must be the last dimension.
        slices (slice | Tuple[slice, ...]): slice for each dimensions
        ndim (int): number of spatial dimensions
        pad (List[int|Tuple]): additional padding of the slice
        padkw (Dict): if unspecified defaults to ``{'mode': 'constant'}``
        return_info (bool, default=False): if True, return extra information
            about the transform.

    Note:
        Negative slices have a different meaning here then they usually do.
        Normally, they indicate a wrap-around or a reversed stride, but here
        they index into out-of-bounds space (which depends on the pad mode).
        For example a slice of -2:1 literally samples two pixels to the left of
        the data and one pixel from the data, so you get two padded values and
        one data value.

    SeeAlso:
        embed_slice - finds the embedded slice and padding

    Returns:

        Sliceable:
            data_sliced: subregion of the input data (possibly with padding,
                depending on if the original slice went out of bounds)


        Tuple[Sliceable, Dict] :
            data_sliced : as above

            transform : information on how to return to the original coordinates

                Currently a dict containing:
                    st_dims: a list indicating the low and high space-time
                        coordinate values of the returned data slice.

                The structure of this dictionary mach change in the future

    Example:
        >>> import kwarray
        >>> data = np.arange(5)
        >>> slices = [slice(-2, 7)]

        >>> data_sliced = kwarray.padded_slice(data, slices)
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 1, 2, 3, 4, 0, 0])

        >>> data_sliced = kwarray.padded_slice(data, slices, pad=[(3, 3)])
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0])

        >>> data_sliced = kwarray.padded_slice(data, slice(3, 4), pad=[(1, 0)])
        >>> print(ub.repr2(data_sliced, with_dtype=False))
        np.array([2, 3])

    """
    if isinstance(slices, slice):
        slices = [slices]

    if padkw is None:
        padkw = {
            'mode': 'constant',
        }

    ndim = len(slices)
    data_dims = data.shape[:ndim]

    # separate requested slice into an in-bounds part and a padding part
    data_slice, extra_padding = embed_slice(slices, data_dims, pad=pad)

    # Crop to the data slice, and then extend with requested padding
    data_sliced = apply_embedded_slice(data, data_slice, extra_padding, **padkw)

    if return_info:
        st_dims = [(sl.start - pad_[0], sl.stop + pad_[1])
                   for sl, pad_ in zip(data_slice, extra_padding)]
        # TODO: return a better transform back to the original space
        transform = {
            'st_dims': st_dims,
            'st_offset': [d[0] for d in st_dims]
        }
        return data_sliced, transform
    else:
        return data_sliced


__TODO__ = """

    - [ ] Could have a kwarray function to expose this inverse slice
          functionality. Also having a top-level call to apply an embedded slice
          would be good.

    chip_index = tuple([slice(tl_y, br_y), slice(tl_x, br_x)])
    data_slice, padding = kwarray.embed_slice(chip_index, imdata.shape)
    # TODO: could have a kwarray function to expose this inverse slice
    # functionality. Also having a top-level call to apply an embedded
    # slice would be good
    inverse_slice = (
        slice(padding[0][0], imdata.shape[0] - padding[0][1]),
        slice(padding[1][0], imdata.shape[1] - padding[1][1]),
    )
    chip = kwarray.padded_slice(imdata, chip_index)
    chip = imdata[chip_index]

    fgdata = function(chip)

    # Apply just the data part back to the original
    imdata[tl_y:br_y, tl_x:br_x, :] = fgdata[inverse_slice]
"""


def apply_embedded_slice(data, data_slice, extra_padding, **padkw):
    """
    Apply a precomputed embedded slice.

    This is used as a subroutine in padded_slice.

    Args:
        data (ndarray): data to slice
        data_slice (Tuple[slice]) first output of embed_slice
        extra_padding (Tuple[slice]) second output of embed_slice

    Returns:
        ndarray
    """
    # Get the parts of the image that are in-bounds
    data_clipped = data[data_slice]
    # Apply the padding part
    data_sliced = _apply_padding(data_clipped, extra_padding, **padkw)
    return data_sliced


def _apply_padding(array, pad_width, **padkw):
    """
    Alternative to numpy pad with different short-cut semantics for
    the "pad_width" argument.

    Unlike numpy pad, you must specify a (start, stop) tuple for each
    dimension. The shortcut is that you only need to specify this for the
    leading dimensions. Any unspecified trailing dimension will get an implicit
    (0, 0) padding.

    TODO: does this get exposed as a public function?
    """
    if sum(map(sum, pad_width)) == 0:
        # No padding was requested
        padded = array
    else:
        trailing_dims = len(array.shape) - len(pad_width)
        if trailing_dims > 0:
            pad_width = pad_width + ([(0, 0)] * trailing_dims)
        padded = np.pad(array, pad_width, **padkw)
    return padded


def embed_slice(slices, data_dims, pad=None):
    """
    Embeds a "padded-slice" inside known data dimension.

    Returns the valid data portion of the slice with extra padding for regions
    outside of the available dimension.

    Given a slices for each dimension, image dimensions, and a padding get the
    corresponding slice from the image and any extra padding needed to achieve
    the requested window size.

    TODO:
        - [ ] Add the option to return the inverse slice

    Args:
        slices (Tuple[slice, ...]):
            a tuple of slices for to apply to data data dimension.

        data_dims (Tuple[int, ...]):
            n-dimension data sizes (e.g. 2d height, width)

        pad (int | List[int | Tuple[int, int]]):
            extra pad applied to (start / end) / (both) sides of each slice dim

    Returns:
        Tuple:
            data_slice - Tuple[slice] a slice that can be applied to an array
                with with shape `data_dims`. This slice will not correspond to
                the full window size if the requested slice is out of bounds.
            extra_padding - extra padding needed after slicing to achieve
                the requested window size.

    Example:
        >>> # Case where slice is inside the data dims on left edge
        >>> import kwarray
        >>> slices = (slice(0, 10), slice(0, 10))
        >>> data_dims  = [300, 300]
        >>> pad        = [10, 5]
        >>> a, b = kwarray.embed_slice(slices, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = (slice(0, 20, None), slice(0, 15, None))
        extra_padding = [(10, 0), (5, 0)]

    Example:
        >>> # Case where slice is bigger than the image
        >>> import kwarray
        >>> slices = (slice(-10, 400), slice(-10, 400))
        >>> data_dims  = [300, 300]
        >>> pad        = [10, 5]
        >>> a, b = kwarray.embed_slice(slices, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = (slice(0, 300, None), slice(0, 300, None))
        extra_padding = [(20, 110), (15, 105)]

    Example:
        >>> # Case where slice is inside than the image
        >>> import kwarray
        >>> slices = (slice(10, 40), slice(10, 40))
        >>> data_dims  = [300, 300]
        >>> pad        = None
        >>> a, b = kwarray.embed_slice(slices, data_dims, pad)
        >>> print('data_slice = {!r}'.format(a))
        >>> print('extra_padding = {!r}'.format(b))
        data_slice = (slice(10, 40, None), slice(10, 40, None))
        extra_padding = [(0, 0), (0, 0)]

    Example:
        >>> # Test error cases
        >>> import kwarray
        >>> import pytest
        >>> slices = (slice(0, 40), slice(10, 40))
        >>> data_dims  = [300, 300]
        >>> with pytest.raises(ValueError):
        >>>     kwarray.embed_slice(slices, data_dims[0:1])
        >>> with pytest.raises(ValueError):
        >>>     kwarray.embed_slice(slices[0:1], data_dims)
        >>> with pytest.raises(ValueError):
        >>>     kwarray.embed_slice(slices, data_dims, pad=[(1, 1)])
        >>> with pytest.raises(ValueError):
        >>>     kwarray.embed_slice(slices, data_dims, pad=[1])
    """
    low_dims = [sl.start for sl in slices]
    high_dims = [sl.stop for sl in slices]

    ndims = len(data_dims)
    if len(low_dims) != ndims:
        raise ValueError('slices and data_dims must have the same length')

    pad_slice = _coerce_pad(pad, ndims)

    # Determine the real part of the image that can be sliced out
    data_slice_st = []
    extra_padding = []

    # Determine the real part of the image that can be sliced out
    for D_img, d_low, d_high, d_pad in zip(data_dims, low_dims, high_dims, pad_slice):
        if d_low is None:
            d_low = 0
        if d_high is None:
            d_high = D_img
        if d_low > d_high:
            raise ValueError('d_low > d_high: {} > {}'.format(d_low, d_high))
        # Determine where the bounds would be if the image size was inf
        raw_low = d_low - d_pad[0]
        raw_high = d_high + d_pad[1]
        # Clip the slice positions to the real part of the image
        sl_low = min(D_img, max(0, raw_low))
        sl_high = min(D_img, max(0, raw_high))
        data_slice_st.append((sl_low, sl_high))

        # Add extra padding when the window extends past the real part
        low_diff = sl_low - raw_low
        high_diff = raw_high - sl_high

        # Hand the case where both raw coordinates are out of bounds
        extra_low = max(0, low_diff + min(0, high_diff))
        extra_high = max(0, high_diff + min(0, low_diff))
        extra = (extra_low, extra_high)
        extra_padding.append(extra)

    data_slice = tuple(slice(s, t) for s, t in data_slice_st)
    return data_slice, extra_padding


def _coerce_pad(pad, ndims):
    if pad is None:
        pad_slice = [(0, 0)] * ndims
    elif isinstance(pad, int):
        pad_slice = [(pad, pad)] * ndims
    else:
        # Normalize to left/right pad value for each dim
        pad_slice = [p if ub.iterable(p) else [p, p] for p in pad]

    if len(pad_slice) != ndims:
        # We could "fix" it, but the user probably made a mistake
        # n_trailing = ndims - len(pad)
        # if n_trailing > 0:
        #     pad = list(pad) + [(0, 0)] * n_trailing
        raise ValueError('pad and data_dims must have the same length')
    return pad_slice
