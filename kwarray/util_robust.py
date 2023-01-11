"""
Functions relating to robust statistical methods for normalizing data.
"""
import numpy as np
import ubelt as ub
import math  # NOQA

try:
    from math import isclose
except Exception:
    from numpy import isclose


def find_robust_normalizers(data, params='auto'):
    """
    Finds robust normalization statistics a set of scalar observations.

    The idea is to estimate "fense" parameters: minimum and maximum values
    where anything under / above these values are likely outliers. For
    non-linear normalizaiton schemes we can also estimate an likely middle and
    extent of the data.

    Args:
        data (ndarray): a 1D numpy array where invalid data has already been removed

        params (str | dict): normalization params.

            When passed as a dictionary valid params are:

                scaling (str):
                    This is the "mode" that will be used in the final
                    normalization. Currently has no impact on the
                    Defaults to 'linear'. Can also be 'sigmoid'.

                extrema (str):
                    The method for determening what the extrama are.
                    Can be "custom-quantile" for an IQR-like method.
                    Can be "tukey" or "IQR" for an exact IQR method.

                low (float): This is the low quantile for likely inliers.

                mid (float): This is the middle quantlie for likely inliers.

                high (float): This is the high quantile for likely inliers.

            Can be specified as a concise string.

            The string "auto" defaults to:
                ``dict(extrema='custom-quantile', scaling='linear', low=0.01, mid=0.5, high=0.9)``.

            The string "tukey" defaults to:
                ``dict(extrema='tukey', scaling='linear')``.

    Returns:
        Dict[str, str | float]:
            normalization parameters that can be passed to
            :func:`kwarray.normalize` containing the keys:

            type (str): which is always 'normalize'

            mode (str): the value of `params['scaling']`

            min_val (float): the determined "robust" minimum inlier value.

            max_val (float): the determined "robust" maximum inlier value.

            beta (float): the determined "robust" middle value for use in
                non-linear normalizers.

            alpha (float): the determined "robust" extent value for use in
                non-linear normalizers.

    Note:
        The defaults and methods of this function are subject to change.

    TODO:
        - [ ] No (or minimal) Magic Numbers! Use first principles to deterimine defaults.
        - [ ] Probably a lot of literature on the subject.
        - [ ] https://arxiv.org/pdf/1707.09752.pdf
        - [ ] https://www.tandfonline.com/doi/full/10.1080/02664763.2019.1671961
        - [ ] https://www.rips-irsp.com/articles/10.5334/irsp.289/

        - [ ] This function is not possible to get write in every case
              (probably can prove this with a NFL theroem), might be useful
              to allow the user to specify a "model" which is specific to some
              domain.

    Example:
        >>> from kwarray.util_robust import *  # NOQA
        >>> data = np.random.rand(100)
        >>> norm_params1 = find_robust_normalizers(data, params='auto')
        >>> norm_params2 = find_robust_normalizers(data, params={'low': 0, 'high': 1.0})
        >>> norm_params3 = find_robust_normalizers(np.empty(0), params='auto')
        >>> print('norm_params1 = {}'.format(ub.repr2(norm_params1, nl=1)))
        >>> print('norm_params2 = {}'.format(ub.repr2(norm_params2, nl=1)))
        >>> print('norm_params3 = {}'.format(ub.repr2(norm_params3, nl=1)))

    Example:
        >>> from kwarray.util_robust import *  # NOQA
        >>> from kwarray.distributions import Mixture
        >>> import ubelt as ub
        >>> # A random mixture distribution for testing
        >>> data = Mixture.random(6).sample(3000)
    """
    if data.size == 0:
        normalizer = {
            'type': None,
            'min_val': np.nan,
            'max_val': np.nan,
        }
    else:
        # should center the desired distribution to visualize on zero
        # beta = np.median(imdata)
        default_params = {
            'extrema': 'custom-quantile',
            'scaling': 'linear',
            'low': 0.01,
            'mid': 0.5,
            'high': 0.9,
        }
        fense_extremes = None
        if isinstance(params, str):
            if params == 'auto':
                params = {}
            elif params in {'tukey', 'iqr'}:
                params = {
                    'extrema': 'tukey',
                }
            elif params == 'std':
                pass
            else:
                raise KeyError(params)

        # hack
        params = ub.dict_union(default_params, params)

        # TODO:
        # https://github.com/derekbeaton/OuRS
        # https://en.wikipedia.org/wiki/Feature_scaling
        if params['extrema'] == 'tukey':
            fense_extremes = _tukey_quantile_fence(data)
        elif params['extrema'] == 'custom-quantile':
            fense_extremes = _custom_quantile_extreme_estimator(data, params)
        else:
            raise KeyError(params['extrema'])

        min_val, mid_val, max_val = fense_extremes

        beta = mid_val
        # division factor
        # from scipy.special import logit
        # alpha = max(abs(old_min - beta), abs(old_max - beta)) / logit(0.998)
        # This chooses alpha such the original min/max value will be pushed
        # towards -1 / +1.
        alpha = max(abs(min_val - beta), abs(max_val - beta)) / 6.212606

        normalizer = {
            'type': 'normalize',
            'mode': params['scaling'],
            'min_val': min_val,
            'max_val': max_val,
            'beta': beta,
            'alpha': alpha,
        }
    return normalizer


def _tukey_quantile_fence(data):
    """
    One might wonder where the 1.5 in the above interval comes from -- Paul
    Velleman, a statistician at Cornell University, was a student of John
    Tukey, who invented this test for outliers. He wondered the same thing.
    When he asked Tukey, "Why 1.5?", Tukey answered, "Because 1 is too small
    and 2 is too large." [OxfordShapeSpread]_.

    References:
        .. [OxfordShapeSpread] http://mathcenter.oxford.emory.edu/site/math117/shapeCenterAndSpread/
        .. [YTFindOutliers] https://www.youtube.com/watch?v=zY1WFMAA-ec
    """
    # Tukey method for outliers
    q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    fence_lower = q1 - 1.5 * iqr
    fence_upper = q1 + 1.5 * iqr
    return fence_lower, q2, fence_upper


def _custom_quantile_extreme_estimator(data, params):
    quant_low = params['low']
    quant_mid = params['mid']
    quant_high = params['high']
    qvals = [0, quant_low, quant_mid, quant_high, 1]
    quantile_vals = np.quantile(data, qvals)

    (quant_low_abs, quant_low_val, quant_mid_val, quant_high_val,
     quant_high_abs) = quantile_vals

    # TODO: we could implement a hueristic where we do a numerical inspection
    # of the intensity distribution. We could apply a normalization that is
    # known to work for data with that sort of histogram distribution.
    # This might involve fitting several parametarized distributions to the
    # data and choosing the one with the best fit. (check how many modes there
    # are).

    # inner_range = quant_high_val - quant_low_val
    # upper_inner_range = quant_high_val - quant_mid_val
    # upper_lower_range = quant_mid_val - quant_low_val

    # Compute amount of weight in each quantile
    quant_center_amount = (quant_high_val - quant_low_val)
    quant_low_amount = (quant_mid_val - quant_low_val)
    quant_high_amount = (quant_high_val - quant_mid_val)

    if isclose(quant_center_amount, 0):
        high_weight = 0.5
        low_weight = 0.5
    else:
        high_weight = quant_high_amount / quant_center_amount
        low_weight = quant_low_amount / quant_center_amount

    quant_high_residual = (1.0 - quant_high)
    quant_low_residual = (quant_low - 0.0)
    # todo: verify, having slight head fog, not 100% sure
    low_pad_val = quant_low_residual * (low_weight * quant_center_amount)
    high_pad_val = quant_high_residual * (high_weight * quant_center_amount)
    min_val = max(quant_low_abs, quant_low_val - low_pad_val)
    max_val = max(quant_high_abs, quant_high_val - high_pad_val)
    mid_val = quant_mid_val
    return (min_val, mid_val, max_val)


def robust_normalize(imdata, return_info=False, nodata=None, axis=None,
                     dtype=np.float32, params='auto', mask=None):
    """
    Normalize data intensities using heuristics to help put sensor data with
    extremely high or low contrast into a visible range.

    This function is designed with an emphasis on getting something that is
    reasonable for visualization.

    TODO:
        - [x] Move to kwarray and renamed to robust_normalize?
        - [ ] Support for M-estimators?

    Args:
        imdata (ndarray): raw intensity data

        return_info (bool):
            if True, return information about the chosen normalization
            heuristic.

        params (str | dict):
            can contain keys, low, high, or center
            e.g. {'low': 0.1, 'center': 0.8, 'high': 0.9}

        axis (None | int):
            The axis to normalize over, if unspecified, normalize jointly

        nodata (None | int):
            A value representing nodata to leave unchanged during
            normalization, for example 0

        dtype (type) : can be float32 or float64

        mask (ndarray | None):
            A mask indicating what pixels are valid and what pixels should be
            considered nodata.  Mutually exclusive with ``nodata`` argument.
            A mask value of 1 indicates a VALID pixel. A mask value of 0
            indicates an INVALID pixel.

    Returns:
        ndarray: a floating point array with values between 0 and 1.

    Note:
        This is effectively a combination of :func:`find_robust_normalizers`
        and :func:`normalize`.

    Example:
        >>> from kwarray.util_robust import *  # NOQA
        >>> from kwarray.distributions import Mixture
        >>> import ubelt as ub
        >>> # A random mixture distribution for testing
        >>> data = Mixture.random(6).sample(3000)
        >>> param_basis = {
        >>>     'scaling': ['linear', 'sigmoid'],
        >>>     'high': [0.6, 0.8, 0.9, 1.0],
        >>> }
        >>> param_grid = list(ub.named_product(param_basis))
        >>> param_grid += ['auto']
        >>> param_grid += ['tukey']
        >>> rows = []
        >>> rows.append({'key': 'orig', 'result': data})
        >>> for params in param_grid:
        >>>     key = ub.repr2(params, compact=1)
        >>>     result, info = robust_normalize(data, return_info=True, params=params)
        >>>     print('key = {}'.format(key))
        >>>     print('info = {}'.format(ub.repr2(info, nl=1)))
        >>>     rows.append({'key': key, 'info': info, 'result': result})
        >>> # xdoctest: +REQUIRES(--show)
        >>> import seaborn as sns
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(rows))
        >>> for row in rows:
        >>>     ax = kwplot.figure(fnum=1, pnum=pnum_()).gca()
        >>>     sns.histplot(data=row['result'], kde=True, bins=128, ax=ax, stat='density')
        >>>     ax.set_title(row['key'])

    Example:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from kwarray.util_robust import *  # NOQA
        >>> import ubelt as ub
        >>> import kwimage
        >>> import kwarray
        >>> s = 512
        >>> bit_depth = 11
        >>> dtype = np.uint16
        >>> max_val = int(2 ** bit_depth)
        >>> min_val = int(0)
        >>> rng = kwarray.ensure_rng(0)
        >>> background = np.random.randint(min_val, max_val, size=(s, s), dtype=dtype)
        >>> poly1 = kwimage.Polygon.random(rng=rng).scale(s / 2)
        >>> poly2 = kwimage.Polygon.random(rng=rng).scale(s / 2).translate(s / 2)
        >>> forground = np.zeros_like(background, dtype=np.uint8)
        >>> forground = poly1.fill(forground, value=255)
        >>> forground = poly2.fill(forground, value=122)
        >>> forground = (kwimage.ensure_float01(forground) * max_val).astype(dtype)
        >>> imdata = background + forground
        >>> normed, info = kwarray.robust_normalize(imdata, return_info=True)
        >>> print('info = {}'.format(ub.repr2(info, nl=1)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(imdata, pnum=(1, 2, 1), fnum=1)
        >>> kwplot.imshow(normed, pnum=(1, 2, 2), fnum=1)
    """
    if axis is not None:
        # Hack, normalize each channel individually. This could
        # be implementd more effciently.
        assert not return_info
        reorg = imdata.swapaxes(0, axis)
        if mask is None:
            parts = []
            for item in reorg:
                part = robust_normalize(item, nodata=nodata, axis=None)
                parts.append(part[None, :])
        else:
            reorg_mask = mask.swapaxes(0, axis)
            parts = []
            for item, item_mask in zip(reorg, reorg_mask):
                part = robust_normalize(item, nodata=nodata, axis=None,
                                           mask=item_mask)
                parts.append(part[None, :])
        recomb = np.concatenate(parts, axis=0)
        final = recomb.swapaxes(0, axis)
        return final

    if imdata.dtype.kind == 'f':
        if mask is None:
            mask = ~np.isnan(imdata)

    if mask is None:
        if nodata is not None:
            mask = imdata != nodata

    if mask is None:
        imdata_valid = imdata
    else:
        imdata_valid = imdata[mask]

    assert not np.any(np.isnan(imdata_valid))

    normalizer = find_robust_normalizers(imdata_valid, params=params)
    imdata_normalized = _apply_robust_normalizer(normalizer, imdata,
                                                 imdata_valid, mask, dtype)

    if mask is not None:
        result = np.where(mask, imdata_normalized, imdata)
    else:
        result = imdata_normalized

    if return_info:
        return result, normalizer
    else:
        return result


def _apply_robust_normalizer(normalizer, imdata, imdata_valid, mask, dtype, copy=True):
    """
    TODO:
        abstract into a scikit-learn-style Normalizer class which can
        fit/predict different types of normalizers.
    """
    import kwarray
    if normalizer['type'] is None:
        imdata_normalized = imdata.astype(dtype, copy=copy)
    elif normalizer['type'] == 'normalize':
        # Note: we are using kwarray normalize, the one in kwimage is deprecated
        imdata_valid_normalized = kwarray.normalize(
            imdata_valid.astype(dtype, copy=copy), mode=normalizer['mode'],
            beta=normalizer['beta'], alpha=normalizer['alpha'],
        )
        if mask is None:
            imdata_normalized = imdata_valid_normalized
        else:
            imdata_normalized = imdata.copy() if copy else imdata
            imdata_normalized[mask] = imdata_valid_normalized
    else:
        raise KeyError(normalizer['type'])
    return imdata_normalized


def normalize(arr, mode='linear', alpha=None, beta=None, out=None,
              min_val=None, max_val=None):
    """
    Normalizes input values based on a specified scheme.

    The default behavior is a linear normalization between 0.0 and 1.0 based on
    the min/max values of the input. Parameters can be specified to achieve
    more general constrat stretching or signal rebalancing. Implements the
    linear and sigmoid normalization methods described in [WikiNorm]_.

    Args:
        arr (NDArray): array to normalize, usually an image

        out (NDArray | None): output array. Note, that we will create an
            internal floating point copy for integer computations.

        mode (str): either linear or sigmoid.

        alpha (float): Only used if mode=sigmoid.  Division factor
            (pre-sigmoid). If unspecified computed as:
            ``max(abs(old_min - beta), abs(old_max - beta)) / 6.212606``.
            Note this parameter is sensitive to if the input is a float or
            uint8 image.

        beta (float): subtractive factor (pre-sigmoid). This should be the
            intensity of the most interesting bits of the image, i.e. bring
            them to the center (0) of the distribution.
            Defaults to ``(max - min) / 2``.  Note this parameter is sensitive
            to if the input is a float or uint8 image.

        min_val: override minimum value

        max_val: override maximum value

    SeeAlso:
        :func:`find_robust_normalizers` - determine robust parameters for
            normalize to mitigate the effect of outliers.

        :func:`robust_normalize` - finds and applies robust normalization
            parameters

    References:
        .. [WikiNorm] https://en.wikipedia.org/wiki/Normalization_(image_processing)

    Example:
        >>> raw_f = np.random.rand(8, 8)
        >>> norm_f = normalize(raw_f)

        >>> raw_f = np.random.rand(8, 8) * 100
        >>> norm_f = normalize(raw_f)
        >>> assert isclose(norm_f.min(), 0)
        >>> assert isclose(norm_f.max(), 1)

        >>> raw_u = (np.random.rand(8, 8) * 255).astype(np.uint8)
        >>> norm_u = normalize(raw_u)

    Example:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> import kwimage
        >>> arr = kwimage.grab_test_image('lowcontrast')
        >>> arr = kwimage.ensure_float01(arr)
        >>> norms = {}
        >>> norms['arr'] = arr.copy()
        >>> norms['linear'] = normalize(arr, mode='linear')
        >>> norms['sigmoid'] = normalize(arr, mode='sigmoid')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(norms))
        >>> for key, img in norms.items():
        >>>     kwplot.imshow(img, pnum=pnum_(), title=key)

    Example:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> arr = np.array([np.inf])
        >>> normalize(arr, mode='linear')
        >>> normalize(arr, mode='sigmoid')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(norms))
        >>> for key, img in norms.items():
        >>>     kwplot.imshow(img, pnum=pnum_(), title=key)

    Benchmark:
        >>> # Our method is faster than standard in-line implementations for
        >>> # uint8 and competative with in-line float32, in addition to being
        >>> # more concise and configurable. In 3.11 all inplace variants are
        >>> # faster.
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> import timerit
        >>> import kwimage
        >>> import kwarray
        >>> ti = timerit.Timerit(1000, bestof=10, verbose=2, unit='ms')
        >>> arr = kwimage.grab_test_image('lowcontrast', dsize=(512, 512))
        >>> #
        >>> arr = kwimage.ensure_float01(arr)
        >>> out = arr.copy()
        >>> for timer in ti.reset('inline_naive(float)'):
        >>>     with timer:
        >>>         (arr - arr.min()) / (arr.max() - arr.min())
        >>> #
        >>> for timer in ti.reset('inline_faster(float)'):
        >>>     with timer:
        >>>         max_ = arr.max()
        >>>         min_ = arr.min()
        >>>         result = (arr - min_) / (max_ - min_)
        >>> #
        >>> for timer in ti.reset('kwarray.normalize(float)'):
        >>>     with timer:
        >>>         kwarray.normalize(arr)
        >>> #
        >>> for timer in ti.reset('kwarray.normalize(float, inplace)'):
        >>>     with timer:
        >>>         kwarray.normalize(arr, out=out)
        >>> #
        >>> arr = kwimage.ensure_uint255(arr)
        >>> out = arr.copy()
        >>> for timer in ti.reset('inline_naive(uint8)'):
        >>>     with timer:
        >>>         (arr - arr.min()) / (arr.max() - arr.min())
        >>> #
        >>> for timer in ti.reset('inline_faster(uint8)'):
        >>>     with timer:
        >>>         max_ = arr.max()
        >>>         min_ = arr.min()
        >>>         result = (arr - min_) / (max_ - min_)
        >>> #
        >>> for timer in ti.reset('kwarray.normalize(uint8)'):
        >>>     with timer:
        >>>         kwarray.normalize(arr)
        >>> #
        >>> for timer in ti.reset('kwarray.normalize(uint8, inplace)'):
        >>>     with timer:
        >>>         kwarray.normalize(arr, out=out)
        >>> print('ti.rankings = {}'.format(ub.urepr(
        >>>     ti.rankings, nl=2, align=':', precision=5)))

    Ignore:
        globals().update(xdev.get_func_kwargs(normalize))
    """
    if out is None:
        out = arr.copy()

    # TODO:
    # - [ ] Parametarize new_min / new_max values
    #     - [ ] infer from datatype
    #     - [ ] explicitly given
    new_min = 0.0
    if arr.dtype.kind in ('i', 'u'):
        # Need a floating point workspace
        float_out = out.astype(np.float32)
        new_max = float(np.iinfo(arr.dtype).max)
    elif arr.dtype.kind == 'f':
        float_out = out
        new_max = 1.0
    else:
        raise TypeError(f'Normalize not implemented for {arr.dtype}')

    # TODO:
    # - [ ] Parametarize old_min / old_max strategies
    #     - [X] explicitly given min and max
    #     - [ ] raw-naive min and max inference
    #     - [ ] outlier-aware min and max inference (see util_robust)
    if min_val is not None:
        old_min = min_val
        float_out[float_out < min_val] = min_val
    else:
        try:
            old_min = np.nanmin(float_out)
        except ValueError:
            old_min = 0

    if max_val is not None:
        old_max = max_val
        float_out[float_out > max_val] = max_val
    else:
        try:
            old_max = np.nanmax(float_out)
        except ValueError:
            old_max = max(0, old_min)

    old_span = old_max - old_min
    new_span = new_max - new_min

    if mode == 'linear':
        # linear case
        # out = (arr - old_min) * (new_span / old_span) + new_min
        factor = 1.0 if old_span == 0 else (new_span / old_span)
        if old_min != 0:
            float_out -= old_min
    elif mode == 'sigmoid':
        # nonlinear case
        # out = new_span * sigmoid((arr - beta) / alpha) + new_min
        from scipy.special import expit as sigmoid
        if beta is None:
            # should center the desired distribution to visualize on zero
            beta = old_max - old_min

        if alpha is None:
            # division factor
            # from scipy.special import logit
            # alpha = max(abs(old_min - beta), abs(old_max - beta)) / logit(0.998)
            # This chooses alpha such the original min/max value will be pushed
            # towards -1 / +1.
            alpha = max(abs(old_min - beta), abs(old_max - beta)) / 6.212606

        if isclose(alpha, 0):
            alpha = 1

        energy = float_out
        energy -= beta
        energy /= alpha
        # Ideally the data of interest is roughly in the range (-6, +6)
        float_out = sigmoid(energy, out=float_out)
        factor = new_span
    else:
        raise KeyError(mode)

    # Stretch / shift to the desired output range
    if factor != 1:
        float_out *= factor

    if new_min != 0:
        float_out += new_min

    if float_out is not out:
        final_out = float_out.astype(out.dtype)
        out.ravel()[:] = final_out.ravel()[:]
    return out
