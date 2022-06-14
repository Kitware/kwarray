"""
Functions relating to robust statistical methods for normalizing data
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
    Finds robust normalization statistics for a single observation

    Args:
        data (ndarray): a 1D numpy array where invalid data has already been removed
        params (str | dict): normalization params

    Returns:
        Dict[str, str | float]: normalization parameters

    TODO:
        - [ ] No Magic Numbers! Use first principles to deterimine defaults.
        - [ ] Probably a lot of literature on the subject.
        - [ ] Is this a kwarray function in general?
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
            elif params == 'tukey':
                params = {
                    'extrema': 'tukey',
                }
            elif params == 'std':
                pass
            else:
                raise KeyError(params)

        # hack
        params = ub.dict_union(default_params, params)

        if params['extrema'] == 'tukey':
            # TODO:
            # https://github.com/derekbeaton/OuRS
            # https://en.wikipedia.org/wiki/Feature_scaling
            fense_extremes = _tukey_quantile_extreme_estimator(data)
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


def _tukey_quantile_extreme_estimator(data):
    # Tukey method for outliers
    # https://www.youtube.com/watch?v=zY1WFMAA-ec
    q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    # One might wonder where the 1.5 in the above interval comes from -- Paul
    # Velleman, a statistician at Cornell University, was a student of John
    # Tukey, who invented this test for outliers. He wondered the same thing.
    # When he asked Tukey, "Why 1.5?", Tukey answered, "Because 1 is too small
    # and 2 is too large."
    # Cite: http://mathcenter.oxford.emory.edu/site/math117/shapeCenterAndSpread/
    fence_lower = q1 - 1.5 * iqr
    fence_upper = q1 + 1.5 * iqr
    return fence_lower, q2, fence_upper


def _custom_quantile_extreme_estimator(data, params):
    quant_low = params['low']
    quant_mid = params['mid']
    quant_high = params['high']
    qvals = [0, quant_low, quant_mid, quant_high, 1]
    quantile_vals = np.quantile(data, qvals)
    print('quantile_vals = {!r}'.format(quantile_vals))

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
                     dtype=np.float32, params='auto'):
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

        return_info (bool, default=False):
            if True, return information about the chosen normalization
            heuristic.

        params (str | dict):
            can contain keys, low, high, or center

        nodata:
            A value representing nodata to leave unchanged during
            normalization, for example 0

        dtype : can be float32 or float64

    Returns:
        ndarray: a floating point array with values between 0 and 1.

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
    """
    import kwarray

    if axis is not None:
        # Hack, normalize each channel individually. This could
        # be implementd more effciently.
        assert not return_info
        reorg = imdata.swapaxes(0, axis)
        parts = []
        for item in reorg:
            part = robust_normalize(item, nodata=nodata, axis=None)
            parts.append(part[None, :])
        recomb = np.concatenate(parts, axis=0)
        final = recomb.swapaxes(0, axis)
        return final

    if nodata is not None:
        mask = imdata != nodata
        imdata_valid = imdata[mask]
    else:
        mask = None
        imdata_valid = imdata

    normalizer = find_robust_normalizers(imdata_valid, params=params)
    # print('normalizer = {!r}'.format(normalizer))

    if normalizer['type'] is None:
        imdata_normalized = imdata.astype(dtype)
    elif normalizer['type'] == 'normalize':
        # Note: we are using kwarray normalize, the one in kwimage is deprecated
        imdata_normalized = kwarray.normalize(
            imdata.astype(dtype), mode=normalizer['mode'],
            beta=normalizer['beta'], alpha=normalizer['alpha'],
        )
    else:
        raise KeyError(normalizer['type'])

    if mask is not None:
        result = np.where(mask, imdata_normalized, imdata)
    else:
        result = imdata_normalized

    if return_info:
        return result, normalizer
    else:
        return result
