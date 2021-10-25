"""

Ignore:
    # Get the appropriate helper function
    from watch.utils import util_kwimage
    import liberator
    lib = liberator.Liberator()
    from watch.utils import util_kwimage
    lib.expand(['watch'])
    lib.add_dynamic(util_kwimage.upweight_center_mask)
    lib.add_dynamic(util_kwimage._auto_kernel_sigma)
    lib.add_dynamic(util_kwimage.morphology)
    print(lib.current_sourcecode())

"""
import numpy as np
import cv2
import numbers
from functools import lru_cache
import kwimage


_CV2_STRUCT_ELEMENTS = {'rect': cv2.MORPH_RECT, 'cross': cv2.MORPH_CROSS, 'ellipse': cv2.MORPH_ELLIPSE}


@lru_cache
def _morph_kernel_core(h, w, element):
    struct_shape = _CV2_STRUCT_ELEMENTS.get(element, element)
    element = cv2.getStructuringElement(struct_shape, (h, w))
    return element


def _morph_kernel(size, element='rect'):
    if isinstance(size, int):
        h = size
        w = size
    else:
        h, w = size
        # raise NotImplementedError
    return _morph_kernel_core(h, w, element)


_CV2_MORPH_MODES = {
    'erode': cv2.MORPH_ERODE, 'dilate': cv2.MORPH_DILATE,
    'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE,
    'gradient': cv2.MORPH_GRADIENT, 'tophat': cv2.MORPH_TOPHAT,
    'blackhat': cv2.MORPH_BLACKHAT,
    'hitmiss': cv2.MORPH_HITMISS}


def morphology(data, mode, kernel=5, element='rect', iterations=1):
    if data.dtype.kind == 'b':
        data = data.astype(np.uint8)
    kernel = _morph_kernel(kernel, element=element)
    if isinstance(mode, str):
        morph_mode = _CV2_MORPH_MODES[mode]
    elif isinstance(mode, int):
        morph_mode = mode
    else:
        raise TypeError(type(mode))

    new = cv2.morphologyEx(
        data, op=morph_mode, kernel=kernel, iterations=iterations)
    return new


def _auto_kernel_sigma(kernel=None, sigma=None, autokernel_mode='ours'):
    if kernel is None and sigma is None:
        kernel = 3

    if kernel is not None:
        if isinstance(kernel, numbers.Integral):
            k_x = k_y = kernel
        else:
            k_x, k_y = kernel

    if sigma is None:
        # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L344
        sigma_x = 0.3 * ((k_x - 1) * 0.5 - 1) + 0.8
        sigma_y = 0.3 * ((k_y - 1) * 0.5 - 1) + 0.8
    else:
        if isinstance(sigma, numbers.Number):
            sigma_x = sigma_y = sigma
        else:
            sigma_x, sigma_y = sigma

    if kernel is None:
        if autokernel_mode == 'zero':
            # When 0 computed internally via cv2
            k_x = k_y = 0
        elif autokernel_mode == 'cv2':
            # if USE_CV2_DEF:
            # This is the CV2 definition
            # https://github.com/egonSchiele/OpenCV/blob/09bab41/modules/imgproc/src/smooth.cpp#L387
            depth_factor = 3  # or 4 for non-uint8
            k_x = int(round(sigma_x * depth_factor * 2 + 1)) | 1
            k_y = int(round(sigma_y * depth_factor * 2 + 1)) | 1
        elif autokernel_mode == 'ours':
            # But I think this definition makes more sense because it keeps
            # sigma and the kernel in agreement more often
            """
            # Our hueristic is computed via solving the sigma heuristic for k
            import sympy as sym
            s, k = sym.symbols('s, k', rational=True)
            sa = sym.Rational('3 / 10') * ((k - 1) / 2 - 1) + sym.Rational('8 / 10')
            sym.solve(sym.Eq(s, sa), k)
            """
            k_x = max(3, round(20 * sigma_x / 3 - 7 / 3)) | 1
            k_y = max(3, round(20 * sigma_y / 3 - 7 / 3)) | 1
        else:
            raise KeyError(autokernel_mode)
    sigma = (sigma_x, sigma_y)
    kernel = (k_x, k_y)
    return kernel, sigma


def upweight_center_mask(shape):
    """
    Example:
        >>> shapes = [32, 64, 96, 128, 256]
        >>> results = {}
        >>> for shape in shapes:
        >>>     results[str(shape)] = upweight_center_mask(shape)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(results))
        >>> for k, result in results.items():
        >>>     kwplot.imshow(result, pnum=pnum_(), title=k)
        >>> kwplot.show_if_requested()
    """
    shape, sigma = _auto_kernel_sigma(kernel=shape)
    sigma_x, sigma_y = sigma
    weights = kwimage.gaussian_patch(shape, sigma=(sigma_x, sigma_y))
    weights = weights / weights.max()
    # weights = kwimage.ensure_uint255(weights)
    kernel = np.maximum(np.array(shape) // 8, 3)
    kernel = kernel + (1 - (kernel % 2))
    weights = morphology(
        weights, kernel=kernel, mode='dilate', element='rect', iterations=1)
    weights = kwimage.ensure_float01(weights)
    weights = np.maximum(weights, 0.001)
    return weights


def _demo_weighted_stitcher():
    """
    Requires kwimage, don't test

    """

    import kwimage
    import kwarray
    import kwplot

    stitch_dims = (512, 512)
    window_dims = (64, 64)

    # Seed a local random number generator
    rng = kwarray.ensure_rng(8022)

    # Create a random heatmap we will use as the dummy "truth" we would like
    # to predict
    heatmap = kwimage.Heatmap.random(dims=stitch_dims, rng=rng)
    truth = heatmap.data['class_probs'][0]

    overlap_to_unweighted = {}
    overlap_to_weighted = {}

    center_weights = upweight_center_mask(window_dims)

    for overlap in [0, 0.1, 0.2, 0.25, 0.3, 0.5]:
        slider = kwarray.SlidingWindow(stitch_dims, window_dims, overlap=overlap,
                                       keepbound=True, allow_overshoot=True)

        unweighted_sticher = kwarray.Stitcher(stitch_dims, device='numpy')
        weighted_sticher = kwarray.Stitcher(stitch_dims, device='numpy')

        # Seed a local random number generator
        rng = kwarray.ensure_rng(8022)
        for space_slice in slider:

            # Make a (dummy) prediction at this slice
            # Our predition will be a perterbed version of the truth
            real_data = truth[space_slice]
            aff = kwimage.Affine.random(rng=rng, theta=0, shear=0)

            # Perterb spatial location
            pred_data = kwimage.warp_affine(real_data, aff)
            pred_data += (rng.randn(*window_dims) * 0.5)
            pred_data = pred_data.clip(0, 1)

            # Add annoying boundary artifacts
            pred_data[0:3, :] = rng.rand()
            pred_data[-3:None, :] = rng.rand()
            pred_data[:, -3:None] = rng.rand()
            pred_data[:, 0:3] = rng.rand()

            pred_data = kwimage.gaussian_blur(pred_data, kernel=9)

            unweighted_sticher.add(space_slice, pred_data)
            weighted_sticher.add(space_slice, pred_data, weight=center_weights)

        unweighted_stiched_pred = unweighted_sticher.finalize()
        weighted_stiched_pred = weighted_sticher.finalize()
        overlap_to_weighted[overlap] = weighted_stiched_pred
        overlap_to_unweighted[overlap] = unweighted_stiched_pred

    kwplot.autompl()
    pnum_ = kwplot.PlotNums(nCols=2, nSubplots=len(overlap_to_unweighted) * 2 + 2)

    kwplot.imshow(truth, fnum=1, pnum=pnum_(), title='(Dummy) Truth')
    kwplot.imshow(center_weights, fnum=1, pnum=pnum_(), title='Window Weights')

    for overlap in overlap_to_unweighted.keys():
        weighted_stiched_pred = overlap_to_weighted[overlap]
        unweighted_stiched_pred = overlap_to_unweighted[overlap]

        kwplot.imshow(unweighted_stiched_pred, fnum=1, pnum=pnum_(), title=f'Unweighted stitched preds: overlap={overlap}')
        kwplot.imshow(weighted_stiched_pred, fnum=1, pnum=pnum_(), title=f'Weighted stitched preds: overlap={overlap}')
