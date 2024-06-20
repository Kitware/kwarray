
def test_memmaped_stitcher():
    import kwarray
    shape = (4, 4)
    device = 'numpy'
    dtype = 'float32'
    nan_policy = 'propogate'
    memmap = True
    self = kwarray.Stitcher(
        shape, device=device, dtype=dtype, nan_policy=nan_policy,
        memmap=memmap)
    import numpy as np
    assert np.all(self.sums == 0)
    assert np.all(self.weights == 0)
    index = (slice(None, None), slice(None, None))
    self.add(index, 2)
    index = (slice(None, None, 2), slice(None, None, 4))
    self.add(index, 3)
    assert np.isclose(self.finalize().sum(), 33)


def test_stitcher_grid():
    # Example of weighted stitching
    # xdoctest: +REQUIRES(module:kwimage)
    import numpy as np

    try:
        import kwimage
    except ImportError:
        import pytest
        pytest.skip('requires kwimage')
    import kwarray
    import ubelt as ub
    data = kwimage.Mask.demo().data.astype(np.float32)
    data_dims = data.shape
    window_dims = (8, 8)
    # We are going to slide a window over the data, do some processing
    # and then stitch it all back together. There are a few ways we
    # can do it. Lets demo the params.
    basis = {
        # Vary the overlap of the slider
        'overlap': (0, 0.5, .9),
        # Vary if we are using weighted stitching or not
        'weighted': ['none', 'gauss'],
        'keepbound': [True, False]
    }
    results = []
    gauss_weights = kwimage.gaussian_patch(window_dims)
    gauss_weights = kwarray.normalize(gauss_weights)
    for params in ub.named_product(basis):
        if params['weighted'] == 'none':
            weights = None
        elif params['weighted'] == 'gauss':
            weights = gauss_weights
        # Build the slider and stitcher
        kwargs = dict(
            shape=data_dims,
            window=window_dims,
            allow_overshoot=True,
            overlap=params['overlap'],
            keepbound=params['keepbound'])
        slider = kwarray.SlidingWindow(**kwargs)
        stitcher = kwarray.Stitcher(data_dims)
        # Loop over the regions
        for sl in list(slider):
            chip = data[sl]
            # This is our dummy function for thie example.
            predicted = np.ones_like(chip) * chip.sum() / chip.size
            stitcher.add(sl, predicted, weight=weights)
        final = stitcher.finalize()
        results.append({
            'final': final,
            'params': params,
        })
    # xdoctest: +REQUIRES(--show)
    # xdoctest: +REQUIRES(module:kwplot)
    import kwplot
    kwplot.autompl()
    pnum_ = kwplot.PlotNums(nCols=3, nSubplots=len(results) + 2)
    kwplot.imshow(data, pnum=pnum_(), title='input image')
    kwplot.imshow(gauss_weights, pnum=pnum_(), title='Gaussian weights')
    pnum_()
    for result in results:
        param_key = ub.urepr(result['params'], compact=1)
        final = result['final']
        canvas = kwarray.normalize(final)
        canvas = kwimage.fill_nans_with_checkers(canvas)
        kwplot.imshow(canvas, pnum=pnum_(), title=param_key)
