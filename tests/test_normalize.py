import kwarray
import numpy as np


def test_normalize_grid():
    import ubelt as ub

    NORMALIZE_PARAM_BASIS = {
        'dtype': [np.float32, np.uint8, int, float],
        'mode': ['linear', 'sigmoid'],
        'use_out': [True, False],
    }
    try:
        import scipy  # NOQA
    except ImportError:
        NORMALIZE_PARAM_BASIS['mode'] = ['linear']

    basis = {
        'shape': [
            [], [0], [0, 1], [1, 0], [3, 8, 0, 2],
            [1], [8, 1], [1, 3], [3, 5], [2, 3, 5]],
        **NORMALIZE_PARAM_BASIS,
    }
    results = []
    for params in ub.named_product(basis):
        arr = np.empty(shape=params['shape'], dtype=params['dtype'])
        if params['use_out']:
            out = arr.copy()
        else:
            out = None
        result = kwarray.normalize(arr, mode=params['mode'], out=out)
        if out is not None:
            assert result is out
        assert result is not arr
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype
        results.append({
            'arr': arr,
            'out': out,
            **params
        })
