
def test_robust_normalize_with_masked_array():
    """
    Test that 3 ways to mask an array are equivalent
    """
    import numpy as np
    import kwarray
    rng = kwarray.ensure_rng(0)
    data = rng.rand(23, 31, 3)

    is_invalid = rng.rand(*data.shape) > 0.5
    is_valid = ~is_invalid

    masked_data = np.ma.MaskedArray(data, is_invalid)
    nanned_data = data.copy()
    nanned_data[is_invalid] = np.nan

    normed1 = kwarray.robust_normalize(masked_data)
    normed2 = kwarray.robust_normalize(data, mask=is_valid)
    normed3 = kwarray.robust_normalize(nanned_data)

    assert isinstance(normed1, np.ma.MaskedArray), 'ma inputs should produce ma outputs'
    assert not isinstance(normed2, np.ma.MaskedArray), 'non-ma inputs should produce non-ma outputs'
    assert not isinstance(normed3, np.ma.MaskedArray), 'nno-ma inputs should produce non-ma outputs'

    assert np.all(normed1.mask == is_invalid)
    assert np.allclose(normed1[is_valid], normed2[is_valid])
    assert np.allclose(normed1[is_valid], normed3[is_valid])
