def _check_np_pad_reqs():
    import numpy as np
    arr = np.zeros((32, 32, 3))

    np.pad(arr, ((1, 1), (1, 1), (1, 1))).shape

    np.pad(arr, ((1, 1), (1, 1))).shape
    np.pad(arr, 1).shape
    np.pad(arr, [1, 1]).shape
