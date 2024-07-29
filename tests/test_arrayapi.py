
def indexable_allclose(dct1, dct2, rel_tol=1e-9, abs_tol=0.0, return_info=False):
    """
    PORT FROM UBELT WITH SUPPORT FOR NDARRAYS

    Walks through two nested data structures and ensures that everything is
    roughly the same.
    """
    import ubelt as ub
    import numpy as np
    from math import isclose
    from functools import partial
    try:
        import torch
    except ImportError:
        torch = None

    isclose_ = partial(isclose, rel_tol=rel_tol, abs_tol=abs_tol)
    np_isclose_ = partial(np.isclose, rtol=rel_tol, atol=abs_tol)

    walker1 = ub.IndexableWalker(dct1)
    walker2 = ub.IndexableWalker(dct2)
    flat_items1 = [
        (path, value) for path, value in walker1
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0]
    flat_items2 = [
        (path, value) for path, value in walker2
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0]

    flat_items1 = sorted(flat_items1)
    flat_items2 = sorted(flat_items2)

    if len(flat_items1) != len(flat_items2):
        info = {
            'faillist': ['length mismatch']
        }
        final_flag = False
    else:
        passlist = []
        faillist = []

        for t1, t2 in zip(flat_items1, flat_items2):
            p1, v1 = t1
            p2, v2 = t2
            assert p1 == p2
            if torch is not None:
                if torch.is_tensor(v1):
                    v1 = v1.numpy()
                if torch.is_tensor(v2):
                    v2 = v2.numpy()

            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                flag = np.all(np_isclose_(v1, v2))
            else:
                flag = (v1 == v2)
                if not flag:
                    if isinstance(v1, float) and isinstance(v2, float) and isclose_(v1, v2):
                        flag = True
            if flag:
                passlist.append(p1)
            else:
                faillist.append((p1, v1, v2))

        final_flag = len(faillist) == 0
        info = {
            'passlist': passlist,
            'faillist': faillist,
        }

    if return_info:
        info.update({
            'walker1': walker1,
            'walker2': walker2,
        })
        return final_flag, info
    else:
        return final_flag


def test_numpy_torch_compat():
    import pytest
    import ubelt as ub
    import numpy as np
    import kwarray
    from kwarray import arrayapi
    try:
        import torch
    except ImportError:
        torch = None

    ArrayAPI = arrayapi.ArrayAPI

    if torch is None:
        pytest.skip('no torch')

    # arrayapi._REGISTERY.registered['numpy']
    rows = list(arrayapi._REGISTERY.registered['api'].values())
    groups = ub.group_items(rows, lambda item: item['func_type'])

    rng = kwarray.ensure_rng()

    basis = {
        'shape': [(3, 5)],
        'dtype': ['float32', 'uint8'],
    }

    for item in ub.named_product(basis):
        np_data1 = rng.rand(3, 5)
        pt_data1 = ArrayAPI.tensor(np_data1)

        blocklist = {
            'take', 'compress', 'repeat', 'tile', 'reshape', 'view',
            'numel', 'atleast_nd', 'full_like',
            #
            'maximum', 'minimum', 'matmul',
            'astype', 'ensure',
            'transpose',
            'pad',
            'dtype_kind',
            'clip',

            'array_equal',
            'isclose',
            'allclose',
        }

        if arrayapi._TORCH_LT_1_7_0:
            # Hack for old torch, works on new torch
            blocklist.update({'all',  'any'})

        errors = []
        for func_type, group in groups.items():
            if func_type == 'data_func':
                for row in group:
                    # TODO: better signature registration so we know how we
                    # need to call the data. For now blocklist non-easy cases

                    func_name = row['func_name']
                    if func_name in blocklist:
                        continue

                    print(f'func_name={func_name}')
                    func = getattr(ArrayAPI, func_name)
                    np_func = getattr(arrayapi.ArrayAPI._numpy, func_name)
                    pt_func = getattr(arrayapi.ArrayAPI._torch, func_name)

                    np_result1 = func(np_data1)
                    np_result2 = np_func(np_data1)
                    pt_result1 = func(pt_data1)
                    pt_result2 = pt_func(pt_data1)

                    results = [np_result1, np_result2, pt_result1, pt_result2]

                    flag = True
                    if isinstance(np_result1, tuple):
                        for a, b in ub.iter_window(results, 2):
                            # flag &= ub.IndexableWalker(a, list_cls=(np.ndarray, torch.Tensor, tuple, list)).allclose(b)
                            flag &= indexable_allclose(a, b)
                    else:
                        results = [ArrayAPI.numpy(r) for r in results]
                        for a, b in ub.iter_window(results, 2):
                            flag &= np.all(np.isclose(a, b))

                    if not flag:
                        errors.append(row)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwarray/tests/test_arrayapi.py
    """
    test_numpy_torch_compat()
