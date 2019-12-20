
def main():
    import kwarray
    import torch
    import ubelt as ub
    import numpy as np

    ti = ub.Timerit(100, bestof=20, verbose=2, unit='us')

    # kwarray.ArrayAPI.coerce('torch').randn
    # TODO: rand / randn for ArrayAPI

    # import torch
    # m1_base = torch.rand(1000, 1000)
    # m2_base = torch.rand(1000, 1000)

    # import netharn as nh
    # xpu = nh.XPU(1)
    # m1_base = xpu.move(m1_base)
    # m2_base = xpu.move(m2_base)

    def make_rand_numpy(dtype=np.float32):
        m1_base = np.random.rand(100, 100)
        m2_base = np.random.rand(100, 100)
        m1 = kwarray.ArrayAPI.astype(m1_base, dtype)
        m2 = kwarray.ArrayAPI.astype(m2_base, dtype)
        return m1, m2

    def make_rand_torch(dtype=np.float32, device=0):
        m1_base = torch.rand(1000, 1000)
        m2_base = torch.rand(1000, 1000)

        import netharn as nh
        xpu = nh.XPU(device)
        m1_base = xpu.move(m1_base)
        m2_base = xpu.move(m2_base)
        m1 = kwarray.ArrayAPI.astype(m1_base, dtype)
        m2 = kwarray.ArrayAPI.astype(m2_base, dtype)
        return m1, m2

    make_rand = make_rand_numpy
    make_rand = make_rand_torch

    m1_base, m2_base = make_rand()

    impl = kwarray.ArrayAPI.coerce(m1_base)

    dtype = 'float32'
    for timer in ti.reset(str(dtype)):
        m1, m2 = make_rand(dtype=dtype)
        with timer:
            impl.matmul(m1, m2)
            if make_rand is make_rand_torch:
                torch.cuda.synchronize()

    dtype = 'float16'
    for timer in ti.reset(str(dtype)):
        m1, m2 = make_rand(dtype=dtype)
        with timer:
            impl.matmul(m1, m2)
            if make_rand is make_rand_torch:
                torch.cuda.synchronize()

    # dtype = 'float64'
    # for timer in ti.reset(str(dtype)):
    #     m1, m2 = make_rand(dtype=dtype)
    #     with timer:
    #         impl.matmul(m1, m2)
    #         if make_rand is make_rand_torch:
    #             torch.cuda.synchronize()

    # dtype = 'float128'
    # for timer in ti.reset(str(dtype)):
    #     m1, m2 = make_rand(dtype=dtype)
    #     with timer:
    #         impl.matmul(m1, m2)
    #         if make_rand is make_rand_torch:
    #             torch.cuda.synchronize()

    # dtype = 'complex256'
    # for timer in ti.reset(str(dtype)):
    #     m1, m2 = make_rand(dtype=dtype)
    #     with timer:
    #         impl.matmul(m1, m2)
    #         if make_rand is make_rand_torch:
    #             torch.cuda.synchronize()

    print(ub.repr2(ti.measures, precision=6))

    import pandas as pd
    print(pd.DataFrame(ti.measures))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwarray/dev/bench_dtype_ops.py
    """
    main()
