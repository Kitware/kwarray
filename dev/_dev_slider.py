def add_fast(stitcher, batch_idxs, values, weight=None, assume_order=True):
    """
    new faster version

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> from kwarray.util_slider import *  # NOQA
        >>> import sys
        >>> # Build a high resolution image and slice it into chips
        >>> frames = np.random.rand(1, 200, 100, 100).astype(np.float32)
        >>> window = (frames.shape[0], 15, 15, 15)
        >>> slider = SlidingWindow(frames.shape, window, stride=(1, 1, 1, 1))
        >>> class SlidingIndexDataset(torch.utils.data.Dataset):
        ...     def __init__(slider_dset, slider, source):
        ...         slider_dset.slider = slider
        ...         slider_dset.source = source
        ...     def __len__(slider_dset):
        ...         return len(slider_dset.slider)
        ...     def __getitem__(slider_dset, index):
        ...         slider = slider_dset.slider
        ...         slices = slider[index]
        ...         basis_idx = np.unravel_index(index, slider.basis_shape)
        ...         chip = slider_dset.source[slices]
        ...         tensor_chip = torch.FloatTensor(chip)
        ...         tensor_basis_idx = torch.LongTensor(np.array(basis_idx))
        ...         return tensor_basis_idx, tensor_chip
        >>> slider_dset = SlidingIndexDataset(slider, frames)
        >>> n_classes = 2
        >>> device = None
        >>> stitcher = Stitcher(slider.basis_shape[1:] + [n_classes], device=device)
        >>> loader = torch.utils.data.DataLoader(slider_dset, batch_size=10)
        >>> batch_iter = iter(loader)
        >>> batch = next(batch_iter)
        >>> batch_idxs_tensors_, chips = batch
        >>> invar = (chips)
        >>> conv = torch.nn.Conv3d(frames.shape[0], n_classes, window[1:])
        >>> values = conv(invar).data
        >>> # remove channel
        >>> weight = None
        >>> batch_idxs = batch_idxs_tensors_[:, 1:]
        >>> stitcher.add_fast(batch_idxs, values, weight, assume_order=True)

    Time:
        torch.cuda.init()

        weight = None

        import ubelt as ub
        device = torch.device(0)
        values = values.to(device)
        stitcher = Stitcher(slider.basis_shape[1:] + [n_classes], device=device)
        for timer in ub.Timerit(100, bestof=10, label='gpu'):
            with timer:
                stitcher.add_fast(batch_idxs, values, weight, assume_order=True)

        stitcher = Stitcher(slider.basis_shape[1:] + [n_classes], device='numpy')
        batch_idxs_np = batch_idxs.numpy()
        values_np = values.cpu().numpy()
        for timer in ub.Timerit(100, bestof=10, label='numpy'):
            with timer:
                stitcher.add_fast(batch_idxs_np, values_np, weight, assume_order=True)

    Benchmark:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> # TODO: refactor to make this work
        >>> from kwarray.util_slider import *  # NOQA
        >>> import sys
        >>> # setup benchmark
        >>> frames = np.random.rand(1, 50, 100, 100).astype(np.float32)
        >>> window = (frames.shape[0], 20, 20, 20)
        >>> slider = SlidingWindow(frames.shape, window, stride=(1, 1, 1, 1))
        >>> class SlidingIndexDataset(torch.utils.data.Dataset):
        ...     def __init__(slider_dset, slider, source):
        ...         slider_dset.slider = slider
        ...         slider_dset.source = source
        ...     def __len__(slider_dset):
        ...         return len(slider_dset.slider)
        ...     def __getitem__(slider_dset, index):
        ...         slider = slider_dset.slider
        ...         slices = slider[index]
        ...         basis_idx = np.unravel_index(index, slider.basis_shape)
        ...         chip = slider_dset.source[slices]
        ...         tensor_chip = torch.FloatTensor(chip)
        ...         tensor_basis_idx = torch.LongTensor(np.array(basis_idx))
        ...         return tensor_basis_idx, tensor_chip
        >>> slider_dset = SlidingIndexDataset(slider, frames)
        >>> loader = torch.utils.data.DataLoader(slider_dset, batch_size=1024)
        >>> n_classes = 2
        >>> device = torch.device(0)
        >>> conv = torch.nn.Conv3d(window[0], n_classes, window[1:])
        >>> conv = conv.to(device)
        >>> #weight = torch.rand(n_classes, 1, 1, 1)[None, :]
        >>> #weight = (weight).to(device)
        >>> #weight_np = weight.cpu().numpy()
        >>> weight = weight_np = None
        >>> # do dummy computation to warm up gpu
        >>> conv(slider_dset[0][1][None, :].to(device))
        >>> torch.set_grad_enabled(False)
        >>> conv.train(False)
        >>> base_shape = slider.basis_shape[1:]
        >>> # ---------------------------------------
        >>> # Benchmark on-gpu stitching with pytorch
        >>> import tqdm
        >>> t1 = ub.Timerit(3, bestof=3, label='gpu')
        >>> for timer in tqdm.tqdm(t1, total=3, leave=True):
        >>>     with timer:
        >>>         stitcher = Stitcher(base_shape + [n_classes], device=device)
        >>>         for batch in loader:
        >>>             batch_idxs_tensors_, chips = batch
        >>>             invar = (chips).to(device)
        >>>             values = conv(invar).data
        >>>             batch_idxs = batch_idxs_tensors_[:, 1:]
        >>>             stitcher.add_fast(batch_idxs, values, weight,
        >>>                               assume_order=True)
        >>> # ---------------------------------------
        >>> # Benchmark on-cpu stitching with numpy
        >>> t2 = ub.Timerit(3, bestof=3, label='numpy')
        >>> for timer in tqdm.tqdm(t2, total=3, leave=True):
        >>>     with timer:
        >>>         stitcher = Stitcher(base_shape + [n_classes], device='numpy')
        >>>         for batch in iter(loader):
        >>>             batch_idxs_tensors_, chips = batch
        >>>             invar = (chips).to(device)
        >>>             values_np = conv(invar).data.cpu().numpy()
        >>>             batch_idxs_np = batch_idxs_tensors_[:, 1:].numpy()
        >>>             stitcher.add_fast(batch_idxs_np, values_np,
        >>>                               weight_np, assume_order=True)
        >>> # VERDICT:
        >>> # Async GPU stitching gives a minor but insignificant speedup
        >>> # GPU:   time per loop: best=4.394 s, mean=4.394 ± 0.0 s
        >>> # NUMPY: time per loop: best=4.876 s, mean=4.876 ± 0.0 s
    """
    # Part of __init__ in original
    # stitcher._cumprod = np.cumprod(list(shape[::-1][:-1]))[::-1]
    # stitcher._cumprod = torch.LongTensor(np.array(stitcher._cumprod))
    if stitcher.device != 'numpy':
        # ON GPU STITCHING
        n_classes = stitcher.shape[-1]
        end = batch_idxs.shape[0] - 1
        t_base_multi_idxs = batch_idxs[[0, end]]

        # we dont need a trailing 1 because we arent padding extra zeros
        cumprod = stitcher._cumprod[None :]
        ravel_idxs_range = (t_base_multi_idxs * cumprod).sum(dim=1)
        first = ravel_idxs_range[0]
        last = ravel_idxs_range[-1] + n_classes
        ravel_sl = slice(first, last)
        ravel_index = ravel_sl

        if weight is None:
            stitcher.sumview[ravel_index] += values.view(-1)
            stitcher.weightview[ravel_index] += 1.0
        else:
            stitcher.sumview[ravel_index] += (values * weight).view(-1)
            stitcher.weightview[ravel_index] += weight.view(-1)
    else:
        # TODO: maybe check if the input is a tensor?
        shape = stitcher.shape
        n_classes = shape[-1]
        # if we assume we get data in order, its even faster
        if assume_order:
            last = batch_idxs.shape[0] - 1
            base_multi_idxs = tuple(batch_idxs[[0, last]].T)
            # Add extra dimension for output classes
            extra_multi_idxs = np.zeros(2, dtype=np.int)
            multi_idxs_range = base_multi_idxs + (extra_multi_idxs,)
            ravel_idxs_range = np.ravel_multi_index(multi_idxs_range, dims=shape)
            first = ravel_idxs_range[0]
            last = ravel_idxs_range[-1] + n_classes
            ravel_sl = slice(first, last)
            ravel_index = ravel_sl
        else:
            base_multi_idxs = tuple(batch_idxs.T)
            extra_multi_idxs = np.zeros(len(batch_idxs), dtype=np.int)
            # The indices for the 0-th class (which should be the last dimension)
            multi_idxs_first = base_multi_idxs + (extra_multi_idxs,)
            ravel_idxs_first = np.ravel_multi_index(multi_idxs_first, dims=shape)

            # The indices for the next classes should be sequentially after
            all_ravel_idxs = [ravel_idxs_first[None, :]]
            for i in range(1, n_classes):
                all_ravel_idxs.append((ravel_idxs_first + i)[None, :])
            # raveled indices that correspond with raveled data
            ravel_idxs = np.vstack(all_ravel_idxs).T.ravel()
            # assert np.sum(1 - np.diff(ravel_idxs)), 'we cant assume order'
            ravel_index = ravel_idxs

        if weight is None:
            stitcher.sumview[ravel_index] += values.ravel()
            stitcher.weightview[ravel_index] += 1.0
        else:
            stitcher.sumview[ravel_index] += (values * weight).ravel()
            stitcher.weightview[ravel_index] += np.ravel(weight)
