"""
Torch specific extensions
"""
import numpy as np
try:
    import torch
except Exception:
    torch = None


def one_hot_embedding(labels, num_classes, dim=1):
    """
    Embedding labels to one-hot form.

    Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.
        dim (int): dimension which will be created, if negative

    Returns:
        Tensor: encoded labels, sized [N,#classes].

    References:
        https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4

    Example:
        >>> # each element in target has to have 0 <= value < C
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> labels = torch.LongTensor([0, 0, 1, 4, 2, 3])
        >>> num_classes = max(labels) + 1
        >>> t = one_hot_embedding(labels, num_classes)
        >>> assert all(row[y] == 1 for row, y in zip(t.numpy(), labels.numpy()))
        >>> import ubelt as ub
        >>> print(ub.repr2(t.numpy().tolist()))
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
        >>> t2 = one_hot_embedding(labels.numpy(), num_classes)
        >>> assert np.all(t2 == t.numpy())
        >>> if torch.cuda.is_available():
        >>>     t3 = one_hot_embedding(labels.to(0), num_classes)
        >>>     assert np.all(t3.cpu().numpy() == t.numpy())

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> nC = num_classes = 3
        >>> labels = (torch.rand(10, 11, 12) * nC).long()
        >>> assert one_hot_embedding(labels, nC, dim=0).shape == (3, 10, 11, 12)
        >>> assert one_hot_embedding(labels, nC, dim=1).shape == (10, 3, 11, 12)
        >>> assert one_hot_embedding(labels, nC, dim=2).shape == (10, 11, 3, 12)
        >>> assert one_hot_embedding(labels, nC, dim=3).shape == (10, 11, 12, 3)
        >>> labels = (torch.rand(10, 11) * nC).long()
        >>> assert one_hot_embedding(labels, nC, dim=0).shape == (3, 10, 11)
        >>> assert one_hot_embedding(labels, nC, dim=1).shape == (10, 3, 11)
        >>> labels = (torch.rand(10) * nC).long()
        >>> assert one_hot_embedding(labels, nC, dim=0).shape == (3, 10)
        >>> assert one_hot_embedding(labels, nC, dim=1).shape == (10, 3)
    """
    if torch is not None and torch.is_tensor(labels):
        in_dims = labels.ndimension()
        if dim < 0:
            dim = in_dims - dim + 1
        if dim == 1 and in_dims == 1:
            # normal case where everything is already flat
            y = torch.eye(int(num_classes), device=labels.device)
            y_onehot = y[labels]
        else:
            # non-flat case (note that this would handle the normal case, but
            # why do extra work?)
            y = torch.eye(int(num_classes), device=labels.device)
            flat_y_onehot = y[labels.view(-1)]
            y_onehot = flat_y_onehot.view(*list(labels.shape) + [num_classes])
            if dim != in_dims:
                dim_order = list(range(in_dims))
                dim_order.insert(dim, in_dims)
                y_onehot = y_onehot.permute(*dim_order)
    else:
        if dim < 0:
            dim = labels.ndim - dim + 1
        flag = (dim != 1 or labels.ndim == 2)
        if flag:
            orig_shape = labels.shape
            labels = labels.reshape(-1)
            # raise NotImplementedError('not implemented for this case')
        y = np.eye(int(num_classes))
        y_onehot = y[labels]
        if flag:
            new_shape =  list(orig_shape) + [num_classes]
            y_onehot = y_onehot.reshape(*new_shape)
            old_axes = list(range(len(orig_shape)))
            new_axes = old_axes
            new_axes.insert(dim, len(orig_shape))
            y_onehot = y_onehot.transpose(*new_axes)
    return y_onehot


def one_hot_lookup(data, indices):
    """
    Return value of a particular column for each row in data.

    Each item in labels corresonds to a row in ``data``. Returns the index
    specified at each row.

    Args:
        data (ArrayLike): N x C float array of values
        indices (ArrayLike): N integer array between 0 and C.
            This is an column index for each row in ``data``.

    Returns:
        ArrayLike: the selected probability for each row

    Notes:
        This is functionally equivalent to
        ``[row[c] for row, c in zip(data, indices)]`` except that it is
        works with pure matrix operations.

    TODO:
        - [ ] Allow the user to specify which dimension indices should be
              zipped over. By default it should be dim=0

        - [ ] Allow the user to specify which dimension indices should select
              from. By default it should be dim=1.

    Example:
        >>> from kwarray.util_torch import *  # NOQA
        >>> data = np.array([
        >>>     [0, 1, 2],
        >>>     [3, 4, 5],
        >>>     [6, 7, 8],
        >>>     [9, 10, 11],
        >>> ])
        >>> indices = np.array([0, 1, 2, 1])
        >>> res = one_hot_lookup(data, indices)
        >>> print('res = {!r}'.format(res))
        res = array([ 0,  4,  8, 10])
        >>> alt = np.array([row[c] for row, c in zip(data, indices)])
        >>> assert np.all(alt == res)


    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> data = torch.from_numpy(np.array([
        >>>     [0, 1, 2],
        >>>     [3, 4, 5],
        >>>     [6, 7, 8],
        >>>     [9, 10, 11],
        >>> ]))
        >>> indices = torch.from_numpy(np.array([0, 1, 2, 1])).long()
        >>> res = one_hot_lookup(data, indices)
        >>> print('res = {!r}'.format(res))
        res = tensor([ 0,  4,  8, 10]...)
        >>> alt = torch.LongTensor([row[c] for row, c in zip(data, indices)])
        >>> assert torch.all(alt == res)

    Ignore:
        >>> # xdoctest: +REQUIRES(module:torch, module:onnx, module:onnx_tf)
        >>> # Test if this converts to ONNX
        >>> from kwarray.util_torch import *  # NOQA
        >>> import torch.onnx
        >>> import io
        >>> import onnx
        >>> import onnx_tf.backend
        >>> import numpy as np
        >>> data = torch.from_numpy(np.array([
        >>>     [0, 1, 2],
        >>>     [3, 4, 5],
        >>>     [6, 7, 8],
        >>>     [9, 10, 11],
        >>> ]))
        >>> indices = torch.from_numpy(np.array([0, 1, 2, 1])).long()
        >>> class TFConvertWrapper(torch.nn.Module):
        >>>     def forward(self, data, indices):
        >>>         return one_hot_lookup(data, indices)
        >>> ###
        >>> # Test the ONNX export
        >>> wrapped = TFConvertWrapper()
        >>> onnx_file = io.BytesIO()
        >>> torch.onnx.export(
        >>>     wrapped, tuple([data, indices]),
        >>>     input_names=['data', 'indices'],
        >>>     output_names=['out'],
        >>>     f=onnx_file,
        >>>     opset_version=11,
        >>>     verbose=1,
        >>> )
        >>> onnx_file.seek(0)
        >>> onnx_model = onnx.load(onnx_file)
        >>> onnx_tf_model = onnx_tf.backend.prepare(onnx_model)
        >>> # Test that the resulting graph tensors are concretely sized.
        >>> import tensorflow as tf
        >>> onnx_gd = onnx_tf_model.graph.as_graph_def()
        >>> output_tensors = tf.import_graph_def(
        >>>     onnx_gd,
        >>>     input_map={},
        >>>     return_elements=[onnx_tf_model.tensor_dict[ol].name for ol in onnx_tf_model.outputs]
        >>> )
        >>> assert all(isinstance(d.value, int) for t in output_tensors for d in t.shape)
        >>> tf_outputs = onnx_tf_model.run([data, indices])
        >>> pt_outputs = wrapped(data, indices)
        >>> print('tf_outputs = {!r}'.format(tf_outputs))
        >>> print('pt_outputs = {!r}'.format(pt_outputs))
        >>> ###
        >>> # Test if data is more than 2D
        >>> shape = (4, 3, 8)
        >>> data = torch.arange(int(np.prod(shape))).view(*shape).float()
        >>> indices = torch.from_numpy(np.array([0, 1, 2, 1])).long()
        >>> onnx_file = io.BytesIO()
        >>> torch.onnx.export(
        >>>     wrapped, tuple([data, indices]),
        >>>     input_names=['data', 'indices'],
        >>>     output_names=['out'],
        >>>     f=onnx_file,
        >>>     opset_version=11,
        >>>     verbose=1,
        >>> )
        >>> onnx_file.seek(0)
        >>> onnx_model = onnx.load(onnx_file)
        >>> onnx_tf_model = onnx_tf.backend.prepare(onnx_model)
        >>> # Test that the resulting graph tensors are concretely sized.
        >>> import tensorflow as tf
        >>> onnx_gd = onnx_tf_model.graph.as_graph_def()
        >>> output_tensors = tf.import_graph_def(
        >>>     onnx_gd,
        >>>     input_map={},
        >>>     return_elements=[onnx_tf_model.tensor_dict[ol].name for ol in onnx_tf_model.outputs]
        >>> )
        >>> assert all(isinstance(d.value, int) for t in output_tensors for d in t.shape)
        >>> tf_outputs = onnx_tf_model.run([data, indices])
        >>> pt_outputs = wrapped(data, indices)
        >>> print('tf_outputs = {!r}'.format(tf_outputs))
        >>> print('pt_outputs = {!r}'.format(pt_outputs))
    """
    if torch is not None and torch.is_tensor(indices):
        if torch.onnx.is_in_onnx_export():

            # Don't use eye for ONNX
            ASSUME_OPTSET = 10

            device = indices.device

            n = data.shape[1]
            # Have to construct eye manually to satisfy onnx
            # Manually construct diag indices
            row_idxs = torch.arange(n, device=device)
            eye_idxs = row_idxs + (row_idxs * n)

            if ASSUME_OPTSET >= 11:
                # With opset 11 we use the "put" operation to directly
                # populate the diagonal elements.
                eye = torch.zeros((n, n), dtype=data.dtype, device=device)
                flat_eye = eye.view(n * n)
                diag_elem = torch.ones(n, dtype=data.dtype, device=device)
                flat_eye[eye_idxs] = diag_elem
            elif ASSUME_OPTSET >= 10:
                # With opset 10 we cannot use "put", so we have to get spicey
                # Construct the flat indexes of an NxN matrix
                flat_idxs = torch.arange(n * n)
                # Broadcast and check if these flat indexes are equal to the
                # target indexes, then sum over the broadcast dimension
                flat_eye = (flat_idxs[:, None] == eye_idxs[None, :]).to(data.dtype).sum(dim=1)
            else:
                raise AssertionError('ASSUME_OPTSET = {}'.format(ASSUME_OPTSET))

            eye = flat_eye.view(n, n)

            # Do the normal lookup in the eye matrix to get the OHE
            ohe = eye[indices]
            # need to pad OHE with extra dimensions for broadcasting
            extra_dims = len(data.shape) - 2
            if extra_dims > 0:
                ohe = ohe[(Ellipsis,) + (None,) * extra_dims]

            # Have to use multiply trick to satisfy onnx
            out = (data * ohe).sum(dim=1)
        else:
            ohe = torch.eye(data.shape[1], dtype=torch.bool, device=indices.device)[indices]
            out = data[ohe]
    else:
        # ohe = kwarray.one_hot_embedding(indices, data.shape[1]).astype(np.bool)
        # Constructing the OHE with a small dtype offers a sizable speed advantage
        ohe = np.eye(data.shape[1], dtype=np.bool)[indices]
        out = data[ohe]
    return out


if __name__ == '__main__':
    """
    CommandLine:
        python -m kwarray.util_torch all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
