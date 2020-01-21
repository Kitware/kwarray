"""
Torch specific extensions
"""
import torch
import numpy as np


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
    if torch.is_tensor(labels):
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


def one_hot_lookup(probs, labels):
    """
    Return probability of a particular label (usually true labels) for each item

    Each item in labels corresonds to a row in probs. Returns the index
    specified at each row.

    Args:
        probs (ArrayLike): N x C float array of probabilities
        labels (ArrayLike): N integer array between 0 and C

    Returns:
        ArrayLike: the selected probability for each row

    Example:
        >>> probs = np.array([
        >>>     [0, 1, 2],
        >>>     [3, 4, 5],
        >>>     [6, 7, 8],
        >>>     [9, 10, 11],
        >>> ])
        >>> labels = np.array([0, 1, 2, 1])
        >>> one_hot_lookup(probs, labels)
        array([ 0,  4,  8, 10])
    """
    assert not torch.is_tensor(labels), 'not implemented yet'
    # ohe = kwarray.one_hot_embedding(labels, probs.shape[1]).astype(np.bool)
    # Constructing the OHE with a small dtype offers a sizable speed advantage
    ohe = np.eye(probs.shape[1], dtype=np.bool)[labels]
    out = probs[ohe]
    return out
