# -*- coding: utf-8 -*-
"""
The ArrayAPI is a common API that works exactly the same on both torch.Tensors
and numpy.ndarrays.


The ArrayAPI is a combination of efficiency and convinience. It is convinient
because you can just use an operation directly, it will type check the data,
and apply the appropriate method. But it is also efficient because it can be
used with minimal type checking by accessing a type-specific backend.

For example, you can do:

.. code:: python

    impl = kwarray.ArrayAPI.coerce(data)

And then impl will give you direct access to the appropriate methods without
any type checking overhead.  e..g. ``impl.<op-you-want>(data)``

But you can also do ``kwarray.ArrayAPI.<op-you-want>(data)`` on anything and it
will do type checking and then do the operation you want.

Example:
    >>> # xdoctest: +REQUIRES(module:torch)
    >>> import torch
    >>> import numpy as np
    >>> data1 = torch.rand(10, 10)
    >>> data2 = data1.numpy()
    >>> # Method 1: grab the appropriate sub-impl
    >>> impl1 = ArrayAPI.impl(data1)
    >>> impl2 = ArrayAPI.impl(data2)
    >>> result1 = impl1.sum(data1, axis=0)
    >>> result2 = impl2.sum(data2, axis=0)
    >>> res1_np = ArrayAPI.numpy(result1)
    >>> res2_np = ArrayAPI.numpy(result2)
    >>> print('res1_np = {!r}'.format(res1_np))
    >>> print('res2_np = {!r}'.format(res2_np))
    >>> assert np.allclose(res1_np, res2_np)
    >>> # Method 2: choose the impl on the fly
    >>> result1 = ArrayAPI.sum(data1, axis=0)
    >>> result2 = ArrayAPI.sum(data2, axis=0)
    >>> res1_np = ArrayAPI.numpy(result1)
    >>> res2_np = ArrayAPI.numpy(result2)
    >>> print('res1_np = {!r}'.format(res1_np))
    >>> print('res2_np = {!r}'.format(res2_np))
    >>> assert np.allclose(res1_np, res2_np)

Example:
    >>> # xdoctest: +REQUIRES(module:torch)
    >>> import torch
    >>> import numpy as np
    >>> data1 = torch.rand(10, 10)
    >>> data2 = data1.numpy()
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import ubelt as ub
import six
from functools import partial
from distutils.version import LooseVersion


try:
    import torch
except Exception:
    torch = None
    _TORCH_HAS_BOOL_COMP = False
else:
    _TORCH_HAS_BOOL_COMP = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')


def _get_funcname(func):
    if six.PY2:
        try:
            return func.func_name
        except AttributeError:
            return func.__name__
    else:
        return func.__name__


class _ImplRegistry(object):
    def __init__(self):
        self.registered = {
            'torch': {},
            'numpy': {},
            'api': {},
        }

    def _register(self, func, func_type, impl):
        func_name = _get_funcname(func)

        assert func_type in {
            'data_func',           # methods where the first argument is an array
            'array_sequence',  # methods that take a sequence of arrays
            'shape_creation',  # methods that allocate new data via shape
        }
        self.registered[impl][func_name] = {
            'func': func,
            'func_name': func_name,
            'func_type': func_type,
        }
        return staticmethod(func)

    def _implmethod(self, func=None, func_type='data_func', impl=None):
        _register = partial(self._register, func_type=func_type, impl=impl)
        if func is None:
            return _register
        else:
            return _register(func)

    def _apimethod(self, key=None, func_type='data_func'):
        """
        Creates wrapper for a "data method" --- i.e. a ArrayAPI function that has
        only one main argument, which is an array.
        """
        if isinstance(key, six.string_types):
            # numpy_func = self.registered['numpy'][key]['func']
            # torch_func = self.registered['torch'][key]['func']
            numpy_func = getattr(NumpyImpls, key)
            if torch is not None:
                torch_func = getattr(TorchImpls, key)
            def func(data, *args, **kwargs):
                if torch is not None and torch.is_tensor(data):
                    return torch_func(data, *args, **kwargs)
                elif isinstance(data, np.ndarray):
                    return numpy_func(data, *args, **kwargs)
                elif isinstance(data, (list, tuple)):
                    return numpy_func(np.asarray(data), *args, **kwargs)
                else:
                    raise TypeError('unknown type {}'.format(type(data)))
            func.__name__ = str(key)  # the str wrap is for python2
        else:
            func = key
        _register = partial(self._register, func_type=func_type, impl='api')
        if func is None:
            return _register
        else:
            return _register(func)

    def _ensure_datamethods_names_are_registered(self):
        """
        Checks to make sure all methods are implemented in both
        torch and numpy implementations as well as exposed in the ArrayAPI.
        """
        # Check that we didn't forget to add anything
        # api_names = {
        #     key for key, value in ArrayAPI.__dict__.items()
        #     if isinstance(value, staticmethod)
        # }
        # import xdev
        # xdev.fix_embed_globals()
        numpy_names = set(self.registered['numpy'].keys())
        torch_names = set(self.registered['torch'].keys())
        api_names = set(self.registered['api'].keys())

        universe = (torch_names | numpy_names)

        grouped = ub.group_items(self.registered['numpy'].values(), lambda x: x['func_type'])
        shape_creation_methods = {item['func_name'] for item in grouped['shape_creation']}

        missing_torch = universe - torch_names
        missing_numpy = universe - numpy_names
        missing_api = universe - api_names - shape_creation_methods

        if missing_numpy or missing_torch:
            message = ub.codeblock(
                '''
                missing_torch = {}
                missing_numpy = {}
                ''').format(missing_torch, missing_numpy)
            raise AssertionError(message)

        if missing_api:
            message = ub.codeblock(
                '''
                WARNING (these will be implicitly created):
                missing_api = {}
                '''.format(missing_api))
            print(message)
            import warnings
            warnings.warn(message)

        # Insert any missing functions into the API implicitly
        autogen = []
        for func_name in missing_api:
            func_type = self.registered['numpy'][func_name]['func_type']
            autogen.append("{} = _apimethod('{}', func_type='{}')".format(func_name, func_name, func_type))
            implicitmethod = self._apimethod(func_name, func_type=func_type)
            setattr(ArrayAPI, func_name, implicitmethod)

        for func_name in universe:
            infos = []
            try:
                for impl in ['numpy', 'torch', 'api']:
                    info = self.registered[impl][func_name]
                    info.pop('func')
                    infos.append(info)
                assert ub.allsame(infos), 'fix {}'.format(infos)
            except KeyError:
                pass

        if autogen:
            print('Please explicitly add the following code:')
            print('\n'.join(autogen))

# alias staticmethod to make it easier to keep track of which IMPL we are in

_REGISTERY = _ImplRegistry()
_torchmethod = partial(_REGISTERY._implmethod, impl='torch')
_numpymethod = partial(_REGISTERY._implmethod, impl='numpy')
_apimethod = _REGISTERY._apimethod


class TorchImpls(object):
    """
    Torch backend for the ArrayAPI API
    """

    is_tensor = True
    is_numpy = False

    @_torchmethod(func_type='array_sequence')
    def cat(datas, axis=-1):
        return torch.cat(datas, dim=axis)

    @_torchmethod(func_type='array_sequence')
    def hstack(datas):
        """
        Concatenates along axis=0 if inputs are are 1-D otherwise axis=1

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> datas1 = [torch.arange(10), torch.arange(10)]
            >>> datas2 = [d.numpy() for d in datas1]
            >>> ans1 = TorchImpls.hstack(datas1)
            >>> ans2 = NumpyImpls.hstack(datas2)
            >>> assert np.all(ans1.numpy() == ans2)
        """
        axis = 1
        if len(datas[0].shape) == 1:
            axis = 0
        return torch.cat(datas, dim=axis)

    @_torchmethod(func_type='array_sequence')
    def vstack(datas):
        """
        Ensures that inputs datas are at least 2D (prepending a dimension of 1)
        and then concats along axis=0.

        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> datas1 = [torch.arange(10), torch.arange(10)]
            >>> datas2 = [d.numpy() for d in datas1]
            >>> ans1 = TorchImpls.vstack(datas1)
            >>> ans2 = NumpyImpls.vstack(datas2)
            >>> assert np.all(ans1.numpy() == ans2)
        """
        datas = [TorchImpls.atleast_nd(d, n=2, front=True) for d in datas]
        return torch.cat(datas, dim=0)

    @_torchmethod(func_type='data_func')
    def atleast_nd(arr, n, front=False):
        ndims = len(arr.shape)
        if n is not None and ndims <  n:
            # append the required number of dimensions to the front or back
            if front:
                expander = (None,) * (n - ndims) + (Ellipsis,)
            else:
                expander = (Ellipsis,) + (None,) * (n - ndims)
            arr = arr[expander]
        return arr

    @_torchmethod(func_type='data_func')
    def view(data, *shape):
        data_ = data.view(*shape)
        return data_

    @_torchmethod(func_type='data_func')
    def take(data, indices, axis=None):
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(data.device)
        if axis is None:
            return data.take(indices)
        else:
            return torch.index_select(data, dim=axis, index=indices)

    @_torchmethod(func_type='data_func')
    def compress(data, flags, axis=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import kwarray
            >>> data = torch.rand(10, 4, 2)
            >>> impl = kwarray.ArrayAPI.coerce(data)

            >>> axis = 0
            >>> flags = (torch.arange(data.shape[axis]) % 2) == 0
            >>> out = impl.compress(data, flags, axis=axis)
            >>> assert tuple(out.shape) == (5, 4, 2)

            >>> axis = 1
            >>> flags = (torch.arange(data.shape[axis]) % 2) == 0
            >>> out = impl.compress(data, flags, axis=axis)
            >>> assert tuple(out.shape) == (10, 2, 2)

            >>> axis = 2
            >>> flags = (torch.arange(data.shape[axis]) % 2) == 0
            >>> out = impl.compress(data, flags, axis=axis)
            >>> assert tuple(out.shape) == (10, 4, 1)

            >>> axis = None
            >>> data = torch.rand(10)
            >>> flags = (torch.arange(data.shape[0]) % 2) == 0
            >>> out = impl.compress(data, flags, axis=axis)
            >>> assert tuple(out.shape) == (5,)
        """
        if not torch.is_tensor(flags):
            flags = np.asarray(flags).astype(np.uint8)
            if _TORCH_HAS_BOOL_COMP:
                flags = torch.BoolTensor(flags).to(data.device)
            else:
                flags = torch.ByteTensor(flags).to(data.device)
        if flags.ndimension() != 1:
            raise ValueError('condition must be a 1-d tensor')
        if axis is None:
            return torch.masked_select(data.view(-1), flags)
        else:
            out_shape = list(data.shape)
            out_shape[axis] = int(flags.sum())

            fancy_shape = [1] * len(out_shape)
            fancy_shape[axis] = flags.numel()
            explicit_flags = flags.view(*fancy_shape)

            flat_out = torch.masked_select(data, explicit_flags)
            out = flat_out.view(*out_shape)
            return out

    @_torchmethod(func_type='data_func')
    def tile(data, reps):
        """
        Implement np.tile in torch

        Example:
            >>> # xdoctest: +SKIP
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> data = torch.arange(10)[:, None]
            >>> ans1 = ArrayAPI.tile(data, [1, 2])
            >>> ans2 = ArrayAPI.tile(data.numpy(), [1, 2])
            >>> assert np.all(ans1.numpy() == ans2)

        Doctest:
            >>> # xdoctest: +SKIP
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> shapes = [(3,), (3, 4,), (3, 5, 7), (1,), (3, 1, 3)]
            >>> for shape in shapes:
            >>>     data = torch.rand(*shape)
            >>>     for axis in range(len(shape)):
            >>>         for reps in it.product(*[range(0, 4)] * len(shape)):
            >>>             ans1 = ArrayAPI.tile(data, reps)
            >>>             ans2 = ArrayAPI.tile(data.numpy(), reps)
            >>>             #print('ans1.shape = {!r}'.format(tuple(ans1.shape)))
            >>>             #print('ans2.shape = {!r}'.format(tuple(ans2.shape)))
            >>>             assert np.all(ans1.numpy() == ans2)
        """
        n_prepend = len(reps) - len(data.shape)
        if n_prepend > 0:
            reps = [1] * n_prepend + list(reps)
        if any(r == 0 for r in reps):
            newshape = np.array([r * s for r, s in zip(reps, data.shape)])
            return torch.empty(tuple(newshape), dtype=data.dtype).to(data.device)
        else:
            return data.repeat(reps)

    @_torchmethod(func_type='data_func')
    def repeat(data, repeats, axis=None):
        """
        I'm not actually sure how to implement this efficiently

        Example:
            >>> # xdoctest: +SKIP
            >>> data = torch.arange(10)[:, None]
            >>> ans1 = ArrayAPI.repeat(data, 2, axis=1)
            >>> ans2 = ArrayAPI.repeat(data.numpy(), 2, axis=1)
            >>> assert np.all(ans1.numpy() == ans2)

        Doctest:
            >>> # xdoctest: +SKIP
            >>> shapes = [(3,), (3, 4,), (3, 5, 7)]
            >>> for shape in shapes:
            >>>     data = torch.rand(*shape)
            >>>     for axis in range(len(shape)):
            >>>         for repeats in range(0, 4):
            >>>             ans1 = ArrayAPI.repeat(data, repeats, axis=axis)
            >>>             ans2 = ArrayAPI.repeat(data.numpy(), repeats, axis=axis)
            >>>             assert np.all(ans1.numpy() == ans2)


            ArrayAPI.repeat(data, 2, axis=0)
            ArrayAPI.repeat(data.numpy(), 2, axis=0)

            x = np.array([[1,2],[3,4]])
            np.repeat(x, [1, 2], axis=0)

            ArrayAPI.repeat(data.numpy(), [1, 2])
        """
        raise NotImplementedError
        if False:
            # hack! may not always work
            return np.repeat(data, repeats, axis=axis)
        else:
            if isinstance(repeats, int):
                if axis is None:
                    return data.view(-1).repeat(repeats)
                else:
                    expander = [1] * len(data.shape)
                    expander[axis] = repeats
                    if repeats == 0:
                        newshape = list(data.shape)
                        newshape[axis] = 0
                        return torch.empty(tuple(newshape), dtype=data.dtype).to(data.device)
                    else:
                        return data.repeat(expander)
            else:
                raise NotImplementedError
        pass

    @_torchmethod(func_type='data_func')
    def T(data):
        ndims = data.ndimension()
        if ndims == 2:
            # torch.t can only handle 2 dims
            return data.t()
        else:
            # use permute for compatability
            axes = list(reversed(range(ndims)))
            return data.permute(*axes)

    @_torchmethod(func_type='data_func')
    def transpose(data, axes):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> data1 = torch.rand(2, 3, 5)
            >>> data2 = data1.numpy()
            >>> res1 = ArrayAPI.transpose(data1, (2, 0, 1))
            >>> res2 = ArrayAPI.transpose(data2, (2, 0, 1))
            >>> assert np.all(res1.numpy() == res2)
        """
        return data.permute(axes)

    @_torchmethod(func_type='data_func')
    def numel(data):
        return data.numel()

    # --- Allocators ----

    @_torchmethod(func_type='data_func')
    def full_like(data, fill_value, dtype=None):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.full_like(data, fill_value, dtype=dtype)

    @_torchmethod(func_type='data_func')
    def empty_like(data, dtype=None):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.empty_like(data, dtype=dtype)

    @_torchmethod(func_type='data_func')
    def zeros_like(data, dtype=None):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.zeros_like(data, dtype=dtype)

    @_torchmethod(func_type='data_func')
    def ones_like(data, dtype=None):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.ones_like(data, dtype=dtype)

    @_torchmethod(func_type='shape_creation')
    def full(shape, fill_value, dtype=float):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.full(shape, fill_value, dtype=dtype)

    @_torchmethod(func_type='shape_creation')
    def empty(shape, dtype=float):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.full(shape, dtype=dtype)

    @_torchmethod(func_type='shape_creation')
    def zeros(shape, dtype=float):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.zeros(shape, dtype=dtype)

    @_torchmethod(func_type='shape_creation')
    def ones(shape, dtype=float):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.ones(shape, dtype=dtype)

    # -------

    @_torchmethod(func_type='data_func')
    def argmax(data, axis=None):
        return torch.argmax(data, dim=axis)

    @_torchmethod(func_type='data_func')
    def argsort(data, axis=-1, descending=False):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> from kwarray.arrayapi import *  # NOQA
            >>> rng = np.random.RandomState(0)
            >>> data2 = rng.rand(5, 5)
            >>> data1 = torch.from_numpy(data2)
            >>> res1 = ArrayAPI.argsort(data1)
            >>> res2 = ArrayAPI.argsort(data2)
            >>> assert np.all(res1.numpy() == res2)
            >>> res1 = ArrayAPI.argsort(data1, axis=1)
            >>> res2 = ArrayAPI.argsort(data2, axis=1)
            >>> assert np.all(res1.numpy() == res2)
            >>> res1 = ArrayAPI.argsort(data1, axis=1, descending=True)
            >>> res2 = ArrayAPI.argsort(data2, axis=1, descending=True)
            >>> assert np.all(res1.numpy() == res2)
            >>> data2 = rng.rand(5)
            >>> data1 = torch.from_numpy(data2)
            >>> res1 = ArrayAPI.argsort(data1, axis=0, descending=True)
            >>> res2 = ArrayAPI.argsort(data2, axis=0, descending=True)
            >>> assert np.all(res1.numpy() == res2)
        """
        return torch.argsort(data, dim=axis, descending=descending)

    @_torchmethod(func_type='data_func')
    def max(data, axis=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> data1 = torch.rand(5, 5, 5, 5, 5, 5)
            >>> data2 = data1.numpy()
            >>> res1 = ArrayAPI.max(data1)
            >>> res2 = ArrayAPI.max(data2)
            >>> assert np.all(res1.numpy() == res2)
            >>> res1 = ArrayAPI.max(data1, axis=(4, 0, 1))
            >>> res2 = ArrayAPI.max(data2, axis=(4, 0, 1))
            >>> assert np.all(res1.numpy() == res2)
            >>> res1 = ArrayAPI.max(data1, axis=(5, -2))
            >>> res2 = ArrayAPI.max(data2, axis=(5, -2))
            >>> assert np.all(res1.numpy() == res2)
        """
        if axis is None:
            return torch.max(data)
        elif isinstance(axis, tuple):
            axis_ = [d if d >= 0 else len(data.shape) + d for d in axis]
            temp = data
            for d in sorted(axis_)[::-1]:
                temp = temp.max(dim=d)[0]
            return temp
        else:
            return torch.max(data, dim=axis)[0]

    @_torchmethod(func_type='data_func')
    def max_argmax(data, axis=None):
        """
        Note: this isn't always gaurenteed to be compatibile with numpy
        if there are equal elements in data. See:
        >>> np.ones(10).argmax()   # xdoctest: +IGNORE_WANT
        0
        >>> torch.ones(10).argmax()   # xdoctest: +IGNORE_WANT
        tensor(9)
        """
        return torch.max(data, dim=axis)

    @_torchmethod
    def maximum(data1, data2, out=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> data1 = torch.rand(5, 5)
            >>> data2 = torch.rand(5, 5)
            >>> result1 = TorchImpls.maximum(data1, data2)
            >>> result2 = NumpyImpls.maximum(data1.numpy(), data2.numpy())
            >>> assert np.allclose(result1.numpy(), result2)
        """
        return torch.max(data1, data2, out=out)

    @_torchmethod
    def minimum(data1, data2, out=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> data1 = torch.rand(5, 5)
            >>> data2 = torch.rand(5, 5)
            >>> result1 = TorchImpls.minimum(data1, data2)
            >>> result2 = NumpyImpls.minimum(data1.numpy(), data2.numpy())
            >>> assert np.allclose(result1.numpy(), result2)
        """
        return torch.min(data1, data2, out=out)

    # @_torchmethod(func_type='data_func')
    # def matmul(data1, data2, out=None):
    #     return torch.matmul(data1, data2, out=out)
    if torch is not None:
        matmul = _torchmethod(torch.matmul)

    @_torchmethod(func_type='data_func')
    def sum(data, axis=None):
        if axis is None:
            return data.sum()
        else:
            return data.sum(dim=axis)

    @_torchmethod(func_type='data_func')
    def nan_to_num(x, copy=True):
        if copy:
            x = x.clone()
        x[torch.isnan(x)] = 0
        return x

    @_torchmethod(func_type='data_func')
    def copy(data):
        return torch.clone(data)

    if torch is not None:
        log = _torchmethod(torch.log)
        log2 = _torchmethod(torch.log2)
        any = _torchmethod(torch.any)
        all = _torchmethod(torch.all)

    @_torchmethod(func_type='data_func')
    def nonzero(data):
        # torch returns an NxD tensor, whereas numpy returns
        # a D-tuple of N-dimensional arrays.
        return tuple(torch.nonzero(data).t())

    @_torchmethod(func_type='data_func')
    def astype(data, dtype, copy=True):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        data = data.to(dtype)
        if copy:
            data = data.clone()
        return data

    @_torchmethod(func_type='data_func')
    def tensor(data, device=ub.NoParam):
        if device is not ub.NoParam:
            data = data.to(device)
        return data

    @_torchmethod(func_type='data_func')
    def numpy(data):
        return data.data.cpu().numpy()

    @_torchmethod(func_type='data_func')
    def tolist(data):
        return data.data.cpu().numpy().tolist()

    @_torchmethod(func_type='data_func')
    def contiguous(data):
        return data.contiguous()

    @_torchmethod(func_type='data_func')
    def pad(data, pad_width, mode='constant'):
        pad = list(ub.flatten(pad_width[::-1]))
        return torch.nn.functional.pad(data, pad, mode=mode)

    @_torchmethod(func_type='data_func')
    def asarray(data, dtype=None):
        """
        Cast data into a tensor representation

        Example:
            >>> data = np.empty((2, 0, 196, 196), dtype=np.float32)
        """
        if not isinstance(data, torch.Tensor):
            data
            try:
                data =  torch.Tensor(data)
            except RuntimeError:
                if data.size == 0:
                    want_shape = data.shape
                    data_ = torch.empty([0])
                    data = data_.view(want_shape)
        if dtype is not None:
            dtype = _torch_dtype_lut().get(dtype, dtype)
            data = data.to(dtype)
        return data

    ensure = asarray

    @_torchmethod(func_type='data_func')
    def dtype_kind(data):
        """ returns the numpy code for the data type kind """
        if data.dtype.is_floating_point:
            return 'f'
        elif data.dtype == torch.uint8:
            return 'u'
        else:
            return 'i'

    @_torchmethod(func_type='data_func')
    def floor(data, out=None):
        return torch.floor(data, out=out)

    @_torchmethod(func_type='data_func')
    def ceil(data, out=None):
        return torch.ceil(data, out=out)

    @_torchmethod(func_type='data_func')
    def ifloor(data, out=None):
        return torch.floor(data, out=out).int()

    @_torchmethod(func_type='data_func')
    def iceil(data, out=None):
        return torch.ceil(data, out=out).int()

    @_torchmethod(func_type='data_func')
    def round(data, decimals=0, out=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:torch)
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> np_data = rng.rand(10) * 100
            >>> pt_data = torch.from_numpy(np_data)
            >>> a = kwarray.ArrayAPI.round(np_data)
            >>> b = kwarray.ArrayAPI.round(pt_data)
            >>> assert np.all(a == b.numpy())
            >>> a = kwarray.ArrayAPI.round(np_data, 2)
            >>> b = kwarray.ArrayAPI.round(pt_data, 2)
            >>> assert np.all(a == b.numpy())
        """
        if decimals == 0:
            return torch.round(data, out=out)
        else:
            factor = 10 ** decimals
            result = torch.round(data * factor, out=out)
            result /= factor
            return result

    @_torchmethod(func_type='data_func')
    def iround(data, out=None, dtype=int):
        dtype = _torch_dtype_lut().get(dtype, dtype)
        return torch.round(data, out=out).to(dtype)

    @_torchmethod(func_type='data_func')
    def clip(data, a_min=None, a_max=None, out=None):
        return torch.clamp(data, a_min, a_max, out=out)

    @_torchmethod(func_type='data_func')
    def softmax(data, axis=None):
        return torch.softmax(data, dim=axis)


class NumpyImpls(object):
    """
    Numpy backend for the ArrayAPI API
    """

    is_tensor = False
    is_numpy = True

    hstack = _numpymethod(func_type='array_sequence')(np.hstack)
    vstack = _numpymethod(func_type='array_sequence')(np.vstack)

    @_numpymethod(func_type='array_sequence')
    def cat(datas, axis=-1):
        return np.concatenate(datas, axis=axis)

    @_numpymethod(func_type='data_func')
    def atleast_nd(arr, n, front=False):
        import kwarray
        return kwarray.atleast_nd(arr, n, front)

    @_numpymethod(func_type='data_func')
    def view(data, *shape):
        data_ = data.reshape(*shape)
        return data_

    @_numpymethod(func_type='data_func')
    def take(data, indices, axis=None):
        return data.take(indices, axis=axis)

    @_numpymethod(func_type='data_func')
    def compress(data, flags, axis=None):
        return data.compress(flags, axis=axis)

    @_numpymethod(func_type='data_func')
    def repeat(data, repeats, axis=None):
        if not (axis is None or isinstance(repeats, int)):
            raise NotImplementedError('torch version of non-int repeats is not implemented')
        return np.repeat(data, repeats, axis=axis)

    @_numpymethod(func_type='data_func')
    def tile(data, reps):
        return np.tile(data, reps)

    @_numpymethod(func_type='data_func')
    def T(data):
        return data.T

    @_numpymethod(func_type='data_func')
    def transpose(data, axes):
        return np.transpose(data, axes)

    @_numpymethod(func_type='data_func')
    def numel(data):
        return data.size

    # --- Allocators ----

    @_numpymethod
    def empty_like(data, dtype=None):
        return np.empty_like(data, dtype=dtype)

    @_numpymethod
    def full_like(data, fill_value, dtype=None):
        return np.full_like(data, fill_value, dtype=dtype)

    @_numpymethod
    def zeros_like(data, dtype=None):
        return np.zeros_like(data, dtype=dtype)

    @_numpymethod
    def ones_like(data, dtype=None):
        return np.ones_like(data, dtype=dtype)

    @_numpymethod(func_type='shape_creation')
    def full(shape, fill_value, dtype=float):
        return np.full(shape, fill_value, dtype=dtype)

    @_numpymethod(func_type='shape_creation')
    def empty(shape, dtype=float):
        return np.full(shape, dtype=dtype)

    @_numpymethod(func_type='shape_creation')
    def zeros(shape, dtype=float):
        return np.zeros(shape, dtype=dtype)

    @_numpymethod(func_type='shape_creation')
    def ones(shape, dtype=float):
        return np.ones(shape, dtype=dtype)

    # -------

    @_numpymethod
    def argmax(data, axis=None):
        return np.argmax(data, axis=axis)

    @_numpymethod
    def argsort(data, axis=-1, descending=False):
        sortx = np.argsort(data, axis=axis)
        if descending:
            sortx = np.flip(sortx, axis=axis)
        return sortx

    @_numpymethod
    def max(data, axis=None):
        return data.max(axis=axis)

    @_numpymethod
    def max_argmax(data, axis=None):
        return data.max(axis=axis), np.argmax(data, axis=axis)

    @_numpymethod
    def sum(data, axis=None):
        return data.sum(axis=axis)

    @_numpymethod
    def maximum(data1, data2, out=None):
        return np.maximum(data1, data2, out=out)

    @_numpymethod
    def minimum(data1, data2, out=None):
        return np.minimum(data1, data2, out=out)

    # @_numpymethod
    # def matmul(data1, data2, out=None):
    #     return np.matmul(data1, data2, out=out)
    matmul = _numpymethod(np.matmul)

    nan_to_num = _numpymethod(np.nan_to_num)

    log = _numpymethod(np.log)
    log2 = _numpymethod(np.log2)

    any = _numpymethod(np.any)
    all = _numpymethod(np.all)

    copy = _numpymethod(np.copy)

    nonzero = _numpymethod(np.nonzero)

    @_numpymethod(func_type='data_func')
    def astype(data, dtype, copy=True):
        return data.astype(dtype, copy=copy)

    @_numpymethod(func_type='data_func')
    def tensor(data, device=ub.NoParam):
        if torch is None:
            raise Exception('torch is not available')
        data = torch.from_numpy(np.ascontiguousarray(data))
        if device is not ub.NoParam:
            data = data.to(device)
        return data

    @_numpymethod(func_type='data_func')
    def numpy(data):
        return data

    @_numpymethod(func_type='data_func')
    def tolist(data):
        return data.tolist()

    @_numpymethod(func_type='data_func')
    def contiguous(data):
        return np.ascontiguousarray(data)

    @_numpymethod(func_type='data_func')
    def pad(data, pad_width, mode='constant'):
        return np.pad(data, pad_width, mode=mode)

    @_numpymethod(func_type='data_func')
    def asarray(data, dtype=None):
        """
        Cast data into a numpy representation
        """
        if torch is not None and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return np.asarray(data, dtype=dtype)

    ensure = asarray

    @_numpymethod(func_type='data_func')
    def dtype_kind(data):
        return data.dtype.kind

    @_numpymethod(func_type='data_func')
    def floor(data, out=None):
        return np.floor(data, out=out)

    @_numpymethod(func_type='data_func')
    def ceil(data, out=None):
        return np.ceil(data, out=out)

    @_numpymethod(func_type='data_func')
    def ifloor(data, out=None):
        return np.floor(data, out=out).astype(np.int32)

    @_numpymethod(func_type='data_func')
    def iceil(data, out=None):
        return np.ceil(data, out=out).astype(np.int32)

    @_numpymethod(func_type='data_func')
    def round(data, decimals=0, out=None):
        return np.round(data, decimals=decimals, out=out)

    @_numpymethod(func_type='data_func')
    def iround(data, out=None, dtype=int):
        return np.round(data, out=out).astype(dtype)

    clip = _numpymethod(np.clip)

    @_numpymethod(func_type='data_func')
    def softmax(data, axis=None):
        from scipy import special
        return special.softmax(data, axis=axis)


class ArrayAPI(object):
    """
    Compatability API between torch and numpy.

    The API defines classmethods that work on both Tensors and ndarrays.  As
    such the user can simply use ``kwarray.ArrayAPI.<funcname>`` and it will
    return the expected result for both Tensor and ndarray types.

    However, this is inefficient because it requires us to check the type of
    the input for every API call. Therefore it is recommended that you use the
    :func:`ArrayAPI.coerce` function, which takes as input the data you want to
    operate on. It performs the type check once, and then returns another
    object that defines with an identical API, but specific to the given data
    type. This means that we can ignore type checks on future calls of the
    specific implementation. See examples for more details.


    Example:
        >>> # Use the easy-to-use, but inefficient array api
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> take = ArrayAPI.take
        >>> np_data = np.arange(0, 143).reshape(11, 13)
        >>> pt_data = torch.LongTensor(np_data)
        >>> indices = [1, 3, 5, 7, 11, 13, 17, 21]
        >>> idxs0 = [1, 3, 5, 7]
        >>> idxs1 = [1, 3, 5, 7, 11]
        >>> assert np.allclose(take(np_data, indices), take(pt_data, indices))
        >>> assert np.allclose(take(np_data, idxs0, 0), take(pt_data, idxs0, 0))
        >>> assert np.allclose(take(np_data, idxs1, 1), take(pt_data, idxs1, 1))

    Example:
        >>> # Use the easy-to-use, but inefficient array api
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> compress = ArrayAPI.compress
        >>> np_data = np.arange(0, 143).reshape(11, 13)
        >>> pt_data = torch.LongTensor(np_data)
        >>> flags = (np_data % 2 == 0).ravel()
        >>> f0 = (np_data % 2 == 0)[:, 0]
        >>> f1 = (np_data % 2 == 0)[0, :]
        >>> assert np.allclose(compress(np_data, flags), compress(pt_data, flags))
        >>> assert np.allclose(compress(np_data, f0, 0), compress(pt_data, f0, 0))
        >>> assert np.allclose(compress(np_data, f1, 1), compress(pt_data, f1, 1))

    Example:
        >>> # Use ArrayAPI to coerce an identical API that doesnt do type checks
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import kwarray
        >>> np_data = np.arange(0, 15).reshape(3, 5)
        >>> pt_data = torch.LongTensor(np_data)
        >>> # The new ``impl`` object has the same API as ArrayAPI, but works
        >>> # specifically on torch Tensors.
        >>> impl = kwarray.ArrayAPI.coerce(pt_data)
        >>> flat_data = impl.view(pt_data, -1)
        >>> print('flat_data = {!r}'.format(flat_data))
        flat_data = tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
        >>> # The new ``impl`` object has the same API as ArrayAPI, but works
        >>> # specifically on numpy ndarrays.
        >>> impl = kwarray.ArrayAPI.coerce(np_data)
        >>> flat_data = impl.view(np_data, -1)
        >>> print('flat_data = {!r}'.format(flat_data))
        flat_data = array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
    """

    _torch = TorchImpls
    _numpy = NumpyImpls

    @staticmethod
    def impl(data):
        """
        Returns a namespace suitable for operating on the input data type

        Args:
            data (ndarray | Tensor): data to be operated on

        """
        if torch is not None and torch.is_tensor(data):
            return TorchImpls
        else:
            return NumpyImpls

    @staticmethod
    def coerce(data):
        """
        Coerces some form of inputs into an array api (either numpy or torch).
        """
        if isinstance(data, six.string_types):
            from kwarray import arrayapi
            if data in ['torch', 'tensor']:
                return arrayapi.TorchImpls
            elif data == 'numpy':
                return arrayapi.NumpyImpls
            else:
                raise KeyError(data)
        else:
            return ArrayAPI.impl(data)

    @_apimethod(func_type='array_sequence')
    def cat(datas, *args, **kwargs):
        impl = ArrayAPI.impl(datas[0])
        return impl.cat(datas, *args, **kwargs)

    @_apimethod(func_type='array_sequence')
    def hstack(datas, *args, **kwargs):
        impl = ArrayAPI.impl(datas[0])
        return impl.hstack(datas, *args, **kwargs)

    @_apimethod(func_type='array_sequence')
    def vstack(datas, *args, **kwargs):
        impl = ArrayAPI.impl(datas[0])
        return impl.vstack(datas, *args, **kwargs)

    take = _apimethod('take')
    compress = _apimethod('compress')

    repeat = _apimethod('repeat')
    tile = _apimethod('tile')

    view = _apimethod('view')
    numel = _apimethod('numel')
    atleast_nd = _apimethod('atleast_nd')

    full_like = _apimethod('full_like')
    ones_like = _apimethod('ones_like')
    zeros_like = _apimethod('zeros_like')
    empty_like = _apimethod('empty_like')

    sum = _apimethod('sum')
    argmax = _apimethod('argmax')
    argsort = _apimethod('argsort')
    max = _apimethod('max')
    maximum = _apimethod('maximum')
    minimum = _apimethod('minimum')

    matmul = _apimethod('matmul')

    astype = _apimethod('astype')
    nonzero = _apimethod('nonzero')

    nan_to_num = _apimethod('nan_to_num')

    tensor = _apimethod('tensor')
    numpy = _apimethod('numpy')
    tolist = _apimethod('tolist')
    asarray = _apimethod('asarray')
    asarray = _apimethod('ensure')

    T = _apimethod('T')
    transpose = _apimethod('transpose')

    contiguous = _apimethod('contiguous')
    pad = _apimethod('pad')

    dtype_kind = _apimethod('dtype_kind')

    max_argmax = _apimethod('max_argmax')

    # ones = _apimethod('ones', func_type='shape_creation')
    # full = _apimethod('full', func_type='shape_creation')
    # empty = _apimethod('empty', func_type='shape_creation')
    # zeros = _apimethod('zeros', func_type='shape_creation')

    any = _apimethod('any', func_type='data_func')
    all = _apimethod('all', func_type='data_func')

    log2 = _apimethod('log2', func_type='data_func')
    log = _apimethod('log', func_type='data_func')
    copy = _apimethod('copy', func_type='data_func')

    iceil = _apimethod('iceil', func_type='data_func')
    ifloor = _apimethod('ifloor', func_type='data_func')
    floor = _apimethod('floor', func_type='data_func')
    ceil = _apimethod('ceil', func_type='data_func')

    round = _apimethod('round', func_type='data_func')
    iround = _apimethod('iround', func_type='data_func')

    clip = _apimethod('clip', func_type='data_func')

    softmax = _apimethod('softmax', func_type='data_func')


TorchNumpyCompat = ArrayAPI  # backwards compat


if __debug__ and torch is not None:
    _REGISTERY._ensure_datamethods_names_are_registered()


@ub.memoize
def _torch_dtype_lut():
    lut = {}

    # Handle nonstandard alias dtype names
    lut['double'] = torch.double
    lut['long'] = torch.long

    # Handle floats
    for k in [np.float16, 'float16']:
        lut[k] = torch.float16
    for k in [np.float32, 'float32']:
        lut[k] = torch.float32
    for k in [np.float64, 'float64']:
        lut[k] = torch.float64

    if torch.float == torch.float32:
        lut['float'] = torch.float32
    else:
        raise AssertionError('dont think this can happen')

    if np.float_ == np.float32:
        lut[float] = torch.float32
    elif np.float_ == np.float64:
        lut[float] = torch.float64
    else:
        raise AssertionError('dont think this can happen')

    # Handle signed integers
    for k in [np.int8, 'int8']:
        lut[k] = torch.int8
    for k in [np.int16, 'int16']:
        lut[k] = torch.int16
    for k in [np.int32, 'int32']:
        lut[k] = torch.int32
    for k in [np.int64, 'int64']:
        lut[k] = torch.int64

    if np.int_ == np.int32:
        lut[int] = torch.int32
    elif np.int_ == np.int64:
        lut[int] = torch.int64
    else:
        raise AssertionError('dont think this can happen')

    if torch.int == torch.int32:
        lut['int'] = torch.int32
    else:
        raise AssertionError('dont think this can happen')

    # Handle unsigned integers
    for k in [np.uint8, 'uint8']:
        lut[k] = torch.uint8

    # import torch.utils.data
    # cant use torch.utils.data.dataloader.numpy_type_map directly because it
    # maps to tensor types not dtypes, but we can use it to check
    check = False
    if check:
        for k, v in torch.utils.data.dataloader.numpy_type_map.items():
            assert lut[k] == v.dtype
    return lut


def dtype_info(dtype):
    """
    Args:
        dtype (type): a numpy, torch, or python numeric data type

    Returns:
        struct: an iinfo of finfo structure depending on the input type.

    References:
        https://higra.readthedocs.io/en/stable/_modules/higra/hg_utils.html#dtype_info

    Example:
        >>> results = []
        >>> results += [dtype_info(float)]
        >>> results += [dtype_info(int)]
        >>> results += [dtype_info(complex)]
        >>> results += [dtype_info(np.float32)]
        >>> results += [dtype_info(np.int32)]
        >>> results += [dtype_info(np.uint32)]
        >>> if hasattr(np, 'complex256'):
        >>>     results += [dtype_info(np.complex256)]
        >>> if torch is not None:
        >>>     results += [dtype_info(torch.float32)]
        >>>     results += [dtype_info(torch.int64)]
        >>>     results += [dtype_info(torch.complex64)]
        >>> for info in results:
        >>>     print('info = {!r}'.format(info))
        >>> for info in results:
        >>>     print('info.bits = {!r}'.format(info.bits))
    """
    if torch is not None and isinstance(dtype, torch.dtype):
        # Using getattr on is_complex for torch 1.4
        if dtype.is_floating_point or getattr(dtype, 'is_complex', False):
            try:
                info = torch.finfo(dtype)
            except Exception:
                # if not hasattr(dtype, 'is_complex'):
                """
                # Helper for dev
                zs = [torch.complex32, torch.complex64, torch.complex128]
                for z in zs:
                    z = torch.finfo(z)
                    args = [f + '=' + repr(getattr(z, f))
                        for f in finfo_ducktype._fields]
                    print('finfo_ducktype({})'.format(', '.join(args)))
                """
                # Return a ducktype for older torches without complex finfo
                # https://github.com/pytorch/pytorch/issues/35954
                from collections import namedtuple
                finfo_ducktype = namedtuple('finfo_ducktype', [
                    'bits', 'dtype', 'eps', 'max', 'min', 'resolution',
                    'tiny'])
                from collections import namedtuple
                finfo_ducktype(bits=32, dtype='float32',
                               eps=1.1920928955078125e-07,
                               max=3.4028234663852886e+38,
                               min=-3.4028234663852886e+38,
                               resolution=1e-06,
                               tiny=1.1754943508222875e-38)
                if dtype == torch.complex32:
                    info = finfo_ducktype(bits=16, dtype='float16',
                                          eps=0.0009765625, max=65504.0,
                                          min=-65504.0, resolution=0.001,
                                          tiny=6.103515625e-05)
                elif dtype == torch.complex64:
                    info = finfo_ducktype(bits=32, dtype='float32',
                                          eps=1.1920928955078125e-07,
                                          max=3.4028234663852886e+38,
                                          min=-3.4028234663852886e+38,
                                          resolution=1e-06,
                                          tiny=1.1754943508222875e-38)
                elif dtype == torch.complex128:
                    info = finfo_ducktype(bits=64, dtype='float64',
                                          eps=2.220446049250313e-16,
                                          max=1.7976931348623157e+308,
                                          min=-1.7976931348623157e+308,
                                          resolution=1e-15,
                                          tiny=2.2250738585072014e-308)
                else:
                    raise TypeError(dtype)
        else:
            info = torch.iinfo(dtype)
    else:
        np_dtype = np.dtype(dtype)
        if np_dtype.kind in {'f', 'c'}:
            info = np.finfo(np_dtype)
        else:
            info = np.iinfo(np_dtype)
    return info
