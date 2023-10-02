import builtins
from typing import Any, Callable, ParamSpec, Sequence, TypeVar

import einops as ei
import jax
import jax._src.dtypes
import jax.config
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.interpreters.batching import BatchTracer
from jaxtyping import Float

from efppo.utils.jax_types import Arr, Shape, BoolScalar

_PyTree = TypeVar("_PyTree")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]


def which_np(*args: Any):
    """Returns which numpy implementation (Numpy or Jax) based on the arguments."""
    checker = lambda a: (isinstance(a, (jnp.ndarray, BatchTracer)) and not isinstance(a, np.ndarray))
    if builtins.any(jtu.tree_leaves(jax.tree_map(checker, args))):
        return jnp
    else:
        return np


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


def tree_copy(tree: _PyTree) -> _PyTree:
    return jtu.tree_map(lambda x: x.copy(), tree)


def tree_stack(trees: list[_PyTree], axis: int = 0) -> _PyTree:
    def tree_stack_inner(*arrs):
        arrs = list(arrs)
        return which_np(*arrs).stack(arrs, axis=axis)

    return jtu.tree_map(tree_stack_inner, *trees)


def jax_vmap(fn: _Fn, rep: int = 1, in_axes: int | Sequence[Any] = 0, **kwargs) -> _Fn:
    for ii in range(rep):
        fn = jax.vmap(fn, in_axes=in_axes, **kwargs)
    return fn


def tree_where(cond: BoolScalar | bool, true_val: _PyTree, false_val: _PyTree) -> _PyTree:
    return jtu.tree_map(lambda x, y: jnp.where(cond, x, y), true_val, false_val)


def concat_at_front(arr1: Float[Arr, "nx"], arr2: Float[Arr, "T nx"], axis: int = 0) -> Float[Arr, "Tp1 nx"]:
    """
    :param arr1: (nx, )
    :param arr2: (T, nx)
    :param axis: Which axis for arr1 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr2_shape = list(arr2.shape)
    del arr2_shape[axis]
    assert np.all(np.array(arr1.shape) == np.array(arr2_shape))

    return jnp.concatenate([jnp.expand_dims(arr1, axis=axis), arr2], axis=axis)


def concat_at_end(arr1: Float[Arr, "T nx"], arr2: Float[Arr, "nx"], axis: int = 0) -> Float[Arr, "Tp1 nx"]:
    """
    :param arr1: (T, nx)
    :param arr2: (nx, )
    :param axis: Which axis for arr1 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr1_shape = list(arr1.shape)
    del arr1_shape[axis]
    assert np.all(np.array(arr1_shape) == np.array(arr2.shape))

    return jnp.concatenate([arr1, jnp.expand_dims(arr2, axis=axis)], axis=axis)


def tree_split_dims(tree: _PyTree, new_dims: Shape) -> _PyTree:
    prod_dims = np.prod(new_dims)

    def tree_split_dims_inner(arr: Arr) -> Arr:
        assert arr.shape[0] == prod_dims
        target_shape = new_dims + arr.shape[1:]
        return arr.reshape(target_shape)

    return jtu.tree_map(tree_split_dims_inner, tree)
