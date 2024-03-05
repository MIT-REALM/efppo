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

from efppo.utils.jax_types import Arr, BoolScalar, FloatScalar, Shape

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


def tree_mac(accum: _PyTree, scalar: float, other: _PyTree, strict: bool = True) -> _PyTree:
    """Tree multiply and accumulate. Return accum + scalar * other, but for pytree."""

    def mac_inner(a, o):
        if strict:
            assert a.shape == o.shape
        return a + scalar * o

    return jtu.tree_map(mac_inner, accum, other)


def tree_add(t1: _PyTree, t2: _PyTree):
    return jtu.tree_map(lambda a, b: a + b, t1, t2)


def tree_inner_product(coefs: list[float], trees: list[_PyTree]) -> _PyTree:
    def tree_inner_product_(*arrs_):
        arrs_ = list(arrs_)
        out = sum([c * a for c, a in zip(coefs, arrs_)])
        return out

    assert len(coefs) == len(trees)
    return jtu.tree_map(tree_inner_product_, *trees)


def poly_clip_max(x, max_val: float):
    """Polynomial symmetric smoothed clipping function that preserves sign and clips positive values only.
    x <= 1
    f(0)   = 0                          f(1)  = 1.0
    f'(0)  = 1.5                        f'(1) = 0.0
    f''(0) = f'''(0) = f''''(0) = 0
    """
    a0 = 3
    a4 = -3
    a5 = 2
    x = jnp.minimum(x, max_val)
    y = x / max_val
    clip_branch = ((a5 * y + a4) * (y**4) + a0) * x / 2
    return jnp.where(x >= 0, clip_branch, 1.5 * x)


def clipped_log1p(h: FloatScalar, min_val: float | FloatScalar) -> FloatScalar:
    """
    Transforms h, a function that has bounded negative part to unbounded negative (but with min val) using log.
    """
    assert min_val < 0, "min_val should be negative to preserve semantics of h."
    log1p_min = -1.0 + 1e-4
    return jnp.clip(jnp.log1p(jnp.clip(h, a_min=log1p_min)), a_min=min_val)


def box_constr_clipmax(x: FloatScalar, bounds: tuple[float, float], scale: float, max_val: float):
    assert len(bounds) == 2, "bounds not length 2"
    hs = jnp.stack([bounds[0] - x, x - bounds[1]]) / scale
    return poly_clip_max(hs, max_val=max_val)


def box_constr_log1p(
    x: FloatScalar,
    bounds: tuple[float, float],
    scale: float,
    min_val: float,
    scale2: float | None = None,
    max_val: float | None = None,
):
    assert len(bounds) == 2, "bounds not length 2"
    if scale2 is None:
        scale2 = 1.0

    hs = jnp.stack([bounds[0] - x, x - bounds[1]]) / (scale * scale2)
    hs = clipped_log1p(hs, min_val=min_val)

    if max_val is not None:
        hs = poly_clip_max(scale2 * hs, max_val=max_val)

    return hs
