from typing import Any, Callable, Literal, Sequence

import flax.linen as nn
import jax.numpy as jnp
import optax
from flax import traverse_util

from efppo.utils.jax_types import AnyFloat, FloatScalar, Shape
from efppo.utils.rng import PRNGKey

ActFn = Callable[[AnyFloat], AnyFloat]

InitFn = Callable[[PRNGKey, Shape, Any], Any]

default_nn_init = nn.initializers.xavier_uniform

HidSizes = Sequence[int]


def scaled_init(initializer: nn.initializers.Initializer, scale: float) -> nn.initializers.Initializer:
    def scaled_init_inner(*args, **kwargs) -> AnyFloat:
        return scale * initializer(*args, **kwargs)

    return scaled_init_inner


ActLiteral = Literal["relu", "tanh", "elu", "swish", "silu", "gelu", "softplus"]


def get_act_from_str(act_str: ActLiteral) -> ActFn:
    act_dict: dict[Literal, ActFn] = dict(
        relu=nn.relu, tanh=nn.tanh, elu=nn.elu, swish=nn.swish, silu=nn.silu, gelu=nn.gelu, softplus=nn.softplus
    )
    return act_dict[act_str]


def wd_mask(params):
    Path = tuple[str, ...]
    flat_params: dict[Path, jnp.ndarray] = traverse_util.flatten_dict(params)
    # Apply weight decay to all parameters except biases and LayerNorm scale and bias.
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)


def optim(learning_rate: float, wd: float, eps: float):
    opt = optax.adamw(learning_rate, eps=eps, weight_decay=wd, mask=wd_mask)
    opt = optax.apply_if_finite(opt, 100)
    return opt


def get_default_tx(
    lr: optax.Schedule | FloatScalar, wd: optax.Schedule | FloatScalar = 1e-4, eps: FloatScalar = 1e-5
) -> optax.GradientTransformation:
    return optax.inject_hyperparams(optim)(learning_rate=lr, wd=wd, eps=eps)
