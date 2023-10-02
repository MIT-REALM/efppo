from typing import Type

import flax.linen as nn
from jaxtyping import Float

from efppo.networks.network_utils import default_nn_init
from efppo.utils.jax_types import Arr
from efppo.utils.shape_utils import assert_shape


class CostValueNet(nn.Module):
    net_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, state: Float[Arr, "* nx"], *args, **kwargs) -> Float[Arr, "*"]:
        batch_shape = state.shape[:-1]
        x = self.net_cls()(state, *args, **kwargs)
        Vl = nn.Dense(1, kernel_init=default_nn_init())(x)
        return assert_shape(Vl.squeeze(-1), batch_shape)


class ConstrValueNet(nn.Module):
    net_cls: Type[nn.Module]
    # Number of constraints.
    nh: int

    @nn.compact
    def __call__(self, state: Float[Arr, "* nx"], *args, **kwargs) -> Float[Arr, "* nh"]:
        batch_shape = state.shape[:-1]
        x = self.net_cls()(state, *args, **kwargs)
        Vhs = nn.Dense(self.nh, kernel_init=default_nn_init())(x)
        return assert_shape(Vhs, batch_shape + (self.nh,))
