from typing import Type

import flax.linen as nn

from efppo.networks.network_utils import default_nn_init
from efppo.task.dyn_types import Obs
from efppo.utils.tfp import tfd


class DiscretePolicyNet(nn.Module):
    base_cls: Type[nn.Module]
    n_actions: int

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        logits = nn.Dense(self.n_actions, kernel_init=default_nn_init(), name="OutputDense")(x)
        return tfd.Categorical(logits=logits)
