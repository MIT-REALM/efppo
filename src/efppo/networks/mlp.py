from typing import Generator, Iterable, TypeVar

_Elem = TypeVar("_Elem")

import flax.linen as nn

from efppo.networks.network_utils import ActFn, HidSizes, default_nn_init, scaled_init
from efppo.utils.jax_types import AnyFloat


def signal_last_enumerate(it: Iterable[_Elem]) -> Generator[tuple[bool, int, _Elem], None, None]:
    iterable = iter(it)
    count = 0
    ret_var = next(iterable)
    for val in iterable:
        yield False, count, ret_var
        count += 1
        ret_var = val
    yield True, count, ret_var


class MLP(nn.Module):
    hid_sizes: HidSizes
    act: ActFn = nn.relu
    act_final: bool = True
    scale_final: float | None = None

    @nn.compact
    def __call__(self, x: AnyFloat) -> AnyFloat:
        nn_init = default_nn_init
        for is_last_layer, ii, hid_size in signal_last_enumerate(self.hid_sizes):
            if is_last_layer and self.scale_final is not None:
                x = nn.Dense(hid_size, kernel_init=scaled_init(nn_init(), self.scale_final))(x)
            else:
                x = nn.Dense(hid_size, kernel_init=nn_init())(x)

            no_activation = is_last_layer and not self.act_final
            if not no_activation:
                x = self.act(x)
        return x
