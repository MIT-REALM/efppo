from typing import Generic, Type, TypeVar

import flax.linen as nn
import jax.numpy as jnp

from efppo.networks.network_utils import default_nn_init
from efppo.task.dyn_types import Obs
from efppo.utils.jax_types import AnyFloat, BFloat, FloatScalar
from efppo.utils.shape_utils import assert_shape

_WrappedModule = TypeVar("_WrappedModule", bound=nn.Module)


class ZEncoder(nn.Module):
    nz: int
    z_mean: float
    z_scale: float

    @nn.compact
    def __call__(self, z: BFloat) -> BFloat:
        # 1: Normalize z.
        norm_z = (z - self.z_mean) / self.z_scale

        # 2: Encode it.
        enc_z = nn.Dense(self.nz, kernel_init=default_nn_init())(norm_z)
        enc_z = nn.tanh(enc_z)
        return assert_shape(enc_z, self.nz, "enc_z")


class EFWrapper(nn.Module, Generic[_WrappedModule]):
    """Wrapper for networks that only take in the state to also take in z."""

    base_cls: Type[_WrappedModule]
    z_encoder_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: Obs, z: FloatScalar) -> AnyFloat:
        assert obs.ndim == (z.ndim + 1)
        z = z[..., None]
        enc_z = self.z_encoder_cls()(z)
        feat = jnp.concatenate([obs, enc_z], axis=-1)
        return self.base_cls()(feat)
