from typing import Callable

import jax.debug as jd
import jax.numpy as jnp

from efppo.utils.chandrupatla import Chandrupatla
from efppo.utils.jax_types import FloatScalar


class Rootfinder:
    def __init__(
        self,
        h_Vh_fn: Callable,
        z_min: float,
        z_max: float,
        h_tgt: float = -0.45,
        h_eps: float = 1e-2,
        n_iters: int = 20,
    ):
        self.h_Vh_fn = h_Vh_fn
        self.z_min = z_min
        self.z_max = z_max
        self.h_tgt = h_tgt - h_eps
        self.n_iters = n_iters

    def get_opt_z(self, obs):
        def z_root_fn(z: FloatScalar):
            h_Vh = self.h_Vh_fn(obs, z)
            Vh = h_Vh.max()
            root = -(Vh - self.h_tgt)
            # jd.print("obs: {}, z: {}, h_Vh: {}    |    root: {}", obs, z, h_Vh, root)
            return root

        def is_bad_root(z):
            eps = 1e-2
            val_curr = z_root_fn(z)
            val_neg_eps = z_root_fn(z - eps)

            grad_est = val_curr - val_neg_eps
            is_neg_grad = grad_est < -1e-2
            return is_neg_grad

        solver = Chandrupatla(z_root_fn, n_iters=self.n_iters, init_t=0.5)
        opt_z, _, init_state = solver.run(self.z_min, self.z_max)

        # The rootfinding is only valid if we bracketed the root.
        #         Vh < h_tgt is safe.
        #         => -(Vh - h_tgt) > 0 is safe.
        # If max(y1, y2) > 0, then both zmin and zmax are safe, so we take zmin.
        # If max(y1, y2) < 0, then both zmin and zmax are unsafe, so we take zmax.
        both_safe = (init_state.y1 > 0) & (init_state.y2 > 0)
        both_unsafe = (init_state.y1 < 0) & (init_state.y2 < 0)
        opt_z = jnp.where(both_safe, self.z_min, jnp.where(both_unsafe, self.z_max, opt_z))

        is_neg_grad_1 = False
        # is_neg_grad_0 = is_bad_root(opt_z)
        # opt_z2_ub = 0.95 * opt_z
        # opt_z2, _ = solver.run(self.z_min, opt_z2_ub)
        #
        # opt_z2_is_root = jnp.abs(z_root_fn(opt_z2)) < 1e-4
        # use_new_z = is_neg_grad_0 & opt_z2_is_root
        # opt_z = jnp.where(use_new_z, opt_z2, opt_z)
        #
        # # Check neg grad again.
        # is_neg_grad_1 = is_bad_root(opt_z)
        #
        # opt_z = jnp.where(is_neg_grad_1, self.z_max, opt_z)
        return opt_z, is_neg_grad_1


class RootfindPolicy:
    def __init__(self, pol_fn: Callable, rootfind: Rootfinder):
        self.pol_fn = pol_fn
        self.rootfind = rootfind

    def __call__(self, obs, z: float):
        del z
        z_opt, is_neg_grad = self.rootfind.get_opt_z(obs)
        dist = self.pol_fn(obs, z_opt)
        return dist
