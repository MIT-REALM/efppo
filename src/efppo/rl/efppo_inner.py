import functools as ft
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from attrs import define
from flax import struct
from loguru import logger

from efppo.networks.ef_wrapper import EFWrapper, ZEncoder
from efppo.networks.mlp import MLP
from efppo.networks.network_utils import ActLiteral, HidSizes, get_act_from_str, get_default_tx
from efppo.networks.policy_net import DiscretePolicyNet
from efppo.networks.train_state import TrainState
from efppo.networks.value_net import ConstrValueNet, CostValueNet
from efppo.rl.collector import Collector, RolloutOutput, collect_single_mode
from efppo.rl.gae_utils import compute_efocp_gae, compute_efocp_V
from efppo.task.dyn_types import BControl, BHFloat, BObs, HFloat, LFloat, ZBBControl, ZBBFloat, ZBTState, ZFloat
from efppo.task.task import Task
from efppo.utils.cfg_utils import Cfg
from efppo.utils.grad_utils import compute_norm_and_clip
from efppo.utils.jax_types import BFloat, FloatScalar
from efppo.utils.jax_utils import jax_vmap, merge01, tree_split_dims, tree_stack
from efppo.utils.rng import PRNGKey
from efppo.utils.schedules import Schedule, as_schedule
from efppo.utils.tfp import tfd


@define
class EFPPOCfg(Cfg):
    @define
    class TrainCfg(Cfg):
        z_min: float
        z_max: float

        gae_lambda: float
        J_max: float

        n_batches: int

        clip_ratio: float

        clip_grad_pol: float
        clip_grad_V: float

    @define
    class EvalCfg(Cfg):
        ...

    @define
    class NetCfg(Cfg):
        pol_lr: Schedule
        val_lr: Schedule

        # Schedules are in units of collect_idx.
        entropy_cf: Schedule
        disc_gamma: Schedule

        act: ActLiteral
        pol_hids: HidSizes
        val_hids: HidSizes

        nz_enc: int
        z_mean: float
        z_scale: float

    net: NetCfg
    train: TrainCfg
    eval: EvalCfg


class EFPPOInner(struct.PyTreeNode):
    update_idx: int
    key: PRNGKey
    policy: TrainState[tfd.Distribution]
    Vl: TrainState[LFloat]
    Vh: TrainState[HFloat]
    disc_gamma: FloatScalar

    task: Task = struct.field(pytree_node=False)
    cfg: EFPPOCfg = struct.field(pytree_node=False)

    ent_cf_sched: optax.Schedule = struct.field(pytree_node=False)
    disc_gamma_sched: optax.Schedule = struct.field(pytree_node=False)

    Cfg = EFPPOCfg

    class Batch(NamedTuple):
        b_obs: BObs
        b_z: BFloat
        b_control: BControl
        b_logprob: BFloat
        b_Ql: BFloat
        bh_Qh: BHFloat
        b_A: BFloat

        @property
        def batch_size(self) -> int:
            assert self.b_logprob.ndim == 1
            return len(self.b_logprob)

    class EvalData(NamedTuple):
        z_zs: ZFloat
        zbb_pol: ZBBControl
        zbb_prob: ZBBFloat
        zbb_Vl: ZBBFloat
        zbb_Vh: ZBBFloat

        zbT_x: ZBTState

        info: dict[str, float]

    @classmethod
    def create(cls, key: jr.PRNGKey, task: Task, cfg: EFPPOCfg):
        key, key_pol, key_Vl, key_Vh = jr.split(key, 4)
        obs, z = np.zeros(task.nobs), np.array(0.0)
        act = get_act_from_str(cfg.net.act)

        # Encoder for z. Params not shared.
        z_base_cls = ft.partial(ZEncoder, nz=cfg.net.nz_enc, z_mean=cfg.net.z_mean, z_scale=cfg.net.z_scale)

        # Define policy network.
        pol_base_cls = ft.partial(MLP, cfg.net.pol_hids, act, act_final=True, scale_final=1e-2)
        pol_cls = ft.partial(DiscretePolicyNet, pol_base_cls, task.n_actions)
        pol_def = EFWrapper(pol_cls, z_base_cls)
        pol_tx = get_default_tx(as_schedule(cfg.net.pol_lr).make())
        pol = TrainState.create_from_def(key_pol, pol_def, (obs, z), pol_tx)

        # Define Vl network.
        Vl_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        Vl_cls = ft.partial(CostValueNet, Vl_base_cls)
        Vl_def = EFWrapper(Vl_cls, z_base_cls)
        Vl_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        Vl = TrainState.create_from_def(key_pol, Vl_def, (obs, z), Vl_tx)

        # Define Vh network.
        Vh_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        Vh_cls = ft.partial(ConstrValueNet, Vh_base_cls, task.nh)
        Vh_def = EFWrapper(Vh_cls, z_base_cls)
        Vh_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        Vh = TrainState.create_from_def(key_pol, Vh_def, (obs, z), Vh_tx)

        ent_cf = as_schedule(cfg.net.entropy_cf).make()
        disc_gamma_sched = as_schedule(cfg.net.disc_gamma).make()
        disc_gamma = disc_gamma_sched(0)

        return EFPPOInner(0, key, pol, Vl, Vh, disc_gamma, task, cfg, ent_cf, disc_gamma_sched)

    @property
    def train_cfg(self) -> EFPPOCfg.TrainCfg:
        return self.cfg.train

    @property
    def z_min(self):
        return self.train_cfg.z_min

    @property
    def z_max(self):
        return self.train_cfg.z_max

    @property
    def ent_cf(self):
        return self.ent_cf_sched(self.update_idx)

    def make_dset(self, data: Collector.Rollout) -> Batch:
        batch_size, T = data.T_control.shape[:2]

        # 1: Compute Vl and h_Vh from data.
        bTp1_Vl = jax_vmap(self.Vl.apply, rep=2)(data.Tp1_obs, data.Tp1_z)
        bTp1h_Vh = jax_vmap(self.Vh.apply, rep=2)(data.Tp1_obs, data.Tp1_z)
        bT_z = data.Tp1_z[:, :-1]

        # 2: Compute Vl, Vh and A using GAE.
        gae_lambda = self.cfg.train.gae_lambda
        J_max = self.cfg.train.J_max
        compute_gae = ft.partial(compute_efocp_gae, disc_gamma=self.disc_gamma, gae_lambda=gae_lambda, J_max=J_max)
        bTh_Qh, bT_Ql, bT_Q = jax_vmap(compute_gae)(data.Th_h, data.T_l, bT_z, bTp1h_Vh, bTp1_Vl)

        # 3: Compute advantage for policy gradient.
        bTh_Vh, bT_Vl = bTp1h_Vh[:, :-1], bTp1_Vl[:, :-1]
        bT_V = jax_vmap(compute_efocp_V, rep=2)(bT_z, bTh_Vh, bT_Vl)
        assert bT_V.shape == (batch_size, T)
        bT_A = bT_Q - bT_V

        # 4: Make the dataset by flattening (b, T) -> (b * T,)
        bT_obs = data.Tp1_obs[:, :-1]
        bT_batch = self.Batch(bT_obs, bT_z, data.T_control, data.T_logprob, bT_Ql, bTh_Qh, bT_A)
        b_batch = jax.tree_map(merge01, bT_batch)
        return b_batch

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, data: Collector.Rollout) -> tuple["EFPPOInner", dict]:
        # Compute GAE values.
        b_dset = self.make_dset(data)

        n_batches = self.train_cfg.n_batches
        assert b_dset.batch_size % n_batches == 0
        batch_size = b_dset.batch_size // self.train_cfg.n_batches
        logger.info(f"Using {n_batches} minibatches each epoch!")
        # 2: Shuffle and reshape
        key_shuffle, key_self = jr.split(self.key, 2)
        rand_idxs = jr.permutation(key_shuffle, jnp.arange(b_dset.batch_size))
        b_dset = jax.tree_map(lambda x: x[rand_idxs], b_dset)
        mb_dset = tree_split_dims(b_dset, (n_batches, batch_size))

        # 3: Perform value function and policy updates.
        def updates_body(alg_: EFPPOInner, b_batch: EFPPOInner.Batch):
            alg_, val_info = alg_.update_value(b_batch)
            alg_, pol_info = alg_.update_policy(b_batch)
            return alg_, val_info | pol_info

        new_self, info = lax.scan(updates_body, self, mb_dset, length=n_batches)
        # Take the mean.
        info = jax.tree_map(jnp.mean, info)

        info["steps/policy"] = self.policy.step
        info["steps/Vl"] = self.Vl.step
        info["anneal/ent_cf"] = self.ent_cf
        return new_self.replace(key=key_self, update_idx=self.update_idx + 1), info

    def update_value(self, batch: Batch) -> tuple["EFPPOInner", dict]:
        def get_Vl_loss(params):
            b_Vl = jax.vmap(ft.partial(self.Vl.apply_with, params=params))(batch.b_obs, batch.b_z)
            assert b_Vl.shape == (batch.batch_size,)

            loss_Vl = jnp.mean((b_Vl - batch.b_Ql) ** 2)

            info = {
                "Loss/Vl": loss_Vl,
                "mean_Vl": b_Vl.mean(),
            }
            return loss_Vl, info

        def get_Vh_loss(params):
            bh_Vh = jax.vmap(ft.partial(self.Vh.apply_with, params=params))(batch.b_obs, batch.b_z)
            assert bh_Vh.shape == (batch.batch_size, self.task.nh)

            loss_Vh = jnp.mean((bh_Vh - batch.bh_Qh) ** 2)

            info = {"Loss/Vh": loss_Vh}
            return loss_Vh, info

        grads_Vl, Vl_info = jax.grad(get_Vl_loss, has_aux=True)(self.Vl.params)
        grads_Vl, Vl_info["Grad/Vl"] = compute_norm_and_clip(grads_Vl, self.train_cfg.clip_grad_V)

        grads_Vh, Vh_info = jax.grad(get_Vh_loss, has_aux=True)(self.Vh.params)
        grads_Vh, Vh_info["Grad/Vh"] = compute_norm_and_clip(grads_Vh, self.train_cfg.clip_grad_V)

        Vl = self.Vl.apply_gradients(grads=grads_Vl)
        Vh = self.Vh.apply_gradients(grads=grads_Vh)
        return self.replace(Vl=Vl, Vh=Vh), Vl_info | Vh_info

    def update_policy(self, batch: Batch) -> tuple["EFPPOInner", dict]:
        def get_pol_loss(pol_params):
            pol_apply = ft.partial(self.policy.apply_with, params=pol_params)

            def get_logprob_entropy(obs, z, control):
                dist = pol_apply(obs, z)
                return dist.log_prob(control), dist.entropy()

            b_logprobs, b_entropy = jax.vmap(get_logprob_entropy)(batch.b_obs, batch.b_z, batch.b_control)
            b_logratios = b_logprobs - batch.b_logprob
            b_is_ratio = jnp.exp(b_logratios)

            b_adv = batch.b_A
            pg_loss_orig = b_adv * b_is_ratio
            pg_loss_clip = b_adv * jnp.clip(b_is_ratio, 1 - clip_ratio, 1 + clip_ratio)
            loss_pg = jnp.maximum(pg_loss_orig, pg_loss_clip).mean()
            pol_clipfrac = jnp.mean(pg_loss_clip > pg_loss_orig)

            mean_entropy = b_entropy.mean()
            loss_entropy = -mean_entropy

            pol_loss = loss_pg + ent_cf * loss_entropy
            info = {
                "loss_pol": pol_loss,
                "entropy": mean_entropy,
                "pol_clipfrac": pol_clipfrac,
            }
            return pol_loss, info

        clip_ratio = self.train_cfg.clip_ratio
        ent_cf, clip_ratio = self.ent_cf, self.train_cfg.clip_ratio
        grads, pol_info = jax.grad(get_pol_loss, has_aux=True)(self.policy.params)

        grads, pol_info["Grad/pol"] = compute_norm_and_clip(grads, self.train_cfg.clip_grad_pol)
        policy = self.policy.apply_gradients(grads=grads)
        return self.replace(policy=policy), pol_info

    @ft.partial(jax.jit, donate_argnums=1)
    def collect(self, collector: Collector) -> tuple[Collector, Collector.Rollout]:
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max
        return collector.collect_batch(ft.partial(self.policy.apply), self.disc_gamma, z_min, z_max)

    @ft.partial(jax.jit, static_argnames=["rollout_T"])
    def eval(self, rollout_T: int) -> EvalData:
        # Evaluate for a range of zs.
        val_zs = np.linspace(self.train_cfg.z_min, self.train_cfg.z_max, num=8)

        Z_datas = []
        for z in val_zs:
            data = self.eval_single_z(z, rollout_T)
            Z_datas.append(data)
        Z_data = tree_stack(Z_datas)

        info = jtu.tree_map(lambda arr: {"0": arr[0], "4": arr[4], "7": arr[7]}, Z_data.info)
        info["update_idx"] = self.update_idx
        return Z_data._replace(info=info)

    def get_mode_and_prob(self, obs, z):
        dist = self.policy.apply(obs, z)
        mode_sample = dist.mode()
        mode_prob = jnp.exp(dist.log_prob(mode_sample))
        return mode_sample, mode_prob

    def get_Vh(self, obs, z):
        h_Vh = self.Vh.apply(obs, z)
        return h_Vh.max()

    def eval_single_z(self, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = self.task.grid_contour()
        bb_obs = jax_vmap(self.task.get_obs, rep=2)(bb_state)
        bb_z = jnp.full(bb_X.shape, z)

        bb_pol, bb_prob = jax_vmap(self.get_mode_and_prob, rep=2)(bb_obs, bb_z)
        bb_Vl = jax_vmap(self.Vl.apply, rep=2)(bb_obs, bb_z)
        bb_Vh = jax_vmap(self.get_Vh, rep=2)(bb_obs, bb_z)

        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = self.task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_mode,
            self.task,
            get_pol=self.policy.apply,
            disc_gamma=self.disc_gamma,
            z_min=z_min,
            z_max=z_max,
            rollout_T=rollout_T,
        )
        b_rollout: RolloutOutput = jax_vmap(collect_fn)(b_x0, b_z0)

        b_h = jnp.max(b_rollout.Th_h, axis=(1, 2))
        assert b_h.shape == (batch_size,)
        b_issafe = b_h <= 0
        p_unsafe = 1 - b_issafe.mean()
        h_mean = jnp.mean(b_h)

        b_l = jnp.sum(b_rollout.T_l, axis=1)
        l_mean = jnp.mean(b_l)

        b_l_final = b_rollout.T_l[:, -1]
        l_final = jnp.mean(b_l_final)
        # --------------------------------------------

        info = {"p_unsafe": p_unsafe, "h_mean": h_mean, "cost sum": l_mean, "l_final": l_final}
        return self.EvalData(z, bb_pol, bb_prob, bb_Vl, bb_Vh, b_rollout.Tp1_state, info)
