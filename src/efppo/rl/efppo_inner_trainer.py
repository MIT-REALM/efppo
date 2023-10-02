import pathlib
import time

import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from loguru import logger
from matplotlib.colors import CenteredNorm

import wandb
from efppo.rl.collector import Collector, CollectorCfg
from efppo.rl.efppo_inner import EFPPOInner
from efppo.task.plotter import Plotter
from efppo.task.task import Task
from efppo.utils.cfg_utils import Cfg
from efppo.utils.ckpt_utils import get_ckpt_manager_sync
from efppo.utils.jax_utils import jax2np
from efppo.utils.path_utils import get_runs_dir, mkdir
from efppo.utils.register_cmaps import register_cmaps
from efppo.utils.rng import PRNGKey
from efppo.utils.wandb_utils import reorder_wandb_name


@define
class TrainerCfg(Cfg):
    n_iters: int
    log_every: int
    eval_every: int
    ckpt_every: int

    ckpt_max_keep: int = 100


class EFPPOInnerTrainer:
    def __init__(self, task: Task):
        self.task = task
        self.plotter = Plotter(task)
        register_cmaps()

    def plot(self, idx: int, plot_dir: pathlib.Path, data: EFPPOInner.EvalData):
        fig_opt = dict(layout="constrained", dpi=200)

        bb_X, bb_Y, _ = self.task.grid_contour()

        # --------------------------------------------
        # Plot the trajectories.
        nz, plot_batch_size, T, nx = data.zbT_x.shape

        figsize = 1.5 * np.array([8, 6])
        fig, axes = plt.subplots(3, 3, figsize=figsize, **fig_opt)
        axes = axes.ravel().tolist()
        for grididx, ax in enumerate(axes[:nz]):
            z = data.z_zs[grididx]
            self.plotter.plot_traj(data.zbT_x[grididx], multicolor=True, ax=ax)
            ax.set_title(f"z={z:.1f}")
        fig_path = mkdir(plot_dir / "phase") / "phase_{:08}.jpg".format(idx)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # --------------------------------------------
        # Contour of Vl.
        cmap = "rocket"
        cmap_Vh = "RdBu_r"

        fig, axes = plt.subplots(3, 3, figsize=figsize, **fig_opt)
        axes = axes.ravel().tolist()
        for grididx, ax in enumerate(axes[:nz]):
            z = data.z_zs[grididx]
            cm = ax.contourf(bb_X, bb_Y, data.zbb_Vl[grididx], levels=32, cmap=cmap)
            self.task.setup_traj_plot(ax)
            fig.colorbar(cm, ax=ax)
            ax.set(xlabel="Position", ylabel="Velocity", title=f"z={z:.1f}")
        fig_path = mkdir(plot_dir / "Vl") / "Vl_{:08}.jpg".format(idx)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # --------------------------------------------
        # Contour of Vh.
        fig, axes = plt.subplots(3, 3, figsize=figsize, **fig_opt)
        axes = axes.ravel().tolist()
        for grididx, ax in enumerate(axes[:nz]):
            z = data.z_zs[grididx]
            cm = ax.contourf(bb_X, bb_Y, data.zbb_Vh[grididx], norm=CenteredNorm(), levels=32, cmap=cmap_Vh)
            self.task.setup_traj_plot(ax)
            fig.colorbar(cm, ax=ax)
            ax.set(xlabel="Position", ylabel="Velocity", title=f"z={z:.1f}")
        fig_path = mkdir(plot_dir / "Vh") / "Vh_{:08}.jpg".format(idx)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

    def train(
        self, key: PRNGKey, alg_cfg: EFPPOInner.Cfg, collect_cfg: CollectorCfg, wandb_name: str, trainer_cfg: TrainerCfg
    ):
        key0, key1 = jr.split(key, 2)
        alg: EFPPOInner = EFPPOInner.create(key0, self.task, alg_cfg)
        collector: Collector = Collector.create(key1, self.task, collect_cfg)

        task_name = self.task.name
        wandb_config = {"alg": alg_cfg.asdict(), "collect": collect_cfg.asdict(), "trainer": trainer_cfg.asdict()}
        wandb.init(project=f"efppo_{task_name}_inner", config=wandb_config)
        wandb_run_name = reorder_wandb_name(wandb_name=wandb_name)

        run_dir = mkdir(get_runs_dir() / f"{task_name}_inner" / wandb_run_name)
        plot_dir = mkdir(run_dir / "plots")
        ckpt_dir = mkdir(run_dir / "ckpts")
        ckpt_manager = get_ckpt_manager_sync(ckpt_dir, max_to_keep=trainer_cfg.ckpt_max_keep)

        idx = 0
        for idx in range(trainer_cfg.n_iters):
            should_log = idx % trainer_cfg.log_every == 0
            should_eval = idx % trainer_cfg.eval_every == 0
            should_ckpt = idx % trainer_cfg.ckpt_every == 0

            t0 = time.time()
            collector, col_data = alg.collect(collector)
            t1 = time.time()
            alg, loss_info = alg.update(col_data)
            t2 = time.time()

            if should_log:
                if should_eval:
                    logger.info("time  |  collect: {:.3f} update: {:.3f}".format(t1 - t0, t2 - t1))
                log_dict = {f"train/{k}": v for k, v in loss_info.items()}
                log_dict["time/collect"] = t1 - t0
                log_dict["time/update"] = t2 - t1
                wandb.log(log_dict, step=idx)

            eval_rollout_T = 128
            if should_eval:
                data = jax2np(alg.eval(eval_rollout_T))
                logger.info(f"[{idx:8}]   {data.info}")

                self.plot(idx, plot_dir, data)

                log_dict = {f"eval/{k}": v for k, v in data.info.items()}
                wandb.log(log_dict, step=idx)

            if should_ckpt:
                ckpt_manager.save_ez(idx, {"alg": alg, "alg_cfg": alg_cfg, "collect_cfg": collect_cfg})
                logger.info("Saved ckpt!")

        # Save at the end.
        ckpt_manager.save_ez(idx, {"alg": alg, "alg_cfg": alg_cfg, "collect_cfg": collect_cfg})
