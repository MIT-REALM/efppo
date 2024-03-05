import functools as ft
import pathlib

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import efppo.run_config.f16
from efppo.rl.collector import RolloutOutput, collect_single_mode
from efppo.rl.efppo_inner import EFPPOInner
from efppo.rl.rootfind_policy import Rootfinder, RootfindPolicy
from efppo.task.f16 import F16GCASFloorCeil
from efppo.task.plotter import Plotter
from efppo.utils.ckpt_utils import get_run_dir_from_ckpt_path, load_ckpt_ez
from efppo.utils.jax_utils import jax2np, jax_vmap, merge01
from efppo.utils.logging import set_logger_format
from efppo.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    set_logger_format()

    run_dir = get_run_dir_from_ckpt_path(ckpt_path)

    task = F16GCASFloorCeil()
    # For prettier trajectories.
    task.dt /= 2
    alg_cfg, collect_cfg = efppo.run_config.f16.get()
    alg: EFPPOInner = EFPPOInner.create(jr.PRNGKey(0), task, alg_cfg)
    ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
    alg = ckpt_dict["alg"]

    rootfind = Rootfinder(alg.Vh.apply, alg.z_min, alg.z_max, h_tgt=-0.70)
    rootfind_pol = RootfindPolicy(alg.policy.apply, rootfind)

    # -----------------------------------------------------
    plot_dir = mkdir(run_dir / "eval_plots")

    rollout_T = 512

    bb_X, bb_Y, bb_x0 = jax2np(task.grid_contour())
    b1, b2 = bb_X.shape
    bb_z0 = np.full((b1, b2), alg.z_max)

    collect_fn = ft.partial(
        collect_single_mode,
        task,
        get_pol=rootfind_pol,
        disc_gamma=alg.disc_gamma,
        z_min=alg.z_min,
        z_max=alg.z_max,
        rollout_T=rollout_T,
    )
    logger.info("Collecting rollouts...")
    vmap_fn = jax.jit(jax_vmap(collect_fn, rep=2))
    bb_rollout: RolloutOutput = jax2np(vmap_fn(bb_x0, bb_z0))
    logger.info("Done collecting rollouts.")

    ###############################3
    # Plot.
    rng = np.random.default_rng(seed=124122)
    plotter = Plotter(task)

    bbTp1_state = bb_rollout.Tp1_state
    bTp1_state = merge01(bbTp1_state)

    idxs = rng.choice(bTp1_state.shape[0], size=100, replace=False)
    bTp1_state = bTp1_state[idxs]
    bb_h = np.max(bb_rollout.Th_h, axis=(2, 3))
    b_issafe = merge01(bb_h)[idxs] < 0

    figsize = np.array([2.8, 2.2])
    fig, ax = plt.subplots(figsize=figsize, dpi=500)
    fig = plotter.plot_traj(bTp1_state, multicolor=True, ax=ax)
    fig_path = plot_dir / "eval_trajs.jpg"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------
    # Plot trajectory in time.
    bTp1_state_safe = bTp1_state[b_issafe]
    fig = plotter.plot_traj2(bTp1_state_safe)
    fig_path = plot_dir / "eval_traj_time.jpg"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
