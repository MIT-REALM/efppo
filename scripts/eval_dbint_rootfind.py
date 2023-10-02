import functools as ft
import pathlib

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer

import efppo.run_config.dbint
from efppo.rl.collector import RolloutOutput, collect_single_mode
from efppo.rl.efppo_inner import EFPPOInner
from efppo.rl.rootfind_policy import Rootfinder, RootfindPolicy
from efppo.task.dbint import DbInt
from efppo.task.plotter import Plotter
from efppo.utils.ckpt_utils import get_run_dir_from_ckpt_path, load_ckpt_ez
from efppo.utils.jax_utils import jax_vmap
from efppo.utils.logging import set_logger_format
from efppo.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    set_logger_format()

    run_dir = get_run_dir_from_ckpt_path(ckpt_path)

    task = DbInt()
    # For prettier trajectories.
    task.dt /= 4
    alg_cfg, collect_cfg = efppo.run_config.dbint.get()
    alg: EFPPOInner = EFPPOInner.create(jr.PRNGKey(0), task, alg_cfg)
    ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
    alg = ckpt_dict["alg"]

    rootfind = Rootfinder(alg.Vh.apply, alg.z_min, alg.z_max)
    rootfind_pol = RootfindPolicy(alg.policy.apply, rootfind)

    # -----------------------------------------------------
    plot_dir = mkdir(run_dir / "eval_plots")

    rollout_T = 512

    b_x0 = task.get_x0_eval(64)
    batch_size = len(b_x0)
    b_z0 = np.full(batch_size, 0.0)

    collect_fn = ft.partial(
        collect_single_mode,
        task,
        get_pol=rootfind_pol,
        disc_gamma=alg.disc_gamma,
        z_min=alg.z_min,
        z_max=alg.z_max,
        rollout_T=rollout_T,
    )
    print("Collecting rollouts...")
    b_rollout: RolloutOutput = jax.jit(jax_vmap(collect_fn))(b_x0, b_z0)
    print("Done collecting rollouts.")

    ###########################################3
    # Plot.
    plotter = Plotter(task)
    figsize = np.array([2.8, 2.2])
    fig, ax = plt.subplots(figsize=figsize, dpi=500)
    fig = plotter.plot_traj(b_rollout.Tp1_state, multicolor=True, ax=ax)
    fig.savefig(plot_dir / "eval_trajs.jpg", bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
