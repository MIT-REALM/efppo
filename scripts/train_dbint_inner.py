import ipdb
import jax.random as jr
import typer

import efppo.run_config.dbint
from efppo.rl.efppo_inner_trainer import EFPPOInnerTrainer, TrainerCfg
from efppo.task.dbint import DbInt
from efppo.utils.logging import set_logger_format


def main(
    name: str = typer.Option(None, help="Name of the run."),
    seed: int = 123445,
):
    set_logger_format()
    task = DbInt()
    alg_cfg, collect_cfg = efppo.run_config.dbint.get()
    trainer = EFPPOInnerTrainer(task)
    trainer_cfg = TrainerCfg(n_iters=100_000, log_every=100, eval_every=1_000, ckpt_every=10_000)
    trainer.train(jr.PRNGKey(seed), alg_cfg, collect_cfg, name, trainer_cfg)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
