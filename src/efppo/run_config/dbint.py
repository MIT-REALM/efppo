from efppo.rl.collector import CollectorCfg
from efppo.rl.efppo_inner import EFPPOCfg
from efppo.utils.schedules import LinDecay


def get():
    zmin, zmax = -1.0, 3.0
    nz_enc = 8
    z_mean = 0.5
    z_scale = 1.0

    pol_hids = val_hids = [256, 256]

    pol_lr = 3e-4
    val_lr = 3e-4
    entropy_cf = LinDecay(1e-2, 1e3, warmup_steps=5_000, trans_steps=35_000)
    disc_gamma = 0.97

    n_batches = 8

    net_cfg = EFPPOCfg.NetCfg(
        pol_lr, val_lr, entropy_cf, disc_gamma, "tanh", pol_hids, val_hids, nz_enc, z_mean, z_scale
    )
    train_cfg = EFPPOCfg.TrainCfg(zmin, zmax, 0.95, 50.0, n_batches, 0.1, 1.0, 1.0)
    eval_cfg = EFPPOCfg.EvalCfg()
    alg_cfg = EFPPOCfg(net_cfg, train_cfg, eval_cfg)

    n_envs = 128
    rollout_T = 64
    mean_age = 80
    max_T = 64
    collect_cfg = CollectorCfg(n_envs, rollout_T, mean_age, max_T)

    return alg_cfg, collect_cfg
