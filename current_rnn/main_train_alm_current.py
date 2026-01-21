"""current_rnn/main_train_alm_current.py

Minimal registry-global training entrypoint (trial-forward + PSTH-from-trial-mean + celltype loss).

Example:
  nohup python -u main_train_alm_current.py \
    --registry_dir /home/jingyi.xu/ALM/results/registry/kd95 \
    --animal kd95 \
    --out_dir /home/jingyi.xu/code_rnn/results_current/smoke_celltype_20260121 \
    --max_epochs 200000 \
    --lr 5e-5 \
    --noise_std 0.005 \
    --noise_std_eval 0.00 \
    --n_exc_virtual 800 \
    --unit_sampling random \
    --lambda_celltype 0.01 \
    --log_celltype_every 100 \
    --log_celltype_topk 20 \
    --dale \
    > train.log 2>&1 &
"""

import os
import argparse
import pandas as pd

from training_current import train_current_alm_global


def _infer_n_obs_from_registry(registry_dir: str, animal: str) -> int:
    """Infer n_obs (global observed inhibitory count) from registry csv's global_idx."""
    cand1 = os.path.join(registry_dir, f"{animal}_registry.csv")
    cand2 = os.path.join(registry_dir, "registry.csv")
    if os.path.isfile(cand1):
        csv_path = cand1
    elif os.path.isfile(cand2):
        csv_path = cand2
    else:
        raise FileNotFoundError(f"Cannot find registry csv in {registry_dir} (tried {cand1} and {cand2})")

    df = pd.read_csv(csv_path)
    if "global_idx" not in df.columns:
        raise KeyError(f"Registry missing 'global_idx'. Got columns={list(df.columns)}")
    g = df["global_idx"].to_numpy()
    if g.size == 0:
        raise ValueError("Registry csv is empty.")
    return int(g.max()) + 1


def parse_args():
    ap = argparse.ArgumentParser()

    # required
    ap.add_argument("--registry_dir", required=True, help="Directory containing <animal>_registry.csv or registry.csv")
    ap.add_argument("--animal", required=True, help="Animal name, e.g., kd95")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    # training
    ap.add_argument("--max_epochs", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # dynamics
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--nonlinearity", type=str, default="relu")

    # architecture
    ap.add_argument("--dale", action="store_true", help="Enable Dale constraint")
    ap.add_argument("--no_dale", action="store_true", help="Disable Dale constraint")

    ap.add_argument("--n_exc_virtual", type=int, default=0, help="If 0, infer from registry and exc_ratio")
    ap.add_argument("--exc_ratio", type=float, default=4.0, help="If n_exc_virtual==0, set n_exc_virtual=exc_ratio*n_obs")

    # data
    ap.add_argument("--max_sessions", type=int, default=0, help="Use only top-K units (0=all)")
    ap.add_argument("--unit_sampling", default="random", choices=["random", "cycle"])
    ap.add_argument("--cond_filter", default=None, help="Comma-separated condition names")
    ap.add_argument("--max_time", type=int, default=None)
    ap.add_argument("--psth_bin_ms", type=float, default=200.0)
    ap.add_argument("--sample_ignore_ms", type=float, default=50.0)
    ap.add_argument("--resp_sec", type=float, default=2.0)

    # trial-level
    ap.add_argument("--trial_keys", default="cell_trials", help="Comma-separated trial dict key candidates")
    ap.add_argument("--trial_batch_per_cond", type=int, default=0, help="0=use all trials")
    ap.add_argument("--trials_bin_ms", type=float, default=200.0)

    # noise
    ap.add_argument("--noise_std", type=float, default=0.01)
    ap.add_argument("--noise_std_eval", type=float, default=0.0)

    # loss
    ap.add_argument("--lambda_celltype", type=float, default=0.05, help="Weight for celltype loss/logging")
    ap.add_argument(
        "--celltype_exclude",
        default="",  # comma-separated
        help="Comma-separated labels to exclude from celltype loss/log (e.g., 'unknown,nan').",
    )

    # logging
    ap.add_argument("--log_celltype_every", type=int, default=10, help="Print per-unit celltype diagnostics every N epochs (0=disable)")
    ap.add_argument("--log_celltype_topk", type=int, default=20, help="Top-K celltypes to print per unit")

    # eval/ckpt
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_latest_every", type=int, default=100)
    ap.add_argument("--save_best_every", type=int, default=50)

    return ap.parse_args()


def main():
    args = parse_args()

    dale = True
    if args.no_dale:
        dale = False
    if args.dale:
        dale = True

    os.makedirs(args.out_dir, exist_ok=True)

    cond_filter = None
    if args.cond_filter is not None and str(args.cond_filter).strip() != "":
        cond_filter = [x.strip() for x in str(args.cond_filter).split(",") if x.strip()]

    trial_keys = tuple([x.strip() for x in str(args.trial_keys).split(",") if x.strip()])

    n_exc_virtual = int(args.n_exc_virtual)
    if n_exc_virtual == 0:
        n_obs = _infer_n_obs_from_registry(args.registry_dir, args.animal)
        n_exc_virtual = int(round(float(args.exc_ratio) * float(n_obs)))
        print(f"[INFO] n_exc_virtual not set; inferred n_obs={n_obs}, set n_exc_virtual={n_exc_virtual} (exc_ratio={args.exc_ratio})")

    max_sessions = None
    if int(args.max_sessions) > 0:
        max_sessions = int(args.max_sessions)

    exclude = tuple([x.strip() for x in str(args.celltype_exclude).split(",") if x.strip()])

    train_current_alm_global(
        registry_dir=args.registry_dir,
        animal=args.animal,
        out_dir=args.out_dir,

        max_sessions=max_sessions,
        seed=int(args.seed),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_epochs=int(args.max_epochs),
        grad_clip=float(args.grad_clip),
        unit_sampling=str(args.unit_sampling),

        dt=float(args.dt),
        tau=float(args.tau),
        substeps=int(args.substeps),
        nonlinearity=str(args.nonlinearity),
        dale=bool(dale),
        n_exc_virtual=int(n_exc_virtual),

        cond_filter=cond_filter,
        max_time=args.max_time,
        psth_bin_ms=float(args.psth_bin_ms),
        sample_ignore_ms=float(args.sample_ignore_ms),
        resp_sec=float(args.resp_sec),

        # trial batch + noise
        use_trials=True,
        trial_keys=trial_keys,
        trial_batch_per_cond=int(args.trial_batch_per_cond),
        trials_bin_ms=float(args.trials_bin_ms),
        noise_std=float(args.noise_std),
        noise_std_eval=float(args.noise_std_eval),

        # loss/log
        lambda_celltype=float(args.lambda_celltype),
        celltype_label_key="cell_subclasses",
        celltype_exclude=exclude if len(exclude) > 0 else ("", "nan", "none", "unknown"),
        log_celltype_every=int(args.log_celltype_every),
        log_celltype_topk=int(args.log_celltype_topk),

        # eval/ckpt
        eval_every=int(args.eval_every),
        save_latest_every=int(args.save_latest_every),
        save_best_every=int(args.save_best_every),
    )


if __name__ == "__main__":
    main()
