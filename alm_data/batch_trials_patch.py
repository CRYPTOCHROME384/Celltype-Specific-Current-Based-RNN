#!/usr/bin/env python
import argparse, os, glob
from typing import Optional


from trials_patch_lib import export_trials_from_stage1


def parse_sid_plane(stage1_npz: str):
    base = os.path.basename(stage1_npz)
    sid_plane = base.replace("psth_", "").replace(".npz", "")
    sid = sid_plane.rsplit(".", 1)[0]
    plane = int(sid_plane.rsplit(".", 1)[1])
    return sid_plane, sid, plane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True)
    ap.add_argument("--stage1-root", default="/home/jingyi.xu/ALM/results/stage1")
    ap.add_argument("--trial-root", default="/allen/aind/scratch/jingyi/2p")
    ap.add_argument("--twnew-prefix", default="{animal}_twNew_{sid}")  # 不含plane
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    # bpod/licks 位置（按 sid 复用）
    ap.add_argument("--bpod-template", default="{trial_dir}/{animal}_twNew_{sid}.bpod.npy")
    ap.add_argument("--licks-template", default="{trial_dir}/{animal}_twNew_{sid}.licks.npy")

    # trial pkl 位置（含 sid_plane）
    ap.add_argument("--trialpkl-template", default="{trial_dir}/{animal}_twNew_{sid_plane}.trial_2p.pkl")

    # 可选：如果 bpod/licks 不存在也想继续（只是不写 per-trial reward events）
    ap.add_argument("--allow-missing-bpod-licks", action="store_true")

    args = ap.parse_args()

    animal = args.animal
    stage1_dir = os.path.join(args.stage1_root, animal)
    trial_dir = os.path.join(args.trial_root, animal)

    stage1_list = sorted(glob.glob(os.path.join(stage1_dir, "psth_*.npz")))
    if not stage1_list:
        raise FileNotFoundError(f"No stage1 npz found under: {stage1_dir}")

    ok = 0
    skip = 0
    fail = 0

    for stage1_npz in stage1_list:
        sid_plane, sid, plane = parse_sid_plane(stage1_npz)

        trial_pkl = args.trialpkl_template.format(trial_dir=trial_dir, animal=animal, sid_plane=sid_plane, sid=sid, plane=plane)
        bpod_npy = args.bpod_template.format(trial_dir=trial_dir, animal=animal, sid=sid, sid_plane=sid_plane, plane=plane)
        licks_npy = args.licks_template.format(trial_dir=trial_dir, animal=animal, sid=sid, sid_plane=sid_plane, plane=plane)

        if args.dry_run:
            print(f"[DRY] {sid_plane}")
            print(f"  stage1: {stage1_npz}")
            print(f"  trial : {trial_pkl}")
            print(f"  bpod  : {bpod_npy}")
            print(f"  licks : {licks_npy}")
            continue

        if not os.path.exists(trial_pkl):
            print(f"[SKIP] {sid_plane}: missing trial_pkl {trial_pkl}")
            skip += 1
            continue

        bpod_arg: Optional[str] = bpod_npy if os.path.exists(bpod_npy) else None
        licks_arg: Optional[str] = licks_npy if os.path.exists(licks_npy) else None

        if (bpod_arg is None or licks_arg is None) and (not args.allow_missing_bpod_licks):
            print(f"[SKIP] {sid_plane}: missing bpod/licks (bpod={os.path.exists(bpod_npy)}, licks={os.path.exists(licks_npy)})")
            skip += 1
            continue

        try:
            out = export_trials_from_stage1(
                stage1_npz=stage1_npz,
                trial_pkl=trial_pkl,
                out_path=None,          # 默认写到 trial_pkl 同目录
                force=bool(args.force),
                max_time=None,
                smooth_ms=0.0,
                bpod_npy=bpod_arg,
                licks_npy=licks_arg,
                swap_p1_p2=False,
            )
            ok += 1
        except Exception as e:
            print(f"[FAIL] {sid_plane}: {e}")
            fail += 1

    if not args.dry_run:
        print(f"\n[SUMMARY] ok={ok}, skip={skip}, fail={fail}")


if __name__ == "__main__":
    main()
