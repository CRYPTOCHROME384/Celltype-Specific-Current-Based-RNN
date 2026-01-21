#!/usr/bin/env python3
# debug_trials_shape.py
'''
python debug.py \
  --stage1_npz /home/jingyi.xu/ALM/results/stage1/kd95/psth_20220823_205730.0.npz \
  --trials_npz /allen/aind/scratch/jingyi/2p/kd95/trials_20220823_205730.0.npz \
  --allow_pickle
'''
import argparse
import os
import glob
import numpy as np


TRIAL_KEYS_DEFAULT = ("cell_trials", "trials", "psth_trials", "F_trials")


def _safe_npz_item(npz, key):
    if key in npz.files:
        return npz[key]
    return None


def _guess_trials_dict(trials_npz, trial_keys):
    """
    Return (trial_dict, used_key) where trial_dict is a Python dict:
      cond_name -> np.ndarray
    Supports:
      - npz[key] is an object array storing a dict (allow_pickle)
      - or trials_npz itself has per-condition arrays
    """
    # 1) try candidate keys: trials_npz[key] is dict-like
    for k in trial_keys:
        if k in trials_npz.files:
            obj = trials_npz[k]
            # common: 0-d object array containing dict
            if obj.dtype == object:
                try:
                    d = obj.item()
                    if isinstance(d, dict) and len(d) > 0:
                        return d, k
                except Exception:
                    pass

    # 2) fallback: treat all entries except obvious meta as cond arrays
    # (only use this if nothing else works)
    d = {}
    for k in trials_npz.files:
        if k in ("meta", "cond_names", "fps", "T"):
            continue
        arr = trials_npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            d[k] = arr
    if len(d) > 0:
        return d, "<per-key arrays>"
    return None, None


def _to_trials_time_neurons_like(a: np.ndarray):
    """
    Mimic your _to_trials_time_neurons heuristics:
    - try all transposes, pick the one with largest T (middle dim)
    - (we also print all candidates)
    Returns best_cand, best_perm
    """
    perms = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    ]
    best = None
    best_perm = None
    best_T = -1

    cands = []
    for p in perms:
        cand = a.transpose(p)
        R, T, N = cand.shape
        cands.append((p, cand.shape))
        if T > best_T:
            best_T = T
            best = cand
            best_perm = p
    return best, best_perm, cands


def _find_trials_npz(stage1_npz_path: str):
    d = os.path.dirname(stage1_npz_path)
    cands = sorted(glob.glob(os.path.join(d, "trials_*.npz")))
    if len(cands) == 1:
        return cands[0], cands
    # also try name transform
    base = os.path.basename(stage1_npz_path)
    guess = base.replace("psth_", "trials_")
    guess_path = os.path.join(d, guess)
    if os.path.isfile(guess_path):
        return guess_path, cands
    return None, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_npz", required=True, help="Path to stage1 psth_*.npz")
    ap.add_argument("--trials_npz", default=None, help="Optional explicit path to trials_*.npz")
    ap.add_argument("--trial_keys", nargs="+", default=list(TRIAL_KEYS_DEFAULT),
                    help="Candidate keys to locate the trials dict inside trials npz")
    ap.add_argument("--print_max", type=int, default=6, help="Max conditions to print (for brevity)")
    ap.add_argument("--allow_pickle", action="store_true", help="Allow pickle when loading trials npz (usually needed).")
    args = ap.parse_args()

    print("\n=== [1] Load stage1 npz ===")
    stage1 = np.load(args.stage1_npz, allow_pickle=True)
    print("stage1:", args.stage1_npz)
    print("stage1 keys:", stage1.files)

    meta = None
    if "meta" in stage1.files:
        try:
            meta = stage1["meta"].item()
            print("meta keys:", list(meta.keys())[:30], "...")
        except Exception as e:
            print("WARN: cannot parse stage1['meta'] as dict:", e)

    if meta is not None:
        cond_names = meta.get("cond_names", None)
        fps = meta.get("fps", None)
        T_meta = meta.get("T", None)
        print("meta.cond_names:", cond_names)
        print("meta.fps:", fps, "meta.T:", T_meta)

    print("\n=== [2] Resolve trials npz ===")
    trials_path = args.trials_npz
    cands = []
    if trials_path is None:
        trials_path, cands = _find_trials_npz(args.stage1_npz)
    print("trials candidates in dir:", cands[:10], ("..." if len(cands) > 10 else ""))
    if trials_path is None:
        raise RuntimeError("Cannot resolve trials npz automatically. Provide --trials_npz explicitly.")
    print("using trials:", trials_path)

    print("\n=== [3] Load trials npz ===")
    trials_npz = np.load(trials_path, allow_pickle=bool(args.allow_pickle))
    print("trials keys:", trials_npz.files)

    trials_dict, used_key = _guess_trials_dict(trials_npz, tuple(args.trial_keys))
    if trials_dict is None:
        raise RuntimeError("Cannot locate trial dict/arrays inside trials npz. Try --allow_pickle or check keys.")
    print("found trials dict via:", used_key)
    print("num conditions in trials_dict:", len(trials_dict))

    # choose cond list to print
    keys = list(trials_dict.keys())
    keys.sort()
    show_keys = keys[: int(args.print_max)]

    print("\n=== [4] Per-condition shape checks ===")
    for cname in show_keys:
        arr = np.asarray(trials_dict[cname])
        print("\n--- cond:", cname, "---")
        print("raw shape:", arr.shape, "dtype:", arr.dtype, "ndim:", arr.ndim)

        if arr.ndim != 3:
            print("ERROR: arr.ndim != 3, skip")
            continue

        # A) check the current *hard-coded* assumption in global preload:
        # assumes arr = [C_keep, T, nTr]
        Ck, Tk, Rk = arr.shape
        print("[hard-coded assumption] treat as [C_keep, T, nTr] ->",
              f"C_keep={Ck}, T={Tk}, nTr={Rk}")

        # B) mimic _to_trials_time_neurons heuristic
        best, best_perm, cands = _to_trials_time_neurons_like(arr)
        print("[heuristic] candidate transposes ->")
        for p, sh in cands:
            print("  perm", p, "=>", sh)
        print("[heuristic] best perm:", best_perm, "best [R,T,N] shape:", best.shape)

        # Optional: quick sanity about what looks like "trial dimension"
        # trial count usually not huge (<< neurons). Warn if trial dim suspicious.
        R, T, N = best.shape
        if R > 500 and N < 200:
            print("WARN: heuristic thinks R is huge; likely trials axis mis-inferred. Check exporter format.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
