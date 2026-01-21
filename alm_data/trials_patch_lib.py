#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch-based trial-level exporter for ALM 2p data.

Key idea:
- Stage1 (psth_*.npz) has already fixed:
  * keep_idx (selected inhibitory neurons)
  * event_frames (S/D/R)
  * fps
  * all_data (bpod-derived trial meta, 21 x nTrials)
  * optionally reward_trace / lick traces (cond-average) already patched in stage1
- This script ONLY adds trial-level neural activity (and optionally per-trial reward events)
  into a separate trials_*.npz, without redoing any celltype mapping.

Outputs:
- trials_<session_id>.<plane>.npz (default in same dir as trial_2p.pkl)
  * cell_trials: dict(cond -> float32[C_keep, T_keep, nTr])
  * trial_indices: dict(cond -> int32[nTr])  (indices in original trial axis)
  * keep_idx, cell_subclasses, cell_clusters, event_frames, fps, cond_names, ...
  * (optional) reward_trace/lick traces copied from stage1 if present
  * (optional) per-trial reward events arrays if bpod/licks provided

Example:
python -u 0.trials.py \
  --trial-pkl /allen/aind/scratch/jingyi/2p/kd95/kd95_twNew_20220823_205730.0.trial_2p.pkl \
  --stage1-npz /home/jingyi.xu/ALM/results/stage1/kd95/psth_20220823_205730.0.npz \
  --force
"""

import argparse
import os
import pickle as pkl
import numpy as np
from typing import Dict, Tuple, Optional, Any, List


# -----------------------------
# Bpod key maps (must match bpod_parse.py and 0.average.py)
# -----------------------------
def keys(name: str = ""):
    d = {
        "outcome": 0,
        "trialtype": 1,
        "earlysample": 2,
        "earlydelay": 3,
        "puff": 4,
        "autowater": 5,
        "totalwater": 6,
        "length": 7,
        "z": 8,
        "power": 9,
        "freelick_pos": 10,
        "licksneeded": 11,
        "protocol": 12,
        "rfl": 13,
        "rt": 14,
        "samplestart": 15,
        "lastdelay": 16,
        "go": 17,
        "reward": 18,
        "freelicktype": 19,
        "reward1": 20,
    }
    return d[name] if name else d


def outcomes(name: str = ""):
    d = {
        "correct": 1,
        "incorrect": 2,
        "ignore": 3,
        "nofollow": 4,
        "droppednotlick": 5,
        "spont": 6,
    }
    return d[name] if name else d


def _infer_left_right_from_trialtype(tt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.unique(tt[~np.isnan(tt)])
    vals_set = set(vals.tolist())

    if vals_set >= {0, 1}:
        is_left = (tt == 0)
        is_right = (tt == 1)
    elif vals_set >= {1, 2}:
        is_left = (tt == 1)
        is_right = (tt == 2)
    else:
        mid = np.median(vals)
        is_left = (tt <= mid)
        is_right = (tt > mid)
    return is_left, is_right


# -----------------------------
# Same logic as 0.average.py: prelim_events_and_trials
# We only keep left_correct / right_correct here.
# -----------------------------
def prelim_events_and_trials_only_LC_RC(all_data: np.ndarray, fps: float):
    behavior = (all_data[keys("protocol")] == 5)
    full_sample = (all_data[keys("earlysample")] == 0)
    full_delay = (all_data[keys("earlydelay")] == 0)
    trials_ok = behavior & full_sample & full_delay

    o = all_data[keys("outcome")]
    tt = all_data[keys("trialtype")]

    CORR = np.isin(o, [outcomes("correct"), outcomes("droppednotlick")])

    FT = trials_ok | (o == outcomes("ignore"))
    ss_sec = np.median(all_data[keys("samplestart")][FT])
    ld_sec = np.median(all_data[keys("lastdelay")][FT])
    go_sec = np.median(all_data[keys("go")][FT])

    ss = int(round(float(ss_sec) * float(fps)))
    ld = int(round(float(ld_sec) * float(fps)))
    go = int(round(float(go_sec) * float(fps)))

    L, R = _infer_left_right_from_trialtype(np.asarray(tt, float))
    LC = np.where(trials_ok & L & CORR)[0].astype(int)
    RC = np.where(trials_ok & R & CORR)[0].astype(int)

    event_frames = {"S": ss, "D": ld, "R": go}
    cond_idx = {
        "left_correct": LC,
        "right_correct": RC,
    }
    # filter empty
    cond_idx = {k: v for k, v in cond_idx.items() if v.size > 0}

    return event_frames, cond_idx, float(go_sec)


# -----------------------------
# dF/F computation (match 0.average.py)
# -----------------------------
def compute_dFoF_presample(F: np.ndarray, fps: float, ss_frames: int) -> np.ndarray:
    C, T, N = F.shape
    ss_frames = max(1, min(int(ss_frames), int(T)))
    base = F[:, :ss_frames, :].mean(axis=1, keepdims=True)
    base = np.maximum(base, 1e-6)
    return (F - base) / base


def _load_trial2p(trial_pkl: str):
    with open(trial_pkl, "rb") as f:
        obj = pkl.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        all_data, F, Fbase = obj[0], obj[1], obj[2]
    else:
        raise ValueError(f"Unexpected trial_2p.pkl structure: {type(obj)}")
    all_data = np.asarray(all_data)
    F = np.asarray(F)
    Fbase = np.asarray(Fbase)
    return all_data, F, Fbase


def _npz_get_obj(z, k: str, default=None):
    if k not in z.files:
        return default
    v = z[k]
    # dicts are usually stored as object arrays
    if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
        return v.item()
    if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        return v.reshape(()).item()
    return v


def _ensure_dir(p: str):
    d = os.path.dirname(os.path.abspath(p))
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


# -----------------------------
# Optional: per-trial reward events (best-effort)
# Uses current_rnn/build_lick_reward_trace.py internals if available.
# -----------------------------
def _compute_per_trial_reward_events_best_effort(
    stage1_all_data: np.ndarray,
    event_frames: Dict[str, int],
    fps: float,
    T: int,
    bpod_npy: Optional[str],
    licks_npy: Optional[str],
    swap_p1_p2: bool,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - reward_events_rel_go_sec: object array len n_trials_2p, each is float array of rel times
      - reward_events_frame: object array len n_trials_2p, each is int array of frame indices
      - reward_first_rel_go_sec: float32 array len n_trials_2p, NaN if none
      - idx_2p_in_bpod: int32 array len n_trials_2p (if alignment possible)

    If any step fails, return {} (and caller will warn).
    """
    if bpod_npy is None or (not os.path.exists(bpod_npy)):
        return {}

    try:
        # Import from your existing helper; this is the same file you used to patch stage1 reward_trace.
        from current_rnn import build_lick_reward_trace as blr  # type: ignore
    except Exception:
        try:
            import build_lick_reward_trace as blr  # type: ignore
        except Exception:
            return {}

    try:
        bpod_output = blr._load_bpod_output(bpod_npy)  # shape (21, n_trials_total) or similar
        n_trials_total = int(np.asarray(bpod_output).shape[1])

        # load licks (prefer fixed licks_npy if provided)
        p1_all, p2_all, _src = blr._load_licks(bpod_npy, licks_npy, n_trials_total=n_trials_total)

        # align stage1 trials -> bpod indexing
        idx_2p_in_bpod = blr._align_stage1_trials_to_bpod(stage1_all_data, bpod_output, atol=1e-6)
        idx_2p_in_bpod = np.asarray(idx_2p_in_bpod, dtype=np.int32)

        p1_2p = [p1_all[int(i)] for i in idx_2p_in_bpod]
        p2_2p = [p2_all[int(i)] for i in idx_2p_in_bpod]

        # apply swap if requested
        if bool(swap_p1_p2):
            p1_2p, p2_2p = p2_2p, p1_2p

        go_each = np.asarray(stage1_all_data[keys("go")], float)
        tt = np.asarray(stage1_all_data[keys("trialtype")], float)
        is_left, is_right = blr._infer_left_right_from_trialtype(tt) if hasattr(blr, "_infer_left_right_from_trialtype") else _infer_left_right_from_trialtype(tt)

        R_frame = int(event_frames["R"])
        max_rel = float((T - 1 - R_frame) / float(fps))
        min_rel = float(-R_frame / float(fps))

        reward_events_rel = np.empty(len(go_each), dtype=object)
        reward_events_frm = np.empty(len(go_each), dtype=object)
        reward_first_rel = np.full(len(go_each), np.nan, dtype=np.float32)

        for tr in range(len(go_each)):
            g = float(go_each[tr])
            if not np.isfinite(g):
                reward_events_rel[tr] = np.asarray([], dtype=np.float32)
                reward_events_frm[tr] = np.asarray([], dtype=np.int32)
                continue

            # correct port for this trial: left -> p1, right -> p2
            # (this matches build_lick_and_reward convention after Tim fix)
            lick_list = p1_2p[tr] if bool(is_left[tr]) else p2_2p[tr]
            if lick_list is None:
                lick_list = []
            lick_arr = np.asarray(lick_list, dtype=float).reshape(-1)
            if lick_arr.size == 0:
                reward_events_rel[tr] = np.asarray([], dtype=np.float32)
                reward_events_frm[tr] = np.asarray([], dtype=np.int32)
                continue

            rel = lick_arr - g  # seconds relative to go
            rel = rel[np.isfinite(rel)]
            # keep only within window that maps into [0, T)
            rel = rel[(rel >= min_rel) & (rel <= max_rel)]
            if rel.size == 0:
                reward_events_rel[tr] = np.asarray([], dtype=np.float32)
                reward_events_frm[tr] = np.asarray([], dtype=np.int32)
                continue

            frm = R_frame + np.round(rel * float(fps)).astype(int)
            frm = frm[(frm >= 0) & (frm < T)]
            rel = rel[(frm >= 0) & (frm < T)]
            order = np.argsort(rel)
            rel = rel[order].astype(np.float32)
            frm = frm[order].astype(np.int32)

            reward_events_rel[tr] = rel
            reward_events_frm[tr] = frm
            reward_first_rel[tr] = float(rel[0])

        return {
            "reward_events_rel_go_sec": reward_events_rel,
            "reward_events_frame": reward_events_frm,
            "reward_first_rel_go_sec": reward_first_rel,
            "idx_2p_in_bpod": idx_2p_in_bpod,
        }

    except Exception:
        return {}


def export_trials_from_stage1(
    stage1_npz: str,
    trial_pkl: str,
    out_path: Optional[str],
    force: bool,
    max_time: Optional[int],
    smooth_ms: float,
    bpod_npy: Optional[str],
    licks_npy: Optional[str],
    swap_p1_p2: bool,
) -> str:
    if (not os.path.exists(stage1_npz)):
        raise FileNotFoundError(f"stage1 npz not found: {stage1_npz}")
    if (not os.path.exists(trial_pkl)):
        raise FileNotFoundError(f"trial pkl not found: {trial_pkl}")

    z = np.load(stage1_npz, allow_pickle=True)

    keep_idx = _npz_get_obj(z, "keep_idx", None)
    if keep_idx is None:
        raise KeyError(f"keep_idx missing in stage1 npz: {stage1_npz}")
    keep_idx = np.asarray(keep_idx, dtype=int)

    stage1_all_data = _npz_get_obj(z, "all_data", None)
    if stage1_all_data is None:
        raise KeyError(f"all_data missing in stage1 npz: {stage1_npz}")
    stage1_all_data = np.asarray(stage1_all_data)

    fps = float(_npz_get_obj(z, "fps", 1.0))
    stage1_event_frames = _npz_get_obj(z, "event_frames", None)
    if stage1_event_frames is None:
        raise KeyError(f"event_frames missing in stage1 npz: {stage1_npz}")
    stage1_event_frames = dict(stage1_event_frames)

    # Decide T_keep:
    # We will use the trial_pkl's time axis length, optionally trunc to max_time.
    all_data_pkl, F, Fbase = _load_trial2p(trial_pkl)
    if F.ndim != 3:
        raise ValueError(f"Expected F shape (cells,time,trials), got {F.shape}")

    C_full, T_full, N_trials = int(F.shape[0]), int(F.shape[1]), int(F.shape[2])

    # Consistency check: all_data should match
    if stage1_all_data.shape != all_data_pkl.shape:
        raise ValueError(
            f"all_data shape mismatch: stage1 {stage1_all_data.shape} vs pkl {all_data_pkl.shape}. "
            "You may have paired the wrong stage1_npz and trial_pkl."
        )

    # Soft check values (do not require exact float equality)
    try:
        if not np.allclose(np.asarray(stage1_all_data, float), np.asarray(all_data_pkl, float), atol=1e-6, equal_nan=True):
            print("[WARN] stage1 all_data != pkl all_data (values differ). Proceeding using stage1 all_data for cond split.")
    except Exception:
        print("[WARN] cannot allclose check all_data; proceeding using stage1 all_data.")

    # Build cond indices from stage1 all_data (same rule as 0.average.py)
    event_frames_from_rule, cond_idx, go_sec_median = prelim_events_and_trials_only_LC_RC(stage1_all_data, fps=fps)

    # Prefer stage1's stored event_frames (truth) but ensure R exists
    event_frames = dict(stage1_event_frames)
    if "R" not in event_frames:
        event_frames = event_frames_from_rule

    # Select time length
    T_keep = int(T_full)
    if max_time is not None and int(max_time) > 0:
        T_keep = min(T_keep, int(max_time))

    # Compute dF/F using presample baseline (ss_frames)
    ss_frames = int(event_frames.get("S", event_frames_from_rule["S"]))
    dFoF = compute_dFoF_presample(F, fps=fps, ss_frames=ss_frames)  # (cells,time,trials)
    dFoF = dFoF[:, :T_keep, :]

    # Optional temporal smoothing (moving average) on trial-level traces
    if smooth_ms is not None and float(smooth_ms) > 0:
        win = int(round((float(smooth_ms) / 1000.0) * float(fps)))
        win = max(1, win)
        if win > 1:
            k = np.ones(win, dtype=np.float32) / float(win)
            # convolve along time for each cell and trial
            # dFoF: (C,T,N)
            C, T, N = dFoF.shape
            out = np.empty_like(dFoF, dtype=np.float32)
            for c in range(C):
                for n in range(N):
                    out[c, :, n] = np.convolve(dFoF[c, :, n].astype(np.float32), k, mode="same")
            dFoF = out

    # Apply keep_idx
    if keep_idx.size == 0:
        raise ValueError(f"keep_idx is empty in stage1 npz: {stage1_npz}")
    if int(keep_idx.max()) >= C_full:
        raise ValueError(
            f"keep_idx out of range: max(keep_idx)={int(keep_idx.max())} >= C_full={C_full}. "
            "You may have paired wrong stage1_npz and trial_pkl."
        )

    dFoF_keep = dFoF[keep_idx, :, :]  # (C_keep, T_keep, N_trials)
    C_keep = int(dFoF_keep.shape[0])

    # Build per-cond trial tensors
    cond_names = ["left_correct", "right_correct"]
    cell_trials: Dict[str, np.ndarray] = {}
    trial_indices: Dict[str, np.ndarray] = {}

    for cname in cond_names:
        idx = cond_idx.get(cname, np.array([], dtype=int))
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            continue
        # bounds check
        if int(idx.max()) >= N_trials:
            raise ValueError(f"cond {cname}: trial index out of range max={int(idx.max())} >= N_trials={N_trials}")
        X = dFoF_keep[:, :, idx]  # (C_keep, T_keep, nTr)
        cell_trials[cname] = X.astype(np.float32, copy=False)
        trial_indices[cname] = idx.astype(np.int32)

    if len(cell_trials) == 0:
        raise RuntimeError("No trials in left_correct/right_correct after filtering. Check all_data/protocol/outcome logic.")

    # Decide output path
    if out_path is None:
        # parse session_id.plane from stage1 filename if possible
        base = os.path.basename(stage1_npz)
        # expected: psth_<session_id>.<plane>.npz
        sid_plane = base.replace("psth_", "").replace(".npz", "")
        out_dir = os.path.dirname(os.path.abspath(trial_pkl))
        out_path = os.path.join(out_dir, f"trials_{sid_plane}.npz")

    if os.path.exists(out_path) and (not force):
        raise FileExistsError(f"Output exists: {out_path}. Use --force to overwrite.")

    _ensure_dir(out_path)

    # Copy meta from stage1 if present
    meta = {}
    for k in [
        "cell_clusters", "cell_subclasses", "kept_table", "n_cells_before",
        "cond_counts", "go_sec", "ss_sec", "ld_sec"
    ]:
        if k in z.files:
            meta[k] = _npz_get_obj(z, k)

    # Copy reward/lick traces (cond-average training inputs) if present in stage1
    for k in [
        "reward_trace", "lick_rate_left", "lick_rate_right", "lick_rate_total", "t_rel_go_sec",
        "swap_p1_p2_used", "auto_swap_used", "licks_npy_used", "licks_source", "align_atol_used"
    ]:
        if k in z.files:
            meta[k] = _npz_get_obj(z, k)

    # Optional: per-trial reward events (future use)
    per_trial_reward = _compute_per_trial_reward_events_best_effort(
        stage1_all_data=stage1_all_data,
        event_frames=event_frames,
        fps=fps,
        T=T_keep,
        bpod_npy=bpod_npy,
        licks_npy=licks_npy,
        swap_p1_p2=swap_p1_p2 or bool(meta.get("swap_p1_p2_used", False)),
    )

    # Prepare save dict
    save_dict: Dict[str, Any] = {
        "stage1_npz": np.array([stage1_npz], dtype=object),
        "trial_pkl": np.array([trial_pkl], dtype=object),
        "fps": float(fps),
        "event_frames": np.array(event_frames, dtype=object),
        "cond_names": np.array(cond_names, dtype=object),
        "keep_idx": keep_idx.astype(np.int32),
        "cell_trials": np.array(cell_trials, dtype=object),
        "trial_indices": np.array(trial_indices, dtype=object),
        "C_keep": np.int32(C_keep),
        "T_keep": np.int32(T_keep),
        "N_trials_total": np.int32(N_trials),
    }

    # add meta fields
    for k, v in meta.items():
        save_dict[k] = v if not isinstance(v, dict) else np.array(v, dtype=object)

    # add per-trial reward event fields if computed
    for k, v in per_trial_reward.items():
        save_dict[k] = v

    np.savez(out_path, **save_dict)

    # Print summary
    print(f"[OK] trials saved -> {out_path}")
    print(f"     C_keep={C_keep}, T_keep={T_keep}, N_trials_total={N_trials}")
    for cname in cond_names:
        ntr = int(trial_indices.get(cname, np.array([], dtype=int)).size)
        print(f"     {cname}: n_trials={ntr}")

    if "reward_trace" in meta:
        rt = np.asarray(meta["reward_trace"])
        print(f"     copied reward_trace from stage1: shape={rt.shape}")
    if per_trial_reward:
        n_have = int(np.isfinite(per_trial_reward["reward_first_rel_go_sec"]).sum())
        print(f"     per-trial reward events computed: n_trials_with_reward_events={n_have}")

    return out_path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--trial-pkl", type=str, required=True)
    ap.add_argument("--stage1-npz", type=str, default=None,
                    help="If not provided, will infer from --animal/--session-id/--plane under --stage1-root.")
    ap.add_argument("--stage1-root", type=str, default="/home/jingyi.xu/ALM/results/stage1")
    ap.add_argument("--animal", type=str, default=None)
    ap.add_argument("--session-id", type=str, default=None)
    ap.add_argument("--plane", type=int, default=None)

    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-time", type=int, default=None, help="truncate time axis (frames)")
    ap.add_argument("--smooth-ms", type=float, default=0.0, help="optional moving-average smoothing on dF/F (ms)")

    # per-trial reward events (optional)
    ap.add_argument("--bpod-npy", type=str, default=None)
    ap.add_argument("--licks-npy", type=str, default=None)
    ap.add_argument("--swap-p1-p2", action="store_true",
                    help="Force swap of lick ports when computing per-trial reward events (training still uses stage1 reward_trace).")

    args = ap.parse_args()

    stage1_npz = args.stage1_npz
    if stage1_npz is None:
        if args.animal is None or args.session_id is None or args.plane is None:
            raise ValueError("Provide --stage1-npz, or provide --animal --session-id --plane to infer.")
        stage1_npz = os.path.join(args.stage1_root, args.animal, f"psth_{args.session_id}.{int(args.plane)}.npz")

    export_trials_from_stage1(
        stage1_npz=stage1_npz,
        trial_pkl=args.trial_pkl,
        out_path=args.out,
        force=bool(args.force),
        max_time=args.max_time,
        smooth_ms=float(args.smooth_ms),
        bpod_npy=args.bpod_npy,
        licks_npy=args.licks_npy,
        swap_p1_p2=bool(args.swap_p1_p2),
    )


if __name__ == "__main__":
    main()
