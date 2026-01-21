#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch version: scan and update lick/reward traces for ALL stage1 PSTH npz of a given animal.

- Reuses the same core logic as build_lick_reward_trace.py:
  * prefer Tim's fixed *.licks.npy
  * robust align stage1 all_data columns -> bpod output columns
  * infer left/right from trialtype like 0.average.py
  * optional swap_p1_p2 safety switch

- Adds:
  * --animal/--manifest/--stage1_root batch mode
  * --auto_swap: evaluate both swap and no-swap consistency and auto-pick
  * --skip_if_has_reward / --force

python /home/jingyi.xu/code_rnn/current_rnn/build_lick_reward_trace.py \
  --animal kd95 \
  --manifest /home/jingyi.xu/ALM/meta/manifest.json \
  --stage1_root /home/jingyi.xu/ALM/results/stage1 \
  --only_correct \
  --lick_smooth_ms 200 \
  --reward_mode correctport \
  --auto_swap \
  --inplace

"""

import argparse
import os
import re
import json
import glob
import shutil
import numpy as np
from typing import Dict, Tuple, Optional, List, Any


# -----------------------------
# Bpod key maps (must match bpod_parse.py)
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


# -----------------------------
# Same logic as 0.average.py: prelim_events_and_trials
# -----------------------------
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


def prelim_events_and_trials(all_data: np.ndarray, fps: float):
    behavior = (all_data[keys("protocol")] == 5)
    freelick = (all_data[keys("protocol")] == 6)
    full_sample = (all_data[keys("earlysample")] == 0)
    full_delay = (all_data[keys("earlydelay")] == 0)
    trials_ok = behavior & full_sample & full_delay

    o = all_data[keys("outcome")]
    tt = all_data[keys("trialtype")]

    # keep same as your pipeline: correct + droppednotlick
    CORR = np.isin(o, [outcomes("correct"), outcomes("droppednotlick")])
    INC = (o == outcomes("incorrect"))
    IG = (o == outcomes("ignore"))
    FT = trials_ok | IG

    ss_sec = np.median(all_data[keys("samplestart")][FT])
    ld_sec = np.median(all_data[keys("lastdelay")][FT])
    go_sec = np.median(all_data[keys("go")][FT])

    ss = int(round(ss_sec * fps))
    ld = int(round(ld_sec * fps))
    go = int(round(go_sec * fps))

    L, R = _infer_left_right_from_trialtype(tt)

    LC = np.where(trials_ok & L & CORR)[0]
    RC = np.where(trials_ok & R & CORR)[0]
    LI = np.where(trials_ok & L & INC)[0]
    RI = np.where(trials_ok & R & INC)[0]

    # freelick subsets (kept for completeness)
    FL = np.where(full_sample & freelick)[0]
    FLt = tt[FL]
    FL_type = all_data[keys("freelicktype")][FL]
    FL_L = FL[(FLt == 0) & (FL_type == 1)]
    FL_R = FL[(FLt == 1) & (FL_type == 1)]

    licks_needed = all_data[keys("licksneeded")]
    puff = all_data[keys("puff")]

    LCo = LC[licks_needed[LC] == 1]
    LCm = LC[licks_needed[LC] > 1]
    RCo = RC[licks_needed[RC] == 1]
    RCm = RC[licks_needed[RC] > 1]

    LIp = LI[puff[LI] == 1]
    LIn = LI[puff[LI] == 0]
    RIp = RI[puff[RI] == 1]
    RIn = RI[puff[RI] == 0]

    agg_one = np.concatenate((LCo, RCo)) if (LCo.size + RCo.size) else np.array([], int)
    agg_multi = np.concatenate((LCm, RCm)) if (LCm.size + RCm.size) else np.array([], int)
    agg_puff = np.concatenate((LIp, RIp)) if (LIp.size + RIp.size) else np.array([], int)
    agg_nopf = np.concatenate((LIn, RIn)) if (LIn.size + RIn.size) else np.array([], int)

    cond_idx = {
        "left_correct": LC,
        "right_correct": RC,
        "left_incorrect": LI,
        "right_incorrect": RI,
        "free_left": FL_L,
        "free_right": FL_R,
        "correct_one_lick": agg_one,
        "correct_multi_licks": agg_multi,
        "incorrect_puff": agg_puff,
        "incorrect_no_puff": agg_nopf,
    }
    cond_idx = {k: v for k, v in cond_idx.items() if v.size > 0}
    return (ss, ld, go), cond_idx, FT, go_sec


# -----------------------------
# Robust alignment: stage1 all_data columns -> bpod output columns
# -----------------------------
def _cols_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    nan_both = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=0.0, atol=atol)
    return bool(np.all(nan_both | close))


def _align_stage1_trials_to_bpod(stage1_all_data: np.ndarray, bpod_output: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """
    stage1_all_data: [21, N2p]
    bpod_output:     [21, Ntotal]
    Return idx_2p_in_bpod: [N2p,]
    """
    if stage1_all_data.ndim != 2 or stage1_all_data.shape[0] != 21:
        raise ValueError(f"stage1 all_data must be [21, N2p], got {stage1_all_data.shape}")
    if bpod_output.ndim != 2 or bpod_output.shape[0] != 21:
        raise ValueError(f"bpod output must be [21, Ntotal], got {bpod_output.shape}")

    N2p = stage1_all_data.shape[1]
    Ntotal = bpod_output.shape[1]

    idx = []
    start = 0
    for j in range(N2p):
        target = stage1_all_data[:, j]
        found = False

        # ordered subsequence search (fast)
        for i in range(start, Ntotal):
            if _cols_equal(bpod_output[:, i], target, atol=atol):
                idx.append(i)
                start = i + 1
                found = True
                break

        # fallback: global search
        if not found:
            for i in range(Ntotal):
                if _cols_equal(bpod_output[:, i], target, atol=atol):
                    idx.append(i)
                    start = i + 1
                    found = True
                    break

        if not found:
            raise ValueError(
                f"Failed to align stage1 trial {j}/{N2p}. "
                f"Try increasing --align_atol (e.g. 1e-4) or verify stage1/bpod are same session."
            )

    return np.asarray(idx, dtype=int)


# -----------------------------
# Load stage1 npz
# -----------------------------
def _unwrap_obj(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        return x.item()
    return x


def _load_stage1_meta(stage1_npz_path: str):
    z = np.load(stage1_npz_path, allow_pickle=True)

    cond_names = list(np.asarray(z["cond_names"]).tolist())
    fps = float(z["fps"])

    event_frames = _unwrap_obj(z["event_frames"])
    if not isinstance(event_frames, dict):
        raise ValueError(f"event_frames should be a dict, got {type(event_frames)}")

    all_data = np.asarray(z["all_data"])
    if all_data.ndim != 2 or all_data.shape[0] != 21:
        raise ValueError(f"Expected all_data to be [21, Ntrials_2p], got {all_data.shape}")

    cell_psth = _unwrap_obj(z["cell_psth"])
    if not isinstance(cell_psth, dict) or len(cell_psth) == 0:
        raise ValueError("cell_psth missing or empty in stage1 npz.")

    any_cond = cond_names[0]
    M = np.asarray(cell_psth[any_cond])
    if M.ndim != 2:
        raise ValueError(f"cell_psth['{any_cond}'] must be (cells,time), got {M.shape}")
    n_cells = int(M.shape[0])
    T = int(M.shape[1])

    R_frame = int(event_frames["R"]) if "R" in event_frames else None

    return z, cond_names, fps, event_frames, R_frame, T, n_cells, all_data


# -----------------------------
# Load bpod output + fixed licks
# -----------------------------
def _load_bpod_output(bpod_npy_path: str) -> Tuple[np.ndarray, int]:
    b = np.load(bpod_npy_path, allow_pickle=True)
    bpod_output = np.asarray(b[0])
    if bpod_output.ndim != 2 or bpod_output.shape[0] != 21:
        raise ValueError(f"Expected bpod_output to be [21, N], got {bpod_output.shape}")
    n_trials_total = int(bpod_output.shape[1])
    return bpod_output, n_trials_total


def _as_list(obj: Any) -> List:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return obj
    return list(obj)


def _is_lick_list_candidate(x: Any, n_trials_total: int) -> bool:
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = x.tolist()
    if not isinstance(x, list):
        return False
    if len(x) != n_trials_total:
        return False

    for el in x[:10]:
        if el is None:
            continue
        if isinstance(el, (list, np.ndarray)):
            return True
        return False
    return True


def _load_fixed_licks_from_licks_npy(licks_npy_path: str, n_trials_total: int) -> Tuple[List, List]:
    l = np.load(licks_npy_path, allow_pickle=True)

    if isinstance(l, np.ndarray) and l.ndim == 1 and l.dtype == object:
        items = l.tolist()
    elif isinstance(l, (list, tuple)):
        items = list(l)
    else:
        items = _as_list(l)

    candidates = []
    for i, it in enumerate(items):
        if _is_lick_list_candidate(it, n_trials_total):
            candidates.append((i, _as_list(it)))

    if len(candidates) < 2:
        raise ValueError(
            f"Cannot find two lick-list arrays of length {n_trials_total} in {licks_npy_path}. "
            f"Found {len(candidates)} candidates. Please inspect the file structure."
        )

    p1 = candidates[0][1]
    p2 = candidates[1][1]
    return p1, p2


def _load_licks(bpod_npy_path: str, licks_npy_path: Optional[str], n_trials_total: int) -> Tuple[List, List, str]:
    if licks_npy_path is not None and os.path.exists(licks_npy_path):
        p1, p2 = _load_fixed_licks_from_licks_npy(licks_npy_path, n_trials_total)
        return p1, p2, f"licks_npy:{licks_npy_path}"

    b = np.load(bpod_npy_path, allow_pickle=True)
    try:
        p1 = _as_list(b[3])
        p2 = _as_list(b[4])
    except Exception as e:
        raise ValueError(f"Failed to load legacy p1/p2 from bpod.npy {bpod_npy_path}: {e}")

    if len(p1) != n_trials_total or len(p2) != n_trials_total:
        raise ValueError(
            f"Legacy bpod.npy p1/p2 length mismatch: len(p1)={len(p1)}, len(p2)={len(p2)}, "
            f"expected n_trials_total={n_trials_total}."
        )

    return p1, p2, "bpod_npy_legacy_p1p2"


# -----------------------------
# Build lick/reward traces
# -----------------------------
def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(int(win), dtype=float) / float(win)
    return np.convolve(np.asarray(x, float), k, mode="same")


def _is_keep_cond_only_correct(cond_name: str) -> bool:
    name = cond_name.lower()
    if "incorrect" in name:
        return False
    if name.startswith("free"):
        return False
    return ("correct" in name)


def build_lick_and_reward(
    cond_names: List[str],
    cond_idx: Dict[str, np.ndarray],
    all_data: np.ndarray,
    p1_licks_2p: List,
    p2_licks_2p: List,
    fps: float,
    R_frame: int,
    T: int,
    only_correct: bool,
    lick_smooth_ms: float,
    reward_mode: str,
    swap_p1_p2: bool = False,
):
    """
    Convention:
      - p1_licks_2p: left lickport (after Tim's fix) by default
      - p2_licks_2p: right lickport (after Tim's fix) by default
    If needed, set swap_p1_p2 to flip.
    """
    if swap_p1_p2:
        p1_licks_2p, p2_licks_2p = p2_licks_2p, p1_licks_2p

    C = len(cond_names)
    dt = 1.0 / float(fps)
    go_each = np.asarray(all_data[keys("go")], float)

    tt = np.asarray(all_data[keys("trialtype")], float)
    is_left, is_right = _infer_left_right_from_trialtype(tt)

    lick_left = np.zeros((C, T), dtype=np.float32)
    lick_right = np.zeros((C, T), dtype=np.float32)
    reward = np.zeros((C, T), dtype=np.float32)

    win = int(round((lick_smooth_ms / 1000.0) * fps))
    win = max(1, win)

    for ci, cname in enumerate(cond_names):
        trials = cond_idx.get(cname, np.array([], dtype=int))

        if only_correct and (not _is_keep_cond_only_correct(cname)):
            continue
        if trials.size == 0:
            continue

        cnt_L = np.zeros(T, dtype=np.float64)
        cnt_R = np.zeros(T, dtype=np.float64)
        cnt_reward = np.zeros(T, dtype=np.float64)

        n_used = 0
        for tr in trials:
            g = go_each[tr]
            if not np.isfinite(g):
                continue

            pL = np.asarray(p1_licks_2p[tr], float) if p1_licks_2p[tr] is not None else np.array([], float)
            pR = np.asarray(p2_licks_2p[tr], float) if p2_licks_2p[tr] is not None else np.array([], float)

            if pL.size > 0:
                fr = R_frame + np.round((pL - g) * fps).astype(int)
                fr = fr[(fr >= 0) & (fr < T)]
                for f in fr:
                    cnt_L[f] += 1.0

            if pR.size > 0:
                fr = R_frame + np.round((pR - g) * fps).astype(int)
                fr = fr[(fr >= 0) & (fr < T)]
                for f in fr:
                    cnt_R[f] += 1.0

            if reward_mode == "correctport":
                if bool(is_left[tr]):
                    src = pL
                elif bool(is_right[tr]):
                    src = pR
                else:
                    src = np.array([], float)

                if src.size > 0:
                    fr = R_frame + np.round((src - g) * fps).astype(int)
                    fr = fr[(fr >= 0) & (fr < T)]
                    for f in fr:
                        cnt_reward[f] += 1.0

            n_used += 1

        if n_used <= 0:
            continue

        rate_L = cnt_L / (n_used * dt)
        rate_R = cnt_R / (n_used * dt)
        rate_reward = cnt_reward / (n_used * dt)

        rate_L = _smooth_1d(rate_L, win)
        rate_R = _smooth_1d(rate_R, win)
        rate_reward = _smooth_1d(rate_reward, win)

        lick_left[ci, :] = rate_L.astype(np.float32)
        lick_right[ci, :] = rate_R.astype(np.float32)
        reward[ci, :] = rate_reward.astype(np.float32)

    lick_total = lick_left + lick_right
    return lick_left, lick_right, lick_total, reward


# -----------------------------
# Save updated npz (optionally inplace)
# -----------------------------
def _save_npz_updated(stage1_npz_path: str, z: np.lib.npyio.NpzFile, extra_fields: Dict[str, Any], inplace: bool):
    base = {k: z[k] for k in z.files}
    base.update(extra_fields)

    if inplace:
        bak = stage1_npz_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(stage1_npz_path, bak)

        tmp = stage1_npz_path + ".tmp.npz"
        np.savez_compressed(tmp, **base)
        os.replace(tmp, stage1_npz_path)
        return stage1_npz_path, bak

    out = stage1_npz_path.replace(".npz", "_lick_reward.npz")
    np.savez_compressed(out, **base)
    return out, None


# -----------------------------
# Batch helpers
# -----------------------------
def _parse_session_id_from_stage1_npz(path: str) -> Optional[str]:
    """
    Expect stage1 naming like: psth_YYYYMMDD_HHMMSS.0.npz  (or variants)
    """
    base = os.path.basename(path)
    m = re.search(r"psth_(\d{8}_\d{6})", base)
    if m:
        return m.group(1)
    return None


def _build_bpod_map_from_manifest(manifest_path: str, animal: str) -> Dict[str, str]:
    """
    Return: session_id -> bpod.npy path
    Uses shared_files['bpod'] if exists; otherwise files['bpod'][0] if present.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    for a in man.get("animals", []):
        if a.get("animal") != animal:
            continue
        out: Dict[str, str] = {}
        for s in a.get("sessions", []):
            sid = s.get("session_id")
            if not sid:
                continue
            shared = s.get("shared_files", {}) or {}
            files = s.get("files", {}) or {}

            bpod = shared.get("bpod", None)
            if (not bpod) and ("bpod" in files) and isinstance(files["bpod"], list) and len(files["bpod"]) > 0:
                bpod = files["bpod"][0]

            if bpod and os.path.exists(bpod):
                # multiple planes may repeat; keep first valid
                out.setdefault(sid, bpod)
        return out

    raise ValueError(f"Animal {animal} not found in manifest: {manifest_path}")


def _cond_index(cond_names: List[str], pattern: str) -> Optional[int]:
    pat = pattern.lower()
    for i, n in enumerate(cond_names):
        if pat in n.lower():
            return i
    return None


def _swap_goodness(lick_L: np.ndarray, lick_R: np.ndarray, cond_names: List[str], fps: float, R_frame: int) -> Optional[float]:
    """
    Higher is better:
      left_correct: lick_L > lick_R
      right_correct: lick_R > lick_L
    Evaluate over [0, 0.5] sec after go (relative).
    """
    iLC = _cond_index(cond_names, "left_correct")
    iRC = _cond_index(cond_names, "right_correct")
    if iLC is None or iRC is None:
        return None

    w0 = R_frame
    w1 = min(lick_L.shape[1], R_frame + int(round(0.5 * fps)))
    if w1 <= w0:
        return None

    lc = float(np.sum(lick_L[iLC, w0:w1]) - np.sum(lick_R[iLC, w0:w1]))
    rc = float(np.sum(lick_R[iRC, w0:w1]) - np.sum(lick_L[iRC, w0:w1]))
    return lc + rc


def _process_one(
    stage1_npz: str,
    bpod_npy: str,
    licks_npy: Optional[str],
    only_correct: bool,
    lick_smooth_ms: float,
    reward_mode: str,
    inplace: bool,
    align_atol: float,
    swap_p1_p2: bool,
    auto_swap: bool,
    verbose: bool = True,
) -> Tuple[bool, str]:
    """
    Return (ok, message). Writes to npz if ok.
    """
    z, cond_names, fps, event_frames, R_frame, T, n_cells, stage1_all_data = _load_stage1_meta(stage1_npz)
    (S, D, R_from_all_data), cond_idx, FT_mask, go_sec_median = prelim_events_and_trials(stage1_all_data, fps)
    if R_frame is None:
        R_frame = int(R_from_all_data)

    bpod_output, n_trials_total = _load_bpod_output(bpod_npy)

    # licks
    if licks_npy is None:
        if bpod_npy.endswith(".bpod.npy"):
            licks_npy = bpod_npy.replace(".bpod.npy", ".licks.npy")
        else:
            root, _ = os.path.splitext(bpod_npy)
            licks_npy = root + ".licks.npy"

    p1_licks_all, p2_licks_all, licks_source = _load_licks(bpod_npy, licks_npy, n_trials_total)

    # align stage1 trials into bpod indexing
    idx_2p_in_bpod = _align_stage1_trials_to_bpod(stage1_all_data, bpod_output, atol=align_atol)
    p1_licks_2p = [p1_licks_all[i] for i in idx_2p_in_bpod]
    p2_licks_2p = [p2_licks_all[i] for i in idx_2p_in_bpod]

    # decide swap
    chosen_swap = bool(swap_p1_p2)
    auto_note = ""
    if auto_swap:
        # compute both and pick the better one (if score computable)
        L0, R0, T0, rew0 = build_lick_and_reward(
            cond_names, cond_idx, stage1_all_data, p1_licks_2p, p2_licks_2p,
            fps=fps, R_frame=R_frame, T=T,
            only_correct=only_correct, lick_smooth_ms=lick_smooth_ms,
            reward_mode=reward_mode, swap_p1_p2=False
        )
        g0 = _swap_goodness(L0, R0, cond_names, fps=fps, R_frame=R_frame)

        L1, R1, T1, rew1 = build_lick_and_reward(
            cond_names, cond_idx, stage1_all_data, p1_licks_2p, p2_licks_2p,
            fps=fps, R_frame=R_frame, T=T,
            only_correct=only_correct, lick_smooth_ms=lick_smooth_ms,
            reward_mode=reward_mode, swap_p1_p2=True
        )
        g1 = _swap_goodness(L1, R1, cond_names, fps=fps, R_frame=R_frame)

        if (g0 is not None) and (g1 is not None):
            chosen_swap = bool(g1 > g0)
            auto_note = f" auto_swap goodness: no-swap={g0:.3f}, swap={g1:.3f}, choose_swap={chosen_swap}"
        else:
            auto_note = " auto_swap skipped (missing left_correct/right_correct in cond_names)."

    # build final traces with chosen swap
    lick_L, lick_R, lick_tot, reward = build_lick_and_reward(
        cond_names=cond_names,
        cond_idx=cond_idx,
        all_data=stage1_all_data,
        p1_licks_2p=p1_licks_2p,
        p2_licks_2p=p2_licks_2p,
        fps=fps,
        R_frame=R_frame,
        T=T,
        only_correct=only_correct,
        lick_smooth_ms=lick_smooth_ms,
        reward_mode=reward_mode,
        swap_p1_p2=chosen_swap,
    )

    t_rel_go_sec = (np.arange(T) - int(R_frame)) / float(fps)

    extra = {
        "lick_rate_left": lick_L,
        "lick_rate_right": lick_R,
        "lick_rate_total": lick_tot,
        "reward_trace": reward,
        "t_rel_go_sec": t_rel_go_sec.astype(np.float32),
        "idx_2p_in_bpod": idx_2p_in_bpod.astype(np.int32),
        "lick_smooth_ms": float(lick_smooth_ms),
        "reward_mode": str(reward_mode),
        "only_correct_lickreward": bool(only_correct),
        "align_atol_used": float(align_atol),
        "licks_source": np.array([licks_source], dtype=object),
        "swap_p1_p2_used": bool(chosen_swap),
        "licks_npy_used": np.array([licks_npy], dtype=object),
        "auto_swap_used": bool(auto_swap),
    }

    out_path, bak = _save_npz_updated(stage1_npz, z, extra, inplace=inplace)
    msg = f"[OK] {os.path.basename(stage1_npz)} -> updated (swap={chosen_swap})."
    if bak:
        msg += f" backup={os.path.basename(bak)}."
    if verbose:
        msg += auto_note
    return True, msg


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # single mode (compatible with your current script)
    ap.add_argument("--stage1_npz", type=str, default=None)
    ap.add_argument("--bpod_npy", type=str, default=None)
    ap.add_argument("--licks_npy", type=str, default=None)

    # batch mode
    ap.add_argument("--animal", type=str, default=None, help="e.g., kd95")
    ap.add_argument("--manifest", type=str, default=os.path.expanduser("~/ALM/meta/manifest.json"))
    ap.add_argument("--stage1_root", type=str, default=os.path.expanduser("~/ALM/results/stage1"),
                    help="Root dir containing <animal>/psth_*.npz")
    ap.add_argument("--glob", type=str, default="psth_*.npz", help="Pattern within stage1 animal dir")

    # behavior
    ap.add_argument("--swap_p1_p2", action="store_true",
                    help="Force swap (manual). If --auto_swap is set, auto decision will override.")
    ap.add_argument("--auto_swap", action="store_true",
                    help="Try both swap/no-swap and pick the more consistent one per session.")
    ap.add_argument("--only_correct", action="store_true")
    ap.add_argument("--lick_smooth_ms", type=float, default=200.0)
    ap.add_argument("--reward_mode", type=str, default="correctport", choices=["correctport"])
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--align_atol", type=float, default=1e-6)

    # batch controls
    ap.add_argument("--skip_if_has_reward", action="store_true",
                    help="If npz already has 'reward_trace', skip unless --force.")
    ap.add_argument("--force", action="store_true", help="Force update even if reward_trace exists.")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be done, no writing.")
    args = ap.parse_args()

    # decide mode
    is_batch = (args.animal is not None)
    is_single = (args.stage1_npz is not None and args.bpod_npy is not None)

    if (not is_batch) and (not is_single):
        raise ValueError("Either provide --stage1_npz and --bpod_npy (single mode), or provide --animal (batch mode).")

    if is_single:
        if args.dry_run:
            print(f"[DRY] would process: stage1={args.stage1_npz} bpod={args.bpod_npy}")
            return
        ok, msg = _process_one(
            stage1_npz=args.stage1_npz,
            bpod_npy=args.bpod_npy,
            licks_npy=args.licks_npy,
            only_correct=args.only_correct,
            lick_smooth_ms=args.lick_smooth_ms,
            reward_mode=args.reward_mode,
            inplace=args.inplace,
            align_atol=args.align_atol,
            swap_p1_p2=args.swap_p1_p2,
            auto_swap=args.auto_swap,
            verbose=True,
        )
        print(msg)
        return

    # batch mode
    animal = args.animal
    stage1_dir = os.path.join(args.stage1_root, animal)
    if not os.path.isdir(stage1_dir):
        raise FileNotFoundError(f"stage1 dir not found: {stage1_dir}")

    bpod_map = _build_bpod_map_from_manifest(args.manifest, animal)

    npz_list = sorted(glob.glob(os.path.join(stage1_dir, args.glob)))
    npz_list = [p for p in npz_list if (not p.endswith(".bak")) and (".tmp." not in p)]
    print(f"[INFO] batch: animal={animal} stage1_dir={stage1_dir} npz={len(npz_list)} manifest_bpod={len(bpod_map)}")

    n_ok, n_skip, n_fail = 0, 0, 0
    for npz_path in npz_list:
        sid = _parse_session_id_from_stage1_npz(npz_path)
        if sid is None:
            n_skip += 1
            print(f"[SKIP] cannot parse session_id from: {os.path.basename(npz_path)}")
            continue
        if sid not in bpod_map:
            n_skip += 1
            print(f"[SKIP] no bpod in manifest for sid={sid} npz={os.path.basename(npz_path)}")
            continue

        # skip if already has reward_trace
        if args.skip_if_has_reward and (not args.force):
            try:
                z0 = np.load(npz_path, allow_pickle=True)
                if "reward_trace" in z0.files:
                    n_skip += 1
                    print(f"[SKIP] reward_trace exists (use --force): {os.path.basename(npz_path)}")
                    continue
            except Exception:
                pass

        bpod = bpod_map[sid]
        licks = (bpod.replace(".bpod.npy", ".licks.npy") if bpod.endswith(".bpod.npy") else None)

        if args.dry_run:
            print(f"[DRY] sid={sid} npz={os.path.basename(npz_path)} bpod={bpod} licks={licks}")
            continue

        try:
            ok, msg = _process_one(
                stage1_npz=npz_path,
                bpod_npy=bpod,
                licks_npy=licks,
                only_correct=args.only_correct,
                lick_smooth_ms=args.lick_smooth_ms,
                reward_mode=args.reward_mode,
                inplace=args.inplace,
                align_atol=args.align_atol,
                swap_p1_p2=args.swap_p1_p2,
                auto_swap=args.auto_swap,
                verbose=True,
            )
            n_ok += 1
            print(msg)
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] sid={sid} npz={os.path.basename(npz_path)} err={repr(e)}")

    print(f"[DONE] ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
