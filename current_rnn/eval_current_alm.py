# current_rnn/eval_current_alm.py
# Supports:
#   (A) single-session eval: --npz <stage1_npz>
#   (B) batch eval across registry: --eval_all --registry_dir <dir> --animal <id>
#
# Plots:
#   - One mosaic figure containing all selected neurons (each neuron is one subplot),
#     and time axis restricted to the SAME time_mask used in training (sample+delay+resp).
#   - In batch mode, neurons are sampled uniformly from all sessions' inhibitory units.
'''
python eval_current_alm.py --eval_all --registry_dir /home/jingyi.xu/ALM/results/registry/kd95 --animal kd95 --n_exc_virtual 800 --model /home/jingyi.xu/code_rnn/results_current/20251230/rnn_current_kd95_global_nobs200_nexc800_ntotal1000.pt --psth_bin_ms 200 --sample_ignore_ms 50 --resp_sec 2.0 --out_dir /home/jingyi.xu/code_rnn/results_global/eval_kd95

python -u eval_current_alm.py \
  --eval_all \
  --registry_dir /home/jingyi.xu/ALM/results/registry/kd95 \
  --animal kd95 \
  --model /home/jingyi.xu/code_rnn/results_current/smoke_celltype_20260121/rnn_current_kd95_global_nobs200_nexc800_ntotal1000.best.pt \
  --n_exc_virtual 800 \
  --noise_std 0 \
  --plot_bin_ms 200

'''
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as tch
import matplotlib.pyplot as plt
import heapq


from data_alm_current import load_alm_psth_npz  # returns psth [C,T,N] + meta dict
from model_current import ALMCurrentRNN
from training_current import (
    _load_default_parameters,
    _maybe_attach_lick_reward_to_meta,
    _build_input_tensor,
    _time_bin_smooth_ctn,
    _build_time_mask_sample_delay_resp,
)

# -----------------------------
# Plot utilities
# -----------------------------
import numpy as np
from typing import Any, Dict, Tuple, Optional

def compute_time_mask_tsec_zero_at_R(
    meta: Dict[str, Any],
    T: int,
    sample_ignore_ms: float,
    resp_sec: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
      mask_np: [T] bool, same loss window mask as training.
      t_sec:   [Tm] float, seconds for masked indices, with R(go cue) as t=0.
      ev_sec:  dict of event times in seconds relative to R, e.g. {"S":..., "D":..., "R":0.0}
    """
    fps = float(meta["fps"])

    # same mask builder as training (you already use it in eval)
    m = _build_time_mask_sample_delay_resp(
        T=T,
        fps=fps,
        meta=meta,
        sample_ignore_ms=float(sample_ignore_ms),
        resp_sec=float(resp_sec),
    )
    idx = np.where(m)[0]  # indices kept by mask

    ev = meta.get("event_frames", {}) or {}
    # robustly fetch event frames
    def _get_ev_frame(name: str, default: int) -> int:
        if isinstance(ev, dict) and (name in ev):
            return int(ev[name])
        return int(default)

    S = _get_ev_frame("S", 0)
    D = _get_ev_frame("D", S)
    R = _get_ev_frame("R", D)  # go cue

    # time axis in seconds for masked indices, zeroed at R
    t_sec = (idx - R) / fps

    # event lines in seconds relative to R
    ev_sec = {
        "S": (S - R) / fps,
        "D": (D - R) / fps,
        "R": 0.0,
    }
    return m, t_sec, ev_sec

import os
import math
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Dict

def plot_psth_comparison_R0_with_events(
    t_sec: np.ndarray,
    psth_used: tch.Tensor,          # [C, Tm, N]
    rates_used: tch.Tensor,         # [C, Tm, N]
    idx_neurons: Sequence[int],
    mse_per_neuron: Optional[np.ndarray],
    cond_names: Sequence[str],
    out_path: str,
    ncols: int = 6,
    title: Optional[str] = None,
    event_sec: Optional[Dict[str, float]] = None,  # {"S":..., "D":..., "R":0.0}
    show_event_labels: bool = True,
    event_linewidth: float = 0.8,
) -> None:
    """
    Mosaic plot:
      - x-axis: seconds, with R(go cue) as t=0 (caller must pass t_sec computed that way).
      - draws vertical event lines for S/D/R if event_sec is provided
      - removes y=0 horizontal reference line (no axhline)
      - GT solid, Fit dashed, same color per condition (as in your current function)
    """
    idx_neurons = list(map(int, idx_neurons))
    n_sel = len(idx_neurons)
    if n_sel == 0:
        raise ValueError("idx_neurons is empty.")

    C = int(psth_used.shape[0])
    if C < 2:
        raise ValueError(f"Expected at least 2 conditions (LC/RC). Got C={C}.")

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_sel / ncols))

    fig_w = 3.2 * ncols
    fig_h = 2.6 * nrows
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h),
        squeeze=False, sharex=False, sharey=False
    )

    cond0 = str(cond_names[0]) if len(cond_names) > 0 else "cond0"
    cond1 = str(cond_names[1]) if len(cond_names) > 1 else "cond1"

    for i, n in enumerate(idx_neurons):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        gt0 = psth_used[0, :, n].detach().cpu().numpy()
        pr0 = rates_used[0, :, n].detach().cpu().numpy()
        gt1 = psth_used[1, :, n].detach().cpu().numpy()
        pr1 = rates_used[1, :, n].detach().cpu().numpy()

        # Same-condition same-color: draw GT first, reuse its color for Fit
        l0, = ax.plot(t_sec, gt0, label=f"{cond0} GT")
        ax.plot(t_sec, pr0, linestyle="--", color=l0.get_color(), label=f"{cond0} Fit")

        l1, = ax.plot(t_sec, gt1, label=f"{cond1} GT")
        ax.plot(t_sec, pr1, linestyle="--", color=l1.get_color(), label=f"{cond1} Fit")

        # --- event lines (S/D/R), with R as 0 ---
        if event_sec is not None:
            # draw R line (t=0) emphasized slightly
            if "R" in event_sec:
                ax.axvline(float(event_sec["R"]), linestyle="-", linewidth=event_linewidth)
            if "S" in event_sec:
                ax.axvline(float(event_sec["S"]), linestyle=":", linewidth=event_linewidth)
            if "D" in event_sec:
                ax.axvline(float(event_sec["D"]), linestyle="--", linewidth=event_linewidth)

            if show_event_labels:
                # put small labels near top-left region inside axes
                y0, y1 = ax.get_ylim()
                y_text = y1 - 0.05 * (y1 - y0)
                # only label within visible x-range
                xmin, xmax = ax.get_xlim()
                for key in ["S", "D", "R"]:
                    if key in event_sec:
                        x = float(event_sec[key])
                        if xmin <= x <= xmax:
                            ax.text(x, y_text, key, fontsize=7, ha="center", va="top")

        # title
        if mse_per_neuron is not None:
            ax.set_title(f"n={n}  mse={float(mse_per_neuron[n]):.2e}", fontsize=9)
        else:
            ax.set_title(f"n={n}", fontsize=9)

        # IMPORTANT: remove y=0 line (do NOT draw axhline)
        # ax.axhline(0.0, linewidth=0.5)  # removed

        if i == 0:
            ax.legend(fontsize=8, loc="best")

    # hide unused axes
    for j in range(n_sel, nrows * ncols):
        rr = j // ncols
        cc = j % ncols
        axes[rr][cc].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _maybe_smooth_trials_like_psth(
    x: tch.Tensor,
    fps: float,
    trials_bin_ms: Optional[float],
) -> tch.Tensor:
    """Optionally smooth a trial tensor shaped [B, T, K] using the same boxcar smoother as training.

    We reuse _time_bin_smooth_ctn which expects [C,T,*]. Here B plays the role of C.
    """
    if trials_bin_ms is None:
        return x
    v = float(trials_bin_ms)
    if v <= 0:
        return x
    return _time_bin_smooth_ctn(x, fps=float(fps), bin_ms=v)


def plot_psth_comparison(
    t_sec: np.ndarray,
    psth_used: tch.Tensor,          # [C, Tm, Nsel]
    rates_used: tch.Tensor,         # [C, Tm, Nsel]
    idx_neurons: Sequence[int],
    mse_per_neuron: Optional[np.ndarray],
    cond_names: Sequence[str],
    out_path: str,
    ncols: int = 6,
    title: Optional[str] = None,
) -> None:
    """
    A single large figure that tiles all selected neurons into a grid.
    Each subplot contains:
      - LC ground-truth vs prediction
      - RC ground-truth vs prediction
    Time axis is already restricted to the training time_mask.
    """
    idx_neurons = list(map(int, idx_neurons))
    n_sel = len(idx_neurons)
    if n_sel == 0:
        raise ValueError("idx_neurons is empty.")

    C = int(psth_used.shape[0])
    if C < 2:
        raise ValueError(f"Expected at least 2 conditions (LC/RC). Got C={C}.")

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_sel / ncols))

    fig_w = 3.2 * ncols
    fig_h = 2.6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharex=False, sharey=False)

    cond0 = str(cond_names[0]) if len(cond_names) > 0 else "cond0"
    cond1 = str(cond_names[1]) if len(cond_names) > 1 else "cond1"

    for i, n in enumerate(idx_neurons):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        gt0 = psth_used[0, :, n].detach().cpu().numpy()
        pr0 = rates_used[0, :, n].detach().cpu().numpy()
        gt1 = psth_used[1, :, n].detach().cpu().numpy()
        pr1 = rates_used[1, :, n].detach().cpu().numpy()

        l0, = ax.plot(t_sec, gt0, label=f"{cond0} GT")                  # GT 实线
        ax.plot(t_sec, pr0, linestyle="--", color=l0.get_color(),       # Fit 虚线，同色
        label=f"{cond0} Fit")

        l1, = ax.plot(t_sec, gt1, label=f"{cond1} GT")
        ax.plot(t_sec, pr1, linestyle="--", color=l1.get_color(),
        label=f"{cond1} Fit")


        if mse_per_neuron is not None:
            ax.set_title(f"n={n}  mse={float(mse_per_neuron[n]):.2e}", fontsize=9)
        else:
            ax.set_title(f"n={n}", fontsize=9)

        ax.axhline(0.0, linewidth=0.5)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    # hide unused axes
    for j in range(n_sel, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved mosaic figure -> {out_path}")


def _select_neurons_by_mse(
    psth_used: tch.Tensor,   # [C,Tm,N]
    rates_used: tch.Tensor,  # [C,Tm,N]
    num_neurons: int,
    mode: str,
    rng_seed: int,
) -> Tuple[tch.Tensor, np.ndarray]:
    """
    mode:
      - best:  choose neurons with smallest MSE
      - worst: choose neurons with largest MSE
      - random: random subset
    """
    with tch.no_grad():
        err = (psth_used - rates_used) ** 2
        mse = err.mean(dim=(0, 1)).detach().cpu().numpy()  # [N]

    N = int(mse.shape[0])
    k = min(int(num_neurons), N)
    if k <= 0:
        raise ValueError("num_neurons must be > 0.")

    if mode == "best":
        idx = np.argsort(mse)[:k]
    elif mode == "worst":
        idx = np.argsort(mse)[-k:][::-1]
    elif mode == "random":
        rng = np.random.default_rng(int(rng_seed))
        idx = rng.choice(N, size=k, replace=False)
    else:
        raise ValueError(f"Unknown mode={mode}")

    idx_t = tch.as_tensor(idx, dtype=tch.long)
    return idx_t, mse
    

# -----------------------------
# Celltype utilities + J plotting
# -----------------------------

def _as_str_array(x):
    if x is None:
        return None
    arr = np.asarray(x)
    # handle object arrays / bytes
    if arr.dtype.kind in ('S', 'O'):
        arr = arr.astype(str)
    return arr


def _extract_kept_cell_subclasses(meta: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return subclasses aligned to the *kept* neuron order (same as psth's last dim).

    Stage1 NPZ may store:
      (A) cell_subclasses already for kept neurons, len == len(keep_idx)
      (B) cell_subclasses for all neurons before keep, len == n_cells_before
          and keep_idx lists the kept array indices.

    We make this deterministic and raise if ambiguous.
    """
    keep_idx = meta.get('keep_idx', None)
    subs = meta.get('cell_subclasses', None)
    if subs is None:
        subs = meta.get('cell_types', None)
    if keep_idx is None or subs is None:
        return None

    keep_idx = np.asarray(keep_idx, dtype=int)
    subs_arr = _as_str_array(subs)
    if subs_arr is None:
        return None

    n_keep = int(len(keep_idx))
    n_sub = int(len(subs_arr))

    if n_sub == n_keep:
        return subs_arr

    n_before = meta.get('n_cells_before', None)
    if n_before is not None:
        try:
            n_before = int(n_before)
        except Exception:
            n_before = None

    if (n_before is not None) and (n_sub == n_before):
        # keep_idx are array indices into the pre-keep list
        if keep_idx.max(initial=-1) >= n_sub or keep_idx.min(initial=0) < 0:
            raise ValueError(
                f"keep_idx out of range for cell_subclasses: keep_idx in [{keep_idx.min()}, {keep_idx.max()}], "
                f"but len(cell_subclasses)={n_sub}"
            )
        return subs_arr[keep_idx]

    raise ValueError(
        "Cannot align cell_subclasses to kept neurons: "
        f"len(cell_subclasses)={n_sub}, len(keep_idx)={n_keep}, n_cells_before={meta.get('n_cells_before', None)}. "
        "Expected len(cell_subclasses)==len(keep_idx) OR len(cell_subclasses)==n_cells_before."
    )


def _parse_superclass(label: str) -> str:
    """Heuristic superclass parsing consistent with your plot_corr_new.py style."""
    s = str(label)
    # common separators
    for sep in ['/', ':', '::', '-', '_']:
        if sep in s:
            return s.split(sep)[0]
    return s


def _order_types_hierarchical(types: list[str]) -> list[str]:
    """Order by (superclass, then subclass) for stable, readable block plots."""
    uniq = sorted(set(map(str, types)))
    pairs = [( _parse_superclass(t), t) for t in uniq]
    pairs.sort(key=lambda x: (x[0], x[1]))
    return [t for _, t in pairs]


def _robust_clim(M: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.0) -> tuple[float, float]:
    x = np.asarray(M, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (-1.0, 1.0)
    lo = float(np.percentile(x, lo_q))
    hi = float(np.percentile(x, hi_q))
    if lo == hi:
        eps = 1e-6 if abs(lo) < 1e-6 else abs(lo) * 1e-3
        lo -= eps
        hi += eps
    return (lo, hi)


def _plot_J_real_by_celltype(
    J_full: tch.Tensor,
    n_exc_virtual: int,
    labels_by_g: Dict[int, str],
    out_dir: str,
    model_tag: str,
    title_prefix: str = "J (real neurons)",
) -> None:
    """Plot J restricted to real (observed inhibitory) neurons, grouped by cell_subclasses.

    - J_full: [N_total, N_total]
    - Real neurons are assumed to be indices [n_exc_virtual:]
      (consistent with training_current: idx_net = n_exc_virtual + global_idx)
    - labels_by_g maps global_idx (0..n_obs-1) -> subclass label string

    Saves:
      - J_real_sorted_by_celltype_<model_tag>.png
      - J_real_celltype_mean_<model_tag>.png
      - J_real_celltype_mean_<model_tag>.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    J_np = J_full.detach().cpu().numpy()
    N_total = int(J_np.shape[0])
    n_exc_virtual = int(n_exc_virtual)
    if n_exc_virtual < 0 or n_exc_virtual > N_total:
        raise ValueError(f"Invalid n_exc_virtual={n_exc_virtual} for N_total={N_total}")

    n_obs = N_total - n_exc_virtual
    if n_obs <= 0:
        raise ValueError(f"No real neurons to plot: N_total={N_total}, n_exc_virtual={n_exc_virtual}")

    # real-real block
    J_real = J_np[n_exc_virtual:, n_exc_virtual:]  # [n_obs, n_obs]

    # labels aligned with global_idx
    labels = ["Unknown"] * n_obs
    for g, lab in labels_by_g.items():
        gi = int(g)
        if 0 <= gi < n_obs:
            labels[gi] = str(lab)

    # types and neuron ordering
    types_present = [lab for lab in labels if (lab is not None) and (str(lab).strip() != "")]
    if len(types_present) == 0:
        print("[WARN] No celltype labels found; skip J-by-celltype plots.")
        return

    ordered_types = _order_types_hierarchical(types_present)

    type_to_inds = {}
    for t in ordered_types:
        inds = [i for i, lab in enumerate(labels) if str(lab) == str(t)]
        if len(inds) > 0:
            type_to_inds[str(t)] = inds

    ordered_types = [t for t in ordered_types if str(t) in type_to_inds]
    if len(ordered_types) == 0:
        print("[WARN] No non-empty type groups; skip J-by-celltype plots.")
        return

    neuron_order = []
    boundaries = []  # cumulative end indices
    for t in ordered_types:
        neuron_order.extend(type_to_inds[str(t)])
        boundaries.append(len(neuron_order))

    J_sorted = J_real[np.ix_(neuron_order, neuron_order)]

    # ---- Plot 1: neuron-level J sorted by celltype ----
    lo, hi = _robust_clim(J_sorted, 2.0, 98.0)

    fig = plt.figure(figsize=(10.5, 9.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(J_sorted, aspect='equal', interpolation='nearest', vmin=lo, vmax=hi)
    ax.set_title(f"{title_prefix} sorted by celltype | {model_tag}\nreal={n_obs}, n_exc_virtual={n_exc_virtual}")
    ax.set_xlabel('pre (sorted)')
    ax.set_ylabel('post (sorted)')

    # boundaries + tick labels at block centers
    start = 0
    tick_pos = []
    tick_lab = []
    for t, end in zip(ordered_types, boundaries):
        if end < J_sorted.shape[0]:
            ax.axhline(end - 0.5, linewidth=0.8)
            ax.axvline(end - 0.5, linewidth=0.8)
        tick_pos.append((start + end - 1) / 2.0)
        tick_lab.append(f"{t} (n={len(type_to_inds[str(t)])})")
        start = end

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=90, fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_png1 = os.path.join(out_dir, f"J_real_sorted_by_celltype_{model_tag}.png")
    fig.savefig(out_png1, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved J sorted-by-celltype -> {out_png1}")

    # ---- Plot 2: type-mean aggregated J ----
    n_types = len(ordered_types)
    J_mean = np.full((n_types, n_types), np.nan, dtype=float)
    for i, ti in enumerate(ordered_types):
        ii = type_to_inds[str(ti)]
        for j, tj in enumerate(ordered_types):
            jj = type_to_inds[str(tj)]
            blk = J_real[np.ix_(ii, jj)]
            if blk.size > 0:
                J_mean[i, j] = float(np.mean(blk))

    lo2, hi2 = _robust_clim(J_mean, 2.0, 98.0)
    fig = plt.figure(figsize=(1.2 * n_types + 5.0, 1.1 * n_types + 4.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(J_mean, aspect='equal', interpolation='nearest', vmin=lo2, vmax=hi2)
    ax.set_title(f"{title_prefix} celltype-mean | {model_tag}\nreal={n_obs}, n_exc_virtual={n_exc_virtual}")
    ax.set_xticks(range(n_types))
    ax.set_xticklabels(ordered_types, rotation=90, fontsize=8)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(ordered_types, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_png2 = os.path.join(out_dir, f"J_real_celltype_mean_{model_tag}.png")
    fig.savefig(out_png2, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved J celltype-mean -> {out_png2}")

    # also save csv
    out_csv = os.path.join(out_dir, f"J_real_celltype_mean_{model_tag}.csv")
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['type'] + ordered_types)
        for t, row in zip(ordered_types, J_mean):
            w.writerow([t] + [f"{x:.6g}" if np.isfinite(x) else "" for x in row])
    print(f"[OK] Saved J celltype-mean csv -> {out_csv}")


# -----------------------------
# Registry helpers (batch eval)
# -----------------------------

@dataclass
class RegistryRow:
    unit_key: str
    global_idx: int
    array_idx: int
    npz_path: str

def _read_registry_csv(registry_dir: str, animal: str) -> List[RegistryRow]:
    """
    Expected columns include: unit_key, global_idx, array_idx, npz_path
    (others are ignored).
    """
    cand1 = os.path.join(registry_dir, f"{animal}_registry.csv")
    cand2 = os.path.join(registry_dir, "registry.csv")
    path = cand1 if os.path.exists(cand1) else cand2
    if not os.path.exists(path):
        raise FileNotFoundError(f"Registry CSV not found: {cand1} or {cand2}")

    rows: List[RegistryRow] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for rr in reader:
            rows.append(
                RegistryRow(
                    unit_key=str(rr["unit_key"]),
                    global_idx=int(float(rr["global_idx"])),
                    array_idx=int(float(rr["array_idx"])),
                    npz_path=str(rr["npz_path"]),
                )
            )
    if len(rows) == 0:
        raise ValueError(f"Empty registry: {path}")
    print(f"[INFO] Loaded registry: {path} (rows={len(rows)})")
    return rows


def _build_keepidx_pos_map(keep_idx: np.ndarray) -> Dict[int, int]:
    keep_idx = np.asarray(keep_idx, dtype=int)
    return {int(a): int(i) for i, a in enumerate(keep_idx.tolist())}


def _group_rows_by_unit(rows: List[RegistryRow]) -> Dict[str, List[RegistryRow]]:
    by_unit: Dict[str, List[RegistryRow]] = defaultdict(list)
    for r in rows:
        by_unit[r.unit_key].append(r)
    return by_unit


# -----------------------------
# Eval core
# -----------------------------

def _infer_device(params_path: Optional[str]) -> tch.device:
    default_parameters = _load_default_parameters(params_path)
    device_str = default_parameters.get("device", "cpu")
    device = tch.device(device_str if tch.cuda.is_available() or device_str == "cpu" else "cpu")
    return device


def _build_model_for_eval(
    model_path: str,
    D_in: int,
    N_total: int,
    params_path: Optional[str],
    device: tch.device,
) -> ALMCurrentRNN:
    default_parameters = _load_default_parameters(params_path)
    dt = float(default_parameters.get("dt", 1.0))
    tau = float(default_parameters.get("tau", 1.0))
    substeps = int(default_parameters.get("substeps", 1))

    net = ALMCurrentRNN(
        N=int(N_total),
        D_in=int(D_in),
        dt=dt,
        tau=tau,
        substeps=substeps,
        nonlinearity="tanh",
        device=device,
        dale_mask=None,  # eval 不强制 Dale；即便训练用了，也不需要从 ckpt 里恢复这块
    ).to(device)

    ckpt = tch.load(model_path, map_location=device)

    # 兼容两种保存格式：
    # (A) 直接保存 state_dict
    # (B) 保存 {"model": state_dict, "opt": ..., ...}
    if isinstance(ckpt, dict) and ("model" in ckpt) and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # 清理不应由 checkpoint 驱动的条目
    if isinstance(state_dict, dict) and "dale_mask" in state_dict:
        state_dict = dict(state_dict)  # copy
        state_dict.pop("dale_mask", None)

    # 更稳妥：如果还有其他多余 key，也不应让 eval 崩溃
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys ignored: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    if len(missing) > 0:
        print(f"[WARN] Missing keys (left default): {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    net.eval()
    return net



def _compute_time_mask_and_tsec(
    meta: Dict[str, Any],
    T: int,
    sample_ignore_ms: float,
    resp_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    fps = float(meta["fps"])
    m = _build_time_mask_sample_delay_resp(
        T=T,
        fps=fps,
        meta=meta,
        sample_ignore_ms=float(sample_ignore_ms),
        resp_sec=float(resp_sec),
    )
    idx = np.where(m)[0]
    ev = meta.get("event_frames", {})
    S = int(ev["S"]) if isinstance(ev, dict) and "S" in ev else 0
    t_sec = (idx - S) / fps
    return m, t_sec


def eval_single_session(
    npz_path: str,
    model_path: str,
    params_path: Optional[str],
    noise_std: float,
    psth_bin_ms: float,
    plot_bin_ms: float,
    sample_ignore_ms: float,
    resp_sec: float,
    num_neurons: int,
    mode: str,
    rng_seed: int,
    out_dir: Optional[str],
    device: tch.device,
) -> None:
    default_parameters = _load_default_parameters(params_path)

    psth, meta = load_alm_psth_npz(
        npz_path=npz_path,
        cond_filter=None,
        max_time=None,
        device=device,
        dtype=tch.float32,
    )
    C, T, N = psth.shape
    meta = _maybe_attach_lick_reward_to_meta(meta=meta, npz_path=npz_path, T=T)

    fps = float(meta.get("fps", 1.0))
    if psth_bin_ms is not None and float(psth_bin_ms) > 0:
        psth = _time_bin_smooth_ctn(psth, fps=fps, bin_ms=float(psth_bin_ms))

    cond_names = list(meta["cond_names"])
    amp_input = float(default_parameters.get("amp_input", 1.0))
    include_go_cue = bool(default_parameters.get("include_go_cue", True))
    go_cue_sec = float(default_parameters.get("go_cue_sec", 0.10))
    amp_go_cue = float(default_parameters.get("amp_go_cue", 1.0))
    include_reward = bool(default_parameters.get("include_reward", True))
    amp_reward = float(default_parameters.get("amp_reward", 1.0))

    u = _build_input_tensor(
        C=C,
        T=T,
        cond_names=cond_names,
        device=device,
        amp_input=amp_input,
        meta=meta,
        amp_stim=amp_input,
        sample_len_sec=1.15,
        on_sec=0.15,
        off_sec=0.10,
        n_bursts=5,
        include_go_cue=include_go_cue,
        go_cue_sec=go_cue_sec,
        amp_go_cue=amp_go_cue,
        include_reward=include_reward,
        amp_reward=amp_reward,
    )
    D_in = int(u.shape[-1])

    # NOTE: single-session eval assumes the model's N_total == N in npz
    net = _build_model_for_eval(
        model_path=model_path,
        D_in=D_in,
        N_total=N,
        params_path=params_path,
        device=device,
    )

    with tch.no_grad():
        out = net(u, h0=None, noise_std=float(noise_std), return_rate=True)
        rates_pred = out["rate"]  # [C,T,N]
        if psth_bin_ms is not None and float(psth_bin_ms) > 0:
            rates_pred = _time_bin_smooth_ctn(rates_pred, fps=fps, bin_ms=float(psth_bin_ms))

    mask_np, t_sec, ev_sec = compute_time_mask_tsec_zero_at_R(
    meta=meta, T=T,
    sample_ignore_ms=sample_ignore_ms,
    resp_sec=resp_sec,
    )
    psth_used = psth[:, mask_np, :]
    rates_used = rates_pred[:, mask_np, :]

    # Optional additional smoothing for visualization only (does not change evaluation loss).
    if plot_bin_ms is not None and float(plot_bin_ms) > 0:
        psth_used = _time_bin_smooth_ctn(psth_used, fps=fps, bin_ms=float(plot_bin_ms))
        rates_used = _time_bin_smooth_ctn(rates_used, fps=fps, bin_ms=float(plot_bin_ms))

    idx_sel, mse = _select_neurons_by_mse(
        psth_used=psth_used,
        rates_used=rates_used,
        num_neurons=int(num_neurons),
        mode=str(mode),
        rng_seed=int(rng_seed),
    )

    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    save_dir = out_dir or os.path.dirname(model_path) or "."
    out_png = os.path.join(save_dir, f"psth_eval_loss_window_{model_tag}.png")

    plot_psth_comparison_R0_with_events(
    t_sec=t_sec,
    psth_used=psth_used,
    rates_used=rates_used,
    idx_neurons=idx_sel.tolist(),
    mse_per_neuron=mse,
    cond_names=cond_names,
    out_path=out_png,
    ncols=6,
    title=f"{model_tag}  unit={os.path.basename(npz_path)}  window=loss (R=0)",
    event_sec=ev_sec,
    show_event_labels=True,
    )


def eval_all_units_from_registry(
    registry_dir: str,
    animal: str,
    model_path: str,
    params_path: Optional[str],
    n_exc_virtual: int,
    noise_std: float,
    psth_bin_ms: float,
    plot_bin_ms: float,
    sample_ignore_ms: float,
    resp_sec: float,
    plot_n: int,
    plot_seed: int,   # 保留参数兼容CLI；best策略下不会影响结果（除非出现完全相等的极少数tie）
    plot_cols: int,
    out_dir: str,
    device: tch.device,
) -> None:
    """
    Batch eval across all units in registry.

    - Computes per-unit loss (MSE on training loss window).
    - For plotting: selects global Top-K neurons with the smallest MSE across ALL sessions' inhibitory neurons,
      and saves ONE mosaic figure.
    - GT and Fit for the same condition share the same color (GT solid, Fit dashed).

    Plot style update:
      - x-axis is zeroed at R(go cue): t=0 at R
      - draw event lines S/D/R
      - remove y=0 horizontal line
    """
    import heapq

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Local helper: time mask + t_sec (R=0) + event lines (sec)
    # -----------------------------
    def _compute_time_mask_tsec_R0_and_events(
        meta: Dict[str, Any],
        T: int,
        sample_ignore_ms: float,
        resp_sec: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        fps = float(meta.get("fps", 1.0))
        m = _build_time_mask_sample_delay_resp(
            T=T,
            fps=fps,
            meta=meta,
            sample_ignore_ms=float(sample_ignore_ms),
            resp_sec=float(resp_sec),
        )
        idx = np.where(m)[0]
        ev = meta.get("event_frames", {}) or {}

        def _get_ev_frame(name: str, default_val: int) -> int:
            if isinstance(ev, dict) and (name in ev):
                try:
                    return int(ev[name])
                except Exception:
                    return int(default_val)
            return int(default_val)

        S = _get_ev_frame("S", 0)
        D = _get_ev_frame("D", S)
        R = _get_ev_frame("R", D)  # if missing R, fallback to D; if missing D, fallback to S; else 0

        t_sec = (idx - R) / fps
        ev_sec = {"S": (S - R) / fps, "D": (D - R) / fps, "R": 0.0}
        return m, t_sec, ev_sec

    # -----------------------------
    # Load registry and group by unit
    # -----------------------------
    rows = _read_registry_csv(registry_dir=registry_dir, animal=animal)
    by_unit = _group_rows_by_unit(rows)

    # -----------------------------
    # Infer N_total and D_in directly from checkpoint (most reliable)
    # -----------------------------
    ckpt = tch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and ("model" in ckpt) and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # remove training-only keys if present
    if isinstance(state_dict, dict) and "dale_mask" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("dale_mask", None)

    if "J" not in state_dict or "W_in" not in state_dict:
        raise KeyError(
            "Cannot infer N_total / D_in from checkpoint. "
            "Expected keys 'J' and 'W_in' in state_dict."
        )

    N_total = int(state_dict["J"].shape[0])
    D_in_ckpt = int(state_dict["W_in"].shape[1])

    # Build model once
    net = _build_model_for_eval(
        model_path=model_path,
        D_in=D_in_ckpt,
        N_total=N_total,
        params_path=params_path,
        device=device,
    )
    print(f"[INFO] Loaded model: N_total={N_total}, D_in={D_in_ckpt}")

    # -----------------------------
    # Eval accumulators
    # -----------------------------
    per_unit_stats: List[Dict[str, Any]] = []

    # Map global_idx -> cell_subclasses label (built during unit iteration)
    labels_by_g: Dict[int, str] = {}


    # Global Top-K best neurons across all sessions:
    # store (-mse, counter, record)
    plot_k = int(plot_n)
    top_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    counter = 0

    # -----------------------------
    # Iterate units
    # -----------------------------
    default_parameters = _load_default_parameters(params_path)
    amp_input = float(default_parameters.get("amp_input", 1.0))
    include_go_cue = bool(default_parameters.get("include_go_cue", True))
    go_cue_sec = float(default_parameters.get("go_cue_sec", 0.10))
    amp_go_cue = float(default_parameters.get("amp_go_cue", 1.0))
    include_reward = bool(default_parameters.get("include_reward", True))
    amp_reward = float(default_parameters.get("amp_reward", 1.0))

    for unit_key, items in by_unit.items():
        # all rows in a unit should share same npz_path
        npz_path = items[0].npz_path
        for rr in items[1:]:
            if rr.npz_path != npz_path:
                raise ValueError(f"unit_key={unit_key} has inconsistent npz_path in registry.")

        # load PSTH + meta
        psth, meta = load_alm_psth_npz(
            npz_path=npz_path,
            cond_filter=None,
            max_time=None,
            device=device,
            dtype=tch.float32,
        )
        C, T, N_kept = psth.shape
        meta = _maybe_attach_lick_reward_to_meta(meta=meta, npz_path=npz_path, T=T)

        fps = float(meta.get("fps", 1.0))

        # time bin / smooth (apply to both GT and prediction to match your training setting)
        if psth_bin_ms is not None and float(psth_bin_ms) > 0:
            psth = _time_bin_smooth_ctn(psth, fps=fps, bin_ms=float(psth_bin_ms))

        # map registry array_idx -> position in keep_idx
        keep_idx = meta.get("keep_idx", None)
        if keep_idx is None:
            raise KeyError(f"meta['keep_idx'] missing in stage1 npz: {npz_path}")
        keep_map = _build_keepidx_pos_map(np.asarray(keep_idx, dtype=int))

        pos_list: List[int] = []
        g_list: List[int] = []
        for rr in items:
            ai = int(rr.array_idx)
            if ai not in keep_map:
                raise ValueError(
                    f"array_idx={ai} not found in keep_idx of {npz_path} (unit_key={unit_key}). "
                    "Check registry builder mapping."
                )
            pos_list.append(int(keep_map[ai]))
            g_list.append(int(rr.global_idx))

        

        # ---- collect cell_subclasses labels for J aggregation ----
        try:
            subs_kept = _extract_kept_cell_subclasses(meta)
        except Exception as e:
            subs_kept = None
            print(f"[WARN] {unit_key}: cannot align cell_subclasses for J plot: {e}")

        if subs_kept is not None:
            for rr in items:
                g = int(rr.global_idx)
                pos = int(keep_map[int(rr.array_idx)])
                lab = str(subs_kept[pos])
                if g in labels_by_g and labels_by_g[g] != lab:
                    print(
                        f"[WARN] global_idx label mismatch: g={g} prev={labels_by_g[g]} new={lab} "
                        f"(unit_key={unit_key}, npz={os.path.basename(npz_path)})"
                    )
                else:
                    labels_by_g[g] = lab
        pos_t = tch.as_tensor(pos_list, dtype=tch.long, device=device)
        psth_sub = psth.index_select(dim=2, index=pos_t)  # [C,T,K]
        K = int(psth_sub.shape[2])

        # build input u with meta (must match training)
        cond_names = list(meta["cond_names"])
        u = _build_input_tensor(
            C=C,
            T=T,
            cond_names=cond_names,
            device=device,
            amp_input=amp_input,
            meta=meta,
            amp_stim=amp_input,
            sample_len_sec=1.15,
            on_sec=0.15,
            off_sec=0.10,
            n_bursts=5,
            include_go_cue=include_go_cue,
            go_cue_sec=go_cue_sec,
            amp_go_cue=amp_go_cue,
            include_reward=include_reward,
            amp_reward=amp_reward,
        )

        D_in = int(u.shape[-1])
        if D_in != D_in_ckpt:
            raise ValueError(
                f"D_in mismatch for unit_key={unit_key}: built u has D_in={D_in}, "
                f"but checkpoint expects D_in={D_in_ckpt}. "
                "This means your eval input construction differs from training."
            )

        # indices into the global net: offset inhibitory by n_exc_virtual
        idx_net = tch.as_tensor(
            [int(n_exc_virtual) + int(g) for g in g_list],
            dtype=tch.long,
            device=device,
        )

        # forward
        with tch.no_grad():
            out = net(u, h0=None, noise_std=float(noise_std), return_rate=True)
            rates_full = out["rate"]  # [C,T,N_total]
            if psth_bin_ms is not None and float(psth_bin_ms) > 0:
                rates_full = _time_bin_smooth_ctn(rates_full, fps=fps, bin_ms=float(psth_bin_ms))
            pred_sub = rates_full.index_select(dim=2, index=idx_net)  # [C,T,K]

        # time mask (same as training loss window) + t_sec zeroed at R
        mask_np, t_sec, ev_sec = _compute_time_mask_tsec_R0_and_events(
            meta=meta, T=T, sample_ignore_ms=sample_ignore_ms, resp_sec=resp_sec
        )
        mask_t = tch.as_tensor(mask_np.astype(np.bool_), device=device)

        # compute unit loss and per-neuron MSE (on loss window)
        with tch.no_grad():
            diff = (pred_sub[:, mask_t, :] - psth_sub[:, mask_t, :]) ** 2  # [C,Tm,K]
            unit_loss = float(diff.mean().detach().cpu().item())
            mse_neuron = diff.mean(dim=(0, 1)).detach().cpu().numpy()  # [K]

        # Precompute traces used for plotting (optionally smoother than the loss window tensors).
        psth_win = psth_sub[:, mask_t, :]
        pred_win = pred_sub[:, mask_t, :]
        if plot_bin_ms is not None and float(plot_bin_ms) > 0:
            psth_win = _time_bin_smooth_ctn(psth_win, fps=fps, bin_ms=float(plot_bin_ms))
            pred_win = _time_bin_smooth_ctn(pred_win, fps=fps, bin_ms=float(plot_bin_ms))

        per_unit_stats.append(
            {
                "unit_key": unit_key,
                "npz_path": npz_path,
                "K": int(K),
                "loss_mse": float(unit_loss),
                "mse_neuron_mean": float(np.mean(mse_neuron)),
                "mse_neuron_median": float(np.median(mse_neuron)),
            }
        )

        # -----------------------------
        # Update global Top-K best neurons for plotting
        # criterion: smallest mse_neuron (loss window)
        # Only materialize traces when the candidate can enter Top-K
        # -----------------------------
        if plot_k > 0:
            cond0 = str(cond_names[0]) if len(cond_names) > 0 else "cond0"
            cond1 = str(cond_names[1]) if len(cond_names) > 1 else "cond1"

            # heap stores (-mse, ...), so -heap[0][0] is the largest mse inside Top-K
            thr = (-top_heap[0][0]) if len(top_heap) >= plot_k else float("inf")

            for local_i in range(K):
                mse_i = float(mse_neuron[local_i])
                if (len(top_heap) < plot_k) or (mse_i < thr):
                    # materialize traces (on loss window)
                    gt0 = psth_win[0, :, local_i].detach().cpu().numpy()
                    pr0 = pred_win[0, :, local_i].detach().cpu().numpy()
                    gt1 = psth_win[1, :, local_i].detach().cpu().numpy()
                    pr1 = pred_win[1, :, local_i].detach().cpu().numpy()

                    rec = {
                        "unit_key": unit_key,
                        "npz_base": os.path.basename(npz_path),
                        "local_i": int(local_i),
                        "mse": float(mse_i),
                        "t_sec": t_sec,        # R=0
                        "ev_sec": ev_sec,      # S/D/R relative to R
                        "cond0": cond0,
                        "cond1": cond1,
                        "gt0": gt0,
                        "pr0": pr0,
                        "gt1": gt1,
                        "pr1": pr1,
                    }

                    counter += 1
                    heapq.heappush(top_heap, (-mse_i, counter, rec))
                    if len(top_heap) > plot_k:
                        heapq.heappop(top_heap)

                    thr = (-top_heap[0][0]) if len(top_heap) >= plot_k else float("inf")

    # -----------------------------
    # Save per-unit stats + summary
    # -----------------------------
    if len(per_unit_stats) == 0:
        raise ValueError("No units evaluated (empty registry?)")

    model_tag = os.path.splitext(os.path.basename(model_path))[0]

    csv_path = os.path.join(out_dir, f"eval_per_unit_{model_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_unit_stats[0].keys()))
        writer.writeheader()
        for r in per_unit_stats:
            writer.writerow(r)
    print(f"[OK] Saved per-unit stats -> {csv_path}")

    losses = np.array([r["loss_mse"] for r in per_unit_stats], dtype=float)
    summary = {
        "animal": animal,
        "model": model_path,
        "registry_dir": registry_dir,
        "n_units": int(len(per_unit_stats)),
        "n_exc_virtual": int(n_exc_virtual),
        "N_total": int(N_total),
        "D_in": int(D_in_ckpt),
        "psth_bin_ms": float(psth_bin_ms),
        "plot_bin_ms": float(plot_bin_ms),
        "plot_bin_ms": float(plot_bin_ms),
        "sample_ignore_ms": float(sample_ignore_ms),
        "resp_sec": float(resp_sec),
        "noise_std": float(noise_std),
        "loss_mean": float(np.mean(losses)),
        "loss_median": float(np.median(losses)),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "plot_strategy": "best_global_topk",
        "plot_n": int(plot_k),
        "time_zero": "R(go_cue)",
        "event_lines": {"S": ":", "D": "--", "R": "-"},
    }
    json_path = os.path.join(out_dir, f"eval_summary_{model_tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved summary -> {json_path}")

    # -----------------------------
    # Build and save ONE mosaic figure using Top-K best neurons
    # -----------------------------
    picked_records = [x[2] for x in top_heap]
    picked_records.sort(key=lambda r: float(r["mse"]))  # best first

    if len(picked_records) == 0:
        print("[WARN] plot_n=0 or no candidates selected; skip mosaic plotting.")
        return

    ncols = max(1, int(plot_cols))
    nrows = int(math.ceil(len(picked_records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)

    for i, rec in enumerate(picked_records):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        t = rec["t_sec"]     # already R=0
        ev = rec.get("ev_sec", None)

        # Same-condition same-color: draw GT first, reuse its color for Fit
        l0, = ax.plot(t, rec["gt0"], label=f"{rec['cond0']} GT")  # solid
        ax.plot(t, rec["pr0"], linestyle="--", color=l0.get_color(), label=f"{rec['cond0']} Fit")

        l1, = ax.plot(t, rec["gt1"], label=f"{rec['cond1']} GT")
        ax.plot(t, rec["pr1"], linestyle="--", color=l1.get_color(), label=f"{rec['cond1']} Fit")

        # --- event lines (relative to R) ---
        if isinstance(ev, dict):
            # R at 0
            if "R" in ev:
                ax.axvline(float(ev["R"]), linestyle="-", linewidth=0.8)
            if "S" in ev:
                ax.axvline(float(ev["S"]), linestyle=":", linewidth=0.8)
            if "D" in ev:
                ax.axvline(float(ev["D"]), linestyle="--", linewidth=0.8)

        # IMPORTANT: remove y=0 line (do NOT draw axhline)
        ax.set_title(
            f"{rec['unit_key']} i={rec['local_i']} mse={float(rec['mse']):.2e}",
            fontsize=8,
        )
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    # hide unused axes
    for j in range(len(picked_records), nrows * ncols):
        rr = j // ncols
        cc = j % ncols
        axes[rr][cc].axis("off")

    fig.suptitle(f"{model_tag}  BEST neurons across all sessions (loss window, R=0)", fontsize=12)
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"psth_eval_all_best_mosaic_{model_tag}_R0.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved best-mosaic -> {png_path}")

    # -----------------------------
    # Plot J (real neurons) aggregated by celltype
    # -----------------------------
    try:
        _plot_J_real_by_celltype(
            J_full=net.J,
            n_exc_virtual=int(n_exc_virtual),
            labels_by_g=labels_by_g,
            out_dir=out_dir,
            model_tag=model_tag,
            title_prefix="J (real observed neurons)",
        )
    except Exception as e:
        print(f"[WARN] J-by-celltype plotting failed: {e}")




    # -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, required=True, help="Path to trained model .pt")
    ap.add_argument("--params", type=str, default=None, help="Path to params json/yaml used by training (optional)")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", type=str, default=None, help="Stage1 psth_*.npz (single-session eval)")
    g.add_argument("--eval_all", action="store_true", help="Evaluate all units in registry (batch)")

    ap.add_argument("--registry_dir", type=str, default=None)
    ap.add_argument("--animal", type=str, default=None)
    ap.add_argument("--n_exc_virtual", type=int, default=0)

    # IMPORTANT: default eval is deterministic; set >0 only if you explicitly want stochastic trajectories.
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--psth_bin_ms", type=float, default=200.0)
    # Extra smoothing for visualization only (applied after the loss-window mask).
    ap.add_argument("--plot_bin_ms", type=float, default=0.0)
    ap.add_argument("--sample_ignore_ms", type=float, default=50.0)
    ap.add_argument("--resp_sec", type=float, default=2.0)

    ap.add_argument("--num_neurons", type=int, default=30)
    ap.add_argument("--mode", default="random", choices=["best", "worst", "random"])
    ap.add_argument("--rng_seed", type=int, default=42)

    ap.add_argument("--plot_n", type=int, default=72)
    ap.add_argument("--plot_seed", type=int, default=42)
    ap.add_argument("--plot_cols", type=int, default=6)

    ap.add_argument("--out_dir", type=str, default=None)

    args = ap.parse_args()

    device = _infer_device(args.params)
    print(f"[INFO] device={device}")

    
    ckpt_dir = os.path.dirname(os.path.abspath(args.model)) or "."
    if args.out_dir is not None:
        od = os.path.abspath(args.out_dir)
        if od != ckpt_dir:
            print(f"[WARN] --out_dir ignored; saving next to checkpoint: {ckpt_dir}")
    save_dir = ckpt_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.eval_all:
        if args.registry_dir is None or args.animal is None:
            raise ValueError("--eval_all requires --registry_dir and --animal")
        eval_all_units_from_registry(
            registry_dir=args.registry_dir,
            animal=args.animal,
            model_path=args.model,
            params_path=args.params,
            n_exc_virtual=int(args.n_exc_virtual),
            noise_std=float(args.noise_std),
            psth_bin_ms=float(args.psth_bin_ms),
            plot_bin_ms=float(args.plot_bin_ms),
            sample_ignore_ms=float(args.sample_ignore_ms),
            resp_sec=float(args.resp_sec),
            plot_n=int(args.plot_n),
            plot_seed=int(args.plot_seed),
            plot_cols=int(args.plot_cols),
            out_dir=save_dir,
            device=device,
        )
    else:
        assert args.npz is not None
        eval_single_session(
            npz_path=args.npz,
            model_path=args.model,
            params_path=args.params,
            noise_std=float(args.noise_std),
            psth_bin_ms=float(args.psth_bin_ms),
            plot_bin_ms=float(args.plot_bin_ms),
            sample_ignore_ms=float(args.sample_ignore_ms),
            resp_sec=float(args.resp_sec),
            num_neurons=int(args.num_neurons),
            mode=str(args.mode),
            rng_seed=int(args.rng_seed),
            out_dir=save_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
