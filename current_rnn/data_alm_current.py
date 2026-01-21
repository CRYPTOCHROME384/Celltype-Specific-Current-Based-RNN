# current_rnn/data_alm_current.py

import os
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch


def build_tone_waveform(
    T: int,
    fps: float,
    S_frame: int,
    sample_len_sec: float = 1.15,
    on_sec: float = 0.15,
    off_sec: float = 0.10,
    n_bursts: int = 5,
) -> np.ndarray:
    
    x = np.zeros(T, dtype=np.float32)

    on_frames = int(round(on_sec * fps))
    off_frames = int(round(off_sec * fps))
    sample_frames = int(round(sample_len_sec * fps))

    start = int(S_frame)
    end = min(T, start + sample_frames)

    t = start
    for k in range(n_bursts):
        if t >= end:
            break
        # ON
        t_on_end = min(end, t + on_frames)
        x[t:t_on_end] = 1.0
        t = t_on_end
        # OFF (last burst can omit off)
        if k < n_bursts - 1:
            t_off_end = min(end, t + off_frames)
            # already zeros
            t = t_off_end

    return x

def _normalize_cond_names(arr) -> List[str]:
    """
    Ensure cond_names loaded from npz is a list of Python strings.
    """
    if isinstance(arr, np.ndarray):
        # Could be array of bytes/str/object
        return [str(x) for x in arr.tolist()]
    # Fallback
    return [str(x) for x in list(arr)]


def build_dale_mask_from_types(is_excitatory: np.ndarray) -> torch.Tensor:
    """
    return dale_mask: [N, N]：
        excitatory → J[:, j] >= 0  (mask = +1)
        inhibitory → J[:, j] <= 0  (mask = -1)
    """
    N = is_excitatory.shape[0]
    dale_mask = np.zeros((N, N), dtype=np.int8)

    exc_idx = np.where(is_excitatory)[0]
    inh_idx = np.where(~is_excitatory)[0]

    # excitatory columns
    dale_mask[:, exc_idx] = 1
    # inhibitory columns
    dale_mask[:, inh_idx] = -1

    # do not constrain self-connection
    np.fill_diagonal(dale_mask, 0)

    return torch.from_numpy(dale_mask)

def load_alm_psth_npz(
    npz_path: str,
    cond_filter: Optional[List[str]] = None,
    max_time: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load trial-averaged PSTH and metadata from a Stage 1 .npz file
    produced by alm_data/0.average.py.

    The .npz is expected to contain (see 0.average.py):
        - session_id, plane, animal
        - cond_names: array of condition names (keys of cell_psth)
        - cell_psth: dict[name] -> ndarray (cells, time)
        - cell_clusters, cell_subclasses
        - fps, t0_frame, event_frames, pre_frames, post_frames
        - keep_idx, n_cells_before
        - cond_counts, source_used, pkl_key_used, ...

    This function:
        1) Selects conditions to use (e.g. ['left_correct', 'right_correct']).
        2) Builds a tensor psth of shape [C, T, N], where:
           - C = number of selected conditions
           - T = number of time points (optionally truncated by max_time)
           - N = number of neurons after cell-type filtering
        3) Returns (psth, meta), where psth is a torch.Tensor and meta is a dict.

    Args:
        npz_path:   path to the Stage 1 npz file.
        cond_filter:
            - If None: try to use ['left_correct', 'right_correct'] if present,
              otherwise use all available conditions found in cell_psth.
            - If list: use the intersection of this list with available keys.
              If nothing matches, raise ValueError.
        max_time:
            - If not None: truncate time dimension to [0:max_time] frames.
        device:
            - Optional torch device to move psth tensor onto.
        dtype:
            - torch dtype for the returned psth.

    Returns:
        psth: torch.Tensor of shape [C, T, N]
        meta: dict containing metadata and numpy arrays (celltype info, etc.)
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"npz file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # -------------------------------------------------------------------------
    # Condition names & cell_psth
    # -------------------------------------------------------------------------
    cond_names_all = _normalize_cond_names(data["cond_names"])
    cell_psth_obj = data["cell_psth"]
    # cell_psth was saved as a dict (object array), need .item() to recover
    if isinstance(cell_psth_obj, np.ndarray):
        cell_psth: Dict[str, np.ndarray] = cell_psth_obj.item()
    else:
        cell_psth = cell_psth_obj

    # --- decide which conditions to use ---
    if cond_filter is None:
        preferred = ["left_correct", "right_correct"]
        cond_names = [c for c in preferred if c in cell_psth]
        if not cond_names:
            # fallback: use all conditions that actually exist in cell_psth
            cond_names = [c for c in cond_names_all if c in cell_psth]
    else:
        cond_names = [c for c in cond_filter if c in cell_psth]
        if not cond_names:
            raise ValueError(
                f"No requested conditions {cond_filter} found in cell_psth keys "
                f"({list(cell_psth.keys())})."
            )

    if not cond_names:
        raise ValueError(
            f"No valid conditions found in npz file {npz_path}. "
            "Check that 0.average.py produced non-empty cell_psth."
        )

    # -------------------------------------------------------------------------
    # Build PSTH array: [C, T, N]
    # Each cell_psth[name] has shape (cells, time)
    # We transpose to (T, N) and then stack -> (C, T, N)
    # -------------------------------------------------------------------------
    psth_list = []
    T_min = None
    N = None

    # First pass: ensure all conditions have consistent (cells, time)
    for name in cond_names:
        M = cell_psth[name]  # (cells, time)
        if M is None:
            raise ValueError(f"cell_psth['{name}'] is None in {npz_path}")

        if N is None:
            N = M.shape[0]
        elif M.shape[0] != N:
            raise ValueError(
                f"Inconsistent neuron count among conditions in {npz_path}: "
                f"condition '{name}' has {M.shape[0]} cells, expected {N}."
            )

        if T_min is None:
            T_min = M.shape[1]
        else:
            T_min = min(T_min, M.shape[1])

    # If max_time is specified, clip by max_time; otherwise use min length across conditions
    if max_time is not None:
        T = min(T_min, max_time)
    else:
        T = T_min

    for name in cond_names:
        M = cell_psth[name]  # (cells, time)
        M = M[:, :T]         # truncate time if needed -> (N, T)
        M = M.T              # (T, N)
        psth_list.append(M)

    # stack to get (C, T, N)
    psth_np = np.stack(psth_list, axis=0)
    psth = torch.as_tensor(psth_np, dtype=dtype)
    if device is not None:
        psth = psth.to(device)

    # -------------------------------------------------------------------------
    # Collect metadata for later analyses
    # -------------------------------------------------------------------------
    # Cell-type info
    cell_clusters = np.asarray(data["cell_clusters"])
    cell_subclasses = np.asarray(data["cell_subclasses"])
    # some npz also stores 'cell_types'; keep if present
    cell_types = np.asarray(data["cell_types"]) if "cell_types" in data else cell_clusters

    # Time / alignment info
    fps = float(data["fps"])
    t0_frame = int(data["t0_frame"])
    pre_frames = int(data["pre_frames"])
    post_frames = int(data["post_frames"])

    # event_frames saved as a dict -> unwrap
    event_frames_obj = data["event_frames"]
    if isinstance(event_frames_obj, np.ndarray):
        # usually a 0-d object array containing a dict
        event_frames = event_frames_obj.item()
    else:
        event_frames = event_frames_obj

    # Index mapping and counts
    keep_idx = np.asarray(data["keep_idx"], dtype=int)
    n_cells_before = int(data["n_cells_before"])
    cond_counts_obj = data["cond_counts"]
    if isinstance(cond_counts_obj, np.ndarray):
        cond_counts = cond_counts_obj.item()
    else:
        cond_counts = cond_counts_obj

    # Session-level info
    session_id = str(data["session_id"])
    plane = str(data["plane"])
    animal = str(data["animal"])
    align_to = str(data["align_to"])
    source_used = str(data["source_used"])
    pkl_key_used = str(data["pkl_key_used"])

    meta: Dict[str, Any] = {
        # Core dimensions
        "cond_names": cond_names,
        "all_cond_names": cond_names_all,
        "C": len(cond_names),
        "T": T,
        "N": N,

        # Time / alignment
        "fps": fps,
        "t0_frame": t0_frame,
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "event_frames": event_frames,  # dict like {'S': ss, 'D': ld, 'R': go}
        "align_to": align_to,

        # Cell-type / indexing
        "cell_clusters": cell_clusters,
        "cell_subclasses": cell_subclasses,
        "cell_types": cell_types,
        "keep_idx": keep_idx,
        "n_cells_before": n_cells_before,

        # Condition counts (per condition)
        "cond_counts": cond_counts,

        # Session identifiers
        "session_id": session_id,
        "plane": plane,
        "animal": animal,

        # Provenance
        "source_used": source_used,
        "pkl_key_used": pkl_key_used,
        "npz_path": os.path.abspath(npz_path),
    }

    return psth, meta
