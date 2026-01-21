# current_rnn/training_current.py

import os
import sys
import json
import time
from typing import Optional, List, Dict, Any

import numpy as np
import torch as tch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Make project root importable so we can reuse losses.py and plotting.py
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from losses import LossAverageTrials          # reuse existing loss
import plotting                               # reuse existing plotting helpers

from model_current import ALMCurrentRNN       # current_rnn/model_current.py
from data_alm_current import load_alm_psth_npz,build_dale_mask_from_types  # current_rnn/data_alm_current.py
from utils_celltype import load_is_excitatory_from_npz  # current_rnn/utils_celltype.py

@tch.no_grad()
def _global_eval_loss_neuron_weighted(
    net,
    units,
    *,
    noise_std_eval: float = 0.0,
) -> float:
    """
    Neuron-weighted global loss:
      For each unit_key/session (one entry in `units`):
        - forward on its u: rates_full [C,T,N_total]
        - select observed neurons: pred_sub [C,T,K]
        - compute diff on its time_mask
        - compute mse_per_neuron = mean_{cond,time}(diff) -> [K]
      Aggregate:
        global_loss = (sum over all neurons of mse_per_neuron) / (total #neurons)
    """
    net.eval()

    total_neurons = 0
    sum_mse_over_neurons = 0.0

    for batch in units:
        u = batch["u"]                 # [C,T,D]
        psth_sub = batch["psth_sub"]   # [C,T,K]
        idx_net = batch["idx_net"]     # [K]
        time_mask = batch["time_mask"] # [T] bool (or indices)

        # numeric sanitation consistent with training
        u = tch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        psth_sub = tch.nan_to_num(psth_sub, nan=0.0, posinf=0.0, neginf=0.0)

        out = net(u, h0=None, noise_std=float(noise_std_eval), return_rate=True)
        rates_full = out["rate"]  # [C,T,N_total]

        pred_sub = rates_full.index_select(dim=2, index=idx_net)  # [C,T,K]

        if time_mask.dtype == tch.bool:
            pred_m = pred_sub[:, time_mask, :]
            targ_m = psth_sub[:, time_mask, :]
        else:
            pred_m = pred_sub.index_select(dim=1, index=time_mask)
            targ_m = psth_sub.index_select(dim=1, index=time_mask)

        diff = (pred_m - targ_m).pow(2)                # [C,Tm,K]
        mse_per_neuron = diff.mean(dim=(0, 1))         # [K]

        sum_mse_over_neurons += float(mse_per_neuron.sum().item())
        total_neurons += int(mse_per_neuron.numel())

    return sum_mse_over_neurons / max(total_neurons, 1)


def _load_default_parameters(params_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load global parameter dictionary from parameters_list.json.
    If params_path is None, uses <ROOT_DIR>/parameters_list.json.
    """
    if params_path is None:
        params_path = os.path.join(ROOT_DIR, "parameters_list.json")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"parameters_list.json not found: {params_path}")
    with open(params_path, "r") as f:
        params = json.load(f)
    return params



def _maybe_attach_lick_reward_to_meta(
    meta: Dict[str, Any],
    npz_path: str,
    T: int,
) -> Dict[str, Any]:
    """Attach lick/reward traces (if present) from the stage1 npz into meta.

    This keeps training/eval code simple even if load_alm_psth_npz does not yet
    return these fields.

    Expected shapes in the stage1 npz:
      reward_trace:      [C_all, T_all]
      lick_rate_left:    [C_all, T_all]
      lick_rate_right:   [C_all, T_all]
      lick_rate_total:   [C_all, T_all]
      t_rel_go_sec:      [T_all]
      idx_2p_in_bpod:    [n_trials_2p]  (informational)

    We slice to the *currently loaded* conditions in meta['cond_names'] and the
    currently loaded time length T.
    """
    if meta is None:
        return meta

    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:
        return meta

    if "cond_names" not in z or "reward_trace" not in z:
        return meta

    cond_all = [str(x) for x in z["cond_names"].tolist()]
    cond_cur = [str(x) for x in meta.get("cond_names", [])]

    name2i = {n: i for i, n in enumerate(cond_all)}
    try:
        idx = [name2i[n] for n in cond_cur]
    except KeyError:
        # If names do not match, fall back to prefix matching (rare but useful)
        idx = []
        for n in cond_cur:
            hit = None
            for i, na in enumerate(cond_all):
                if na == n or na.startswith(n) or n.startswith(na):
                    hit = i
                    break
            if hit is None:
                raise KeyError(
                    f"Cannot map cond '{n}' to stage1 npz cond_names.\n"
                    f"meta.cond_names={cond_cur}\n"
                    f"npz.cond_names={cond_all}"
                )
            idx.append(hit)

    def _slice_CT(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(f"Expected [C,T] array, got shape {arr.shape}")
        arr = arr[idx, :]
        if arr.shape[1] >= T:
            arr = arr[:, :T]
        else:
            # If T is longer (shouldn't happen), pad with zeros
            pad = T - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
        return arr.astype(np.float32, copy=False)

    for k in ["reward_trace", "lick_rate_left", "lick_rate_right", "lick_rate_total"]:
        if k in z:
            meta[k] = _slice_CT(np.asarray(z[k]))

    if "t_rel_go_sec" in z:
        t = np.asarray(z["t_rel_go_sec"]).astype(np.float32, copy=False)
        meta["t_rel_go_sec"] = t[:T] if t.shape[0] >= T else np.pad(t, (0, T - t.shape[0]))

    if "idx_2p_in_bpod" in z:
        meta["idx_2p_in_bpod"] = np.asarray(z["idx_2p_in_bpod"])

    return meta



def _build_sample_waveforms(
    T: int,
    fps: float,
    S_frame: int,
    sample_len_sec: float = 1.15,
    on_sec: float = 0.15,
    off_sec: float = 0.10,
    n_bursts: int = 5,
) -> np.ndarray:
    """Return tone_wave[T], noise_wave[T] as float32."""
    tone = np.zeros(T, dtype=np.float32)
    noise = np.zeros(T, dtype=np.float32)

    sample_frames = int(round(sample_len_sec * fps))
    start = int(S_frame)
    end = min(T, start + sample_frames)

    # noise: constant during sample
    noise[start:end] = 1.0

    # tone bursts: ON/OFF pattern
    on_f = int(round(on_sec * fps))
    off_f = int(round(off_sec * fps))

    t = start
    for k in range(n_bursts):
        if t >= end:
            break
        t_on_end = min(end, t + on_f)
        tone[t:t_on_end] = 1.0
        t = t_on_end
        if k < n_bursts - 1:
            t = min(end, t + off_f)

    return tone, noise


def _build_input_tensor(
    C: int,
    T: int,
    cond_names: List[str],
    device: tch.device,
    amp_input: float = 1.0,
    meta: Optional[Dict[str, Any]] = None,
    amp_stim: Optional[float] = None,
    sample_len_sec: float = 1.15,
    on_sec: float = 0.15,
    off_sec: float = 0.10,
    n_bursts: int = 5,
    include_go_cue: bool = True,
    go_cue_sec: float = 0.10,
    amp_go_cue: Optional[float] = None,
    include_reward: bool = True,
    amp_reward: Optional[float] = None,
) -> tch.Tensor:
    """Build external input u for each condition.

    Output u has shape [C, T, D_in] where:

      u(t) = [cond_onehot] + [stim_left, stim_right] + [go_cue] + [reward]

    Notes on trainability:
      - The temporal traces (go_cue, reward_trace) are fixed functions of time
        (trial averages, aligned to the go cue).
      - The *mapping* from these traces into neuron currents is learned via W_in,
        i.e., each input channel corresponds to a trainable N-vector (one weight
        per neuron).
    """
    if amp_stim is None:
        amp_stim = amp_input
    if amp_go_cue is None:
        amp_go_cue = amp_input
    if amp_reward is None:
        amp_reward = 1.0

    # -----------------------------
    # (1) condition one-hot: [C,T,C]
    # -----------------------------
    u = tch.zeros((C, T, C), device=device, dtype=tch.float32)
    for i in range(C):
        u[i, :, i] = float(amp_input)

    # -----------------------------
    # (2) sample stimulus waveforms: [C,T,2]
    # -----------------------------
    stim = tch.zeros((C, T, 2), device=device, dtype=tch.float32)

    if meta is None:
        raise ValueError("meta is required to build stimulus/go/reward inputs (need event_frames, fps, etc.)")

    fps = float(meta.get("fps", 1.0))
    S_frame = int(meta.get("event_frames", {}).get("S", 0))

    tone, noise = _build_sample_waveforms(
    T=T,
    fps=fps,
    S_frame=S_frame,
    sample_len_sec=sample_len_sec,
    on_sec=on_sec,
    off_sec=off_sec,
    n_bursts=n_bursts,
    )

    # scale in numpy (or torch都行)，然后转 torch 到同一 device
    tone = (tone * float(amp_stim)).astype(np.float32, copy=False)
    noise = (noise * float(amp_stim)).astype(np.float32, copy=False)

    tone_t = tch.from_numpy(tone).to(device=device, dtype=tch.float32).view(T, 1)   # [T,1]
    noise_t = tch.from_numpy(noise).to(device=device, dtype=tch.float32).view(T, 1) # [T,1]
    # assign per condition
    for i, name in enumerate(cond_names):
        s = str(name).lower()

        is_lc = ("left_correct" in s) or (s.startswith("lc")) or ("_lc" in s)
        is_rc = ("right_correct" in s) or (s.startswith("rc")) or ("_rc" in s)
        if is_lc and not is_rc:
            stim[i, :, 0:1] = tone_t
            stim[i, :, 1:2] = noise_t
        elif is_rc and not is_lc:
            stim[i, :, 0:1] = noise_t
            stim[i, :, 1:2] = tone_t
        else:
            pass

    u = tch.cat([u, stim], dim=-1)  # [C,T,C+2]

    # -----------------------------
    # (3) go cue: [C,T,1] (0.1s pulse starting at R frame)
    # -----------------------------
    if include_go_cue:
        R = _get_event_frame(meta, ["R", "go", "go_cue", "go_cue_onset", "G"])
        if R is None:
            raise KeyError("Go cue frame not found in meta['event_frames'] (expected key 'R').")
        go_frames = int(round(float(go_cue_sec) * fps))
        go_frames = max(1, go_frames)
        go = tch.zeros((T, 1), device=device, dtype=tch.float32)
        a0 = max(0, int(R))
        a1 = min(T, int(R) + go_frames)
        if a1 > a0:
            go[a0:a1, 0] = float(amp_go_cue)
        go = go.unsqueeze(0).repeat(C, 1, 1)  # [C,T,1]
        u = tch.cat([u, go], dim=-1)  # +1

    # -----------------------------
    # (4) reward: [C,T,1] (per-cond trace, aligned to go cue)
    # -----------------------------
    if include_reward:
        if "reward_trace" not in meta:
            raise KeyError(
                "meta['reward_trace'] missing. Run build_lick_reward_trace.py --inplace "
                "to write reward_trace into the stage1 npz, then re-load, or call "
                "_maybe_attach_lick_reward_to_meta(meta, npz_path, T) before building u."
            )
        rew = np.asarray(meta["reward_trace"], dtype=np.float32)
        if rew.shape[0] != C:
            raise ValueError(f"reward_trace first dim must match C={C}, got {rew.shape}")
        if rew.shape[1] != T:
            # allow mild mismatch
            rew = rew[:, :T] if rew.shape[1] >= T else np.pad(rew, ((0, 0), (0, T - rew.shape[1])), mode="constant")

        reward = tch.from_numpy(rew).to(device=device, dtype=tch.float32).unsqueeze(-1)  # [C,T,1]
        reward = reward * float(amp_reward)
        u = tch.cat([u, reward], dim=-1)  # +1

    return u


def _kernel_size_from_bin_ms(fps: float, bin_ms: float) -> int:
    """Convert bin_ms to an odd kernel size in frames (>=1), so output length stays T."""
    if bin_ms is None or bin_ms <= 0:
        return 1
    k = int(round(float(bin_ms) / 1000.0 * float(fps)))
    k = max(1, k)
    if (k % 2) == 0:
        k += 1
    return k


def _time_bin_smooth_ctn(x: tch.Tensor, fps: float, bin_ms: float) -> tch.Tensor:
    """Boxcar smooth along time for x shaped [C, T, N] (or [C, T, D]). Length-preserving."""
    k = _kernel_size_from_bin_ms(fps=fps, bin_ms=bin_ms)
    if k <= 1:
        return x

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C,T,*], got {tuple(x.shape)}")

    C, T, D = x.shape
    y = x.permute(0, 2, 1).contiguous().view(C * D, 1, T)   # [C*D,1,T]
    pad = k // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=k, stride=1)            # [C*D,1,T]
    y = y.view(C, D, T).permute(0, 2, 1).contiguous()       # [C,T,D]
    return y

def _regularization_l2(
    net: ALMCurrentRNN, lam_J: float = 0.0, lam_W: float = 0.0
) -> tch.Tensor:
    """
    Simple L2 regularization on J and W_in.
    """
    reg = tch.zeros((), device=net.J.device)
    if lam_J > 0.0:
        reg = reg + lam_J * net.J.pow(2).mean()
    if lam_W > 0.0:
        reg = reg + lam_W * net.W_in.pow(2).mean()
    return reg

import numpy as np

def _get_event_frame(meta: Dict[str, Any], keys: List[str]) -> Optional[int]:
    """Try multiple keys in meta['event_frames'] and return the first found."""
    ev = meta.get("event_frames", None)
    if not isinstance(ev, dict):
        return None
    for k in keys:
        if k in ev:
            return int(ev[k])
    return None


def _build_time_mask_sample_delay_resp(
    T: int,
    fps: float,
    meta: dict,
    sample_ignore_ms: float = 50.0,
    resp_sec: float = 2.0,
) -> np.ndarray:
    """
    mask = [S+ignore, D) + [D, R) + [R, R+resp_sec]
      S = sample onset
      D = delay onset
      R = go cue
    """
    # 事件帧（按你现有 meta.event_frames 的习惯，优先用 'S','D','G'）
    S = _get_event_frame(meta, ["S", "sample", "sample_on", "sample_onset"])
    D = _get_event_frame(meta, ["D", "delay", "delay_on", "delay_onset"])
    G = _get_event_frame(meta, ["R", "go", "go_cue", "response", "response_onset"])

    if S is None:
        raise KeyError("Cannot find sample onset frame in meta['event_frames'] (expected key like 'S').")
    if G is None:
        raise KeyError("Cannot find go cue frame in meta['event_frames'] (expected key like 'G').")

    ignore_frames = int(round(sample_ignore_ms * fps / 1000.0))
    start = S + ignore_frames

    if D is None:
        D = start

    m = np.zeros(T, dtype=bool)

    # sample: [start, D)
    a0 = max(0, start)
    a1 = min(T, D)
    if a1 > a0:
        m[a0:a1] = True

    # delay: [D, G)
    b0 = max(0, D)
    b1 = min(T, G)
    if b1 > b0:
        m[b0:b1] = True

    # response: [G, G + resp_sec]
    resp_frames = int(round(resp_sec * fps))
    c0 = max(0, G)
    c1 = min(T, G + resp_frames)
    if c1 > c0:
        m[c0:c1] = True

    return m


def _build_time_mask_sample_phase(
    T: int,
    meta: Dict[str, Any],
    device: tch.device,
    ignore_ms: float = 0.0,
    sample_window_ms: Optional[float] = None,
) -> tch.Tensor:
    """
    """
    fps = float(meta["fps"])
    event_frames = meta["event_frames"]  # dict like {'S': ss, 'D': ld, 'R': go}

    if "S" not in event_frames:
        raise KeyError("event_frames does not contain key 'S' (sample onset).")

    S_frame = int(event_frames["S"])

    # Number of frames to ignore after sample onset
    ignore_frames = int(round(ignore_ms / 1000.0 * fps))

    start_frame = S_frame + ignore_frames

    # Decide end_frame
    if sample_window_ms is not None:
        window_frames = int(round(sample_window_ms / 1000.0 * fps))
        end_frame = start_frame + window_frames
    else:
        # If delay onset 'D' is available, use it as the end of sample phase; otherwise use T.
        if "D" in event_frames:
            end_frame = int(event_frames["D"])
        else:
            end_frame = T

    # Clip to valid range
    start_frame = max(0, min(start_frame, T))
    end_frame = max(0, min(end_frame, T))

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid time window for sample phase: "
            f"start_frame={start_frame}, end_frame={end_frame}, T={T}"
        )

    time_mask = tch.zeros(T, dtype=tch.bool, device=device)
    time_mask[start_frame:end_frame] = True

    return time_mask


def train_current_alm(
    npz_path: str,
    cond_filter=None,
    max_time=None,
    lr: float = 1e-3,
    max_epochs: int = 50000,
    seed: int = 42,
    noise_std: float = 0.0,
    psth_bin_ms: float = 200.0,
    lam_J: float = 0.0,
    lam_W: float = 0.0,
    params_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    tag: str = "",
    # time masking options:
    use_time_mask: bool = True,
    sample_ignore_ms: float = 50.0,
    resp_sec: float = 2.0,
    # input channels:
    include_go_cue: bool = True,
    go_cue_sec: float = 0.10,
    include_reward: bool = True,
    amp_reward: Optional[float] = None,
    amp_go_cue: Optional[float] = None,
) -> None:
    """
    Train a single-session N-dimensional current-based RNN on trial-averaged ALM data.

    Args:
        npz_path:
            Path to the Stage 1 .npz file produced by 0.average.py.
        cond_filter:
            Optional list of condition names to use, e.g. ['left_correct', 'right_correct'].
            If None, load_alm_psth_npz will choose a reasonable default.
        max_time:
            Optional truncation of the time axis to max_time frames.
        lr:
            Learning rate for Adam.
        max_epochs:
            Number of training epochs.
        seed:
            Random seed for reproducibility.
        noise_std:
            Standard deviation of additive Gaussian noise on h in the RNN.
            For the first deterministic model, this can be kept at 0.0.
        lam_J:
            L2 regularization weight on J.
        lam_W:
            L2 regularization weight on W_in.
        params_path:
            Path to parameters_list.json. If None, uses <ROOT_DIR>/parameters_list.json.
        out_dir:
            Directory to save models and loss plots. If None, uses <ROOT_DIR>/results_current.
        tag:
            String tag appended to filenames for this training run.

        use_time_mask:
            If True, restrict the loss computation to a subset of time points
            specified by mask_mode and related parameters.
        mask_mode:
            Currently only "sample" is implemented: select the sample phase.
        sample_ignore_ms:
            When mask_mode == "sample": number of milliseconds after sample onset
            to exclude from the training window (e.g., 50.0 ms).
        sample_window_ms:
            When mask_mode == "sample": duration of the training window (in ms)
            after the ignored period. If None, the window extends until delay onset
            (event_frames['D']) if present, otherwise until the end of the trial.

    Returns:
        A dictionary with:
            - 'net':         trained model (ALMCurrentRNN)
            - 'psth':        ground-truth psth tensor [C, T, N]
            - 'meta':        metadata dict from load_alm_psth_npz
            - 'loss_history': numpy array of training loss values
            - 'time_mask':   torch.BoolTensor of shape [T] (or None if not used)
    """
    # -------------------------------------------------------------------------
    # Setup and configuration
    # -------------------------------------------------------------------------
    if out_dir is None:
        out_dir = os.path.join(ROOT_DIR, "results_current")
    os.makedirs(out_dir, exist_ok=True)

    default_parameters = _load_default_parameters(params_path)

    # Device selection
    device_str = default_parameters.get("device", "cpu")
    device = tch.device(device_str if tch.cuda.is_available() or device_str == "cpu" else "cpu")

    # Set random seeds
    np.random.seed(seed)
    tch.manual_seed(seed)
    if device.type == "cuda":
        tch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # Load trial-averaged PSTH and metadata
    # -------------------------------------------------------------------------
    psth, meta = load_alm_psth_npz(
        npz_path=npz_path,
        cond_filter=cond_filter,
        max_time=max_time,
        device=device,
        dtype=tch.float32,
    )
    # psth: [C, T, N]
    C, T, N = psth.shape
    cond_names = meta["cond_names"]

    # Attach lick/reward traces (if present) so _build_input_tensor can add reward channel
    meta = _maybe_attach_lick_reward_to_meta(meta=meta, npz_path=npz_path, T=T)

    # -------------------------------------------------------------------------
    # Optional: boxcar time-binning (length-preserving smoothing) on PSTH
    # -------------------------------------------------------------------------
    fps = float(meta.get("fps", 1.0))
    if psth_bin_ms is not None and psth_bin_ms > 0:
        psth = _time_bin_smooth_ctn(psth, fps=fps, bin_ms=float(psth_bin_ms))
        print(f"[INFO] Applied PSTH boxcar smoothing: bin_ms={psth_bin_ms} (fps={fps})")
    else:
        print("[INFO] PSTH boxcar smoothing disabled (psth_bin_ms<=0).")

    # -------------------------------------------------------------------------
    # Build time mask (optional)
    # -------------------------------------------------------------------------
    time_mask = None
    if use_time_mask:
        time_mask = _build_time_mask_sample_delay_resp(
        T=T,
        fps=float(meta["fps"]),
        meta=meta,
        sample_ignore_ms=sample_ignore_ms,
        resp_sec=2.0,
        )
    else:
        time_mask = np.ones(T, dtype=bool)


    # -------------------------------------------------------------------------
    # Build external input tensor u: [C, T, D_in]
    # For now, D_in = C and u is condition one-hot.
    # -------------------------------------------------------------------------
    amp_input = float(default_parameters.get("amp_input", 1.0))
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

    D_in = u.shape[-1]

    # -------------------------------------------------------------------------
    # Instantiate the RNN
    # -------------------------------------------------------------------------
    dt = float(default_parameters.get("dt", 1.0))
    tau = float(default_parameters.get("tau", 1.0))
    substeps = int(default_parameters.get("substeps", 1))

    is_exc = load_is_excitatory_from_npz(npz_path) 

    dale_mask = build_dale_mask_from_types(is_exc)

    net = ALMCurrentRNN(
        N=N,
        D_in=D_in,
        dt=dt,
        tau=tau,
        substeps=substeps,
        nonlinearity="tanh",
        device=device,
        dale_mask=dale_mask.to(device),
    )
    # -------------------------------------------------------------------------
    # Loss and optimizer
    # -------------------------------------------------------------------------
    loss_trials = LossAverageTrials()
    optimizer = tch.optim.Adam(net.parameters(), lr=lr)

    loss_history = np.zeros(max_epochs, dtype=np.float32)

    # Use session id and tag to build filenames
    session_id = meta.get("session_id", "session")
    plane = meta.get("plane", "plane")
    animal = meta.get("animal", "animal")

    if tag:
        run_tag = f"{animal}_{session_id}_{plane}_{tag}"
    else:
        run_tag = f"{animal}_{session_id}_{plane}"

    model_path = os.path.join(out_dir, f"rnn_current_{run_tag}.pt")
    loss_plot_path = os.path.join(out_dir, f"loss_{run_tag}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"[INFO] Training ALMCurrentRNN on {npz_path}")
    print(f"[INFO] Conditions: {cond_names}")
    print(f"[INFO] psth shape: C={C}, T={T}, N={N}, device={device}")
    print(f"[INFO] dt={dt}, tau={tau}, amp_input={amp_input}, lr={lr}")
    print(f"[INFO] Saving model to: {model_path}")
    print(f"[INFO] Training for {max_epochs} epochs...")

    start_time = time.time()
    best_loss = float("inf")
    best_state_dict = None

    for epoch in range(max_epochs):
        net.train()
        optimizer.zero_grad()

        # Forward: u has shape [C, T, D_in]
        out = net(u, h0=None, noise_std=noise_std, return_rate=True)
        rates_pred = out["rate"]  # shape [C, T, N]

        # Optionally apply time mask on the time dimension
        if time_mask is not None:
            psth_used = psth[:, time_mask, :]         # [C, T_mask, N]
            rates_used = rates_pred[:, time_mask, :]  # [C, T_mask, N]
        else:
            psth_used = psth
            rates_used = rates_pred

        # Loss: trial-averaged reconstruction + L2 regularization
        loss_fit = loss_trials(psth_used, rates_used)
        loss_reg = _regularization_l2(net, lam_J=lam_J, lam_W=lam_W)
        loss = loss_fit + loss_reg

        loss.backward()
        optimizer.step()

        # Optionally enforce Dale's law by projecting J if a mask is set
        net.apply_dale_mask()

        # Record loss
        loss_val = float(loss.item())
        loss_history[epoch] = loss_val

        # Track best model
        if loss_val < best_loss:
            best_loss = loss_val
            best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        # Logging
        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Loss = {loss_val:.6f} | "
                f"Fit = {float(loss_fit.item()):.6f} | "
                f"Reg = {float(loss_reg.item()):.6f} | "
                f"Elapsed = {elapsed/60:.1f} min"
            )

        # Optionally plot loss curve periodically
        if (epoch + 1) % 1000 == 0:
            plotting.plot_loss(epoch + 1, loss_history, title="Total loss", tag=loss_plot_path)

    total_time = time.time() - start_time
    print(f"[INFO] Training finished in {total_time/60:.1f} minutes.")
    print(f"[INFO] Best loss = {best_loss:.6f}")

    # -------------------------------------------------------------------------
    # Save best model and final loss curve
    # -------------------------------------------------------------------------
    if best_state_dict is not None:
        tch.save(best_state_dict, model_path)
        print(f"[OK] Best model saved to {model_path}")

    # Final loss plot
    plotting.plot_loss(max_epochs, loss_history, title="Total loss", tag=loss_plot_path)
    print(f"[OK] Loss curve saved to {loss_plot_path}.png")

    result = {
        "net": net,
        "psth": psth,
        "meta": meta,
        "loss_history": loss_history,
        "time_mask": time_mask,
    }
    return result

# ===========================
# Global-registry training
# ===========================
import csv
from collections import defaultdict

def _time_bin_smooth_psth(x: tch.Tensor, fps: float, bin_ms: float) -> tch.Tensor:
    return _time_bin_smooth_ctn(x, fps=fps, bin_ms=bin_ms)

def _read_registry_csv(registry_csv_path):
    rows = []
    with open(registry_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if len(rows) == 0:
        raise ValueError("Empty registry csv: %s" % registry_csv_path)
    return rows

def _group_registry_rows(rows):
    """
    Group rows by unit_key. Minimal required keys:
      - unit_key, npz_path, global_idx, array_idx
    """
    by_unit = defaultdict(list)
    max_g = -1
    for r in rows:
        if "unit_key" not in r or "npz_path" not in r or "global_idx" not in r or "array_idx" not in r:
            raise KeyError("Registry row missing required columns. Need unit_key/npz_path/global_idx/array_idx.")
        g = int(r["global_idx"])
        aidx = int(r["array_idx"])
        max_g = max(max_g, g)
        by_unit[str(r["unit_key"])].append((g, aidx, r))
    n_obs = max_g + 1
    return by_unit, n_obs

def _build_keepidx_pos_map(keep_idx_arr):
    # keep_idx_arr: np.ndarray[int], length N_kept
    d = {}
    for i in range(int(keep_idx_arr.shape[0])):
        d[int(keep_idx_arr[i])] = i
    return d


def _extract_celltype_labels_keep(
    *,
    npz_path: str,
    keep_idx: np.ndarray,
    label_key: str = "cell_subclasses",
    allow_pickle: bool = True,
) -> np.ndarray:
    """Return string labels aligned to keep_idx.

    The source arrays may be either:
      - already aligned to keep_idx (len == len(keep_idx))
      - aligned to the full cell axis (len == n_cells_before), in which case we subset by keep_idx

    Returns:
      labels_keep: np.ndarray[object] of length len(keep_idx), with each element a normalized string.
    """
    z = np.load(npz_path, allow_pickle=bool(allow_pickle))
    if label_key not in z.files:
        raise KeyError(f"{npz_path}: missing '{label_key}' required for cell-type loss.")

    raw = z[label_key]
    arr = np.asarray(raw)
    # handle object scalar storing list/dict
    if arr.ndim == 0:
        try:
            arr = np.asarray(arr.item())
        except Exception:
            arr = np.asarray([arr.item()])

    if arr.ndim != 1:
        raise ValueError(f"{npz_path}: expected 1D '{label_key}', got shape={arr.shape}")

    keep_idx = np.asarray(keep_idx, dtype=int)
    if arr.shape[0] == keep_idx.shape[0]:
        labels_keep = arr
    else:
        mx = int(keep_idx.max())
        if arr.shape[0] <= mx:
            raise ValueError(
                f"{npz_path}: cannot subset '{label_key}' by keep_idx: len(labels)={arr.shape[0]} <= max(keep_idx)={mx}"
            )
        labels_keep = arr[keep_idx]

    def _norm(x) -> str:
        if x is None:
            return "unknown"
        if isinstance(x, bytes):
            try:
                x = x.decode("utf-8", "ignore")
            except Exception:
                x = str(x)
        s = str(x)
        s2 = s.strip()
        if s2 == "":
            return "unknown"
        if s2.lower() in {"nan", "none", "null"}:
            return "unknown"
        return s2

    labels_keep = np.asarray([_norm(v) for v in labels_keep], dtype=object)
    return labels_keep

def _preload_units_from_registry(
    by_unit,
    n_exc_virtual,
    device,
    cond_filter,
    max_time,
    psth_bin_ms,
    sample_ignore_ms,
    # ---- optional knobs (won't break existing call sites) ----
    amp_input: float = 1.0,
    include_go_cue: bool = True,
    go_cue_sec: float = 0.10,
    include_reward: bool = True,
    reward_mode: str = "correctport",
    resp_sec: float = 2.0,
    # ---- NEW: trial-level targets from separate trials_*.npz ----
    use_trials: bool = False,
    trial_keys: tuple = ("cell_trials",),     # keys inside trials_*.npz
    trials_bin_ms: float = None,              # if None: follow psth_bin_ms
    trials_roots: list = None,                # optional search roots for trials_*.npz
    require_trials: bool = True,              # if True: raise if a unit has no trials file
):
    """
    Returns:
      units: list[dict] with keys:
        - unit_key, npz_path
        - psth_sub:  [C,T,K]  (cond-avg target; inhibitory-only subset for this unit)
        - trials_sub: dict[str, Tensor] each [R,T,K]  (optional; trial-level targets per cond)
        - u:         [C,T,D]  (inputs; tone/go/reward)
        - idx_net:   [K] long (indices into full net of size N_total)
        - time_mask: [T] bool (torch)
        - meta: dict
      shared: dict with keys:
        - C, T, fps
        - N_inh_total, N_total
    """
    import os
    import re
    import glob
    import numpy as np
    import torch as tch

    def _get_smoother():
        if "_time_bin_smooth_psth" in globals() and callable(globals()["_time_bin_smooth_psth"]):
            return globals()["_time_bin_smooth_psth"]
        if "_time_bin_smooth_ctn" in globals() and callable(globals()["_time_bin_smooth_ctn"]):
            return globals()["_time_bin_smooth_ctn"]
        return None

    smoother = _get_smoother()

    if trials_bin_ms is None:
        trials_bin_ms = float(psth_bin_ms)

    # -----------------------------
    # helper: parse session_id + plane from stage1 name
    # stage1 example: psth_20221004_104619.0.npz
    # -----------------------------
    _re_stage1 = re.compile(r"psth_(?P<sid>.+?)\.(?P<plane>\d+)\.npz$")

    def _infer_animal_from_stage1_path(p: str) -> str:
        # common: .../stage1/<animal>/psth_....npz
        parts = os.path.normpath(p).split(os.sep)
        if "stage1" in parts:
            i = parts.index("stage1")
            if i + 1 < len(parts):
                return parts[i + 1]
        # fallback: unknown
        return ""

    def _expected_trials_basename(stage1_npz: str) -> str:
        b = os.path.basename(stage1_npz)
        m = _re_stage1.search(b)
        if not m:
            raise ValueError(f"Cannot parse session_id/plane from stage1 basename: {b}")
        sid = m.group("sid")
        plane = m.group("plane")
        return f"trials_{sid}.{plane}.npz"

    # -----------------------------
    # helper: load trials dict from trials_*.npz
    # -----------------------------
    # -----------------------------
    # helper: load trials dict + required meta from trials_*.npz
    # -----------------------------
    def _load_trials_from_trials_npz(trials_npz_path: str) -> tuple[dict, dict]:
        """Load trial-level arrays and minimal metadata.

        Expected in trials_*.npz:
          - one of `trial_keys`: dict[cond_name] -> np.ndarray with shape [C_keep, T_keep, nTr]
          - keep_idx: np.ndarray[int] length C_keep
          - cond_names: np.ndarray[str] length C (optional but strongly recommended)

        Returns:
          trials_dict: dict[cond_name] -> np.ndarray [C_keep, T_keep, nTr]
          trials_meta: dict with keys: keep_idx, cond_names
        """
        z = np.load(trials_npz_path, allow_pickle=True)

        # (a) trial dict
        trials_dict = None
        used_key = None
        for k in trial_keys:
            if k in z.files:
                used_key = k
                obj = z[k]
                if isinstance(obj, np.ndarray) and obj.dtype == object:
                    obj = obj.item()
                if not isinstance(obj, dict):
                    raise TypeError(
                        f"{trials_npz_path}: '{k}' must be a dict(cond->array), got {type(obj)}"
                    )
                trials_dict = obj
                break
        if trials_dict is None:
            raise KeyError(
                f"{trials_npz_path}: no trial dict found in keys={trial_keys}. "
                f"Available keys={list(z.files)[:60]}"
            )

        # (b) required metadata
        if "keep_idx" not in z.files:
            raise KeyError(
                f"{trials_npz_path}: missing required key 'keep_idx'. "
                "The trials exporter must save keep_idx to guarantee alignment with stage1." 
            )
        keep_idx = np.asarray(z["keep_idx"], dtype=int)

        cond_names = None
        if "cond_names" in z.files:
            cond_names = [str(x) for x in np.asarray(z["cond_names"]).tolist()]

        trials_meta = {
            "used_key": used_key,
            "keep_idx": keep_idx,
            "cond_names": cond_names,
        }
        return trials_dict, trials_meta

    def _raise_trials_alignment_error(
        *,
        stage1_npz: str,
        trials_npz: str,
        stage1_keep_idx: np.ndarray,
        trials_keep_idx: np.ndarray,
        trials_dict: dict,
        extra: str = "",
    ) -> None:
        """Raise a hard error with full debug payload (no guessing)."""
        cond_shapes: list[str] = []
        try:
            for k in list(trials_dict.keys()):
                try:
                    a = np.asarray(trials_dict[k])
                    cond_shapes.append(f"  - {k}: shape={tuple(a.shape)} dtype={a.dtype} ndim={a.ndim}")
                except Exception as e:
                    cond_shapes.append(f"  - {k}: <shape_read_failed> err={e}")
        except Exception as e:
            cond_shapes.append(f"  <cannot_list_trials_dict_keys> err={e}")

        msg_lines = [
            "Trials/stage1 alignment check FAILED (hard error).",
            f"stage1_npz: {stage1_npz}",
            f"trials_npz: {trials_npz}",
            f"len(stage1.keep_idx)={int(np.asarray(stage1_keep_idx).shape[0])}",
            f"len(trials.keep_idx)={int(np.asarray(trials_keep_idx).shape[0])}",
            "per-cond arrays in trials_dict:",
            *cond_shapes,
        ]
        if extra:
            msg_lines += ["", "Details:", str(extra)]
        raise ValueError("\n".join(msg_lines))

    # -----------------------------
    # helper: find trials_*.npz for a given stage1 npz
    # -----------------------------
    trials_index_by_basename = {}  # basename -> path (filled lazily per root)
    roots_indexed = set()

    def _index_root(root: str):
        if root in roots_indexed:
            return
        roots_indexed.add(root)
        if not root or (not os.path.isdir(root)):
            return
        # build a basename->path index (first hit wins)
        for p in glob.glob(os.path.join(root, "**", "trials_*.npz"), recursive=True):
            bn = os.path.basename(p)
            if bn not in trials_index_by_basename:
                trials_index_by_basename[bn] = p

    def _resolve_trials_npz(stage1_npz: str) -> str:
        bn = _expected_trials_basename(stage1_npz)

        # (1) same directory as stage1
        p1 = os.path.join(os.path.dirname(stage1_npz), bn)
        if os.path.isfile(p1):
            return p1

        # (2) user-provided roots
        if trials_roots is not None:
            for r in trials_roots:
                if not r:
                    continue
                p2 = os.path.join(r, bn)
                if os.path.isfile(p2):
                    return p2
            # if not found, build index for these roots and try by basename
            for r in trials_roots:
                _index_root(str(r))
            if bn in trials_index_by_basename and os.path.isfile(trials_index_by_basename[bn]):
                return trials_index_by_basename[bn]

        # (3) automatic roots (best-effort)
        animal_guess = _infer_animal_from_stage1_path(stage1_npz)
        auto_roots = []
        if animal_guess:
            auto_roots.append(f"/allen/aind/scratch/jingyi/2p/{animal_guess}")
        auto_roots.append("/allen/aind/scratch/jingyi/2p")

        for r in auto_roots:
            p3 = os.path.join(r, bn)
            if os.path.isfile(p3):
                return p3
        for r in auto_roots:
            _index_root(r)
        if bn in trials_index_by_basename and os.path.isfile(trials_index_by_basename[bn]):
            return trials_index_by_basename[bn]

        tried = [p1]
        if trials_roots is not None:
            tried += [os.path.join(str(r), bn) for r in trials_roots if r]
        tried += [os.path.join(r, bn) for r in auto_roots]
        raise FileNotFoundError(
            f"Cannot find trials file for stage1={stage1_npz}\n"
            f"Expected basename: {bn}\n"
            f"Tried (direct):\n  " + "\n  ".join(tried[:20]) + ("\n  ...(truncated)" if len(tried) > 20 else "") + "\n"
            f"Hint: pass trials_roots=[<dir containing trials_*.npz>] to train_current_alm_global, "
            f"or move/copy trials_*.npz into stage1 folder."
        )

    # -----------------------------
    # Pass 1: load each unit, build psth_sub (pre-clip), record (C,T,fps)
    # -----------------------------
    pre = []
    C_ref = None
    fps_ref = None
    g_max = -1
    T_min = None

    for unit_key, items in by_unit.items():
        if len(items) == 0:
            continue

        npz_path = str(items[0][2]["npz_path"])
        for (_, _, rr) in items[1:]:
            if str(rr["npz_path"]) != npz_path:
                raise ValueError(f"unit_key {unit_key} has inconsistent npz_path in registry.")

        psth, meta = load_alm_psth_npz(
            npz_path=npz_path,
            cond_filter=cond_filter,
            max_time=None,      # handle clipping ourselves
            device=device,
        )

        if max_time is not None:
            psth = psth[:, : int(max_time), :]

        keep_idx = meta.get("keep_idx", None)
        if keep_idx is None:
            raise KeyError(
                f"meta['keep_idx'] missing for {npz_path}. "
                "Stage1 npz must contain keep_idx for registry mapping."
            )
        keep_map = _build_keepidx_pos_map(np.asarray(keep_idx, dtype=int))

        g_list = []
        pos_list = []
        for (g, array_idx, _) in items:
            ai = int(array_idx)
            if ai not in keep_map:
                raise ValueError(
                    f"array_idx={ai} not found in keep_idx of {npz_path} (unit_key={unit_key}). "
                    "Check registry builder mapping."
                )
            pos_list.append(int(keep_map[ai]))
            g_list.append(int(g))

        if len(g_list) == 0:
            continue

        pos_t = tch.as_tensor(pos_list, dtype=tch.long, device=device)
        psth_sub = psth.index_select(dim=2, index=pos_t)  # [C,T,K]

        if psth_bin_ms is not None and float(psth_bin_ms) > 0:
            if smoother is None:
                raise NameError("No smoothing function found. Define _time_bin_smooth_psth or _time_bin_smooth_ctn.")
            fps = float(meta["fps"])
            psth_sub = smoother(psth_sub, fps=fps, bin_ms=float(psth_bin_ms))

        C = int(psth_sub.shape[0])
        T = int(psth_sub.shape[1])
        fps = float(meta["fps"])

        if C_ref is None:
            C_ref = C
        elif C != C_ref:
            raise ValueError(f"Inconsistent C across units: unit {unit_key} has C={C}, ref C={C_ref}")

        if fps_ref is None:
            fps_ref = fps
        else:
            if abs(fps - fps_ref) > 1e-3:
                raise ValueError(f"Inconsistent fps across units: unit {unit_key} fps={fps}, ref fps={fps_ref}")

        T_min = T if T_min is None else min(T_min, T)
        g_max = max(g_max, max(g_list))

        pre.append(
            dict(
                unit_key=unit_key,
                npz_path=npz_path,
                psth_sub=psth_sub,
                meta=meta,
                g_list=g_list,
                pos_list=pos_list,   # IMPORTANT: position in keep_idx for trial-cell selection
                T_ref=T,             # stage1 T (post max_time clip)
            )
        )

    if len(pre) == 0:
        raise ValueError("No units loaded from registry (by_unit is empty after filtering).")
    if T_min is None or C_ref is None or fps_ref is None:
        raise RuntimeError("Failed to determine shared (C,T,fps).")

    T_shared = int(T_min)
    N_inh_total = int(g_max + 1)
    N_total = int(n_exc_virtual) + N_inh_total

    # -----------------------------
    # Pass 2: clip, attach traces, build u/time_mask/idx_net (+ trials_sub if use_trials)
    # -----------------------------
    units = []
    trials_cache = {}  # stage1_npz -> {path, trials_dict, trials_keep_idx, trials_cond_names}

    for item in pre:
        unit_key = item["unit_key"]
        npz_path = item["npz_path"]
        meta = item["meta"]
        g_list = item["g_list"]
        pos_list = item["pos_list"]
        T_ref = int(item["T_ref"])

        psth_sub = item["psth_sub"][:, :T_shared, :]  # [C,T,K]

        meta = _maybe_attach_lick_reward_to_meta(meta, npz_path=npz_path, T=T_shared)

        cond_names = list(meta["cond_names"])
        if len(cond_names) != int(psth_sub.shape[0]):
            raise ValueError(
                f"cond_names length mismatch for {unit_key}: "
                f"len(cond_names)={len(cond_names)} but C={int(psth_sub.shape[0])}"
            )

        u = _build_input_tensor(
            C=len(cond_names),
            T=T_shared,
            cond_names=cond_names,
            device=device,
            amp_input=float(amp_input),
            include_go_cue=bool(include_go_cue),
            go_cue_sec=float(go_cue_sec),
            include_reward=bool(include_reward),
            meta=meta,
        )

        time_mask_np = _build_time_mask_sample_delay_resp(
            T=T_shared,
            fps=float(meta["fps"]),
            meta=meta,
            sample_ignore_ms=float(sample_ignore_ms),
            resp_sec=float(resp_sec),
        )
        time_mask = tch.as_tensor(time_mask_np.astype(np.bool_), device=device)

        idx_net = tch.as_tensor(
            [int(g) + int(n_exc_virtual) for g in g_list],
            dtype=tch.long,
            device=device,
        )

        # ---- NEW: trial-level targets from separate trials_*.npz ----
        trials_sub = None
        if bool(use_trials):
            # Resolve + load once per stage1 file
            if npz_path not in trials_cache:
                trials_npz = _resolve_trials_npz(npz_path)
                trials_dict, trials_meta = _load_trials_from_trials_npz(trials_npz)
                trials_cache[npz_path] = {
                    "path": trials_npz,
                    "trials_dict": trials_dict,
                    "trials_keep_idx": np.asarray(trials_meta["keep_idx"], dtype=int),
                    "trials_cond_names": trials_meta.get("cond_names", None),
                    "used_key": trials_meta.get("used_key", None),
                }

            cache = trials_cache[npz_path]
            trials_npz = str(cache["path"])
            trials_dict = cache["trials_dict"]
            trials_keep_idx = np.asarray(cache["trials_keep_idx"], dtype=int)

            # -------------------------
            # Strict alignment checks
            # -------------------------
            stage1_keep_idx = np.asarray(meta.get("keep_idx", None), dtype=int)
            if stage1_keep_idx is None:
                raise KeyError(f"{npz_path}: meta['keep_idx'] missing; cannot align trials.")

            if stage1_keep_idx.shape[0] != trials_keep_idx.shape[0]:
                _raise_trials_alignment_error(
                    stage1_npz=npz_path,
                    trials_npz=trials_npz,
                    stage1_keep_idx=stage1_keep_idx,
                    trials_keep_idx=trials_keep_idx,
                    trials_dict=trials_dict,
                    extra="keep_idx length mismatch",
                )

            if not np.array_equal(stage1_keep_idx, trials_keep_idx):
                _raise_trials_alignment_error(
                    stage1_npz=npz_path,
                    trials_npz=trials_npz,
                    stage1_keep_idx=stage1_keep_idx,
                    trials_keep_idx=trials_keep_idx,
                    trials_dict=trials_dict,
                    extra="keep_idx values mismatch (arrays are not identical)",
                )

            # Optional but helpful: cond_names consistency
            trials_cond_names = cache.get("trials_cond_names", None)
            if trials_cond_names is not None:
                if list(trials_cond_names) != list(cond_names):
                    _raise_trials_alignment_error(
                        stage1_npz=npz_path,
                        trials_npz=trials_npz,
                        stage1_keep_idx=stage1_keep_idx,
                        trials_keep_idx=trials_keep_idx,
                        trials_dict=trials_dict,
                        extra=(f"cond_names mismatch\n  stage1.cond_names={list(cond_names)}\n  trials.cond_names={list(trials_cond_names)}"),
                    )

            # Build per-condition [R, T, K] tensors
            trials_sub = {}
            K = int(len(pos_list))
            if K == 0:
                raise RuntimeError(f"{npz_path}: empty pos_list; registry mapping produced no observed neurons.")

            max_pos = int(max(pos_list))
            if max_pos >= int(trials_keep_idx.shape[0]):
                _raise_trials_alignment_error(
                    stage1_npz=npz_path,
                    trials_npz=trials_npz,
                    stage1_keep_idx=stage1_keep_idx,
                    trials_keep_idx=trials_keep_idx,
                    trials_dict=trials_dict,
                    extra=f"pos_list index out of range: max(pos_list)={max_pos} but C_keep={int(trials_keep_idx.shape[0])}",
                )

            for cname in cond_names:
                if cname not in trials_dict:
                    _raise_trials_alignment_error(
                        stage1_npz=npz_path,
                        trials_npz=trials_npz,
                        stage1_keep_idx=stage1_keep_idx,
                        trials_keep_idx=trials_keep_idx,
                        trials_dict=trials_dict,
                        extra=f"condition '{cname}' missing in trials_dict keys={list(trials_dict.keys())[:40]}",
                    )

                arr = np.asarray(trials_dict[cname])
                if arr.ndim != 3:
                    _raise_trials_alignment_error(
                        stage1_npz=npz_path,
                        trials_npz=trials_npz,
                        stage1_keep_idx=stage1_keep_idx,
                        trials_keep_idx=trials_keep_idx,
                        trials_dict=trials_dict,
                        extra=f"trials_dict[{cname}] must be 3D [C_keep,T_keep,nTr], got shape={arr.shape}",
                    )

                Ck, Tk, Rk = [int(x) for x in arr.shape]
                if Ck != int(stage1_keep_idx.shape[0]):
                    _raise_trials_alignment_error(
                        stage1_npz=npz_path,
                        trials_npz=trials_npz,
                        stage1_keep_idx=stage1_keep_idx,
                        trials_keep_idx=trials_keep_idx,
                        trials_dict=trials_dict,
                        extra=f"C_keep mismatch in trials array for cond={cname}: Ck={Ck} vs len(keep_idx)={int(stage1_keep_idx.shape[0])}",
                    )
                if Tk < T_shared:
                    _raise_trials_alignment_error(
                        stage1_npz=npz_path,
                        trials_npz=trials_npz,
                        stage1_keep_idx=stage1_keep_idx,
                        trials_keep_idx=trials_keep_idx,
                        trials_dict=trials_dict,
                        extra=f"time dim too short in trials array for cond={cname}: Tk={Tk} < T_shared={T_shared}",
                    )
                if Rk <= 0:
                    _raise_trials_alignment_error(
                        stage1_npz=npz_path,
                        trials_npz=trials_npz,
                        stage1_keep_idx=stage1_keep_idx,
                        trials_keep_idx=trials_keep_idx,
                        trials_dict=trials_dict,
                        extra=f"nTr must be >0 for cond={cname}, got Rk={Rk}",
                    )

                # exporter saves [C_keep, T_keep, nTr] (cells,time,trials)
                arr_k = arr[pos_list, :T_shared, :]          # [K, T, R]
                arr_rtk = np.transpose(arr_k, (2, 1, 0))     # [R, T, K]

                xt = tch.as_tensor(arr_rtk, dtype=tch.float32, device=device)

                if trials_bin_ms is not None and float(trials_bin_ms) > 0:
                    if smoother is None:
                        raise NameError("No smoothing function found. Define _time_bin_smooth_psth or _time_bin_smooth_ctn.")
                    xt = smoother(xt, fps=float(meta["fps"]), bin_ms=float(trials_bin_ms))

                trials_sub[cname] = xt

            # Final guard
            if len(trials_sub) != len(cond_names):
                _raise_trials_alignment_error(
                    stage1_npz=npz_path,
                    trials_npz=trials_npz,
                    stage1_keep_idx=stage1_keep_idx,
                    trials_keep_idx=trials_keep_idx,
                    trials_dict=trials_dict,
                    extra=f"trials_sub missing some conditions: got={list(trials_sub.keys())} expected={list(cond_names)}",
                )

        # ---- NEW: cell-type labels for the observed K neurons (string groups) ----
        # We use stage1_npz as source of labels; labels are aligned to keep_idx first.
        keep_idx_arr = np.asarray(meta.get("keep_idx"), dtype=int)
        if keep_idx_arr.size == 0:
            raise RuntimeError(f"{npz_path}: meta['keep_idx'] missing/empty; cannot align cell_subclasses.")

        try:
            labels_keep = _extract_celltype_labels_keep(
                npz_path=npz_path,
                keep_idx=keep_idx_arr,
                label_key="cell_subclasses",
                allow_pickle=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load/align cell_subclasses for cell-type loss.\n"
                f"  stage1_npz: {npz_path}\n"
                f"  unit_key:   {unit_key}\n"
                "  hint: ensure stage1 npz contains a 1D 'cell_subclasses' and that keep_idx aligns.\n"
                f"  original error: {e}"
            )

        # pos_list indexes into keep-axis positions (0..C_keep-1)
        labels_K = np.asarray([str(labels_keep[int(p)]) for p in pos_list], dtype=object)

        # Build boolean masks per subclass (exclude unknown)
        type_masks: Dict[str, tch.Tensor] = {}
        type_counts: Dict[str, int] = {}
        for lab in sorted(set(labels_K.tolist())):
            if str(lab).strip().lower() in {"unknown", "nan", "none", ""}:
                continue
            msk = tch.as_tensor([str(x) == str(lab) for x in labels_K], dtype=tch.bool, device=device)
            c = int(msk.sum().item())
            if c > 0:
                type_masks[str(lab)] = msk
                type_counts[str(lab)] = c
        type_names = list(type_masks.keys())

        units.append(
            dict(
                unit_key=unit_key,
                npz_path=npz_path,
                psth_sub=psth_sub,
                trials_sub=trials_sub,   # None if use_trials=False
                u=u,
                idx_net=idx_net,
                time_mask=time_mask,
                meta=meta,
                # cell-type grouping info for K observed neurons
                subclasses_K=labels_K.tolist(),
                type_names=type_names,
                type_masks=type_masks,
                type_counts=type_counts,
            )
        )

    if len(units) == 0:
        raise RuntimeError("No units left after preload (possibly all missing trials files).")

    shared = dict(C=int(C_ref), T=int(T_shared), fps=float(fps_ref), N_inh_total=N_inh_total, N_total=N_total)
    return units, shared




def train_current_alm_global(
    registry_dir: str,
    animal: str,
    out_dir: str,
    *,
    max_sessions: Optional[int] = None,
    seed: int = 42,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_epochs: int = 200000,
    grad_clip: Optional[float] = 1.0,
    unit_sampling: str = "random",  # random|cycle
    # model
    dt: float = 1e-2,
    tau: float = 5e-2,
    substeps: int = 1,
    nonlinearity: str = "relu",
    dale: bool = True,
    n_exc_virtual: int = 800,
    # inputs
    amp_input: float = 1.0,
    include_go_cue: bool = True,
    go_cue_sec: float = 0.10,
    include_reward: bool = True,
    # data selection
    cond_filter=None,
    max_time=None,
    psth_bin_ms: float = 200.0,
    # time masking
    sample_ignore_ms: float = 50.0,
    resp_sec: float = 2.0,
    # trials + losses
    use_trials: bool = True,
    trial_keys: tuple[str, ...] = ("cell_trials",),
    trial_batch_per_cond: int = 0,      # 0 means use all trials
    trials_bin_ms: float = 0.0,
    noise_std: float = 0.03,
    noise_std_eval: float = 0.0,
    # cell-type loss
    lambda_celltype: float = 0.1,
    celltype_label_key: str = "cell_subclasses",  # string labels
    celltype_exclude: tuple[str, ...] = ("", "nan", "none", "unknown"),
    # logging
    log_celltype_every: int = 10,
    log_celltype_topk: int = 20,
    # bookkeeping
    eval_every: int = 1,
    save_latest_every: int = 1,
    save_best_every: int = 50,
):
    """Global (registry-based) training.

    Requested loss structure:
      1) Do a *single trial-level forward* per step (with process noise) using trials_sub.
      2) PSTH loss: average predictions over trials *per condition* and fit stage1 condition-average PSTH via L2.
      3) Cell-type loss: for each condition and each cell_subclass group, average over neurons in the group
         (NOT over trials) and apply L2 on trial-level population traces.

    Notes:
      - This function assumes trials_*.npz exist for each stage1 session and will raise if missing.
      - Cell-type labels are assumed to be string arrays, sourced from meta[celltype_label_key] or loaded from stage1 npz.
    """

    import os, time, json
    import numpy as np
    import pandas as pd
    import torch as tch

    if not bool(use_trials):
        raise ValueError(
            "This training mode requires use_trials=True because PSTH is fitted from trial predictions (trial-mean)."
        )

    os.makedirs(out_dir, exist_ok=True)

    dev = tch.device("cuda" if tch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(int(seed))

    def _atomic_torch_save(obj, path: str):
        tmp = path + ".tmp"
        tch.save(obj, tmp)
        os.replace(tmp, path)

    def _normalize_str_label(x) -> str:
        if x is None:
            return "none"
        try:
            s = str(x)
        except Exception:
            return "none"
        s2 = s.strip()
        return s2

    def _load_cell_labels_for_keep(npz_path: str, keep_idx: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        """Return labels_keep with length C_keep aligned to keep_idx positions [0..C_keep-1]."""
        # 1) try meta
        labels = meta.get(celltype_label_key, None)
        if labels is None:
            try:
                z = np.load(npz_path, allow_pickle=True)
                if celltype_label_key in z.files:
                    labels = z[celltype_label_key]
            except Exception:
                labels = None

        if labels is None:
            raise KeyError(
                f"{npz_path}: cannot find '{celltype_label_key}' in meta or stage1 npz. "
                "Cell-type loss requires this label array."
            )

        # unpack object arrays
        if isinstance(labels, np.ndarray) and labels.dtype == object:
            try:
                labels = labels.tolist()
            except Exception:
                pass

        labels_arr = np.asarray(labels)

        # If labels already aligned to keep (len==C_keep), keep it.
        C_keep = int(keep_idx.shape[0])
        if labels_arr.shape[0] == C_keep:
            labels_keep = labels_arr
        else:
            # Otherwise treat as full-cell labels and subset by keep_idx
            if int(keep_idx.max()) >= int(labels_arr.shape[0]):
                raise ValueError(
                    f"{npz_path}: '{celltype_label_key}' length={int(labels_arr.shape[0])} but keep_idx.max={int(keep_idx.max())}. "
                    "Cannot subset labels by keep_idx."
                )
            labels_keep = labels_arr[keep_idx]

        # normalize to string array
        out = np.array([_normalize_str_label(x) for x in labels_keep], dtype=object)
        return out

    def _build_trial_batch_with_cond_id(
        u_cond: tch.Tensor,
        trials_sub: Dict[str, tch.Tensor],
        cond_names: List[str],
    ) -> tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
        """Build trial-level batch.

        Args:
          u_cond: [C,T,D]
          trials_sub[cname]: [R,T,K]

        Returns:
          u_trials: [B,T,D]
          y_trials: [B,T,K]
          cond_id:  [B] int64 in [0..C-1]
        """
        u_list, y_list, c_list = [], [], []
        C, T, D = u_cond.shape

        for ci, cname in enumerate(cond_names):
            if cname not in trials_sub:
                continue
            Y = trials_sub[cname]  # [R,T,K]
            R = int(Y.shape[0])
            if int(trial_batch_per_cond) > 0 and R > int(trial_batch_per_cond):
                idx = tch.randint(low=0, high=R, size=(int(trial_batch_per_cond),), device=Y.device)
                Yb = Y.index_select(dim=0, index=idx)
            else:
                Yb = Y
            Bc = int(Yb.shape[0])
            Ub = u_cond[ci : ci + 1, :, :].expand(Bc, T, D)
            u_list.append(Ub)
            y_list.append(Yb)
            c_list.append(tch.full((Bc,), int(ci), dtype=tch.long, device=Y.device))

        if len(u_list) == 0:
            raise RuntimeError("No trial data found for any condition in cond_names (after filtering).")

        u_trials = tch.cat(u_list, dim=0)
        y_trials = tch.cat(y_list, dim=0)
        cond_id = tch.cat(c_list, dim=0)
        return u_trials, y_trials, cond_id

    def _masked_mse_per_neuron_from_psth_trialmean(
        pred_sub: tch.Tensor,   # [B,T,K]
        cond_id: tch.Tensor,    # [B]
        psth_sub: tch.Tensor,   # [C,T,K]
        time_mask: tch.Tensor,  # [T] bool or indices
        C: int,
    ) -> tuple[tch.Tensor, tch.Tensor]:
        """Compute neuron-wise MSE between per-condition trial-mean predictions and PSTH target.

        Returns:
          mse_per_neuron: [K]
          mse_sum_neurons: scalar (sum over neurons)
        """
        if time_mask.dtype == tch.bool:
            pred_m = pred_sub[:, time_mask, :]   # [B,Tm,K]
            targ_psth_m = psth_sub[:, time_mask, :]  # [C,Tm,K]
        else:
            pred_m = pred_sub.index_select(dim=1, index=time_mask)
            targ_psth_m = psth_sub.index_select(dim=1, index=time_mask)

        mse_per_neuron = tch.zeros((pred_sub.shape[2],), device=pred_sub.device, dtype=tch.float32)
        n_used = 0
        for ci in range(int(C)):
            sel = (cond_id == int(ci))
            if not bool(sel.any()):
                continue
            pm = pred_m[sel].mean(dim=0)          # [Tm,K]
            tm = targ_psth_m[ci]                  # [Tm,K]
            diff2 = (pm - tm).pow(2)              # [Tm,K]
            mse_per_neuron = mse_per_neuron + diff2.mean(dim=0)  # [K]
            n_used += 1

        if n_used == 0:
            raise RuntimeError("No trials for any condition when computing PSTH loss.")

        mse_per_neuron = mse_per_neuron / float(n_used)
        mse_sum_neurons = mse_per_neuron.sum()
        return mse_per_neuron, mse_sum_neurons

    def _celltype_sum_neurons_triallevel(
        pred_sub: tch.Tensor,      # [B,T,K]
        targ_sub: tch.Tensor,      # [B,T,K]
        cond_id: tch.Tensor,       # [B]
        time_mask: tch.Tensor,     # [T] bool or indices
        type_masks: Dict[str, tch.Tensor],  # type -> bool[K]
        C: int,
    ) -> tch.Tensor:
        """Neuron-weighted cell-type loss on trial-level data.

        For each condition and each cell type g:
          pop_pred[r,t] = mean_k in g pred[r,t,k]
          pop_targ[r,t] = mean_k in g targ[r,t,k]
          mse_g = mean_{r,t}( (pop_pred-pop_targ)^2 )
        Accumulate neuron-weighted sum: sum_g (n_g * mse_g), averaged over conditions.
        """
        if time_mask.dtype == tch.bool:
            pred_m = pred_sub[:, time_mask, :]
            targ_m = targ_sub[:, time_mask, :]
        else:
            pred_m = pred_sub.index_select(dim=1, index=time_mask)
            targ_m = targ_sub.index_select(dim=1, index=time_mask)

        total = tch.zeros((), device=pred_sub.device, dtype=tch.float32)
        n_used_cond = 0

        # precompute float masks
        float_masks = {}
        counts = {}
        for name, m in type_masks.items():
            if m.dtype != tch.bool:
                m = m.bool()
            n = int(m.sum().item())
            if n <= 0:
                continue
            float_masks[name] = m.to(dtype=tch.float32).view(1, 1, -1)  # [1,1,K]
            counts[name] = tch.tensor(float(n), device=pred_sub.device, dtype=tch.float32)

        if len(float_masks) == 0:
            return total

        for ci in range(int(C)):
            sel = (cond_id == int(ci))
            if not bool(sel.any()):
                continue
            p_ci = pred_m[sel]  # [Rci,Tm,K]
            t_ci = targ_m[sel]  # [Rci,Tm,K]

            cond_sum = tch.zeros((), device=pred_sub.device, dtype=tch.float32)
            for name, mf in float_masks.items():
                n = counts[name]
                pop_p = (p_ci * mf).sum(dim=-1) / n  # [Rci,Tm]
                pop_t = (t_ci * mf).sum(dim=-1) / n
                mse_g = (pop_p - pop_t).pow(2).mean()
                cond_sum = cond_sum + n * mse_g

            total = total + cond_sum
            n_used_cond += 1

        if n_used_cond == 0:
            return tch.zeros((), device=pred_sub.device, dtype=tch.float32)
        return total / float(n_used_cond)

    def _celltype_metrics_triallevel(
        pred_sub: tch.Tensor,      # [B,T,K]
        targ_sub: tch.Tensor,      # [B,T,K]
        cond_id: tch.Tensor,       # [B]
        time_mask: tch.Tensor,     # [T] bool or indices
        type_masks: Dict[str, tch.Tensor],  # type -> bool[K]
        C: int,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-celltype metrics for logging.

        Returns dict: {type_name: {"n_cells": int, "mse": float}}
        where mse is averaged over trials and time, and then averaged over conditions that have trials.
        """
        if time_mask.dtype == tch.bool:
            pred_m = pred_sub[:, time_mask, :]
            targ_m = targ_sub[:, time_mask, :]
        else:
            pred_m = pred_sub.index_select(dim=1, index=time_mask)
            targ_m = targ_sub.index_select(dim=1, index=time_mask)

        out: Dict[str, Dict[str, float]] = {}

        # prepare masks + counts
        float_masks = {}
        counts = {}
        for name, m in type_masks.items():
            if m.dtype != tch.bool:
                m = m.bool()
            n = int(m.sum().item())
            if n <= 0:
                continue
            float_masks[name] = m.to(dtype=tch.float32).view(1, 1, -1)
            counts[name] = float(n)

        if len(float_masks) == 0:
            return out

        # compute per type mse (condition-averaged)
        for name, mf in float_masks.items():
            n = counts[name]
            mse_sum = 0.0
            n_cond = 0
            for ci in range(int(C)):
                sel = (cond_id == int(ci))
                if not bool(sel.any()):
                    continue
                p_ci = pred_m[sel]  # [Rci,Tm,K]
                t_ci = targ_m[sel]
                pop_p = (p_ci * mf).sum(dim=-1) / float(n)  # [Rci,Tm]
                pop_t = (t_ci * mf).sum(dim=-1) / float(n)
                mse_g = (pop_p - pop_t).pow(2).mean()
                mse_sum += float(mse_g.detach().cpu().item())
                n_cond += 1
            if n_cond > 0:
                out[name] = {"n_cells": float(n), "mse": mse_sum / float(n_cond)}
        return out

    

    def _celltype_metrics_for_logging(
        pred_sub: tch.Tensor,      # [B,T,K]
        targ_sub: tch.Tensor,      # [B,T,K]
        cond_id: tch.Tensor,       # [B]
        time_mask: tch.Tensor,     # [T] bool or indices
        type_masks: Dict[str, tch.Tensor],  # type -> bool[K]
        C: int,
    ) -> Dict[str, Any]:
        """Return detailed per-celltype metrics for logging.

        Outputs:
          {
            'n_effective': int,
            'counts': {type: int},
            'mse': {type: float},   # condition-averaged MSE of population traces
          }
        """
        if time_mask.dtype == tch.bool:
            pred_m = pred_sub[:, time_mask, :]
            targ_m = targ_sub[:, time_mask, :]
        else:
            pred_m = pred_sub.index_select(dim=1, index=time_mask)
            targ_m = targ_sub.index_select(dim=1, index=time_mask)

        # precompute float masks + counts
        float_masks = {}
        counts = {}
        for name, m in type_masks.items():
            if m.dtype != tch.bool:
                m = m.bool()
            n = int(m.sum().item())
            if n <= 0:
                continue
            float_masks[name] = m.to(dtype=tch.float32).view(1, 1, -1)  # [1,1,K]
            counts[name] = n

        mse_out: Dict[str, float] = {}
        if len(float_masks) == 0:
            return {"n_effective": 0, "counts": {}, "mse": {}}

        for name, mf in float_masks.items():
            n_used_cond = 0
            mse_sum = 0.0
            n = float(counts[name])
            for ci in range(int(C)):
                sel = (cond_id == int(ci))
                if not bool(sel.any()):
                    continue
                p_ci = pred_m[sel]
                t_ci = targ_m[sel]
                pop_p = (p_ci * mf).sum(dim=-1) / n
                pop_t = (t_ci * mf).sum(dim=-1) / n
                mse_g = (pop_p - pop_t).pow(2).mean()
                mse_sum += float(mse_g.detach().cpu().item())
                n_used_cond += 1
            if n_used_cond > 0:
                mse_out[name] = mse_sum / float(n_used_cond)

        return {
            "n_effective": int(len(mse_out)),
            "counts": {k: int(v) for k, v in counts.items()},
            "mse": mse_out,
        }

    def _build_type_masks_from_labels(labels_K: np.ndarray, device: tch.device) -> Dict[str, tch.Tensor]:
        # labels_K: object/str array length K
        exc = set([s.lower() for s in celltype_exclude])
        masks: Dict[str, tch.Tensor] = {}
        uniq = []
        for x in labels_K.tolist():
            s = _normalize_str_label(x)
            if s.lower() in exc:
                continue
            if s not in uniq:
                uniq.append(s)
        for s in uniq:
            m = np.array([_normalize_str_label(x) == s for x in labels_K.tolist()], dtype=bool)
            masks[s] = tch.as_tensor(m, dtype=tch.bool, device=device)
        return masks

    def _eval_combined_loss(net, units, *, noise_std_eval: float) -> Dict[str, float]:
        net.eval()
        total_neurons = 0
        sum_psth_neurons = 0.0
        sum_type_neurons = 0.0
        sum_total_neurons = 0.0

        with tch.no_grad():
            for batch in units:
                u_cond = batch["u"]
                psth_sub = batch["psth_sub"]
                idx_net = batch["idx_net"]
                time_mask = batch["time_mask"]

                trials_sub = batch.get("trials_sub", None)
                if trials_sub is None:
                    raise KeyError(f"unit_key={batch.get('unit_key','NA')}: trials_sub missing; eval requires trial data.")

                cond_names = list(batch["meta"]["cond_names"])
                u_trials, y_trials, cond_id = _build_trial_batch_with_cond_id(u_cond=u_cond, trials_sub=trials_sub, cond_names=cond_names)

                u_trials = tch.nan_to_num(u_trials, nan=0.0, posinf=0.0, neginf=0.0)
                y_trials = tch.nan_to_num(y_trials, nan=0.0, posinf=0.0, neginf=0.0)

                out = net(u_trials, h0=None, noise_std=float(noise_std_eval), return_rate=True)
                rates_full = out["rate"]
                pred_sub = rates_full.index_select(dim=2, index=idx_net)

                C = int(psth_sub.shape[0])
                mse_per_neuron, psth_sum = _masked_mse_per_neuron_from_psth_trialmean(
                    pred_sub=pred_sub,
                    cond_id=cond_id,
                    psth_sub=psth_sub,
                    time_mask=time_mask,
                    C=C,
                )

                type_masks = batch.get("type_masks", None)
                if type_masks is None:
                    type_masks = _build_type_masks_from_labels(np.asarray(batch.get("subclasses_K", []), dtype=object), device=dev)

                type_sum = _celltype_sum_neurons_triallevel(
                    pred_sub=pred_sub,
                    targ_sub=y_trials,
                    cond_id=cond_id,
                    time_mask=time_mask,
                    type_masks=type_masks,
                    C=C,
                )

                K = int(mse_per_neuron.numel())
                total_neurons += K
                sum_psth_neurons += float(psth_sum.detach().cpu().item())
                sum_type_neurons += float(type_sum.detach().cpu().item())
                sum_total_neurons += float((psth_sum + float(lambda_celltype) * type_sum).detach().cpu().item())

        denom = max(total_neurons, 1)
        return {
            "psth": sum_psth_neurons / float(denom),
            "type": sum_type_neurons / float(denom),
            "total": sum_total_neurons / float(denom),
        }

    # ----------------------------
    # 1) Load registry
    # ----------------------------
    registry_csv = os.path.join(registry_dir, f"{animal}_registry.csv")
    if not os.path.isfile(registry_csv):
        registry_csv = os.path.join(registry_dir, "registry.csv")
    if not os.path.isfile(registry_csv):
        raise FileNotFoundError(
            f"Cannot find registry csv in {registry_dir} (tried {animal}_registry.csv and registry.csv)"
        )

    df = pd.read_csv(registry_csv)

    required_cols = ["unit_key", "global_idx", "array_idx", "npz_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Registry missing columns: {missing}. Got columns={list(df.columns)}")

    by_unit: Dict[str, List[Any]] = {}
    for _, r in df.iterrows():
        uk = str(r["unit_key"])
        g = int(r["global_idx"])
        a = int(r["array_idx"])
        by_unit.setdefault(uk, []).append((g, a, dict(r)))

    unit_keys = sorted(by_unit.keys())
    if max_sessions is not None and int(max_sessions) > 0:
        unit_keys = unit_keys[: int(max_sessions)]
        by_unit = {k: by_unit[k] for k in unit_keys}

    # ----------------------------
    # 2) Preload unit tensors (+ trials_sub)
    # ----------------------------
    units, shared = _preload_units_from_registry(
        by_unit=by_unit,
        n_exc_virtual=int(n_exc_virtual),
        device=dev,
        cond_filter=cond_filter,
        max_time=max_time,
        psth_bin_ms=psth_bin_ms,
        sample_ignore_ms=sample_ignore_ms,
        resp_sec=float(resp_sec),
        use_trials=True,
        trial_keys=tuple(trial_keys),
        trials_bin_ms=trials_bin_ms,
    )

    if len(units) == 0:
        raise RuntimeError("No units loaded from registry. Check registry csv and filters.")

    # observed inhibitory count
    all_g = []
    for u0 in units:
        all_g.append((u0["idx_net"].detach().cpu().numpy() - int(n_exc_virtual)).astype(int))
    all_g = np.concatenate(all_g, axis=0)
    n_obs = int(all_g.max()) + 1
    n_total = int(n_exc_virtual) + int(n_obs)

    D_in = int(units[0]["u"].shape[-1])

    # ----------------------------
    # 3) Dale mask
    # ----------------------------
    dale_mask = None
    if bool(dale):
        dale_mask = tch.zeros((n_total, n_total), dtype=tch.int8, device=dev)
        dale_mask[:, : int(n_exc_virtual)] = 1
        dale_mask[:, int(n_exc_virtual) :] = -1

    # ----------------------------
    # 4) Build model
    # ----------------------------
    net = ALMCurrentRNN(
        N=int(n_total),
        D_in=int(D_in),
        dt=float(dt),
        tau=float(tau),
        substeps=int(substeps),
        nonlinearity=str(nonlinearity),
        device=dev,
        dale_mask=dale_mask,
    ).to(dev)

    opt = tch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    tag = f"{animal}_global_nobs{n_obs}_nexc{int(n_exc_virtual)}_ntotal{n_total}"
    best_path = os.path.join(out_dir, f"rnn_current_{tag}.best.pt")
    latest_path = os.path.join(out_dir, f"rnn_current_{tag}.latest.pt")
    meta_path = os.path.join(out_dir, f"rnn_current_{tag}.meta.json")

    # ----------------------------
    # 5) Training loop
    # ----------------------------
    best_eval_total = float("inf")
    best_state = None
    unit_order = list(range(len(units)))

    t0 = time.time()
    for ep in range(int(max_epochs)):
        net.train()
        opt.zero_grad(set_to_none=True)

        if unit_sampling == "random":
            rng.shuffle(unit_order)
        elif unit_sampling == "cycle":
            pass
        else:
            raise ValueError(f"unit_sampling must be 'random' or 'cycle', got {unit_sampling}")

        total_neurons = 0
        sum_total_neurons_detached = 0.0
        sum_psth_neurons_detached = 0.0
        sum_type_neurons_detached = 0.0

        for ui in unit_order:
            batch = units[ui]
            u_cond = batch["u"]
            psth_sub = batch["psth_sub"]
            idx_net = batch["idx_net"]
            time_mask = batch["time_mask"]
            trials_sub = batch["trials_sub"]
            cond_names = list(batch["meta"]["cond_names"])
            type_masks = batch["type_masks"]

            # build combined trial batch
            u_trials, y_trials, cond_id = _build_trial_batch_with_cond_id(u_cond=u_cond, trials_sub=trials_sub, cond_names=cond_names)

            u_trials = tch.nan_to_num(u_trials, nan=0.0, posinf=0.0, neginf=0.0)
            y_trials = tch.nan_to_num(y_trials, nan=0.0, posinf=0.0, neginf=0.0)

            out = net(u_trials, h0=None, noise_std=float(noise_std), return_rate=True)
            rates_full = out["rate"]
            pred_sub = rates_full.index_select(dim=2, index=idx_net)  # [B,T,K]

            C = int(psth_sub.shape[0])
            mse_per_neuron, psth_sum = _masked_mse_per_neuron_from_psth_trialmean(
                pred_sub=pred_sub,
                cond_id=cond_id,
                psth_sub=psth_sub,
                time_mask=time_mask,
                C=C,
            )

            type_sum = _celltype_sum_neurons_triallevel(
                pred_sub=pred_sub,
                targ_sub=y_trials,
                cond_id=cond_id,
                time_mask=time_mask,
                type_masks=type_masks,
                C=C,
            )

            total_sum = psth_sum + float(lambda_celltype) * type_sum

            # ---- cell-type diagnostics ----
            if int(log_celltype_every) > 0 and (ep == 0 or ((ep + 1) % int(log_celltype_every) == 0)):
                with tch.no_grad():
                    try:
                        mlog = _celltype_metrics_for_logging(
                            pred_sub=pred_sub.detach(),
                            targ_sub=y_trials.detach(),
                            cond_id=cond_id.detach(),
                            time_mask=time_mask,
                            type_masks=type_masks,
                            C=C,
                        )
                        counts = mlog['counts']
                        mse_by = mlog['mse']
                        n_eff = int(mlog['n_effective'])
                        # sort by cell count
                        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                        topk = items[: int(log_celltype_topk)] if int(log_celltype_topk) > 0 else items
                        parts = []
                        for name, n in topk:
                            mse = float(mse_by.get(name, float('nan')))
                            parts.append(f"{name}(n={n},mse={mse:.4g})")
                        uk = str(batch.get('unit_key', 'NA'))
                        print(f"[celltype] ep={ep+1:06d} unit={uk} K={int(pred_sub.shape[2])} n_types={n_eff} :: " + ', '.join(parts), flush=True)
                    except Exception as e:
                        uk = str(batch.get('unit_key', 'NA'))
                        print(f"[celltype] ep={ep+1:06d} unit={uk} (logging failed): {e}", flush=True)

            # ---- celltype logging (per unit) ----
            if int(log_celltype_every) > 0 and ((ep == 0) or ((ep + 1) % int(log_celltype_every) == 0)):
                with tch.no_grad():
                    uk = str(batch.get("unit_key", "NA"))
                    # type distribution
                    dist = {k: int(v.sum().detach().cpu().item()) for k, v in type_masks.items()}
                    # effective types
                    eff = {k: v for k, v in dist.items() if v > 0}
                    metrics = _celltype_metrics_triallevel(
                        pred_sub=pred_sub.detach(),
                        targ_sub=y_trials.detach(),
                        cond_id=cond_id.detach(),
                        time_mask=time_mask,
                        type_masks=type_masks,
                        C=C,
                    )
                    # sort by count desc
                    items = sorted(((k, eff[k], metrics.get(k, {}).get("mse", float('nan'))) for k in eff.keys()), key=lambda x: (-x[1], str(x[0])))
                    topk = int(log_celltype_topk)
                    if topk > 0:
                        items = items[:topk]
                    parts = [f"{k}(n={n},mse={mse:.6g})" for k, n, mse in items]
                    print(
                        f"[celltype][ep {ep+1:06d}] unit={uk} n_types={len(eff)} dist_top={parts}",
                        flush=True,
                    )

            if not tch.isfinite(total_sum):
                with tch.no_grad():
                    msg = {
                        "ep": ep + 1,
                        "unit_key": batch.get("unit_key", "NA"),
                        "psth_sum": float(psth_sum.detach().cpu().item()),
                        "type_sum": float(type_sum.detach().cpu().item()),
                        "lambda_celltype": float(lambda_celltype),
                        "u_minmax": (float(u_trials.min().cpu()), float(u_trials.max().cpu())),
                        "targ_minmax": (float(y_trials.min().cpu()), float(y_trials.max().cpu())),
                        "rate_minmax": (float(rates_full.min().cpu()), float(rates_full.max().cpu())),
                        "J_norm": float(net.J.norm().detach().cpu()),
                        "W_norm": float(net.W_in.norm().detach().cpu()),
                        "dt_over_tau": float(net.dt / net.tau),
                        "substeps": int(getattr(net, "substeps", 1)),
                    }
                raise FloatingPointError(f"Non-finite loss detected:\n{json.dumps(msg, indent=2)}")

            total_sum.backward()

            K = int(mse_per_neuron.numel())
            total_neurons += K
            sum_total_neurons_detached += float(total_sum.detach().cpu().item())
            sum_psth_neurons_detached += float(psth_sum.detach().cpu().item())
            sum_type_neurons_detached += float(type_sum.detach().cpu().item())

        denom = max(total_neurons, 1)
        for p in net.parameters():
            if p.grad is not None:
                p.grad.div_(float(denom))

        if grad_clip is not None and float(grad_clip) > 0:
            tch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))

        opt.step()

        if bool(dale):
            net.apply_dale_mask()

        train_total = sum_total_neurons_detached / float(denom)
        train_psth = sum_psth_neurons_detached / float(denom)
        train_type = sum_type_neurons_detached / float(denom)

        # ---- eval ----
        eval_dict = None
        if int(eval_every) > 0 and ((ep + 1) % int(eval_every) == 0):
            eval_dict = _eval_combined_loss(net, units, noise_std_eval=float(noise_std_eval))

            if float(eval_dict["total"]) < best_eval_total:
                best_eval_total = float(eval_dict["total"])
                best_state = {
                    "model": net.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": int(ep + 1),
                    "eval": eval_dict,
                    "tag": tag,
                }

        # ---- saves ----
        if int(save_latest_every) > 0 and ((ep + 1) % int(save_latest_every) == 0):
            _atomic_torch_save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": int(ep + 1)}, latest_path)

        if best_state is not None and int(save_best_every) > 0 and ((ep + 1) % int(save_best_every) == 0):
            _atomic_torch_save(best_state, best_path)

        # ---- meta log ----
        if (ep == 0) or ((ep + 1) % 10 == 0) or ((ep + 1) == int(max_epochs)):
            now = time.time()
            meta_out = {
                "animal": animal,
                "epoch": int(ep + 1),
                "elapsed_sec": float(now - t0),
                "train": {"total": float(train_total), "psth": float(train_psth), "type": float(train_type)},
                "eval": eval_dict,
                "best_eval_total": float(best_eval_total),
                "lambda_celltype": float(lambda_celltype),
                "noise_std": float(noise_std),
                "noise_std_eval": float(noise_std_eval),
                "n_obs": int(n_obs),
                "n_exc_virtual": int(n_exc_virtual),
                "n_total": int(n_total),
                "D_in": int(D_in),
            }
            with open(meta_path, "w") as f:
                json.dump(meta_out, f, indent=2)

            # console
            if eval_dict is not None:
                print(
                    f"[ep {ep+1:06d}] train total={train_total:.6f} psth={train_psth:.6f} type={train_type:.6f} | "
                    f"eval total={eval_dict['total']:.6f} psth={eval_dict['psth']:.6f} type={eval_dict['type']:.6f} | "
                    f"best={best_eval_total:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[ep {ep+1:06d}] train total={train_total:.6f} psth={train_psth:.6f} type={train_type:.6f} | best={best_eval_total:.6f}",
                    flush=True,
                )

    # final best save
    if best_state is not None:
        _atomic_torch_save(best_state, best_path)

    return {
        "best_path": best_path,
        "latest_path": latest_path,
        "meta_path": meta_path,
        "best_eval_total": float(best_eval_total),
    }
