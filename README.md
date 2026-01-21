code_rnn/
├─ parameters_list.json          
├─ rnn_digest.txt                

├─ alm_data/                     
│  ├─ 0.average.py               
│  ├─ bpod_parse.py              
│  ├─ utils_alm.py               

├─ legacy_rnn/                   # original SingleTrialLicks, for reference
│  ├─ model.py
│  ├─ training.py
│  ├─ main_train.py
│  ├─ main_test.py
│  ├─ cross_validating.py
│  ├─ utils.py
│  ├─ losses.py                  
│  ├─ plotting.py                

├─ current_rnn/                  # current-based RNN for ALM
│  ├─ model_current.py           
│  ├─ data_alm_current.py        
│  ├─ training_current.py        
│  ├─ losses.py                  # reused
│  ├─ plotting.py                # reused
│  ├─ main_train_alm_current.py  
│  ├─ eval_current_alm.py            
│  ├─ multi_session.py           # TODO


# ALM Current-RNN Data Layout

This document describes the file locations, naming conventions, and tensor formats
used by the ALM current-based RNN pipeline.

---

## 1. Directory structure

### 1.1 Stage1 (trial-averaged) NPZ
Default root (per animal):
- `/home/jingyi.xu/ALM/results/stage1/<animal>/`

Files:
- `psth_<session_id>.<plane>.npz`

Example:
- `/home/jingyi.xu/ALM/results/stage1/kd95/psth_20220823_205730.0.npz`

Stage1 is the **canonical source** for:
- `keep_idx` (selected inhibitory neurons)
- `event_frames` (S/D/R alignment)
- `fps`
- `all_data` (bpod-derived trial meta)
- trial-averaged PSTHs
- (patched) trial-averaged behavioral traces (reward/lick kernels)

---

### 1.2 Trials (trial-level) NPZ
Default roots (best effort search at training time):
- user-provided `trials_roots=[...]`
- else auto:
  - `/allen/aind/scratch/jingyi/2p/<animal>/`
  - `/allen/aind/scratch/jingyi/2p/`

Naming convention (must match stage1):
- `trials_<session_id>.<plane>.npz`

Example:
- `/allen/aind/scratch/jingyi/2p/kd95/trials_20220823_205730.0.npz`

The exporter creates this file **in the same directory as** the corresponding
`*.trial_2p.pkl` by default.

---

### 1.3 Trial PKL + bpod/licks sidecars (source data for trials export)

Default templates (batch exporter):
- trial pkl:
  - `{trial_root}/{animal}/{animal}_twNew_{session_id}.{plane}.trial_2p.pkl`
- bpod npy:
  - `{trial_root}/{animal}/{animal}_twNew_{session_id}.bpod.npy`
- licks npy:
  - `{trial_root}/{animal}/{animal}_twNew_{session_id}.licks.npy`

(These are used by the exporter to build trial-level traces and optional per-trial reward events.)

---

## 2. File formats

### 2.1 Stage1 NPZ: `psth_<sid>.<plane>.npz`

Stage1 contains trial-averaged (condition-averaged) supervision targets and
aligned metadata.

### Required keys for training
- `cell_psth` or equivalent PSTH tensor for LC/RC (condition dimension)
- `cond_names`: list of condition names (e.g., `["left_correct","right_correct"]`)
- `keep_idx`: int array of selected cells (inhibitory neurons)
- `fps`: float
- `event_frames`: dict with keys such as `"S"`, `"D"`, `"R"` (frame indices)
- `all_data`: array of bpod trial metadata

### Behavioral traces (trial-averaged inputs)
- `reward_trace`: float[T] (trial-averaged kernel aligned to go cue)
- `lick_rate_left`, `lick_rate_right`, `lick_rate_total`: float[T]
- `t_rel_go_sec`: float[T] (time axis relative to go cue)

Notes:
- `reward_trace` is constructed from behavior and written back into stage1 during preprocessing.

---

### 2.2 Trials NPZ: `trials_<sid>.<plane>.npz`

Trials NPZ stores **trial-level neural activity** in a strict format.

### Core keys
- `cell_trials`: dict mapping `cond_name -> float32[C_keep, T_keep, nTr]`
  - `C_keep` is the number of kept cells (must match `len(keep_idx)`)
  - `T_keep` is number of frames in a trial (can be truncated by exporter)
  - `nTr` can differ across conditions
- `trial_indices`: dict mapping `cond_name -> int32[nTr]` (original trial indices)
- `keep_idx`: int32[C_keep] (copied from stage1; must match stage1 exactly)
- `cond_names`: list of condition names used by exporter (typically LC/RC)
- `fps`: float
- `event_frames`: dict (copied from stage1)

### Convenience / copied metadata
- `cell_clusters`, `cell_subclasses`, `n_cells_before`, `cond_counts`, etc.

### Copied behavioral traces (trial-averaged; for training inputs)
- `reward_trace`, `lick_rate_left`, `lick_rate_right`, `lick_rate_total`, `t_rel_go_sec`
  - these are copied from stage1 if present

### Optional per-trial reward events (if bpod/licks provided)
- exporter may include arrays describing per-trial reward timing relative to go cue

---

## 3. Tensor shapes used in training

### 3.1 Trial-averaged supervision (stage1)
- PSTH target used by training:
  - `psth_sub`: float[C, T, K]
    - C = number of conditions (LC/RC)
    - T = frames
    - K = number of observed inhibitory neurons sampled for this step

### 3.2 Trial-level supervision (trials npz)
- Raw per-condition trial data in file:
  - `cell_trials[cname]`: float[C_keep, T_keep, nTr]
- After selecting the K sampled neurons (pos_list) and slicing time:
  - `arr_k = arr[pos_list, :T_shared, :]`: float[K, T, nTr]
- Converted to training batch layout:
  - `trials_sub[cname] = transpose(arr_k, (2,1,0))`: float[R, T, K]
    - R = nTr (trials)

---

## 4. Alignment rules (strict)

When training with trial-level data:
- `stage1.keep_idx` must be **exactly equal** to `trials.keep_idx`
- for each condition:
  - `cell_trials[cname].shape[0] == len(keep_idx)`
  - `cell_trials[cname].shape[1] >= T_shared`
- the sampled indices `pos_list` must satisfy:
  - `0 <= pos_list < len(keep_idx)`

If any of these fail, training should raise an error printing:
- stage1 path, trials path
- keep_idx lengths
- per-condition shapes


