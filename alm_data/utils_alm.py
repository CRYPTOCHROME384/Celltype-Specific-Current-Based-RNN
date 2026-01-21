# utils_alm.py
import os
import pickle as pkl
import numpy as np
import torch as tch

from bpod_parse import keys, outcomes

FPS_FALLBACK = 29.1
SESSION_TO_PATH = {}


def load_trial2p(trial_pkl_path):
    """读取 ALM 的 trial_2p.pkl: 返回 all_data, F, Fbase"""
    with open(trial_pkl_path, 'rb') as f:
        obj = pkl.load(f)
    all_data, F, Fbase = obj[0], obj[1], obj[2]
    # F: (cells, time, trials)
    return all_data, F, Fbase

def compute_dFoF_presample(F, fps, ss_frames):
    C, T, N = F.shape
    ss_frames = max(1, min(int(ss_frames), T))
    base = F[:, :ss_frames, :].mean(axis=1, keepdims=True)
    base = np.maximum(base, 1e-6)
    return (F - base) / base

def prelim_events_and_trials(all_data, fps):
    behavior = (all_data[keys('protocol')] == 5)
    freelick = (all_data[keys('protocol')] == 6)
    full_sample = (all_data[keys('earlysample')] == 0)
    full_delay  = (all_data[keys('earlydelay')]  == 0)
    trials_ok = behavior & full_sample & full_delay
    o  = all_data[keys('outcome')]
    tt = all_data[keys('trialtype')]
    CORR = np.isin(o, [outcomes('correct'), outcomes('droppednotlick')])
    INC  = (o == outcomes('incorrect'))
    IG   = (o == outcomes('ignore'))
    FT   = trials_ok | IG
    ss_sec = np.median(all_data[keys('samplestart')][FT])
    ld_sec = np.median(all_data[keys('lastdelay')][FT])
    go_sec = np.median(all_data[keys('go')][FT])
    ss = int(round(ss_sec * fps)); ld = int(round(ld_sec * fps)); go = int(round(go_sec * fps))
    vals = np.unique(tt)
    if set(vals.tolist()) >= {0,1}: L = (tt == 0); R = (tt == 1)
    elif set(vals.tolist()) >= {1,2}: L = (tt == 1); R = (tt == 2)
    else: mid = np.median(vals); L = (tt <= mid); R = (tt > mid)
    LC = np.where(trials_ok & L & CORR)[0]; RC = np.where(trials_ok & R & CORR)[0]
    LI = np.where(trials_ok & L & INC )[0]; RI = np.where(trials_ok & R & INC )[0]
    FL = np.where(full_sample & freelick)[0]; FLt = tt[FL]; FL_type = all_data[keys('freelicktype')][FL]
    FL_L = FL[(FLt == 0) & (FL_type == 1)]; FL_R = FL[(FLt == 1) & (FL_type == 1)]
    licks_needed = all_data[keys('licksneeded')]; puff = all_data[keys('puff')]
    LCo = LC[licks_needed[LC] == 1]; LCm = LC[licks_needed[LC] > 1]
    RCo = RC[licks_needed[RC] == 1]; RCm = RC[licks_needed[RC] > 1]
    LIp = LI[puff[LI] == 1];        LIn = LI[puff[LI] == 0]
    RIp = RI[puff[RI] == 1];        RIn = RI[puff[RI] == 0]
    agg_one   = np.concatenate((LCo, RCo)) if (LCo.size + RCo.size) else np.array([], int)
    agg_multi = np.concatenate((LCm, RCm)) if (LCm.size + RCm.size) else np.array([], int)
    agg_puff  = np.concatenate((LIp, RIp)) if (LIp.size + RIp.size) else np.array([], int)
    agg_nopf  = np.concatenate((LIn, RIn)) if (LIn.size + RIn.size) else np.array([], int)
    cond_idx = {
        'left_correct': LC, 'right_correct': RC, 'left_incorrect': LI, 'right_incorrect': RI,
        'free_left': FL_L, 'free_right': FL_R, 'correct_one_lick': agg_one, 'correct_multi_licks': agg_multi,
        'incorrect_puff': agg_puff, 'incorrect_no_puff': agg_nopf,
    }
    cond_idx = {k: v for k, v in cond_idx.items() if v.size > 0}
    return (ss, ld, go), cond_idx, FT, go_sec

def trial_average_all_neurons(list_sessions, hemi='left', max_time_index=350, fps=FPS_FALLBACK):
    """
    生成 trial-average neural activity:
    返回 trial_average: (2, T, N_total_neurons), indexes_list: {session_name: range}
    这里不再区分左右半球，hemi 参数仅用于接口兼容。
    """
    mean_rates = []
    indexes_list = {}
    s = 0

    for trial_pkl in list_sessions:
        ses_name = os.path.basename(trial_pkl)
        SESSION_TO_PATH[ses_name] = trial_pkl
        all_data, F, Fbase = load_trial2p(trial_pkl)
        C, T, N = F.shape

        T_keep = min(max_time_index, T)

        # 事件 & 条件划分
        (ss, ld, go), cond_idx, FT, go_sec = prelim_events_and_trials(all_data, fps)

        # ΔF/F (baseline: sample 前)
        dFoF = compute_dFoF_presample(F, fps, ss_frames=ss)  # (C, T, N)
        dFoF = dFoF[:, :T_keep, :]

        left_trials  = cond_idx.get('left_correct',  np.array([], int))
        right_trials = cond_idx.get('right_correct', np.array([], int))
        if left_trials.size == 0 or right_trials.size == 0:
            print(f"[WARN] {ses_name} 没有 left/right correct trial，跳过")
            continue

        left_avg  = np.nanmean(dFoF[:, :, left_trials],  axis=2)  # (C, T_keep)
        right_avg = np.nanmean(dFoF[:, :, right_trials], axis=2)

        left_TN  = left_avg.T    # (T, C)
        right_TN = right_avg.T
        rates = np.stack([left_TN, right_TN], axis=0)  # (2, T, C)

        mean_rates.append(rates)
        indexes_list[ses_name] = np.arange(s, s + C)
        s += C

    if not mean_rates:
        raise RuntimeError("trial_average_all_neurons: 没有有效 session")

    trial_average = np.concatenate(mean_rates, axis=2)  # (2, T, N_total)
    return trial_average, indexes_list



def mean_licks_inputs(list_sessions, hemi='left', max_time_index=350, fps=FPS_FALLBACK, device='cuda'):
    """
    简化版：目前先不给真实 lick 输入，返回全 0。
    接口保持和原 utils.mean_licks_inputs 完全一致。
    """
    trial_average, indexes = trial_average_all_neurons(
        list_sessions, hemi=hemi, max_time_index=max_time_index, fps=fps
    )
    mean_licks = np.zeros_like(trial_average, dtype=np.float32)
    mean_licks = tch.tensor(mean_licks, dtype=tch.float32, device=device)
    return mean_licks




def licks_inputs(list_sessions, hemi='left', max_time_index=350, fps=FPS_FALLBACK, device='cuda'):
    """
    简化版：单 trial lick 输入全部设为 0。
    返回: {session_name: tensor(2, n_trials, T)}
    """
    licks_per_session = {}
    for trial_pkl in list_sessions:
        ses_name = os.path.basename(trial_pkl)
        all_data, F, Fbase = load_trial2p(trial_pkl)
        C, T, N = F.shape
        T_keep = min(max_time_index, T)

        # 2 通道(left/right)，这里先全 0 占位
        # 注意原 DANDI 版本 shape 是 (2, n_trials, T)
        sess_licks = tch.zeros((2, N, T_keep), dtype=tch.float32, device=device)
        licks_per_session[ses_name] = sess_licks
    return licks_per_session



def load_me_data(list_sessions, default_parameters, hemi='left', max_time_index=350, fps=FPS_FALLBACK):
    """
    封装成和原 DANDI 版本接口一致的字典：
    - rates: (2, T, N_total)
    - indexes: {session_name: np.arange(...)}
    - n_trials, trial_types, time_average, neurons_average, responses
    """
    device = default_parameters['device']
    trial_average, indexes = trial_average_all_neurons(
        list_sessions, hemi=hemi, max_time_index=max_time_index, fps=fps
    )

    ALM_data = {}
    ALM_data['rates']   = trial_average
    ALM_data['indexes'] = indexes

    trial_types = {}
    n_trials = {}
    responses = {}
    time_average = {}
    neurons_average = {}

    for trial_pkl in list_sessions:
        ses_name = os.path.basename(trial_pkl)
        all_data, F, Fbase = load_trial2p(trial_pkl)
        C, T, N = F.shape
        T_keep = min(max_time_index, T)

        tt = all_data[keys('trialtype')]
        vals = np.unique(tt)
        if set(vals.tolist()) >= {0, 1}:
            L = (tt == 0)
            R = (tt == 1)
        elif set(vals.tolist()) >= {1, 2}:
            L = (tt == 1)
            R = (tt == 2)
        else:
            mid = np.median(vals)
            L = (tt <= mid)
            R = (tt > mid)

        ttypes = np.full_like(tt, fill_value=-1, dtype=int)
        ttypes[L] = 0
        ttypes[R] = 1
        trial_types[ses_name] = ttypes
        n_trials[ses_name] = tt.shape[0]

        # time_average / neurons_average 先给一个安全占位
        # 后面如果要严格复现合作者的 Loss 再精细化
        dFoF = compute_dFoF_presample(F, fps, ss_frames=int(np.median(all_data[keys('samplestart')] * fps)))
        dFoF = dFoF[:, :T_keep, :]  # (C, T_keep, N)

        # time_average: 直接用 trial_average 中该 session 的部分
        time_average[ses_name] = tch.tensor(
            trial_average[:, :T_keep, indexes[ses_name]],
            dtype=tch.float32,
            device=device
        )

        # neurons_average: 这里先填零张量占位
        neurons_avg = np.zeros((2, n_trials[ses_name], T_keep), dtype=np.float32)
        neurons_average[ses_name] = tch.tensor(neurons_avg, dtype=tch.float32, device=device)

        responses[ses_name] = all_data[keys('outcome')]

    ALM_data['n_trials']        = n_trials
    ALM_data['trial_types']     = trial_types
    ALM_data['time_average']    = time_average
    ALM_data['neurons_average'] = neurons_average
    ALM_data['responses']       = responses
    return ALM_data


def toch_version_ratates_trial_types(ALM_hemi, default_parameters, session_names):
    """
    把 numpy 的 rates / trial_types 转成 torch tensor。
    """
    device = default_parameters['device']
    rates_np = ALM_hemi['rates']   # (2, T, N_total)
    rates = tch.tensor(rates_np, dtype=tch.float32, device=device)

    trial_types = {}
    for ses_name in session_names:
        t_types = ALM_hemi['trial_types'][ses_name]
        t_types = tch.tensor(t_types, dtype=tch.float32, device=device)
        trial_types[ses_name] = t_types
    return rates, trial_types


def toch_single_trials(session_names, default_parameters, max_time_index=350, fps=FPS_FALLBACK):
    """
    session_names: 训练脚本传进来的 session 名（字典 key），例如 'kd95_twNew_20220823_205730.0.trial_2p.pkl'
    我们通过 SESSION_TO_PATH 找到对应的完整 trial_2p.pkl 路径。
    """
    device = default_parameters['device']
    single_trials_activity = {}

    for ses_name in session_names:
        # 从全局映射里取出完整路径；如果取不到，就退回用 ses_name 当路径（以防万一）
        trial_pkl = SESSION_TO_PATH.get(ses_name, ses_name)

        # 这里 trial_pkl 应该就是 '/allen/.../kd95_twNew_20220823_205730.0.trial_2p.pkl'
        all_data, F, Fbase = load_trial2p(trial_pkl)
        C, T, N = F.shape
        T_keep = min(max_time_index, T)

        # 和前面 trial_average 一样，先做 dFoF
        ss_sec = np.median(all_data[keys('samplestart')])
        ss_frames = int(round(ss_sec * fps))
        dFoF = compute_dFoF_presample(F, fps, ss_frames=ss_frames)  # (cells, time, trials)
        dFoF = dFoF[:, :T_keep, :]

        # 转成 (n_trials, T, n_neurons)
        trials_TCN = np.transpose(dFoF, (2, 1, 0))   # (N_trials, T_keep, C)
        data_tensor = tch.tensor(trials_TCN, dtype=tch.float32, device=device)

        single_trials_activity[ses_name] = data_tensor

    return single_trials_activity

