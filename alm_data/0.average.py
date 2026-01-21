#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 (v5): averaging with CSV default (cluster+subclass), PKL optional
-----------------------------------------------------------------------
Adds:
  (1) Save both per-cell Predicted_Cluster (fine) and Predicted_Subclass (coarse).
  (2) --dump-kept table.csv to export kept ROI mapping:
      ROI_1based, array_idx, ex_vivo_id, Predicted_Subclass, Predicted_Cluster, source_slice

Notes:
- CSV columns expected (case-insensitive): mRNA_ID, mRNA_Slice, Predicted_Subclass, Predicted_Cluster
- ROI id → array index conversion is 1-based → 0-based.
- Per-session match files are searched in match_ids/<animal>_s{1,2}/ and unioned.
"""

import os, sys, json, csv, argparse, pickle as pkl, glob, re
import numpy as np
import pandas as pd
from collections import Counter

ROOT        = os.path.expanduser('~/ALM')
MANIFEST    = os.path.join(ROOT, 'meta', 'manifest.json')
OUT_DIR     = os.path.join(ROOT, 'results', 'stage1')
SUMMARY_CSV = os.path.join(OUT_DIR, 'summary_stage1.csv')

FPS_FALLBACK   = 29.1
SMOOTH_DEFAULT = 1
TRIAL_STAT_DEF = 'mean'

sys.path.append(os.path.join(ROOT, 'scripts'))
from bpod_parse import keys, outcomes  # noqa: E402

def load_manifest():
    with open(MANIFEST, 'r', encoding='utf-8') as f:
        return json.load(f)

def first_or_none(d, key):
    v = d.get(key)
    return v[0] if v else None

def load_trial2p_pkl(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    all_data, F, Fbase = obj[0], obj[1], obj[2]
    assert all_data.ndim == 2 and F.ndim == 3 and Fbase.shape == F.shape
    return all_data, F, Fbase

def get_fps(rec):
    fps = FPS_FALLBACK
    print(f'[info] using fps={fps} for {rec["session_id"]}.{rec["plane"]}')
    return fps

def _smooth_time(CxT, win=1):
    if win <= 1: return CxT
    k = np.ones(win, dtype=float) / win
    def _conv(v):
        v = np.asarray(v, float)
        m = np.isfinite(v)
        if not m.any(): return v
        vv = np.copy(v); vv[~m] = 0.0
        num = np.convolve(vv, k, mode='same')
        den = np.convolve(m.astype(float), k, mode='same')
        out = np.full_like(num, np.nan); ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out
    return np.apply_along_axis(_conv, 1, CxT)

def _trial_reduce(dFoF, tidx, stat='mean'):
    if len(tidx) == 0: return None
    if stat == 'median': return np.nanmedian(dFoF[:, :, tidx], axis=2)
    return np.nanmean(dFoF[:, :, tidx], axis=2)

def zero_center_pre(cell_psth, fps, pre_frames):
    if pre_frames is None or pre_frames <= 0: return cell_psth
    out = {}
    for name, M in cell_psth.items():
        if M is None: continue
        pf = min(pre_frames, M.shape[1])
        base = np.nanmean(M[:, :pf], axis=1, keepdims=True)
        out[name] = M - base
    return out

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

def align_freelick_trials_inplace(F, all_data, fps, go_sec_median, freelick_indices):
    if freelick_indices.size == 0: return
    go_each = all_data[keys('go')][freelick_indices]
    offsets = np.round(fps * (go_each - go_sec_median)).astype(int)
    T = F.shape[1]
    for trial_num, offset in zip(freelick_indices, offsets):
        if offset < 0:
            off = abs(int(offset))
            F[:, 0:T-off, trial_num] = F[:, off:T, trial_num]
            F[:, T-off:T, trial_num] = 0
        elif offset > 0:
            nkeep = max(0, T - offset)
            F[:, 0:nkeep, trial_num] = F[:, offset:offset+nkeep, trial_num]
            F[:, nkeep:T, trial_num] = 0

def compute_dFoF_presample(F, fps, ss_frames):
    C, T, N = F.shape
    ss_frames = max(1, min(int(ss_frames), T))
    base = F[:, :ss_frames, :].mean(axis=1, keepdims=True)
    base = np.maximum(base, 1e-6)
    return (F - base) / base

# ---- PKL backend ----
def _load_cell2k_from_pkl(animal, session_id, plane):
    p_sess = os.path.join(ROOT, 'data', 'cell_type', f'out_{animal}', f'{animal}.sess_z_cell.pkl')
    if not os.path.exists(p_sess):
        return None, None
    try:
        with open(p_sess, 'rb') as f:
            obj = pkl.load(f)
        for key in (f'{animal}_twNew_{session_id}.{plane}', f'{animal}_{session_id}.{plane}'):
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
                return obj[key], key
    except Exception:
        pass
    return None, None

def select_cells_by_pkl(cell2k, n_cells):
    keep_idx, clusters, subclasses, table = [], [], [], []
    for k_id, k_val in cell2k.items():
        try:
            roi_id = int(k_id)       # 1-based
            array_idx = roi_id - 1   # 0-based
        except Exception:
            continue
        if array_idx < 0 or array_idx >= n_cells: continue
        try: kv = int(k_val)
        except Exception: continue
        if kv < 0: continue
        keep_idx.append(array_idx)
        clusters.append(str(kv))     # cluster = K number (string)
        subclasses.append('NA')      # no subclass in PKL
        table.append({'ROI_1based': roi_id, 'array_idx': array_idx, 'ex_vivo_id': '', 'Predicted_Subclass': 'NA', 'Predicted_Cluster': str(kv), 'source_slice': ''})
    if not keep_idx: return [], [], [], []
    order = np.argsort(np.asarray(keep_idx))
    keep_idx_sorted = [int(np.asarray(keep_idx)[order][j]) for j in range(len(order))]
    clusters_sorted = [str(np.asarray(clusters)[order][j]) for j in range(len(order))]
    subclasses_sorted= [str(np.asarray(subclasses)[order][j]) for j in range(len(order))]
    table_sorted = [table[int(o)] for o in order]
    return keep_idx_sorted, clusters_sorted, subclasses_sorted, table_sorted

# ---- CSV backend ----
def _find_match_files(animal, session_id, plane, slices=(1,2)):
    hits = []
    for sl in slices:
        mdir = os.path.join(ROOT, 'data', 'match_ids', f'{animal}_s{sl}')
        pattern = os.path.join(mdir, f'{animal}_twNew_{session_id}.{plane}.tps.*.match.txt')
        fs = sorted(glob.glob(pattern))
        if fs:
            hits.append((sl, fs[0]))
    return hits

def _load_roi_to_exvivo(match_path):
    try:
        df = pd.read_csv(match_path, sep=None, engine='python')
        if df.shape[1] < 2:
            raise ValueError("match file must have >=2 columns")
        c0, c1 = df.columns[:2]
        rois = pd.to_numeric(df[c0], errors='coerce').astype('Int64').dropna().astype(int)
        exvs = pd.to_numeric(df[c1], errors='coerce').astype('Int64').dropna().astype(int)
        n = min(len(rois), len(exvs))
        return {int(rois.iloc[i]): int(exvs.iloc[i]) for i in range(n)}
    except Exception:
        mapping = {}
        with open(match_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'): continue
                parts = re.split(r'[\s,]+', s)
                if len(parts) < 2: continue
                try:
                    r = int(float(parts[0])); e = int(float(parts[1]))
                    mapping[r] = e
                except Exception:
                    continue
        return mapping

def _build_ex_maps(csv_path):
    df = pd.read_csv(csv_path, sep=None, engine='python')
    cols = {c.lower(): c for c in df.columns}
    need = ['mrna_id', 'predicted_cluster', 'predicted_subclass']
    for k in need:
        if k not in cols:
            raise KeyError(f'CSV must contain {k}. Got: {list(df.columns)}')
    col_id = cols['mrna_id']
    col_pc = cols['predicted_cluster']
    col_ps = cols['predicted_subclass']
    df_ok = df.loc[df[col_pc].notna()].copy()
    ex_ids  = pd.to_numeric(df_ok[col_id], errors='coerce').astype('Int64').dropna().astype(int).tolist()
    clusts  = df_ok[col_pc].astype(str).tolist()
    subcls  = df_ok[col_ps].astype(str).tolist()
    ex2cluster  = {int(e): c for e, c in zip(ex_ids, clusts)}
    ex2subclass = {int(e): s for e, s in zip(ex_ids, subcls)}
    return ex2cluster, ex2subclass

def select_cells_by_csv(animal, session_id, plane, n_cells, csv_path):
    hits = _find_match_files(animal, session_id, plane, slices=(1,2))
    if not hits:
        print(f'[WARN] no match files for {animal} {session_id}.{plane}')
        return [], [], [], []
    ex2cluster, ex2subclass = _build_ex_maps(csv_path)
    roi2info = {}  # array_idx -> (roi1b, ex_id, subclass, cluster, slice)
    for sl, mpath in hits:
        roi2ex = _load_roi_to_exvivo(mpath)
        for roi_1b, ex_id in roi2ex.items():
            idx = roi_1b - 1  # 1-based -> 0-based
            if 0 <= idx < n_cells and (ex_id in ex2cluster):
                roi2info[idx] = (roi_1b, ex_id, ex2subclass.get(ex_id, 'NA'), ex2cluster[ex_id], sl)
    if not roi2info: return [], [], [], []
    keep_idx_sorted = sorted(roi2info.keys())
    clusters = [roi2info[i][3] for i in keep_idx_sorted]
    subclasses = [roi2info[i][2] for i in keep_idx_sorted]
    table = [{
        'ROI_1based': roi2info[i][0],
        'array_idx': i,
        'ex_vivo_id': roi2info[i][1],
        'Predicted_Subclass': roi2info[i][2],
        'Predicted_Cluster': roi2info[i][3],
        'source_slice': roi2info[i][4],
    } for i in keep_idx_sorted]
    return keep_idx_sorted, clusters, subclasses, table

def aggregate_by_type(cell_psth, labels, group_map=None, n_boot=1000):
    if group_map is None:
        uniq = sorted(set(labels))
        group_map = {u: [u] for u in uniq}
    labels = np.asarray(labels)
    masks = {g: np.isin(labels, members) for g, members in group_map.items()}
    rng = np.random.default_rng(2025)
    def boot_mean(X, B=1000):
        if X.shape[0] == 0: return None
        mean = X.mean(axis=0)
        if X.shape[0] == 1: return mean, mean, mean, 1
        boots = np.empty((B, X.shape[1]), float)
        for b in range(B):
            idx = rng.integers(0, X.shape[0], size=X.shape[0])
            boots[b] = X[idx].mean(axis=0)
        lo = np.percentile(boots, 2.5, axis=0); hi = np.percentile(boots, 97.5, axis=0)
        return mean, lo, hi, X.shape[0]
    out = {}
    for name, CF in cell_psth.items():
        res = {}
        for g, mask in masks.items():
            X = CF[mask]
            if X.size == 0: continue
            r = boot_mean(X, B=n_boot)
            if r is None: continue
            m, lo, hi, n = r
            res[g] = {'mean': m, 'ci_low': lo, 'ci_high': hi, 'n_cells': int(n)}
        out[name] = res
    return out

def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST, 'r', encoding='utf-8') as f:
        man = json.load(f)

    rows = []
    for animal in man['animals']:
        aname = animal['animal']
        # 每只小鼠单独的子目录
        out_dir_animal = os.path.join(OUT_DIR, aname)
        os.makedirs(out_dir_animal, exist_ok=True)

        # 若未指定 --em-csv，则按 animal 自动推断
        auto_em_csv = os.path.join(
            ROOT, 'data', 'cell_type',
            f'cell_typing_em_results_{aname}_alm_visp_combined.csv'
        )
        use_em_csv = args.em_csv if (args.em_csv and os.path.exists(args.em_csv)) else auto_em_csv

        for rec in animal['sessions']:
            trial_pkl = rec['files'].get('trial_2p', [None])[0]
            if not trial_pkl or not os.path.exists(trial_pkl):
                continue

            sid, plane = rec['session_id'], str(rec['plane'])
            tag = f'{sid}.{plane}'
            out_npz = os.path.join(out_dir_animal, f'psth_{tag}.npz')

            try:
                all_data, F, Fbase = load_trial2p_pkl(trial_pkl)
            except Exception as e:
                print(f'[SKIP] {tag}: trial_2p load failed - {e}')
                continue

            fps = get_fps(rec)
            (ss, ld, go), cond_idx, FT_mask, go_sec_median = prelim_events_and_trials(all_data, fps)

            # align freelick
            freelick_all = np.array([], dtype=int)
            if 'free_left' in cond_idx:
                freelick_all = np.concatenate((freelick_all, cond_idx['free_left']))
            if 'free_right' in cond_idx:
                freelick_all = np.concatenate((freelick_all, cond_idx['free_right']))
            if freelick_all.size > 0:
                align_freelick_trials_inplace(F, all_data, fps, go_sec_median, freelick_all)

            # ΔF/F
            dFoF = compute_dFoF_presample(F, fps, ss_frames=ss)

            # trial reduce + smooth + zero-center
            cell_psth = {}
            for name, tidx in cond_idx.items():
                M = _trial_reduce(dFoF, tidx, stat=args.trial_stat)
                if M is None:
                    continue
                M = _smooth_time(M, win=args.smooth_win)
                cell_psth[name] = M
            if not cell_psth:
                print(f'[SKIP] {tag}: empty conditions')
                continue

            zc_frames = int(round(args.zero_center_pre * fps)) if args.zero_center_pre > 0 else ss
            cell_psth = zero_center_pre(cell_psth, fps, pre_frames=zc_frames)

            # --- cell selection ---
            C_full = F.shape[0]
            source_used = args.celltype_source
            keep_idx = []; cell_clusters = []; cell_subclasses = []; kept_table = []
            pkl_key_used = None

            if args.celltype_source == 'csv':
                if not os.path.exists(use_em_csv):
                    print(f'[WARN] {aname} {tag}: EM CSV missing at {use_em_csv}; fallback to pkl')
                    source_used = 'pkl'
                else:
                    keep_idx, cell_clusters, cell_subclasses, kept_table = \
                        select_cells_by_csv(aname, sid, plane, C_full, use_em_csv)
                    if len(keep_idx) == 0:
                        print(f'[WARN] {aname} {tag}: CSV selection empty; fallback to pkl')
                        source_used = 'pkl'

            if source_used == 'pkl':
                cell2k, key_used = _load_cell2k_from_pkl(aname, sid, plane)
                if not cell2k:
                    print(f'[SKIP] {aname} {tag}: no mapping in pkl')
                    continue
                keep_idx, cell_clusters, cell_subclasses, kept_table = \
                    select_cells_by_pkl(cell2k, n_cells=C_full)
                pkl_key_used = key_used

            if len(keep_idx) == 0:
                print(f'[SKIP] {aname} {tag}: no cells kept')
                continue
            if len(keep_idx) < args.min_cells:
                print(f'[SKIP] {aname} {tag}: only {len(keep_idx)} cells (<{args.min_cells}), skip saving npz')
                continue

            # 过滤细胞
            cell_psth = {name: M[keep_idx, :] for name, M in cell_psth.items()}

            # 聚合：cluster + subclass
            type_psth_cluster  = aggregate_by_type(cell_psth, cell_clusters,  n_boot=1000)
            type_psth_subclass = aggregate_by_type(cell_psth, cell_subclasses, n_boot=1000)

            # meta
            any_cond = next(iter(cell_psth.keys()))
            T = cell_psth[any_cond].shape[1]
            pre_sec  = ss / float(fps)
            post_sec = max(0.0, T / float(fps) - pre_sec)
            event_frames = {'S': ss, 'D': ld, 'R': go}
            cond_counts = {k: int(len(v)) for k, v in cond_idx.items()}

            # dump-kept：如果传入的是目录，则放在每只小鼠子目录下
            if args.dump_kept:
                dump_path = args.dump_kept
                if os.path.isdir(dump_path):
                    dump_path = os.path.join(out_dir_animal, f'kept_{tag}.csv')
                if source_used == 'csv':
                    kept_rows = []
                    by_idx = {row['array_idx']: row for row in kept_table}
                    for idx in keep_idx:
                        if idx in by_idx: kept_rows.append(by_idx[idx])
                    pd.DataFrame(
                        kept_rows,
                        columns=['ROI_1based','array_idx','ex_vivo_id',
                                 'Predicted_Subclass','Predicted_Cluster','source_slice']
                    ).to_csv(dump_path, index=False)
                    print(f'[dump] kept mapping -> {dump_path} ({len(kept_rows)} rows)')

            # 保存 npz 到 animal 子目录
            np.savez_compressed(
                out_npz,
                session_id=sid,
                plane=plane,
                animal=aname,
                cond_names=np.array(list(cell_psth.keys())),
                cell_types=np.asarray(cell_clusters),
                cell_clusters=np.asarray(cell_clusters),
                cell_subclasses=np.asarray(cell_subclasses),
                cell_psth=cell_psth,
                type_psth_cluster=type_psth_cluster,
                type_psth_subclass=type_psth_subclass,
                type_psth=type_psth_cluster,
                all_data=all_data,
                fps=float(fps),
                t0_frame=int(ss),
                event_frames=event_frames,
                align_to='pre-sample',
                pre_sec=float(pre_sec),
                post_sec=float(post_sec),
                pre_frames=int(round(pre_sec*fps)),
                post_frames=int(round(post_sec*fps)),
                cond_counts=cond_counts,
                keep_idx=np.asarray(keep_idx, dtype=int),
                n_cells_before=int(C_full),
                source_used=source_used,
                pkl_key_used=pkl_key_used if pkl_key_used else '',
                allow_pickle=True
            )

            # summary 行
            from collections import Counter
            type_counts = Counter(cell_clusters)
            sub_counts  = Counter(cell_subclasses)
            row = {
                'animal': aname, 'session_id': sid, 'plane': plane,
                'n_cells_before': C_full, 'n_cells_kept': len(keep_idx),
                'n_frames': F.shape[1],
                **{f'cond_{k}': int(v) for k, v in cond_counts.items()},
                **{f'ncluster_{t}': c for t, c in type_counts.items()},
                **{f'nsub_{t}': c for t, c in sub_counts.items()},
                'fps': float(fps), 'pre_sec': float(pre_sec), 'post_sec': float(post_sec),
                'trial_stat': args.trial_stat, 'smooth_win': int(args.smooth_win),
                'zero_center_pre': float(args.zero_center_pre if args.zero_center_pre > 0 else 0.0),
                'source_used': source_used, 'pkl_key_used': pkl_key_used if pkl_key_used else '',
                'out_npz': out_npz
            }
            rows.append(row)
            print(f'[OK] {aname} {tag}: kept {len(keep_idx)}/{C_full} by {source_used}; saved -> {out_npz}')

    # 写 summary（全局）
    if rows:
        all_keys = set().union(*[r.keys() for r in rows])
        header = ['animal','session_id','plane','n_cells_before','n_cells_kept','n_frames'] + \
                 sorted([k for k in all_keys if k.startswith('cond_')]) + \
                 sorted([k for k in all_keys if k.startswith('ncluster_')]) + \
                 sorted([k for k in all_keys if k.startswith('nsub_')]) + \
                 ['fps','pre_sec','post_sec','trial_stat','smooth_win','zero_center_pre','source_used','pkl_key_used','out_npz']
        with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\nSummary written: {SUMMARY_CSV}')
    else:
        print('\nNo record processed.')



def parse_args():
    ap = argparse.ArgumentParser(
        description="Stage-1 averaging with CSV/PKL, per-animal folders, dump-kept, and min-cells filter"
    )
    ap.add_argument('--trial-stat', choices=['mean','median'], default=TRIAL_STAT_DEF)
    ap.add_argument('--smooth-win', type=int, default=SMOOTH_DEFAULT)
    ap.add_argument('--zero-center-pre', type=float, default=0.0)

    # 默认优先用 CSV；若未找到则自动回退到 PKL
    ap.add_argument('--celltype-source', choices=['csv','pkl'], default='csv')

    # 若不传 --em-csv，将自动按 animal 选择：
    ap.add_argument('--em-csv', default=None,
                    help='Path to a specific EM CSV; if not provided, use per-animal default under data/cell_type/')

    # kept 映射表：可传目录或完整文件路径
    ap.add_argument('--dump-kept', default=None,
                    help='If set: if a directory is given, write kept_<sid>.<plane>.csv into per-animal folder; '
                         'if a file path is given, write exactly to that file (CSV mode only)')

    ap.add_argument('--min-cells', type=int, default=40,
                    help='Minimum cells required to save npz (default 40)')
    return ap.parse_args()



if __name__ == '__main__':
    args = parse_args()
    run(args)
