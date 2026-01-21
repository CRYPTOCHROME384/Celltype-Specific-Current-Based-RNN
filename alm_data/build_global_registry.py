#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a mouse-level global registry (entity cells -> global indices) from Stage-1 NPZ files.

Key ideas:
- Each entity cell is uniquely identified by (animal, session_id, plane, array_idx).
- We choose top-K sessions by available neurons (n_cells_kept), and then sample cells into a
  global observed pool of size N_obs (or use all if insufficient).
- Output:
  1) registry.csv
  2) unit_to_obs_idx.json  (unit_key -> list of global_idx)
  3) stats.json

python build_global_registry.py \
  --animal kd95 \
  --stage1_dir /home/jingyi.xu/ALM/results/stage1 \
  --out_dir /home/jingyi.xu/ALM/results/registry/kd95 \
  --top_sessions 1 \
  --session_agg sum \
  --n_global_obs 200 \
  --n_min_per_unit 10 \
  --n_max_per_unit 100 \
  --cell_select random \
  --seed 42

Stage-1 NPZ is assumed to contain (at least):
  session_id, plane, animal, keep_idx, cell_subclasses, cell_clusters
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------

NPZ_RE = re.compile(r"^psth_(\d{8}_\d{6})\.(\w+)\.npz$")


@dataclass
class UnitRec:
    animal: str
    session_id: str
    plane: str
    npz_path: str
    n_kept: int
    keep_idx: np.ndarray              # (n_kept,)
    cell_subclasses: np.ndarray       # (n_kept,) dtype=str/object
    cell_clusters: np.ndarray         # (n_kept,) dtype=str/object
    manifest_flags: Optional[Dict[str, Any]] = None
    manifest_match_summary: Optional[Dict[str, Any]] = None

    @property
    def unit_key(self) -> str:
        return f"{self.session_id}.{self.plane}"


def _as_str_array(x: Any, n: int) -> np.ndarray:
    """Convert npz-loaded arrays to a safe string array of length n."""
    if x is None:
        return np.array([""] * n, dtype=object)
    arr = np.asarray(x)
    if arr.shape[0] != n:
        # be tolerant: pad/truncate if needed
        out = np.array([""] * n, dtype=object)
        m = min(n, arr.shape[0])
        out[:m] = arr[:m]
        return out.astype(object)
    return arr.astype(object)


def load_manifest_optional(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] manifest not found at: {path} (ignored)")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_manifest(manifest: dict, animal: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Return dict keyed by (session_id, plane) -> record containing flags + match_summary.
    Manifest sessions are built per (session_id, plane). See build_manifest.py structure.
    """
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for a in manifest.get("animals", []):
        if str(a.get("animal")) != animal:
            continue
        for srec in a.get("sessions", []):
            sid = str(srec.get("session_id"))
            plane = str(srec.get("plane"))
            out[(sid, plane)] = {
                "flags": srec.get("flags", {}),
                "match_summary": srec.get("match_summary", {}),
            }
    return out


def scan_stage1_units(stage1_animal_dir: str,
                      animal: str,
                      manifest_idx: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
                      require_flags: bool = False,
                      require_has_bpod: bool = True,
                      require_has_check: bool = True) -> List[UnitRec]:
    """
    Scan psth_*.npz under stage1_animal_dir and build UnitRec list.
    Optionally filter by manifest flags.
    """
    units: List[UnitRec] = []
    if not os.path.isdir(stage1_animal_dir):
        raise FileNotFoundError(f"Stage1 dir not found: {stage1_animal_dir}")

    for fn in sorted(os.listdir(stage1_animal_dir)):
        m = NPZ_RE.match(fn)
        if not m:
            continue
        sid, plane = m.group(1), m.group(2)
        npz_path = os.path.join(stage1_animal_dir, fn)

        # Manifest filter (optional)
        flags = None
        match_summary = None
        if manifest_idx is not None and (sid, plane) in manifest_idx:
            flags = manifest_idx[(sid, plane)].get("flags", {}) or {}
            match_summary = manifest_idx[(sid, plane)].get("match_summary", {}) or {}
            if require_flags:
                if require_has_bpod and not bool(flags.get("has_bpod", False)):
                    continue
                if require_has_check and not bool(flags.get("has_check", False)):
                    continue

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] failed to load {npz_path}: {e}")
            continue

        # Required fields in your stage1 outputs (see digest)
        keep_idx = np.asarray(data["keep_idx"], dtype=int)
        n_kept = int(keep_idx.shape[0])

        cell_subclasses = _as_str_array(data.get("cell_subclasses", None), n_kept)
        cell_clusters = _as_str_array(data.get("cell_clusters", None), n_kept)

        # sanity: stage1 also saved session_id/plane/animal; but trust filename first
        units.append(UnitRec(
            animal=animal,
            session_id=str(data.get("session_id", sid)),
            plane=str(data.get("plane", plane)),
            npz_path=npz_path,
            n_kept=n_kept,
            keep_idx=keep_idx,
            cell_subclasses=cell_subclasses,
            cell_clusters=cell_clusters,
            manifest_flags=flags,
            manifest_match_summary=match_summary
        ))

    return units


def rank_sessions_by_neurons(units: List[UnitRec],
                            session_agg: str = "sum") -> List[Tuple[str, int]]:
    """
    Compute per-session neuron counts (aggregated across planes) and return sorted list desc.
    session_agg: "sum" or "max"
    """
    by_sess: Dict[str, List[int]] = defaultdict(list)
    for u in units:
        by_sess[u.session_id].append(u.n_kept)

    sess_scores = []
    for sid, lst in by_sess.items():
        if session_agg == "max":
            score = int(max(lst))
        else:
            score = int(sum(lst))
        sess_scores.append((sid, score))

    sess_scores.sort(key=lambda x: (-x[1], x[0]))
    return sess_scores


def choose_top_sessions(units: List[UnitRec],
                        top_sessions: int,
                        session_agg: str = "sum") -> Tuple[List[UnitRec], List[Tuple[str, int]]]:
    """
    Keep only units whose session_id is in top-K sessions by neuron availability.
    """
    sess_scores = rank_sessions_by_neurons(units, session_agg=session_agg)
    if top_sessions <= 0 or top_sessions >= len(sess_scores):
        chosen = set([sid for sid, _ in sess_scores])
    else:
        chosen = set([sid for sid, _ in sess_scores[:top_sessions]])
    kept_units = [u for u in units if u.session_id in chosen]
    kept_scores = [(sid, sc) for sid, sc in sess_scores if sid in chosen]
    return kept_units, kept_scores


def sample_registry(units: List[UnitRec],
                    n_global_obs: int,
                    n_min_per_unit: int,
                    n_max_per_unit: int,
                    celltype_field: str = "subclass",
                    cell_select: str = "random",
                    seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, List[int]], dict]:
    """
    Build registry by sampling entity cells from each unit.

    Strategy:
    - Phase 1: take at least n_min_per_unit from each unit (if available).
    - Phase 2: fill remaining quota with a round-robin across units up to n_max_per_unit.

    celltype_field: "subclass" or "cluster"
    cell_select: "random" | "first"
    """
    rng = np.random.default_rng(seed)

    # Prepare per-unit candidate indices (positions in kept array)
    per_unit_candidates: Dict[str, List[int]] = {}
    per_unit_taken: Dict[str, List[int]] = {}
    for u in units:
        key = u.unit_key
        idxs = list(range(u.n_kept))
        if cell_select == "random":
            rng.shuffle(idxs)
        # else "first": keep natural order
        per_unit_candidates[key] = idxs
        per_unit_taken[key] = []

    def _take_from_unit(u: UnitRec, k: int):
        key = u.unit_key
        cand = per_unit_candidates[key]
        take = cand[:k]
        per_unit_candidates[key] = cand[k:]
        per_unit_taken[key].extend(take)

    # Phase 1: minimum coverage
    for u in units:
        k = min(n_min_per_unit, u.n_kept, n_max_per_unit)
        if k > 0:
            _take_from_unit(u, k)

    # Total taken so far
    total_taken = sum(len(v) for v in per_unit_taken.values())

    # If user wants fewer than minimum total, truncate later
    # Phase 2: fill remaining quota with round-robin up to per-unit max
    target = n_global_obs if n_global_obs > 0 else total_taken
    if target < total_taken:
        # We'll truncate globally at the end; still keep per-unit mapping coherent
        pass
    else:
        remaining = target - total_taken
        # Units that still can contribute (respecting n_max_per_unit)
        unit_order = [u.unit_key for u in sorted(units, key=lambda x: (-x.n_kept, x.unit_key))]
        while remaining > 0:
            progressed = False
            for u in units:
                key = u.unit_key
                if remaining <= 0:
                    break
                if len(per_unit_taken[key]) >= min(n_max_per_unit, u.n_kept):
                    continue
                if len(per_unit_candidates[key]) == 0:
                    continue
                _take_from_unit(u, 1)
                remaining -= 1
                progressed = True
                if remaining <= 0:
                    break
            if not progressed:
                break  # no more candidates anywhere

    # Build registry rows
    rows = []
    global_idx = 0
    unit_to_obs: Dict[str, List[int]] = defaultdict(list)

    # Flatten chosen entities in a stable order (by session_id, plane) for reproducibility
    units_sorted = sorted(units, key=lambda u: (u.session_id, str(u.plane)))
    for u in units_sorted:
        key = u.unit_key
        chosen_pos = per_unit_taken[key]

        # If need to truncate globally (n_global_obs < min-coverage total), do it here
        if n_global_obs > 0 and global_idx >= n_global_obs:
            break

        # Determine celltype vector
        if celltype_field == "cluster":
            ctype_vec = u.cell_clusters
        else:
            ctype_vec = u.cell_subclasses

        for pos in chosen_pos:
            if n_global_obs > 0 and global_idx >= n_global_obs:
                break
            array_idx = int(u.keep_idx[pos])
            roi_1based = int(array_idx + 1)

            subclass = str(u.cell_subclasses[pos]) if pos < len(u.cell_subclasses) else ""
            cluster = str(u.cell_clusters[pos]) if pos < len(u.cell_clusters) else ""

            rows.append({
                "global_idx": global_idx,
                "animal": u.animal,
                "session_id": u.session_id,
                "plane": str(u.plane),
                "unit_key": key,
                "array_idx": array_idx,
                "ROI_1based": roi_1based,
                "cell_subclass": subclass,
                "cell_cluster": cluster,
                "npz_path": os.path.abspath(u.npz_path),
                # optional manifest annotations
                "has_bpod": (u.manifest_flags or {}).get("has_bpod", None),
                "has_check": (u.manifest_flags or {}).get("has_check", None),
                "n_pairs_total": (u.manifest_match_summary or {}).get("n_pairs_total", None),
                "ge40_s1": ((u.manifest_match_summary or {}).get("ge40", {}) or {}).get("s1", None),
                "ge40_s2": ((u.manifest_match_summary or {}).get("ge40", {}) or {}).get("s2", None),
            })
            unit_to_obs[key].append(global_idx)
            global_idx += 1

    df = pd.DataFrame(rows)

    # Stats
    stats = {
        "n_units": int(len(units)),
        "n_global_obs": int(df.shape[0]),
        "n_min_per_unit": int(n_min_per_unit),
        "n_max_per_unit": int(n_max_per_unit),
        "cell_select": cell_select,
        "celltype_field": celltype_field,
        "unit_sizes_kept": {u.unit_key: int(u.n_kept) for u in units},
        "unit_sampled": {k: int(len(v)) for k, v in unit_to_obs.items()},
        "cell_subclass_counts": dict(Counter(df["cell_subclass"].astype(str).tolist())),
        "cell_cluster_counts": dict(Counter(df["cell_cluster"].astype(str).tolist())),
    }
    return df, dict(unit_to_obs), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True, help="e.g. kd95")
    ap.add_argument("--stage1_dir", required=True, help="e.g. /home/jingyi.xu/ALM/results/stage1")
    ap.add_argument("--out_dir", required=True, help="output directory for registry files")
    ap.add_argument("--manifest", default=None, help="optional: ~/ALM/meta/manifest.json")
    ap.add_argument("--require_manifest_flags", action="store_true",
                    help="if set, filter units by manifest flags (bpod/check).")
    ap.add_argument("--require_has_bpod", action="store_true", help="used with --require_manifest_flags")
    ap.add_argument("--require_has_check", action="store_true", help="used with --require_manifest_flags")

    # Session selection
    ap.add_argument("--top_sessions", type=int, default=0,
                    help="Keep only top-K sessions by available neurons (0 = keep all).")
    ap.add_argument("--session_agg", choices=["sum", "max"], default="sum",
                    help="How to aggregate neuron counts across planes for session ranking.")

    # Registry sampling
    ap.add_argument("--n_global_obs", type=int, default=800, help="Total observed (entity) neurons in registry.")
    ap.add_argument("--n_min_per_unit", type=int, default=10, help="Min sampled neurons per unit (session.plane).")
    ap.add_argument("--n_max_per_unit", type=int, default=40, help="Max sampled neurons per unit (session.plane).")
    ap.add_argument("--celltype_field", choices=["subclass", "cluster"], default="subclass")
    ap.add_argument("--cell_select", choices=["random", "first"], default="random")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    stage1_animal_dir = os.path.join(args.stage1_dir, args.animal)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = load_manifest_optional(args.manifest)
    manifest_idx = index_manifest(manifest, args.animal) if manifest is not None else None

    units = scan_stage1_units(
        stage1_animal_dir=stage1_animal_dir,
        animal=args.animal,
        manifest_idx=manifest_idx,
        require_flags=args.require_manifest_flags,
        require_has_bpod=args.require_has_bpod,
        require_has_check=args.require_has_check,
    )

    if len(units) == 0:
        raise RuntimeError(f"No stage1 units found under: {stage1_animal_dir}")

    # Choose top-K sessions (by total neurons across planes)
    units_kept, sess_scores = choose_top_sessions(
        units, top_sessions=args.top_sessions, session_agg=args.session_agg
    )

    # Print session ranking summary
    print(f"\n[INFO] Found {len(units)} units total. Keeping {len(units_kept)} units after session selection.")
    print("[INFO] Top sessions (session_id, score):")
    for sid, sc in sess_scores[:min(len(sess_scores), max(10, args.top_sessions or 10))]:
        print(f"  {sid}: {sc}")

    # Build registry
    df, unit_to_obs, stats = sample_registry(
        units=units_kept,
        n_global_obs=args.n_global_obs,
        n_min_per_unit=args.n_min_per_unit,
        n_max_per_unit=args.n_max_per_unit,
        celltype_field=args.celltype_field,
        cell_select=args.cell_select,
        seed=args.seed,
    )

    out_csv = os.path.join(args.out_dir, f"{args.animal}.registry.csv")
    out_map = os.path.join(args.out_dir, f"{args.animal}.unit_to_obs_idx.json")
    out_stats = os.path.join(args.out_dir, f"{args.animal}.registry.stats.json")

    df.to_csv(out_csv, index=False)
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(unit_to_obs, f, ensure_ascii=False, indent=2)
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[OK] Registry written:")
    print("  ", out_csv)
    print("  ", out_map)
    print("  ", out_stats)

    # Quick report
    print("\n[REPORT] Sampled neurons per unit (top 15 by sampled):")
    sampled = sorted(stats["unit_sampled"].items(), key=lambda x: -x[1])[:15]
    for k, v in sampled:
        print(f"  {k}: {v} / {stats['unit_sizes_kept'].get(k, '?')}")

    print("\n[REPORT] Cell subclass counts (top 20):")
    c = Counter(df["cell_subclass"].astype(str).tolist())
    for name, cnt in c.most_common(20):
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    main()
