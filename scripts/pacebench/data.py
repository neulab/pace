"""Data loading: source matrix X, target matrix Y, per-target instance tensor P_target."""

import numpy as np
import pandas as pd

from pacebench.config import (
    BASE_DIR, MODELS, TARGET_DATASETS, EXCLUDE_DATASETS,
)


def _get_leaf_dirs(base_dir):
    """Enumerate benchmark leaf directories (containing CSVs)."""
    leaves = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir(): continue
        if list(d.glob("*.csv")): leaves.append((d.name, d))
        else:
            for sub in sorted(d.iterdir()):
                if sub.is_dir() and list(sub.glob("*.csv")):
                    leaves.append((d.name, sub))
    return leaves


def _load_leaf(leaf_path, models):
    """Load one benchmark's per-instance scores for the given models into a (n_models, n_inst) matrix."""
    model_series = {}; all_ids = set()
    for model in models:
        p = leaf_path / f"{model}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p, dtype=str)
        # Legacy 2-column CSVs (id, score) sometimes use the metric_name as `id`
        # and put the actual numeric score under `id`; only when there is NO
        # explicit `metric_name` column AND the `id` column is fully numeric do
        # we swap them. With the modern 3-column format (id, score, metric_name)
        # we MUST trust the columns as-is — earlier code's blanket swap silently
        # zeroed out aime25/gpqa/ifeval/livecodebench/acp_gen.
        if "metric_name" not in df.columns:
            if pd.to_numeric(df["id"], errors="coerce").notna().all():
                df = df.rename(columns={"id": "score", "metric_name": "id"})
        df = df.drop_duplicates(subset="id")
        s = df.set_index("id")["score"]
        if not isinstance(s, pd.Series): s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        model_series[model] = s
        all_ids.update(s.index.tolist())
    if not all_ids:
        return [], np.zeros((len(models), 0), dtype=np.float32)
    ids = sorted(all_ids, key=str)
    id_pos = {iid: j for j, iid in enumerate(ids)}
    scores = np.zeros((len(models), len(ids)), dtype=np.float32)
    for i, model in enumerate(models):
        if model in model_series:
            for iid, val in model_series[model].items():
                scores[i, id_pos[iid]] = float(val)
    return ids, scores


def build_data(base_dir=BASE_DIR, models=MODELS,
               target_datasets=TARGET_DATASETS,
               exclude_datasets=EXCLUDE_DATASETS,
               source_datasets=None,
               verbose=True):
    """
    Load source matrix X (n, |S|), target mean scores Y (n, T), target names,
    and per-target instance tensor list P_target [T × (n, n_t_inst)].

    Parameters
    ----------
    target_datasets : set[str]
        Benchmarks to treat as targets (must exist under base_dir).
    exclude_datasets : set[str]
        Benchmarks to skip entirely (neither source nor target).
    source_datasets : set[str] or None
        If provided, restrict source candidates to this set. If None (default),
        use all non-target, non-excluded leaf directories as source.
    """
    target_datasets = set(target_datasets)
    exclude_datasets = set(exclude_datasets) if exclude_datasets else set()
    if source_datasets is not None:
        source_datasets = set(source_datasets)

    leaves = _get_leaf_dirs(base_dir)
    source_blocks = []
    source_names_seen = []
    source_col_info = []   # per-column metadata: (benchmark, subdir, instance_id)
    target_blocks = {t: [] for t in target_datasets}
    if verbose:
        print(f"Loading {len(leaves)} leaf directories …", flush=True)
    for dname, leaf_path in leaves:
        if dname in exclude_datasets: continue
        is_target = dname in target_datasets
        if not is_target and source_datasets is not None and dname not in source_datasets:
            continue
        ids, scores = _load_leaf(leaf_path, models)
        if len(ids) == 0: continue
        if is_target:
            target_blocks[dname].append(scores)
        else:
            source_blocks.append(scores)
            source_names_seen.append(dname)
            subdir = leaf_path.name if leaf_path.name != dname else ""
            for iid in ids:
                source_col_info.append((dname, subdir, str(iid)))

    X = (np.hstack(source_blocks).astype(np.float32) if source_blocks
         else np.zeros((len(models), 0), dtype=np.float32))
    if verbose:
        n_src = len(source_names_seen)
        print(f"  Source: {X.shape[1]:,} instances from {n_src} benchmark(s)"
              + (f"  [{', '.join(sorted(set(source_names_seen)))}]" if n_src <= 6 else ""),
              flush=True)

    target_names = sorted(target_datasets)
    Y = np.zeros((len(models), len(target_names)), dtype=np.float64)
    P_target = []
    for j, tname in enumerate(target_names):
        if target_blocks[tname]:
            all_sc = np.hstack(target_blocks[tname]).astype(np.float32)
            Y[:, j] = all_sc.mean(axis=1)
            P_target.append(all_sc)
            if verbose:
                print(f"  Target {tname}: {all_sc.shape[1]:,} instances", flush=True)
        else:
            P_target.append(np.zeros((len(models), 0), dtype=np.float32))
    return X, Y, target_names, P_target, source_col_info


def filter_valid_columns(X, eps=1e-10, return_mask=False):
    """Drop source columns with near-zero variance (non-informative).

    If return_mask=True, also return the boolean validity mask over the original
    columns — useful for applying the same filter to parallel per-column metadata.
    """
    var = X.var(axis=0)
    valid = var > eps
    X_v = X[:, valid].astype(np.float64)
    if return_mask:
        return X_v, valid
    return X_v
