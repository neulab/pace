import os
import pandas as pd
from typing import List, Dict, Tuple

# SWE CSV schema
SWE_ID_COL = "metadata.instance_id"
SWE_SCORE_COL = "metadata.scores.resolved"  # 0/1 or "unknown"

DEFAULT_SWE_CSV_DIR = "swebench"


def _load_swe_csv_as_dict(csv_path: str, id_col: str = SWE_ID_COL, score_col: str = SWE_SCORE_COL) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or score_col not in df.columns:
        raise ValueError(
            f"CSV {csv_path} missing required columns. Need {id_col} and {score_col}. Got: {list(df.columns)[:30]}"
        )
    inst = df[id_col].astype(str)
    sc_num = pd.to_numeric(df[score_col], errors="coerce")
    sc = sc_num.fillna(0.0).clip(lower=0.0, upper=1.0)
    return dict(zip(inst.tolist(), sc.astype(float).tolist()))


def list_canonical_models(swe_csv_dir: str = DEFAULT_SWE_CSV_DIR) -> List[str]:
    if not os.path.isdir(swe_csv_dir):
        return []
    models = []
    for fn in os.listdir(swe_csv_dir):
        if fn.endswith(".csv"):
            models.append(fn[:-4])
    models.sort()
    return models


def load_model_outputs_for_models(model_names: List[str], swe_csv_dir: str = DEFAULT_SWE_CSV_DIR) -> Tuple[List[str], List[Dict[str, float]]]:
    kept_models = []
    dicts = []
    for m in model_names:
        p = os.path.join(swe_csv_dir, f"{m}.csv")
        if not os.path.exists(p):
            continue
        dicts.append(_load_swe_csv_as_dict(p))
        kept_models.append(m)
    return kept_models, dicts
