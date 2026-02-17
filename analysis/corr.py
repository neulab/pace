import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd

def calc_swebench_corr(
    json1: dict,
    model_acc: dict,
    method: str = "pearson",
    verbose: bool = True,
):
    """
    Compute correlation between SWE-bench values in json1 and external model accuracies.

    Args:
        json1 (dict): model -> metrics dict (must contain "swebench")
        model_acc (dict): model -> accuracy
        method (str): "pearson", "spearman", or "kendall"
        verbose (bool): print matched models and values

    Returns:
        corr (float)
        p_value (float)
        used_models (list)
    """

    xs, ys, used_models = [], [], []

    for model, acc in model_acc.items():
        if model not in json1:
            continue

        swe = json1[model]
        if swe is None:
            continue

        xs.append(float(swe))
        ys.append(float(acc))
        used_models.append(model)

    if len(xs) < 2:
        raise ValueError("Need at least 2 overlapping models to compute correlation")

    xs = np.array(xs)
    ys = np.array(ys)

    if method == "pearson":
        corr, p = pearsonr(xs, ys)
    elif method == "spearman":
        corr, p = spearmanr(xs, ys)
    elif method == "kendall":
        corr, p = kendalltau(xs, ys)
    else:
        raise ValueError(f"Unknown method: {method}")

    if verbose:
        print(f"\nCorrelation method: {method}")
        print(f"Models used ({len(used_models)}):")
        for m, x, y in zip(used_models, xs, ys):
            print(f"  {m:30s} swebench={x:6.2f}  acc={y:6.2f}")
        print(f"\nCorrelation = {corr:.4f}, p-value = {p:.4g}")

    return corr, p, used_models


def calc_swebench_acc(csv_path: str) -> int:
    df = pd.read_csv(csv_path)

    # Coerce to numeric, invalid → NaN
    col = pd.to_numeric(df["metadata.scores.resolved"], errors="coerce")

    # Keep only exact 1s (0 stays 0, everything else becomes 0)
    resolved = (col == 1).astype(int)

    return int(resolved.sum() / 500)

models = [model.replace(".csv", "") for model in os.listdir("/home/yueqis/proxybench/swebench") if ".csv" in model]
swe = {model: calc_swebench_acc(f"/home/yueqis/proxybench/swebench/{model}.csv") for model in models}

def calc_bfcl_corr():
    out = {}
    model_map = {
        "claude-opus-4-1-20250805": "Claude-Opus-4", "claude-sonnet-4-5-20250929": "Claude-Sonnet-4.5", "Gemini-2.5-Pro": "Gemini-2.5-Pro", 
        "DeepSeek-V3.2-Exp-thinking": "DeepSeek-V3.2-Reasoner", "GPT-5__high": "GPT-5", "o3__high": "o3", "Gemini-3-Pro-Preview": "Gemini-3-Pro-Preview", 
        "gpt-5-mini-2025-08-07": "GPT-5-mini", "o4-mini__high": "o4-mini"
    }
    model_map = {val: key for key, val in model_map.items()}
    print(model_map)
    for model in swe.keys():
        if model not in model_map: 
            print(model)
            continue
        model = model_map[model]
        if model in os.listdir("/home/yueqis/proxybench/BFCL/BFCL"):
            with open(f"/home/yueqis/proxybench/BFCL/BFCL/{model}/live/BFCL_v4_live_multiple_score.json") as file: 
                acc = json.loads(file.readlines()[0])["accuracy"]
            out[model] = acc
        else: print(f"{model} not avail")
    calc_swebench_corr(swe, out)
calc_bfcl_corr()