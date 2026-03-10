#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def find_summary_files(root: Path) -> List[Path]:
    # Look for any */summary.json recursively one level or more
    return list(root.rglob("summary.json"))


def load_summary(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def collect_rows(root: Path, only_full: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    for summ_path in find_summary_files(root):
        pair = summ_path.parent.name  # e.g., humaneval_to_gaia
        try:
            data = load_summary(summ_path)
        except Exception as e:
            print(f"[WARN] Failed to load {summ_path}: {e}")
            continue

        for tag, res in data.items():
            # Optionally filter to FULL__vs__FULL
            if only_full and not ("_FULL__vs__" in tag and tag.endswith("_FULL")):
                continue

            params = res.get("params", {})
            corr_train = res.get("corr_train", {})
            corr_eval = res.get("corr_eval", {})

            row = {
                "pair": pair,
                "tag": tag,
                "k_source": params.get("k_source"),
                "target_train_size": params.get("target_train_size"),
                "target_eval_size": params.get("target_eval_size"),
                "pearson_train": corr_train.get("pearson"),
                "pearson_eval": corr_eval.get("pearson"),
                "spearman_train": corr_train.get("spearman"),
                "spearman_eval": corr_eval.get("spearman"),
                "json_path": str(summ_path),
            }
            rows.append(row)
    return rows


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair",
        "tag",
        "k_source",
        "target_train_size",
        "target_eval_size",
        "pearson_train",
        "pearson_eval",
        "spearman_train",
        "spearman_eval",
        "json_path",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(rows: List[Dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        ("pair", "Pair"),
        ("tag", "Tag"),
        ("k_source", "k_src"),
        ("target_train_size", "TGT_train"),
        ("target_eval_size", "TGT_eval"),
        ("pearson_train", "r_train"),
        ("pearson_eval", "r_eval"),
        ("spearman_train", "rho_train"),
        ("spearman_eval", "rho_eval"),
    ]
    with open(out_md, "w") as f:
        # header
        f.write("| " + " | ".join(h for _, h in cols) + " |\n")
        f.write("|" + "|".join([" --- "] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize train/eval correlations from bootstrap summary.json files")
    parser.add_argument("--root", required=True, help="Root directory containing <source>_to_<target>/summary.json folders")
    parser.add_argument("--out_csv", default="", help="Path to save CSV table")
    parser.add_argument("--out_md", default="", help="Path to save Markdown table")
    parser.add_argument("--only_full", action="store_true", help="Include only FULL__vs__FULL tags")
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_rows(root, only_full=args.only_full)

    # Sort for readability: by pair then tag
    rows.sort(key=lambda r: (r.get("pair", ""), r.get("tag", "")))

    # Print a compact preview to stdout
    print("pair, tag, pearson_train, pearson_eval, spearman_train, spearman_eval")
    for r in rows:
        print(
            f"{r.get('pair')}, {r.get('tag')}, "
            f"{r.get('pearson_train')}, {r.get('pearson_eval')}, "
            f"{r.get('spearman_train')}, {r.get('spearman_eval')}"
        )

    if args.out_csv:
        write_csv(rows, Path(args.out_csv))
        print(f"Saved CSV to: {args.out_csv}")

    if args.out_md:
        write_markdown(rows, Path(args.out_md))
        print(f"Saved Markdown to: {args.out_md}")


if __name__ == "__main__":
    main()
