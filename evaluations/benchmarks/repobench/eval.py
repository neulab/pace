import os
import json
import csv
from fuzzywuzzy import fuzz
from evaluation.metrics import exact_match_score, edit_similarity_score, codebleu_score
import fire


def compute_sample_exact_match(pred, gt):
    """Compute exact match for a single sample."""
    return 1.0 if pred.split() == gt.split() else 0.0


def compute_sample_edit_similarity(pred, gt):
    """Compute edit similarity for a single sample."""
    return fuzz.ratio(pred, gt)


def eval(
    path="results/deepseek-coder-1.3b-base-python",
    language="python", # to calculate codebleu, we need to specify the language
    output_dir="repobench_results",  # base output directory for CSV files
    metric="em"
):
    # Mapping from level names to output directory names
    level_to_dir = {
        "cross_file_first": "repobench_xff_python",
        "cross_file_random": "repobench_xfr_python",
        "in_file": "repobench_if_python"
    }

    metrics_to_compute = {
        "em": "exact_match",
        "es": "edit_similarity",
        "cb": "codebleu"
    }

    chosen_metric = metrics_to_compute.get(metric, "exact_match")

    # Extract model name from path
    model_name = os.path.basename(path.rstrip("/"))

    total_data_points = 0
    total_em_model, total_es_model, total_cb_model = 0, 0, 0

    for level in ["cross_file_first", "cross_file_random", "in_file"]:
        filepath = os.path.join(path, f"{level}.jsonl")
        seen_indices = set()  # Track seen indices for the current level

        # check if the file exists
        if not os.path.exists(filepath):
            print(f"Level: {level} not found for the model")
            continue

        with open(filepath, "r") as f:

            data = []
            for line in f:
                entry = json.loads(line.strip())
                idx = entry["idx"]

                # Skip duplicate indices based on the chosen policy (here, keeping the former)
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    data.append(entry)

            data_points = len(data)

            if data_points == 0:
                continue

            ground_truth = [d["gt"] for d in data]
            generated = [d["pred"] for d in data]
            idx = [d["idx"] for d in data]

            em_model = round(exact_match_score(ground_truth, generated) * 100, 2)
            es_model = round(edit_similarity_score(ground_truth, generated), 2)
            cb_model = round(codebleu_score(generated, ground_truth, language) * 100, 2)

            # Compute sample-level metrics and save to CSV
            sample_metrics = []
            for _idx, pred, gt in zip(idx, generated, ground_truth):
                em_sample = compute_sample_exact_match(pred, gt)
                es_sample = compute_sample_edit_similarity(pred, gt)
                cb_sample = codebleu_score([pred], [gt], language) * 100  # CodeBLEU for a single sample
                sample_metrics.append({
                    "idx": _idx,
                    "exact_match": em_sample,
                    "edit_similarity": es_sample,
                    "codebleu": cb_sample,
                })

            # Create output directory and save CSV
            csv_dir = os.path.join(output_dir, level_to_dir[level])
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"{model_name}-{chosen_metric}.csv")

            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["idx", "score", "metric_name"])
                for metrics in sample_metrics:
                    writer.writerow([metrics["idx"], metrics[chosen_metric], chosen_metric])
                    # writer.writerow([metrics["exact_match"], "exact_match"])
                    # writer.writerow([metrics["edit_similarity"], "edit_similarity"])
                    # writer.writerow([metrics["codebleu"], "codebleu"])

            print(f"Saved sample-level metrics to {csv_path}")

            # accumulate the data points and the metrics
            total_data_points += data_points
            total_em_model += em_model * data_points
            total_es_model += es_model * data_points
            total_cb_model += cb_model * data_points

            print(f"Level: {level} with {data_points} data points")
            print(f"EM: {em_model}, ES: {es_model}, CB: {cb_model}")
            print("-" * 30)

    # calculate the weighted averages
    if total_data_points > 0:
        avg_em_model = round(total_em_model / total_data_points, 2)
        avg_es_model = round(total_es_model / total_data_points, 2)
        avg_cb_model = round(total_cb_model / total_data_points, 2)

        print("Weighted Averages:")
        print(f"EM: {avg_em_model}, ES: {avg_es_model}, CB: {avg_cb_model}\n")

    else:
        print("No data points were found for evaluation.")
        
if __name__ == "__main__":
    fire.Fire(eval)