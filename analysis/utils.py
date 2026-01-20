import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

with open("swebench.json") as f: swebench = json.load(f)
def get_swe(model):
    return swebench[model]["swebench"]

def get_task_count_by_diff(task):
    if task == "codeexecution": return {"easy": 216, "medium": 254, "hard": 9}
    if task == "codegeneration": return {"easy": 322, "medium": 383, "hard": 350}
    if task == "selfrepair": return {"easy": 322, "medium": 383, "hard": 350}
    if task == "testoutputprediction": return {"easy": 147, "medium": 223, "hard": 72}

def get_task_key(task, line):
    if task == "codeexecution": return line["id"]
    if task == "codegeneration": return line["question_id"]
    if task == "selfrepair": return line["question_id"]
    if task == "testoutputprediction": return f'{line["question_id"]}_{line["test_id"]}'

def load_model_results_0(model, task, d):
    model_dir = f"{d}/{model}"
    if not os.path.exists(f"{model_dir}"): return {}, 0
    output = {}
    count = 0
    for i in range(1):
        if task == "codeexecution": file_path = f"{model_dir}/{i}/{task}_1_cot_eval_all.json"
        else: file_path = f"{model_dir}/{i}/{task}_1_eval_all.json"
        if not os.path.exists(file_path): 
            continue
        count += 1
        with open(file_path) as f: f = json.load(f)
        for line in f:
            line_key = get_task_key(task, line)
            if line_key not in output: output[line_key] = {}
            output[line_key] = line
    return output, count

def load_task_results(model, task):
    model_dir = f"/home/yueqis/LiveCodeBench/output/{model}"
    if not os.path.exists(f"{model_dir}"): return {}, 0
    output = {}
    count = 0
    for i in range(10):
        if task == "codeexecution": file_path = f"{model_dir}/{i}/{task}_1_cot_eval_all.json"
        else: file_path = f"{model_dir}/{i}/{task}_1_eval_all.json"
        if not os.path.exists(file_path): 
            continue
        count += 1
        with open(file_path) as f: f = json.load(f)
        for line in f:
            line_key = get_task_key(task, line)
            if line_key not in output: output[line_key] = {}
            output[line_key][i] = line
    return output, count

def get_accuracy_by_diff(model, task):
    output, num_iter = load_task_results(model, task)
    results = {"easy": 0, "medium": 0, "hard": 0}
    for iters in output.values():
        for i in iters:
            results[iters[i]["difficulty"]] += iters[i]["pass@1"]
    task_count = get_task_count_by_diff(task)
    accs = {"easy": 0, "medium": 0, "hard": 0}
    for diff in results:
        accs[diff] = round(results[diff] / num_iter / task_count[diff] * 100, 1)
    accs["avg"] = round(sum(results.values()) / num_iter / sum(task_count.values()) * 100, 1)
    # print(accs)
    return accs

def analyze_and_plot(x, y, save_path):
    x = np.array(x)
    y = np.array(y)

    # Compute correlations
    pearson_corr, pearson_p = pearsonr(x, y)
    spearman_corr, spearman_p = spearmanr(x, y)

    print(f"Pearson r = {pearson_corr:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman ρ = {spearman_corr:.4f}, p = {spearman_p:.4e}")

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.7)

    # Add labels and title
    plt.xlabel("Accuracy")
    plt.ylabel("SWE Bench")
    plt.title(
        f"Correlation Plot ({save_path})\n"
        f"Pearson = {pearson_corr:.3f} (p={pearson_p:.4e}), Spearman = {spearman_corr:.3f} (p={spearman_p:.4e})"
    )

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color="red", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def plot_four_correlations(y, x_lists, save_path="four_corr_plots.png"):
    """
    y: list or numpy array of target values
    x_lists: list of four lists [x1, x2, x3, x4]
    titles: optional list of subplot titles
    save_path: file path to save the figure
    """

    y = np.array(y)
    assert len(x_lists) == 4, "You must provide exactly 4 x-lists."

    titles = ["easy", "medium", "hard", "avg"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    correlations = []

    for i, ax in enumerate(axes):
        x = np.array(x_lists[i])

        # Compute correlations
        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)
        correlations.append((pearson_corr, spearman_corr))

        # Scatter plot
        ax.scatter(x, y, alpha=0.7)

        # Best fit line
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color="red", linewidth=2)

        # Labels + title
        ax.set_title(
            f"Correlation Plot ({save_path}-{titles[i]})\n"
            f"Pearson = {pearson_corr:.3f} (p={pearson_p:.4e}), Spearman = {spearman_corr:.3f} (p={spearman_p:.4e})"
        )
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("SWE Bench")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved 4-panel figure to {save_path}")
    return correlations

def get_accuracy_corr(task):
    output_dir = "/home/yueqis/LiveCodeBench/output/"
    models = os.listdir(output_dir)
    x = {"easy": [], "medium": [], "hard": [], "avg": []}
    y = []
    for model in models:
        accs = get_accuracy_by_diff(model, task)
        for diff in x:
            x[diff].append(accs[diff])
        y.append(swebench[model]["swebench"])
    print(list(x.values()), y)
    plot_four_correlations(y, list(x.values()), f"figures/{task}_{diff}.jpg")

# get_accuracy_corr("testoutputprediction")
