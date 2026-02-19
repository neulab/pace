#!/usr/bin/env python3
"""
Explore BigCodeBench Dataset

Load and display examples from the BigCodeBench dataset.
"""

from datasets import load_dataset
import json

def load_bigcodebench(subset: str = "full", split: str = "v0.1.4"):
    """Load BigCodeBench dataset from HuggingFace."""
    if subset == "full":
        hf_path = "bigcode/bigcodebench"
    else:
        hf_path = f"bigcode/bigcodebench-{subset}"

    data = load_dataset(hf_path, split=split)
    print(f"Loaded {len(data)} tasks from {hf_path}")
    return data


def get_task_by_id(data, task_id: str):
    """Get a specific task by its task_id."""
    for task in data:
        if task["task_id"] == task_id:
            return task
    return None


def display_task(task: dict, show_test: bool = False):
    """Display a task's details."""
    print("=" * 80)
    print(f"TASK ID: {task['task_id']}")
    print("=" * 80)

    print("\n" + "-" * 40)
    print("ENTRY POINT:")
    print("-" * 40)
    print(task.get("entry_point", "N/A"))

    print("\n" + "-" * 40)
    print("INSTRUCT PROMPT:")
    print("-" * 40)
    print(task.get("instruct_prompt", "N/A"))

    print("\n" + "-" * 40)
    print("COMPLETE PROMPT (function signature):")
    print("-" * 40)
    print(task.get("complete_prompt", "N/A"))

    print("\n" + "-" * 40)
    print("CANONICAL SOLUTION:")
    print("-" * 40)
    print(task.get("canonical_solution", "N/A"))

    print("\n" + "-" * 40)
    print("CODE PROMPT (for calibrated evaluation):")
    print("-" * 40)
    print(task.get("code_prompt", "N/A"))

    if show_test:
        print("\n" + "-" * 40)
        print("TEST CASES:")
        print("-" * 40)
        print(task.get("test", "N/A"))

    print("\n" + "-" * 40)
    print("LIBRARIES USED:")
    print("-" * 40)
    print(task.get("libs", "N/A"))


if __name__ == "__main__":
    # Load the dataset
    print("Loading BigCodeBench dataset...")
    data = load_bigcodebench(subset="full")

    # Show specific example
    task_id = "BigCodeBench/1084"
    print(f"\n\nLooking for task: {task_id}")

    task = get_task_by_id(data, task_id)

    if task:
        display_task(task, show_test=True)
    else:
        print(f"Task {task_id} not found!")

    # Also show available fields
    print("\n" + "=" * 80)
    print("AVAILABLE FIELDS IN DATASET:")
    print("=" * 80)
    if len(data) > 0:
        print(list(data[0].keys()))
