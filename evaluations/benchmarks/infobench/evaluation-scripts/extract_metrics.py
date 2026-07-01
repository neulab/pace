import json
import os
from os.path import join, exists

def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_metrics(eval_dir):
    """Extract accuracy and token metrics from evaluation files."""

    # Find all evaluation files (only DecomposeEval files, not metrics files)
    eval_files = [f for f in os.listdir(eval_dir) if ('DecomposeEval' in f) and (f.endswith('.json') or f.endswith('.jsonl'))]

    print(f"Found {len(eval_files)} evaluation files")

    for eval_file in eval_files:
        file_path = join(eval_dir, eval_file)
        model_name = eval_file.replace('_DecomposeEval.json', '').replace('_DecomposeEval.jsonl', '')

        # Skip infobench-project files (duplicates)
        if model_name == 'infobench-project':
            print(f"Skipping {eval_file} (duplicate)")
            continue

        print(f"\nProcessing: {model_name}")

        data = load_jsonl(file_path)
        print(f"  Loaded {len(data)} instances")

        metrics = []
        for entry in data:
            instance_id = entry.get('id', 'unknown')
            eval_results = entry.get('eval', [])
            prompt_tokens = entry.get('prompt_tokens', 0) or 0
            completion_tokens = entry.get('completion_tokens', 0) or 0
            total_tokens = prompt_tokens + completion_tokens

            # Calculate accuracy
            if eval_results:
                true_count = sum(1 for r in eval_results if r is True)
                total_questions = len(eval_results)
                accuracy = (true_count / total_questions) * 100 if total_questions > 0 else 0
            else:
                true_count = 0
                total_questions = 0
                accuracy = 0

            metrics.append({
                'id': instance_id,
                'accuracy_percent': round(accuracy, 2),
                'correct': true_count,
                'total_questions': total_questions,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            })

        # Calculate overall stats
        if metrics:
            avg_accuracy = sum(m['accuracy_percent'] for m in metrics) / len(metrics)
            total_prompt = sum(m['prompt_tokens'] for m in metrics)
            total_completion = sum(m['completion_tokens'] for m in metrics)
            print(f"  Average accuracy: {avg_accuracy:.2f}%")
            print(f"  Total tokens: {total_prompt + total_completion:,}")

        # Write metrics file
        output_file = join(eval_dir, f"{model_name}_metrics.json")
        with open(output_file, 'w') as f:
            json.dump({
                'model': model_name,
                'num_instances': len(metrics),
                'average_accuracy': round(avg_accuracy, 2) if metrics else 0,
                'total_prompt_tokens': total_prompt if metrics else 0,
                'total_completion_tokens': total_completion if metrics else 0,
                'instances': metrics
            }, f, indent=2)

        print(f"  Saved to: {output_file}")

if __name__ == "__main__":
    eval_dir = "/Users/vincentlo/InfoBench/evaluation/litellm_proxy_azure_gpt-4.1"
    extract_metrics(eval_dir)
    print("\nDone!")
