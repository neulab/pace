import json
import os
from os.path import join

def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_output_tokens(output_dir):
    """Extract token counts from model output files."""

    output_files = [f for f in os.listdir(output_dir) if f.startswith('test_output_') and f.endswith('.jsonl')]

    print(f"Found {len(output_files)} output files\n")
    print(f"{'Model':<45} {'Instances':>10} {'Prompt':>12} {'Completion':>12} {'Total':>12}")
    print("-" * 95)

    all_results = {}

    for output_file in sorted(output_files):
        file_path = join(output_dir, output_file)
        model_name = output_file.replace('test_output_', '').replace('.jsonl', '')

        data = load_jsonl(file_path)

        total_prompt = 0
        total_completion = 0

        for entry in data:
            prompt_tokens = entry.get('prompt_tokens', 0) or 0
            completion_tokens = entry.get('completion_tokens', 0) or 0
            total_prompt += prompt_tokens
            total_completion += completion_tokens

        total_tokens = total_prompt + total_completion

        print(f"{model_name:<45} {len(data):>10} {total_prompt:>12,} {total_completion:>12,} {total_tokens:>12,}")

        all_results[model_name] = {
            'instances': len(data),
            'prompt_tokens': total_prompt,
            'completion_tokens': total_completion,
            'total_tokens': total_tokens
        }

    # Save summary to JSON
    summary_file = join(output_dir, 'token_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("-" * 95)
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    output_dir = "/Users/vincentlo/InfoBench/output_files"
    extract_output_tokens(output_dir)
