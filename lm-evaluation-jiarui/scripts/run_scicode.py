#!/usr/bin/env python3
"""
SciCode Benchmark Runner

Wrapper script for running SciCode benchmark with lm-eval-harness style interface.
Supports all models via the LiteLLM proxy.

Usage:
    python run_scicode.py --model azure/gpt-4o --output_path ./results --limit 5
"""

import argparse
import json
import logging
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


from model_api import (
    ProxyModelAPI,
    ModelConfig,
    GenerationResult,
    ResultsLogger,
    SUPPORTED_MODELS,
)

# Local evaluation module
try:
    from scicode_eval import evaluate_scicode
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============ SciCode-specific helpers ============

# Prompt template when background is provided (with_background=True)
BACKGROUND_PROMPT_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```.
"""

# Default prompt template (without background - model generates background as comment)
DEFAULT_PROMPT_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
"""


def extract_python_script(response: str) -> str:
    """Extract Python code from model response."""
    if '```' in response:
        if '```python' in response:
            python_script = response.split("```python")[1].split("```")[0]
        else:
            python_script = response.split('```')[1].split('```')[0]
    else:
        logger.warning("No code block found in response")
        python_script = response

    # Remove import statements (they come from dependencies)
    python_script = re.sub(
        r'^\s*(import .*|from .*\s+import\s+.*)',
        '',
        python_script,
        flags=re.MULTILINE
    )
    return python_script.strip()


def extract_function_name(function_header: str) -> str:
    """Extract function name from function header."""
    match = re.search(r'def\s+(\w+)\s*\(', function_header)
    if match:
        return match.group(1)
    return ""


def get_function_from_code(code: str, function_name: str) -> str:
    """Extract a specific function from code."""
    lines = code.split('\n')
    in_function = False
    function_lines = []
    indent_level = 0

    for line in lines:
        if f'def {function_name}(' in line or f'def {function_name} (' in line:
            in_function = True
            # Determine indent level
            indent_level = len(line) - len(line.lstrip())
            function_lines.append(line)
        elif in_function:
            if line.strip() == '':
                function_lines.append(line)
            elif line.startswith(' ' * (indent_level + 1)) or line.startswith('\t'):
                function_lines.append(line)
            elif line.strip().startswith('#'):
                function_lines.append(line)
            else:
                # Check if this is a new definition at same or higher level
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    break
                function_lines.append(line)

    return '\n'.join(function_lines)


class SciCodeRunner:
    """Runner for SciCode benchmark."""

    def __init__(
        self,
        model: str,
        output_path: str,
        base_url: str = "https://cmu.litellm.ai/v1/chat/completions",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        with_background: bool = False,
        use_cache: bool = True,
        cache_dir: str = None,
    ):
        self.model = model
        self.output_path = Path(output_path)
        self.with_background = with_background

        # Initialize model API with caching
        config = ModelConfig(
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        self.api = ProxyModelAPI(config)

        # Track previous LLM code for multi-step problems
        self.previous_llm_code: Dict[str, List[Optional[str]]] = {}

    def process_problem_code(self, prob_data: dict, step_idx: int) -> str:
        """Process problem code for a step."""
        sub_step = prob_data['sub_steps'][step_idx]
        header_docstring = sub_step['function_header']
        return_str = sub_step['return_line']
        return f"{header_docstring}\n\n{return_str}"

    def process_problem_steps(self, problem_data: dict, num_steps: int, prob_id: str) -> tuple:
        """Process problem data and return previous steps and next step."""
        output_lines = []
        previous_code = []

        # Process previous steps
        for i in range(num_steps - 1):
            sub_step = problem_data["sub_steps"][i]
            step_desc = sub_step["step_description_prompt"]
            if self.with_background:
                step_desc += '\n' + sub_step.get("step_background", "")
            output_lines.append(step_desc)

            # Get previous code
            prev_code = self.previous_llm_code.get(prob_id, [None] * len(problem_data["sub_steps"]))[i]
            if prev_code:
                output_lines.append(prev_code)
                previous_code.append(prev_code)
            output_lines.append("------")

        # Process next step
        next_sub_step = problem_data["sub_steps"][num_steps - 1]
        next_step_desc = next_sub_step["step_description_prompt"]
        if self.with_background:
            next_step_desc += '\n' + next_sub_step.get("step_background", "")

        next_step_lines = [
            next_step_desc,
            self.process_problem_code(problem_data, num_steps - 1)
        ]

        problem_steps_str = "\n\n".join(output_lines[:-1]) if output_lines else ""  # Remove last "------"
        next_step_str = "\n\n".join(next_step_lines)
        previous_code_str = "\n".join(previous_code)

        return problem_steps_str, next_step_str, previous_code_str

    def generate_prompt(self, prob_data: dict, num_steps: int, prob_id: str) -> tuple:
        """Generate prompt for a step."""
        problem_steps_str, next_step_str, previous_code_str = self.process_problem_steps(
            prob_data, num_steps, prob_id
        )
        dependencies = prob_data.get("required_dependencies", "")

        # Use appropriate template based on with_background flag
        # (faithful to original gencode.py behavior)
        template = BACKGROUND_PROMPT_TEMPLATE if self.with_background else DEFAULT_PROMPT_TEMPLATE

        prompt = template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        )

        return prompt, f'{dependencies}\n{previous_code_str}\n'

    def generate_step(self, prob_data: dict, step_num: int) -> tuple[str, str, str]:
        """Generate code for a single step."""
        prob_id = prob_data["problem_id"]
        tot_steps = len(prob_data["sub_steps"])

        # Initialize previous code tracking
        if prob_id not in self.previous_llm_code:
            self.previous_llm_code[prob_id] = [None] * tot_steps

        # Generate prompt
        prompt, previous_code = self.generate_prompt(prob_data, step_num, prob_id)

        # Call model
        messages = [{"role": "user", "content": prompt}]
        response = self.api.generate(messages)

        # Extract code
        extracted_code = extract_python_script(response)

        # Store for next steps
        self.previous_llm_code[prob_id][step_num - 1] = extracted_code

        return prompt, response, extracted_code


def run_scicode(
    model: str,
    output_path: str,
    split: str = "test",
    limit: Optional[int] = None,
    apply_chat_template_flag: bool = True,
    log_samples: bool = True,
    base_url: str = "https://cmu.litellm.ai/v1/chat/completions",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    with_background: bool = False,
    evaluate: bool = False,
    h5py_file_path: Optional[str] = None,
    eval_timeout: int = 1800,
    use_cache: bool = True,
    cache_dir: str = None,
):
    """Run SciCode benchmark evaluation."""
    from datasets import load_dataset

    logger.info(f"Running SciCode benchmark")
    logger.info(f"  Model: {model}")
    logger.info(f"  Split: {split}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Limit: {limit}")
    logger.info(f"  With background: {with_background}")
    logger.info(f"  Cache enabled: {use_cache}")

    # Load dataset from HuggingFace
    try:
        data = load_dataset("SciCode1/SciCode", split=split)
        logger.info(f"Loaded {len(data)} problems from SciCode1/SciCode")
    except Exception as e:
        logger.error(f"Could not load SciCode dataset: {e}")
        return None

    # Apply limit
    if limit is not None and limit > 0:
        data = data.select(range(min(limit, len(data))))
        logger.info(f"Limited to {len(data)} problems")

    # Initialize runner
    runner = SciCodeRunner(
        model=model,
        output_path=output_path,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        with_background=with_background,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    # Initialize results logger
    benchmark_name = f"scicode_{'bg' if with_background else 'nobg'}"
    results_logger = ResultsLogger(
        output_path=output_path,
        model_name=model,
        benchmark_name=benchmark_name,
    )

    # Prepare SciCode native output directory
    model_sanitized = model.replace("/", "--").replace(":", "_")
    bg_suffix = "with_background" if with_background else "without_background"
    native_output_dir = Path(output_path) / results_logger.model_name_sanitized / "scicode_format" / bg_suffix
    native_output_dir.mkdir(parents=True, exist_ok=True)

    # Prompt output directory
    prompt_output_dir = Path(output_path) / results_logger.model_name_sanitized / "prompts" / bg_suffix
    prompt_output_dir.mkdir(parents=True, exist_ok=True)

    # Run generation
    all_results = []
    total_steps = 0
    completed_steps = 0

    from tqdm import tqdm

    for prob_idx, prob_data in enumerate(tqdm(data, desc="Processing problems")):
        prob_data = dict(prob_data)
        prob_id = prob_data["problem_id"]
        sub_steps = prob_data["sub_steps"]
        num_steps = len(sub_steps)

        logger.info(f"Processing problem {prob_id} with {num_steps} steps")

        problem_results = []

        for step_num in range(1, num_steps + 1):
            step_id = f"{prob_id}.{step_num}"
            step_idx = step_num - 1  # 0-indexed

            # Skip special steps that use reference code (faithful to original)
            # These steps have known issues and use pre-computed reference solutions
            if (prob_id == "13" and step_idx == 5) or \
               (prob_id == "62" and step_idx == 0) or \
               (prob_id == "76" and step_idx == 2):
                logger.info(f"Skipping {step_id} (uses reference code in original)")
                continue

            total_steps += 1

            try:
                # Generate code for this step
                prompt, response, extracted_code = runner.generate_step(prob_data, step_num)

                # Save prompt
                prompt_file = prompt_output_dir / f"{prob_id}.{step_num}.txt"
                prompt_file.write_text(prompt, encoding="utf-8")

                # Save generated code (with dependencies prefix)
                dependencies = prob_data.get("required_dependencies", "")
                prev_code = "\n".join([
                    c for c in runner.previous_llm_code.get(prob_id, [])[:step_num-1]
                    if c is not None
                ])
                full_code = f"{dependencies}\n{prev_code}\n{extracted_code}"

                code_file = native_output_dir / f"{prob_id}.{step_num}.py"
                code_file.write_text(full_code, encoding="utf-8")

                step_result = {
                    "step_id": step_id,
                    "prompt": prompt,
                    "response": response,
                    "extracted_code": extracted_code,
                    "status": "generated",
                }
                problem_results.append(step_result)
                completed_steps += 1

                # Log sample
                if log_samples:
                    result = GenerationResult(
                        doc_id=step_id,
                        doc={
                            "problem_id": prob_id,
                            "step_num": step_num,
                            "function_header": sub_steps[step_num-1].get("function_header", ""),
                        },
                        prompt=[{"role": "user", "content": prompt}],
                        response=response,
                        metadata={"extracted_code": extracted_code},
                    )
                    results_logger.add_sample(result)

            except Exception as e:
                logger.error(f"Error on step {step_id}: {e}")
                problem_results.append({
                    "step_id": step_id,
                    "error": str(e),
                    "status": "failed",
                })

        all_results.append({
            "problem_id": prob_id,
            "num_steps": num_steps,
            "steps": problem_results,
        })

    # Calculate metrics
    errors = total_steps - completed_steps
    metrics = {
        "num_problems": len(data),
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "total_errors": errors,
        "generation_rate": completed_steps / total_steps if total_steps > 0 else 0,
        "with_background": with_background,
        "cache_hits": runner.api.cache_hits,
        "cache_misses": runner.api.cache_misses,
    }

    # Run evaluation if requested
    if evaluate:
        if not EVAL_AVAILABLE:
            logger.error("Evaluation module not available. Check scicode_eval.py")
            metrics["note"] = "Evaluation skipped - module not available"
        elif h5py_file_path is None or not Path(h5py_file_path).exists():
            logger.error(f"H5PY test data file not found: {h5py_file_path}")
            logger.error("Please download test_data.h5 from SciCode repository")
            metrics["note"] = "Evaluation skipped - test data file not found"
        else:
            logger.info("\n" + "="*50)
            logger.info("Running local evaluation...")
            logger.info("="*50)

            try:
                # Set up log directory for caching test results
                log_dir = Path(output_path) / results_logger.model_name_sanitized / "eval_logs" / bg_suffix

                eval_results = evaluate_scicode(
                    code_dir=native_output_dir,
                    split=split,
                    with_background=with_background,
                    h5py_file_path=h5py_file_path,
                    log_dir=log_dir,
                    timeout=eval_timeout,
                    verbose=True,
                )

                # Add evaluation metrics to results
                metrics["evaluation"] = {
                    "correct_problems": eval_results["correct_problems"],
                    "total_problems": eval_results["total_problems"],
                    "problem_accuracy": eval_results["problem_accuracy"],
                    "correct_steps": eval_results["correct_steps"],
                    "total_steps_tested": eval_results["total_steps"],
                    "step_accuracy": eval_results["step_accuracy"],
                    "test_duration_seconds": eval_results["test_duration_seconds"],
                }

                # Save detailed eval results
                eval_output_file = native_output_dir.parent / f"eval_results_{bg_suffix}.json"
                with open(eval_output_file, "w") as f:
                    json.dump(eval_results, f, indent=2)
                logger.info(f"Saved evaluation results to {eval_output_file}")

            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                metrics["evaluation_error"] = str(e)
    else:
        metrics["note"] = "Run with --evaluate flag for execution metrics"

    # Save results
    if log_samples:
        results_logger.save_samples()
    results_logger.save_results(metrics)

    # Save detailed results
    results_file = native_output_dir.parent / "generation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nGeneration Complete!")
    logger.info(f"  Problems: {len(data)}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Completed: {completed_steps}")
    if errors > 0:
        logger.info(f"  Errors: {errors} (re-run to retry with cache)")
    logger.info(f"  Cache: {runner.api.cache_hits} hits, {runner.api.cache_misses} misses")
    logger.info(f"  Output directory: {native_output_dir}")

    if not evaluate:
        logger.info(f"\nTo evaluate execution:")
        logger.info(f"  python run_scicode.py --model {model} --evaluate --h5py_file_path <path/to/test_data.h5>")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run SciCode benchmark with lm-eval-harness style interface"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., azure/gpt-4o, gemini/gemini-2.5-pro)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/scicode",
        help="Path to save results (default: ./results/scicode)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems to evaluate (default: None = all)",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=True,
        help="Apply chat template to prompts (default: True)",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=True,
        help="Log individual samples to output (default: True)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://cmu.litellm.ai/v1/chat/completions",
        help="Base URL for LiteLLM proxy",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )
    parser.add_argument(
        "--with_background",
        action="store_true",
        default=False,
        help="Include background information in prompts (default: False)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Run evaluation after generation (default: False)",
    )
    parser.add_argument(
        "--h5py_file_path",
        type=str,
        default=None,
        help="Path to test_data.h5 file for evaluation (required if --evaluate is set)",
    )
    parser.add_argument(
        "--eval_timeout",
        type=int,
        default=1800,
        help="Timeout per test in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Enable response caching to avoid redundant API calls (default: True)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Disable response caching",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory (default: ~/.cache/llm_responses)",
    )

    args = parser.parse_args()

    # Handle cache flag
    use_cache = args.use_cache and not args.no_cache

    metrics = run_scicode(
        model=args.model,
        output_path=args.output_path,
        split=args.split,
        limit=args.limit,
        apply_chat_template_flag=args.apply_chat_template,
        log_samples=args.log_samples,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        with_background=args.with_background,
        evaluate=args.evaluate,
        h5py_file_path=args.h5py_file_path,
        eval_timeout=args.eval_timeout,
        use_cache=use_cache,
        cache_dir=args.cache_dir,
    )

    return 0 if metrics else 1


if __name__ == "__main__":
    sys.exit(main())
