"""InfoBench handler for ProxyBench."""

import json
import os
import sys

from openai import OpenAI

from evaluations.handlers._compat import chat_completion

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

INFOBENCH_DIR = os.path.join(BENCHMARKS_DIR, "infobench")
INFOBENCH_OUTPUT_FILES_DIR = os.path.join(INFOBENCH_DIR, "output_files")

INFOBENCH_SYS_MSG = (
    "Based on the provided Input (if any) and Generated Text, answer the ensuing "
    "Questions with either a YES or NO choice. Your selection should be based on your "
    "judgment as well as the following rules:\n\n"
    "- YES: Select 'YES' if the generated text entirely fulfills the condition specified "
    "in the question. However, note that even minor inaccuracies exclude the text from "
    "receiving a 'YES' rating.\n\n"
    "- NO: Opt for 'NO' if the generated text fails to meet the question's requirements "
    "or provides no information that could be utilized to answer the question."
)


def _run_infobench(
    model_name: str,
    base_url: str,
    api_key: str,
    instance_id: str,
) -> list:
    """Run a single InfoBench instance: generate model output then evaluate decomposed questions.

    instance_id is the task id, e.g. "user_oriented_task_167".
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    task_entry = None
    for fname in os.listdir(INFOBENCH_OUTPUT_FILES_DIR):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(INFOBENCH_OUTPUT_FILES_DIR, fname)
        with open(fpath) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("id") == instance_id:
                    task_entry = {
                        k: entry[k] for k in
                        ("id", "input", "category", "instruction",
                         "decomposed_questions", "subset", "question_label")
                        if k in entry
                    }
                    break
        if task_entry:
            break

    if task_entry is None:
        raise ValueError(
            f"Instance id '{instance_id}' not found in any output file under "
            f"{INFOBENCH_OUTPUT_FILES_DIR}."
        )

    instruction = task_entry.get("instruction", "")
    input_text = task_entry.get("input", "")
    decomposed_questions = task_entry.get("decomposed_questions", [])

    gen_prompt = instruction
    if input_text:
        gen_prompt = f"Input:\n{input_text}\n\nInstruction:\n{instruction}"

    gen_response = chat_completion(
        client,
        model=model_name,
        messages=[{"role": "user", "content": gen_prompt}],
        temperature=0,
        max_tokens=2048,
        timeout=300,
    )
    output = gen_response.choices[0].message.content or ""
    gen_prompt_tokens = getattr(gen_response.usage, "prompt_tokens", 0)
    gen_completion_tokens = getattr(gen_response.usage, "completion_tokens", 0)

    bool_results = []
    eval_prompt_tokens = 0
    eval_completion_tokens = 0

    for question in decomposed_questions:
        if input_text:
            content = (
                f"{INFOBENCH_SYS_MSG}\n\n"
                f"Input:\n\"{input_text}\"\n\n"
                f"Generated Text:\n\"{output}\"\n\n"
                f"Question:\n{question}\n"
            )
        else:
            content = (
                f"{INFOBENCH_SYS_MSG}\n\n"
                f"Generated Text:\n\"{output}\"\n\n"
                f"Question:\n{question}\n"
            )

        try:
            eval_response = chat_completion(
                client,
                model=model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=50,
                timeout=60,
            )
            generation = eval_response.choices[0].message.content or ""
            eval_prompt_tokens += getattr(eval_response.usage, "prompt_tokens", 0)
            eval_completion_tokens += getattr(eval_response.usage, "completion_tokens", 0)

            g = generation.strip().lower()
            if g.startswith("yes"):
                bool_results.append(True)
            elif g.startswith("no"):
                bool_results.append(False)
            elif "yes" in g and "no" not in g:
                bool_results.append(True)
            elif "no" in g and "yes" not in g:
                bool_results.append(False)
            else:
                bool_results.append(None)
        except Exception as e:
            print(f"Warning: eval call failed for question '{question}': {e}", file=sys.stderr)
            bool_results.append(None)

    correct = sum(1 for r in bool_results if r is True)
    total = len(bool_results)
    accuracy_percent = round(correct / total * 100, 4) if total > 0 else 0.0

    result = dict(task_entry)
    result["output"] = output
    result["eval"] = bool_results
    result["correct"] = correct
    result["total_questions"] = total
    result["accuracy_percent"] = accuracy_percent
    result["prompt_tokens"] = gen_prompt_tokens + eval_prompt_tokens
    result["completion_tokens"] = gen_completion_tokens + eval_completion_tokens
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]

    return [result]
