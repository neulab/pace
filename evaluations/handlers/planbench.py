"""PlanBench handler for ProxyBench."""

import json
import os
import re
import sys
import tempfile

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_HANDLERS_DIR, "..", ".."))

PLANBENCH_RAW_DIR = os.path.join(_ROOT_DIR, "results", "raw_results", "planbench")

# Vendored PlanBench package (original grading code we reuse for authentic scoring).
_PLANBENCH_PKG = os.path.join(_ROOT_DIR, "evaluations", "benchmarks", "planbench", "plan-bench")
_PLANBENCH_CFG = os.path.join(_PLANBENCH_PKG, "configs", "blocksworld.yaml")

# task_7 compares final states (no VAL); everything except task_3 is a plan task.
_PLANBENCH_STATE_TASKS = ("task_7_plan_execution",)

# The canonical PlanBench blocksworld grading domain (== instances/blocksworld/
# generated_domain.pddl, which is not shipped). Its predicate/action names MUST
# match configs/blocksworld.yaml — verified: actions pick-up/put-down/stack/unstack,
# predicates ontable/clear/handempty/holding/on.
_BLOCKSWORLD_DOMAIN = """(define (domain blocksworld-4ops)
  (:requirements :strips)
  (:predicates (clear ?x) (ontable ?x) (handempty) (holding ?x) (on ?x ?y))
  (:action pick-up
    :parameters (?ob)
    :precondition (and (clear ?ob) (ontable ?ob) (handempty))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (ontable ?ob)) (not (handempty))))
  (:action put-down
    :parameters (?ob)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (handempty) (ontable ?ob) (not (holding ?ob))))
  (:action stack
    :parameters (?ob ?underob)
    :precondition (and (clear ?underob) (holding ?ob))
    :effect (and (handempty) (clear ?ob) (on ?ob ?underob)
                 (not (clear ?underob)) (not (holding ?ob))))
  (:action unstack
    :parameters (?ob ?underob)
    :precondition (and (on ?ob ?underob) (clear ?ob) (handempty))
    :effect (and (holding ?ob) (clear ?underob)
                 (not (on ?ob ?underob)) (not (clear ?ob)) (not (handempty)))))
"""


def _planbench_grade(subtask, query, llm_raw_response, ground_truth_plan):
    """Authentic PlanBench grading, reusing the vendored original functions.

    Returns (llm_correct: int|None, extracted, extras: dict). Requires VAL for the
    plan tasks (env var $VAL pointing at a dir with the `validate` binary); task_7
    is state-comparison only. The problem's init/goal are reconstructed from the
    query's last [STATEMENT] block (equivalent to the generated instance file, since
    the query was produced from it), then graded with the original text_to_plan /
    validate_plan / text_to_state and the canonical blocksworld domain.
    """
    # Default $VAL to where install.sh / install_val.py puts it, so grading works
    # even if the user forgot to export it. An explicit $VAL still wins.
    os.environ.setdefault("VAL", os.path.expanduser("~/.planutils/packages/val/bin"))
    if _PLANBENCH_PKG not in sys.path:
        sys.path.insert(0, _PLANBENCH_PKG)
    import yaml
    # validate_plan / text_to_plan / text_to_state are all re-exported by the
    # utils package __init__ (mirrors the original `from utils import *`).
    from utils import text_to_plan, text_to_state, validate_plan
    from response_evaluation import ResponseEvaluator   # _extract_state_text (staticmethod)
    from tarski.io import PDDLReader
    with open(_PLANBENCH_CFG) as f:
        data = yaml.safe_load(f)

    # ---- task_7: final-state set equality (mirrors evaluate_state) ----
    if subtask in _PLANBENCH_STATE_TASKS:
        state_text = ResponseEvaluator._extract_state_text(llm_raw_response)
        llm_state = text_to_state(state_text, data)
        gt = ground_truth_plan if isinstance(ground_truth_plan, list) \
            else text_to_state(ground_truth_plan, data)
        return int(sorted(gt) == sorted(llm_state)), llm_state, {}

    # ---- plan tasks (t1/t2/t4/t5/t6/t8_*): VAL validation (mirrors evaluate_plan) ----
    block = query.rsplit("[STATEMENT]", 1)[-1]
    m_init = re.search(r"As initial conditions I have that,\s*(.*?)\.\s*\n", block, re.S)
    m_goal = re.search(r"My goal is to have that\s*(.*?)\.", block, re.S)
    init_preds = text_to_state(m_init.group(1), data) if m_init else []
    goal_preds = text_to_state(m_goal.group(1), data) if m_goal else []

    def _to_pddl(preds):
        return ["(" + " ".join(p.split("_")) + ")" for p in preds]

    objs = sorted({tok for p in (init_preds + goal_preds) for tok in p.split("_")[1:]})
    problem_pddl = (
        "(define (problem reconstructed)\n(:domain blocksworld-4ops)\n"
        "(:objects " + " ".join(objs) + ")\n"
        "(:init " + " ".join(_to_pddl(init_preds)) + ")\n"
        "(:goal (and " + " ".join(_to_pddl(goal_preds)) + ")))\n"
    )

    extras = {}
    with tempfile.TemporaryDirectory() as wd:
        domain_file = os.path.join(wd, "domain.pddl")
        problem_file = os.path.join(wd, "problem.pddl")
        plan_file = os.path.join(wd, "llm_plan")
        with open(domain_file, "w") as f:
            f.write(_BLOCKSWORLD_DOMAIN)
        with open(problem_file, "w") as f:
            f.write(problem_pddl)
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain_file)
        problem = reader.parse_instance(problem_file)
        try:
            llm_plan, _ = text_to_plan(llm_raw_response, problem.actions, plan_file, data)
            extracted = llm_plan
            correct = int(validate_plan(domain_file, problem_file, plan_file))
            if "optimality" in subtask and correct:
                actual = sum(1 for ln in llm_plan.split("\n") if len(ln) > 0)
                optimal = sum(1 for ln in (ground_truth_plan or "").split("\n") if ln.strip())
                extras = {"actual_cost_of_llm_plan": actual, "optimal_cost": optimal}
                correct = int(actual == optimal)
        except Exception:
            extracted = None
            correct = 0
    return correct, extracted, extras

PLANBENCH_TASKS = (
    "task_1_plan_generation",
    "task_2_plan_optimality",
    "task_3_plan_verification",
    "task_4_plan_reuse",
    "task_5_plan_generalization",
    "task_6_replanning",
    "task_7_plan_execution",
    "task_8_1_goal_shuffling",
    "task_8_2_full_to_partial",
    "task_8_3_partial_to_full",
)

PLANBENCH_DEFAULT_SUBTASK = "task_1_plan_generation"


def _run_planbench(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single PlanBench instance.

    instance_id is the integer instance_id stored in each JSON entry.
    Query prompts are loaded from any existing raw-result file for the given subtask.

    Evaluation:
      task_3_plan_verification: binary valid/invalid string match.
      All other tasks: returns llm_correct=None (requires VAL + Fast Downward).
    """
    from openai import OpenAI
    from evaluations.handlers._compat import chat_completion

    if subtask not in PLANBENCH_TASKS:
        raise ValueError(
            f"PlanBench subtask '{subtask}' not recognized. "
            f"Valid subtasks: {PLANBENCH_TASKS}"
        )

    target_id = int(instance_id)

    instance_data = None
    for provider in sorted(os.listdir(PLANBENCH_RAW_DIR)):
        provider_dir = os.path.join(PLANBENCH_RAW_DIR, provider)
        if not os.path.isdir(provider_dir):
            continue
        for model_dir in sorted(os.listdir(provider_dir)):
            task_file = os.path.join(provider_dir, model_dir, f"{subtask}.json")
            if not os.path.exists(task_file):
                continue
            with open(task_file) as f:
                data = json.load(f)
            for inst in data.get("instances", []):
                if inst.get("instance_id") == target_id:
                    instance_data = {
                        "instance_id":       inst["instance_id"],
                        "query":             inst["query"],
                        "ground_truth_plan": inst.get("ground_truth_plan", ""),
                        "task":              data.get("task", subtask),
                        "domain":            data.get("domain", ""),
                    }
                    for k in ("parsed_ground_truth_plan", "example_instance_ids"):
                        if k in inst:
                            instance_data[k] = inst[k]
                    break
            if instance_data:
                break
        if instance_data:
            break

    if instance_data is None:
        raise ValueError(
            f"PlanBench instance not found: subtask='{subtask}', instance_id={target_id}. "
            f"Check that raw results exist under {PLANBENCH_RAW_DIR}."
        )

    query = instance_data["query"]
    ground_truth_plan = instance_data["ground_truth_plan"]

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = chat_completion(
        client,
        model=model_name,
        messages=[{"role": "user", "content": query}],
        temperature=0,
        max_tokens=2048,
        timeout=300,
    )
    llm_raw_response = response.choices[0].message.content or ""

    result = dict(instance_data)
    result["llm_raw_response"] = llm_raw_response
    result["extracted_llm_plan"] = None

    if subtask == "task_3_plan_verification":
        def _parse_validity(text: str):
            t = text.lower()
            if re.search(r'\bthe (above )?plan is (not valid|invalid)\b', t):
                return False
            if re.search(r'\bthe (above )?plan is valid\b', t):
                return True
            if re.search(r'\binvalid\b', t):
                return False
            if re.search(r'\bvalid\b', t):
                return True
            return None

        gt_valid = _parse_validity(ground_truth_plan)
        llm_valid = _parse_validity(llm_raw_response)
        if gt_valid is not None and llm_valid is not None:
            result["llm_correct_binary"] = (gt_valid == llm_valid)
        else:
            result["llm_correct_binary"] = None
        result["llm_correct"] = None
    else:
        correct, extracted, extras = _planbench_grade(
            subtask, query, llm_raw_response, ground_truth_plan
        )
        result["extracted_llm_plan"] = extracted
        result["llm_correct"] = correct        # 0/1 (int), matching the standardized CSVs
        result.update(extras)                  # actual_cost_of_llm_plan / optimal_cost (task_2)

    return [result]
