import re

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    def extract_code(r: str) -> str:
        # Handle responses that include markdown code blocks (GPT-4o, Gemini style)
        if r.strip().startswith("```"):
            # Extract code from within the code block
            pattern = r"```(?:python)?\s*\n?([\s\S]*?)```"
            matches = re.findall(pattern, r)
            if matches:
                return matches[0].strip()
        # Handle responses without code blocks (Claude style) - truncate at closing ```
        if r.find("```") != -1:
            return r[: r.find("```")]
        return r

    return [
        [doc["prompt"] + extract_code(r) for r in resp]
        for resp, doc in zip(resps, docs)
    ]


def build_predictions_chat(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """
    For OpenAI models that return full code in markdown blocks.
    Extracts the code from ```python ... ``` blocks.
    """
    import re

    def extract_code(response: str, prompt: str) -> str:
        # Try to extract code from markdown code block
        # Use a more robust pattern that handles various code block formats
        pattern = r"```(?:python)?\s*\n([\s\S]*?)```"
        matches = re.findall(pattern, response)

        if matches:
            # Find the code block that contains the function definition
            func_match = re.search(r"def\s+(\w+)", prompt)
            func_name = func_match.group(1) if func_match else None

            for code in matches:
                code = code.strip()
                # If this code block contains the expected function, use it
                if func_name and f"def {func_name}" in code:
                    return code

            # If no matching function found, use the first code block
            code = matches[0].strip()
            if func_name and f"def {func_name}" not in code:
                # The code block might just contain the implementation,
                # so prepend the prompt
                return prompt + code
            return code

        # No code block found, try to extract function directly from response
        func_match = re.search(r"def\s+(\w+)", prompt)
        if func_match:
            func_name = func_match.group(1)
            if f"def {func_name}" in response:
                # Find the function definition and extract it
                func_start = response.find(f"def {func_name}")
                # Try to find the end of the function (next function or end of response)
                remaining = response[func_start:]
                return remaining

        # Fallback: prepend prompt to response
        return prompt + response

    return [[extract_code(r, doc["prompt"]) for r in resp] for resp, doc in zip(resps, docs)]
