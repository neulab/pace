def doc_to_text(doc) -> str:
    """
    Format the document into a CoT prompt for chat completions.
    """
    choices = ["A", "B", "C", "D"]
    prompt = "Read the following passage and answer the question.\n\n"
    prompt += f"Passage: {doc['context']}\n\n"
    prompt += f"Question: {doc['question']}\n\n"
    prompt += "Choices:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"({choice}) {option}\n"
    prompt += "\nThink step by step, then provide your final answer in the format: \"The answer is (X)\" where X is A, B, C, or D."
    return prompt


def doc_to_target(doc) -> str:
    """
    Convert the label to (A), (B), (C), (D) format.
    """
    label_map = {"a": "(A)", "b": "(B)", "c": "(C)", "d": "(D)"}
    return label_map[doc["label"].strip().lower()]
