import re
from typing import Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from datasets import load_dataset as hf_load_dataset

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
    "Respond in the following format: <think>\n...\n</think>\n, then answer."
)

_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_boxed_answer(solution: str) -> str:
    """Extract the last \\boxed{} content from a solution string."""
    # Find all \boxed{ positions and extract the innermost content of the last one
    matches = list(_BOXED_RE.finditer(solution))
    if not matches:
        return solution.strip()
    # Take the last match and extract balanced braces
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start:i - 1].strip()


def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[Optional[float]]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(sol)
        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            reward = float(content.strip().lower() == sol.strip().lower())
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    pattern = r"</think>"
    contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in contents]
    for content, match in zip(contents, matches):
        if not match:
            truncated = content[:200] + "..." if len(content) > 200 else content
            print(f"Mismatch: {truncated}")
            print("-" * 100)
    return [1.0 if match else 0.0 for match in matches]


def _make_conversation(example):
    answer = _extract_boxed_answer(example["solution"])
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": answer,
    }


def load_math_dataset(dataset_name_or_path: str, example_numbers: int = None):
    train_dataset = hf_load_dataset("hendrycks/competition_math", split="train")
    test_dataset = hf_load_dataset("hendrycks/competition_math", split="test")

    train_dataset = train_dataset.map(_make_conversation)
    test_dataset = test_dataset.map(_make_conversation)

    if example_numbers is not None and len(train_dataset) > example_numbers:
        train_dataset = train_dataset.select(range(example_numbers))

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "reward_functions": [accuracy_reward, format_reward],
        "reward_weights": [1.0, 1.0],
    }
