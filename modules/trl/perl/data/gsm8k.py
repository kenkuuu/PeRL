import re

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the following math problem step by step. "
    "Show your reasoning inside <think>...</think> tags, then provide only the final "
    "numeric answer inside <answer>...</answer> tags."
)

# Units and symbols to strip before numeric comparison.
_UNIT_PATTERN = re.compile(
    r"[$€£¥₹]"                         # currency symbols (prefix)
    r"|"
    r"(?<!\S)"                          # unit suffixes (only when preceded by space or start)
    r"(?:dollars?|cents?|euros?|pounds?|yen|yuan"
    r"|km|m|cm|mm|kg|g|mg|lbs?|oz|ft|in|mi"
    r"|hours?|hrs?|minutes?|mins?|seconds?|secs?"
    r"|days?|weeks?|months?|years?"
    r"|people|students?|items?|units?|pieces?|apples?|oranges?"
    r"|%|percent|per\s+cent)"
    r"\b.*$",
    flags=re.IGNORECASE,
)


def _normalize_number(text: str) -> str:
    """Strip units, currency and formatting from a candidate numeric string."""
    text = text.strip()
    text = _UNIT_PATTERN.sub("", text)
    text = text.replace(",", "")   # remove thousand separators
    return text.strip()


def _numbers_equal(pred: str, gold: str) -> bool:
    """Return True if pred and gold represent the same number after normalization."""
    p, g = _normalize_number(pred), _normalize_number(gold)
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, AttributeError):
        return p == g


def _extract_hash_answer(text: str) -> str | None:
    """Extract answer from GSM8K '#### <answer>' format."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def _extract_xml_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    try:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    except IndexError:
        return ""


def format_reward(completions, **kwargs) -> list[float]:
    """Reward 1.0 if the completion contains both <think> and <answer> tags."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL) for content in contents]
    return [1.0 if match else 0.0 for match in matches]


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """Reward 1.0 if the extracted answer matches ground truth numerically."""
    contents = [completion[0]["content"] for completion in completions]
    return [
        1.0 if _numbers_equal(_extract_xml_answer(c), str(gt)) else 0.0
        for c, gt in zip(contents, answer)
    ]


def _make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": _extract_hash_answer(example["answer"]),
    }


def load_gsm8k_dataset(dataset_name_or_path: str, example_numbers: int = None):
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

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
