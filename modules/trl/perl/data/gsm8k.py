import re

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the following math problem step by step. "
    "Show your reasoning inside <think>...</think> tags, then provide only the final "
    "numeric answer inside <answer>...</answer> tags."
)


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
    rewards = []
    for content, gt in zip(contents, answer):
        extracted = _extract_xml_answer(content)
        try:
            pred = float(extracted.replace(",", ""))
            gold = float(str(gt).replace(",", ""))
            rewards.append(1.0 if pred == gold else 0.0)
        except (ValueError, AttributeError):
            rewards.append(1.0 if extracted.strip() == str(gt).strip() else 0.0)
    return rewards


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
        "reward_weights": [2.0, 1.0],
    }
