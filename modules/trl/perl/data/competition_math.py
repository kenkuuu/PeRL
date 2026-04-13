"""
Dataset loader for qwedsacf/competition_math (MATH benchmark).

Split strategy:
  train[:7500]   -> training set
  train[-5000:]  -> validation set (full), sampled to eval_sample_size for per-step eval

Dataset fields:
  problem  : str  - problem statement
  solution : str  - full worked solution containing \\boxed{...} at the end
  level    : str  - "Level 1" ... "Level 5"
  type     : str  - "Algebra", "Geometry", etc.
"""

import re
from datasets import load_dataset as hf_load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

COMPETITION_MATH_SYSTEM_PROMPT = (
    "You are an expert at solving competition mathematics problems. "
    "Think through the problem carefully inside <think> </think> tags, "
    "then give your final answer inside \\\\boxed{}."
)


def make_conversation(example):
    """Convert competition_math example to prompt/solution dict."""
    prompt = [
        {"role": "system", "content": COMPETITION_MATH_SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
    ]
    return {"prompt": prompt, "solution": example["solution"]}


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def accuracy_reward(completions, solution, **kwargs):
    """Binary reward: 1.0 if the boxed answer matches the gold solution."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    extraction_cfg = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(units=True),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]
    for content, sol in zip(contents, solution):
        gold_parsed = parse(sol, extraction_config=extraction_cfg, extraction_mode="first_match")
        if len(gold_parsed) == 0:
            rewards.append(0.0)
            continue
        answer_parsed = parse(content, extraction_config=extraction_cfg, extraction_mode="first_match")
        try:
            import signal

            def _timeout(signum, frame):
                raise TimeoutError

            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(5)
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            finally:
                signal.alarm(0)
        except Exception:
            reward = 0.0
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward 1.0 if response contains </think> and \\boxed{...}."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in contents:
        has_think = bool(re.search(r"</think>", content))
        has_boxed = bool(re.search(r"\\boxed\{", content))
        rewards.append(1.0 if (has_think and has_boxed) else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_competition_math_dataset(
    dataset_name_or_path: str,
    example_numbers: int = None,
    eval_sample_size: int = 200,
):
    """
    Load qwedsacf/competition_math and return train/val split.

    Args:
        dataset_name_or_path : HuggingFace dataset name or local path.
        example_numbers      : Optional cap on training examples.
        eval_sample_size     : Number of validation examples used for
                               per-step evaluation (sampled from train[-5000:]).
                               Set to None to use all 5000.
    """
    raw = hf_load_dataset(dataset_name_or_path)
    train_data = raw["train"]

    n = len(train_data)
    train_end = min(7500, n)
    val_start = max(0, n - 5000)

    train_split = train_data.select(range(train_end))
    val_split = train_data.select(range(val_start, n))

    # Optionally cap training examples
    if example_numbers is not None and len(train_split) > example_numbers:
        train_split = train_split.select(range(example_numbers))

    # Sample validation set for efficient per-step evaluation
    if eval_sample_size is not None and len(val_split) > eval_sample_size:
        # Fixed seed for reproducibility across runs
        val_split = val_split.shuffle(seed=42).select(range(eval_sample_size))

    train_split = train_split.map(make_conversation, remove_columns=train_split.column_names)
    val_split = val_split.map(make_conversation, remove_columns=val_split.column_names)

    return {
        "train_dataset": train_split,
        "test_dataset": val_split,
        "reward_functions": [format_reward, accuracy_reward],
        "reward_weights": [0.5, 1.0],
    }
