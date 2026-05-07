import json
import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer, set_seed
from tqdm import tqdm

from perl.data import load_dataset as load_perl_dataset
from perl.utils.logging import init_logger, logger


@dataclass
class EvalConfig:
    model_name_or_path: str = None
    checkpoint_path: str = None      # LoRA checkpoint dir; None = base model only
    dataset_name_or_path: str = "gsm8k"
    max_new_tokens: int = 1024
    dtype: str = "bfloat16"
    seed: int = 42
    output_file: Optional[str] = None
    # vLLM settings
    use_vllm: bool = True
    lora_rank: int = 16              # must match training r; used by vLLM enable_lora
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    # transformers fallback settings
    batch_size: int = 8


def _to_prompt_string(prompt, tokenizer):
    """Convert prompt to string regardless of format (string or message list)."""
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


def _score(completion, example):
    """Extract <answer> and compare with ground truth numerically."""
    try:
        extracted = completion.split("<answer>")[-1].split("</answer>")[0].strip()
    except IndexError:
        extracted = ""

    gt = str(example.get("answer", example.get("target", ""))).replace(",", "")
    try:
        return float(extracted.replace(",", "")) == float(gt), extracted
    except (ValueError, AttributeError):
        return extracted.strip() == gt.strip(), extracted


def _generate_vllm(config, prompts):
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=config.model_name_or_path,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    )
    if config.checkpoint_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = config.lora_rank

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0, max_tokens=config.max_new_tokens)

    lora_request = None
    if config.checkpoint_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, config.checkpoint_path)

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def _generate_transformers(config, prompts, tokenizer):
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if config.checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.checkpoint_path)
    model.eval()

    completions = []
    for i in tqdm(range(0, len(prompts), config.batch_size)):
        batch_texts = prompts[i : i + config.batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(batch_texts)):
            completions.append(tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True))
    return completions


def evaluate(config: EvalConfig):
    init_logger()
    set_seed(config.seed)

    logger.info(f"Loading tokenizer from {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset: {config.dataset_name_or_path}")
    dataset_dict = load_perl_dataset(config.dataset_name_or_path, tokenizer=tokenizer)
    test_dataset = dataset_dict["test_dataset"]
    total = len(test_dataset)

    # Build prompt strings
    prompts = [_to_prompt_string(example["prompt"], tokenizer) for example in test_dataset]

    # Generate completions
    if config.use_vllm:
        logger.info(f"Generating with vLLM (tensor_parallel_size={config.tensor_parallel_size})")
        completions = _generate_vllm(config, prompts)
    else:
        logger.info(f"Generating with transformers (batch_size={config.batch_size})")
        completions = _generate_transformers(config, prompts, tokenizer)

    # Score
    correct = 0
    results = []
    for i, (completion, example) in enumerate(zip(completions, test_dataset)):
        is_correct, extracted = _score(completion, example)
        correct += int(is_correct)
        results.append({
            "prompt": prompts[i],
            "ground_truth": example.get("answer", example.get("target", "")),
            "completion": completion,
            "extracted": extracted,
            "correct": is_correct,
        })

    accuracy = correct / total
    logger.info(f"Accuracy: {correct}/{total} = {accuracy * 100:.2f}%")

    if config.output_file:
        out_dir = os.path.dirname(config.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(config.output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": accuracy, "correct": correct, "total": total, "results": results},
                f, indent=2, ensure_ascii=False,
            )
        logger.info(f"Results saved to {config.output_file}")

    return accuracy
