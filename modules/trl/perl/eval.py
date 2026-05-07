import json
import os
import torch
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm

from perl.data.gsm8k import _extract_xml_answer, load_gsm8k_dataset
from perl.utils.logging import init_logger, logger


@dataclass
class EvalConfig:
    model_name_or_path: str = None
    checkpoint_path: str = None  # LoRA checkpoint dir; None = base model only
    dataset_name_or_path: str = "gsm8k"
    batch_size: int = 8
    max_new_tokens: int = 1024
    dtype: str = "bfloat16"
    seed: int = 42
    output_file: Optional[str] = None


def evaluate(config: EvalConfig):
    init_logger()
    set_seed(config.seed)
    torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    logger.info(f"Loading tokenizer from {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading GSM8K test dataset")
    test_dataset = load_gsm8k_dataset(config.dataset_name_or_path)["test_dataset"]

    logger.info(f"Loading model from {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    if config.checkpoint_path:
        logger.info(f"Loading LoRA from {config.checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.checkpoint_path)

    model.eval()

    correct = 0
    total = len(test_dataset)
    results = []

    logger.info(f"Evaluating on {total} examples (batch_size={config.batch_size})")
    for i in tqdm(range(0, total, config.batch_size)):
        batch = test_dataset[i : i + config.batch_size]
        prompts = batch["prompt"]
        answers = batch["answer"]

        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for j in range(len(answers)):
            completion = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            extracted = _extract_xml_answer(completion)
            try:
                pred = float(extracted.replace(",", ""))
                gold = float(str(answers[j]).replace(",", ""))
                is_correct = pred == gold
            except (ValueError, AttributeError):
                is_correct = extracted.strip() == str(answers[j]).strip()

            correct += int(is_correct)
            results.append({
                "question": batch["question"][j],
                "ground_truth": answers[j],
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
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Results saved to {config.output_file}")

    return accuracy
