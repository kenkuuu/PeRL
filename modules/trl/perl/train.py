
import inspect
import torch
import os

from typing import List, Optional
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from fire import Fire

from perl.utils.logging import init_logger, logger
from perl.data import load_dataset
from perl.config.config import TrainConfig

def fuzzy_jobs(
    args: TrainConfig
):
    init_logger()
    args.training.output_dir = args.training.output_dir or "output"
    args.training.run_name = args.training.run_name or args.training.output_dir # training run name is the output_dir
    if not os.path.exists(args.training.output_dir): # check if output_dir exists
        os.makedirs(args.training.output_dir, exist_ok=True)
    else:
        logger.info(f"Output directory {args.training.output_dir} already exists, using it")
    set_seed(args.common.seed)

    if args.common.debug:
        args.training.report_to = []

    # only initialize for rank 0 when process group is available
    is_main_process = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0

    if is_main_process:
        if "trackio" in args.training.report_to:
            import trackio
            trackio.init(
                project=args.logging.trackio_project,
                space_id=args.logging.trackio_space_id,
                config=vars(args.training)
            )
            logger.info(f"Trackio initialized successfully")
        elif "wandb" in args.training.report_to:
            import wandb
            wandb.init(
                project=args.logging.wandb_project,
                name=args.training.run_name,
                config=vars(args.training),
            )
            logger.info(f"Wandb initialized successfully")

    return args

def train(
    config: TrainConfig = None
):
    # 0. parse args and prepare logger
    print(config)
    args = fuzzy_jobs(config)

    # 1. load tokenizer and dataset
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    tokenizer.padding_side = "left"  # Configure for decoder-only architecture: use left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    logger.info(f"Loading dataset from {args.dataset.dataset_name_or_path}")
    dataset = load_dataset(
        args.dataset.dataset_name_or_path,
        example_numbers=args.dataset.example_numbers,
        tokenizer=tokenizer
    )
    train_dataset = dataset["train_dataset"]
    test_dataset = dataset["test_dataset"]
    reward_functions = dataset["reward_functions"]

    if "reward_weights" in dataset:
        reward_weights = dataset["reward_weights"]
    else:
        reward_weights = [1.0] * len(reward_functions)
    args.training.reward_weights = reward_weights

    # 2. load and configure model
    logger.info(f"Loading model from {args.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model.model_name_or_path,
        torch_dtype= torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2"
    )
    logger.info(f"Model loaded successfully")

    # 3. configure lora
    optimizer = None
    if args.peft.use_peft:
        logger.info(f"Detected PEFT configuration, configuring {args.peft.type}")
        from perl.lora.adapter import apply_peft
        optimizer, model = apply_peft(model, args)
        logger.info(f"PEFT ({args.peft.type}) configured successfully")

    # 4.Training configuration - filter to only params accepted by this TRL version
    # gradient_checkpointing with PEFT requires use_reentrant=False to avoid frozen-param issues
    if args.training.gradient_checkpointing and not args.training.gradient_checkpointing_kwargs:
        args.training.gradient_checkpointing_kwargs = {"use_reentrant": False}
    grpo_params = set(inspect.signature(GRPOConfig).parameters.keys())
    training_dict = {k: v for k, v in vars(args.training).items() if k in grpo_params}
    training_args = GRPOConfig(**training_dict)

    # 5.Train
    # Use a single param group to prevent any empty-group removal by DeepSpeed/Accelerate.
    if optimizer is None:
        from torch.optim import AdamW
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.training.learning_rate)

    # Create scheduler explicitly from the same optimizer so param_group counts are always in sync.
    # Passing (optimizer, scheduler) prevents Trainer/DeepSpeed from creating a mismatched scheduler.
    from transformers.optimization import get_scheduler as _get_scheduler
    _num_warmup = int(args.training.warmup_ratio * args.training.max_steps)
    _sched_kwargs: dict = dict(
        name=args.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=_num_warmup,
        num_training_steps=args.training.max_steps,
    )
    if "scheduler_specific_kwargs" in inspect.signature(_get_scheduler).parameters \
            and args.training.lr_scheduler_kwargs:
        _sched_kwargs["scheduler_specific_kwargs"] = args.training.lr_scheduler_kwargs
    scheduler = _get_scheduler(**_sched_kwargs)

    logger.info(f"Training model with GRPO")
    trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, scheduler),
    )
    if "reward_weights" in trainer_params:
        trainer_kwargs["reward_weights"] = reward_weights
    trainer = GRPOTrainer(**trainer_kwargs)
    
    # 支持从 checkpoint 恢复训练
    resume_checkpoint = args.training.resume_from_checkpoint
    if resume_checkpoint == "true":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training completed successfully")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
