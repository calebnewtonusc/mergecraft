"""
training/train.py — Stage 1: SFT for MergeCraft.

Fine-tunes Qwen2.5-7B-Coder on 500k+ labeled (task, merged_contribution) pairs.
The model learns what merged contributions look like across 1000 repos.

Launch:
    deepspeed --num_gpus=18 training/train.py \
        --deepspeed training/configs/ds_config.json \
        --output_dir checkpoints/sft
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


@dataclass
class SFTConfig_:
    base_model: str = "Qwen/Qwen2.5-7B-Coder-Instruct"
    output_dir: str = "./checkpoints/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = (
        1e-4  # MC-18: 2e-4 was too high and caused training instability
    )
    max_seq_length: int = 8192
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    data_dir: str = "./data/synthesized"
    wandb_project: str = "mergecraft-sft"


SYSTEM_PROMPT = """\
You are MergeCraft, a specialist AI for generating open source contributions that get merged.
You understand project conventions, maintainer preferences, PR scope requirements, and the
social process of open source contribution. Your goal is contributions that are accepted on
first review — not just code that works.
"""


def format_example(example: dict) -> str:
    """Format a training pair as a chat message."""
    task = example.get("task", "")
    contribution = example.get("contribution", "")
    repo = example.get("repo", "")
    conventions = (
        json.dumps(example.get("conventions", {}), indent=2)
        if example.get("conventions")
        else ""
    )

    user_msg = f"Repository: {repo}\nTask: {task}"
    if conventions:
        user_msg += f"\n\nConventions:\n{conventions}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{contribution}<|im_end|>"
    )


def load_training_data(config: SFTConfig_) -> tuple[Dataset, Dataset]:
    examples = []
    data_dir = Path(config.data_dir)
    for jsonl_file in data_dir.glob("**/*.jsonl"):
        with jsonl_file.open() as fh:
            for line in fh:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        if ex.get("outcome") == "merged":
                            examples.append(ex)
                    except json.JSONDecodeError:
                        pass

    if not examples:
        raise RuntimeError(
            f"No merged contribution examples found in {config.data_dir!r}. "
            "Run the discovery and synthesis pipeline first."
        )
    logger.info(f"Loaded {len(examples):,} merged contribution examples")
    # MC-2: Shuffle before splitting so train/val sets are representative, not chronological
    random.seed(42)
    random.shuffle(examples)
    texts = [format_example(ex) for ex in examples]
    split = int(len(texts) * 0.95)
    return Dataset.from_dict({"text": texts[:split]}), Dataset.from_dict(
        {"text": texts[split:]}
    )


def train(config: SFTConfig_) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)  # nosec B615
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds, val_ds = load_training_data(config)

    sft_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=True,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        eval_strategy="steps",
        eval_steps=500,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else [],
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_args,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    logger.success(f"SFT complete → {config.output_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./checkpoints/sft")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train(SFTConfig_(output_dir=args.output_dir))


if __name__ == "__main__":
    main()
