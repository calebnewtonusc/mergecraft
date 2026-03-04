"""
training/train_dpo.py — Stage 3: DPO for MergeCraft.

Preference pairs:
  preferred: high merge probability + clean, idiomatic code + good description
  rejected:  lower merge probability OR messy code OR inadequate description

Launch:
    deepspeed --num_gpus=18 training/train_dpo.py \
        --base_model checkpoints/grpo \
        --output_dir checkpoints/final
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


@dataclass
class DPOConfig_:
    base_model: str = "checkpoints/grpo"
    output_dir: str = "./checkpoints/final"
    beta: float = 0.1
    learning_rate: float = 5e-7
    data_dir: str = "./data/synthesized"


def load_preference_data(config: DPOConfig_) -> Dataset:
    """
    Build DPO preference pairs from labeled PR outcomes.

    preferred = merged PR (high quality)
    rejected = rejected PR with same task (low quality)
    """
    pairs = []
    data_dir = Path(config.data_dir)

    # MC-1: Group by repo (not task string) so merged/rejected from same repo are paired.
    # Grouping by exact task string means merged and rejected almost never share the same key.
    by_repo: dict[str, list[dict]] = {}
    for f in data_dir.glob("**/*.jsonl"):
        with f.open() as fh:
            for line in fh:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        repo = ex.get("repo", "")
                        if repo not in by_repo:
                            by_repo[repo] = []
                        by_repo[repo].append(ex)
                    except json.JSONDecodeError:
                        pass

    # Create contrastive pairs
    for repo, examples in by_repo.items():
        merged = [e for e in examples if e.get("outcome") == "merged"]
        rejected = [e for e in examples if e.get("outcome") == "rejected"]
        if merged and rejected:
            pairs.append(
                {
                    "prompt": f"Repository: {repo}\nTask: {merged[0].get('task', '')}",
                    "chosen": merged[0].get("contribution", ""),
                    "rejected": rejected[0].get("contribution", ""),
                }
            )

    if not pairs:
        logger.warning("No contrastive pairs found — using synthetic fallback")
        pairs = [
            {
                "prompt": "Repository: fastapi/fastapi\nTask: Add response_model_exclude_none parameter",
                "chosen": "Complete, tested PR with proper description linking issue and explaining why",
                "rejected": "Quick fix without tests or description",
            }
        ]

    # MC-30: Shuffle pairs before returning to avoid ordering bias
    random.shuffle(pairs)
    logger.info(f"DPO pairs: {len(pairs):,}")
    return Dataset.from_list(pairs)


def train(config: DPOConfig_) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    # Read adapter_config.json to find the true base model, then wrap with the
    # PEFT adapter.  Loading a PEFT adapter directory via AutoModelForCausalLM
    # silently ignores the LoRA weights — PeftModel.from_pretrained is required.
    adapter_cfg = json.load(open(Path(config.base_model) / "adapter_config.json"))
    true_base = adapter_cfg["base_model_name_or_path"]
    _base = AutoModelForCausalLM.from_pretrained(
        true_base,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = PeftModel.from_pretrained(_base, config.base_model)
    model.enable_input_require_grads()
    # ref_model must be a separate frozen copy — load the base independently so
    # the two models share no state.
    _ref_base = AutoModelForCausalLM.from_pretrained(
        true_base,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    ref_model = PeftModel.from_pretrained(_ref_base, config.base_model)
    ref_model.enable_input_require_grads()
    # MC-6: ref_model must be in eval mode — it is frozen and only used for KL divergence
    ref_model.eval()

    dataset = load_preference_data(config)

    dpo_args = DPOConfig(
        output_dir=config.output_dir,
        beta=config.beta,
        learning_rate=config.learning_rate,
        bf16=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else [],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        args=dpo_args,
        train_dataset=dataset,
    )
    try:
        trainer.train()
    finally:
        trainer.save_model(config.output_dir)
    logger.success(f"DPO complete → {config.output_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/grpo")
    parser.add_argument("--output_dir", default="./checkpoints/final")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train(
        DPOConfig_(
            base_model=args.base_model, output_dir=args.output_dir, beta=args.beta
        )
    )


if __name__ == "__main__":
    main()
