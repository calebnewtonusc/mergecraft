"""
training/train_rl.py — Stage 2: GRPO with maintainer simulator reward.

The reward function is the maintainer simulator — a learned model of merge probability.
For each candidate contribution, the simulator predicts the merge probability.
GRPO optimizes toward contributions that maintainers are most likely to accept.

Launch:
    deepspeed --num_gpus=18 training/train_rl.py \
        --base_model checkpoints/sft \
        --output_dir checkpoints/grpo \
        --group_size 8
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

# MC-15: renamed GRPOConfig_ to GRPOTrainingConfig below to avoid shadowing the trl import
from trl import GRPOConfig, GRPOTrainer

from synthesis.maintainer_simulator import MaintainerSimulator


# MC-15: renamed from GRPOConfig_ to GRPOTrainingConfig to avoid shadowing the trl GRPOConfig import
@dataclass
class GRPOTrainingConfig:
    base_model: str = "checkpoints/sft"
    output_dir: str = "./checkpoints/grpo"
    group_size: int = 8
    learning_rate: float = 5e-6
    temperature: float = 0.9
    max_new_tokens: int = 4096
    kl_coeff: float = 0.1  # MC-17: raised from 0.01 — too low caused reward hacking
    data_dir: str = "./data/synthesized"


def build_reward_fn(simulator: MaintainerSimulator):
    """Build the GRPO reward function using the maintainer simulator."""

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        # GRPOTrainer passes completions as a flat list of strings (not nested).
        # There are num_generations completions per prompt, interleaved in order.
        rewards = []
        n = len(completions) // len(prompts)  # num_generations
        for i, prompt in enumerate(prompts):
            repo = _extract_repo(prompt)
            conventions = _extract_conventions(prompt)
            for completion in completions[i * n : (i + 1) * n]:
                score = simulator.score(
                    repo=repo,
                    pr_title="",
                    pr_description=_extract_pr_description(completion),
                    code_diff=_extract_code_changes(completion),
                    metadata=_estimate_metadata(completion),
                    conventions=conventions,
                )
                rewards.append(score.merge_probability)

        return rewards

    return reward_fn


def _extract_repo(prompt: str) -> str:
    import re

    m = re.search(r"Repository:\s*(\S+)", prompt)
    return m.group(1) if m else ""


def _extract_conventions(prompt: str) -> dict:
    # MC-16: greedy `.*` regex can grab multiple JSON objects; use raw_decode instead
    try:
        start = prompt.index("Conventions:")
        brace_pos = prompt.index("{", start)
        obj, _ = json.JSONDecoder().raw_decode(prompt, brace_pos)
        return obj
    except (ValueError, json.JSONDecodeError):
        return {}


def _extract_pr_description(completion: str) -> str:
    import re

    m = re.search(r"<pr_description>(.*?)</pr_description>", completion, re.DOTALL)
    return m.group(1).strip() if m else completion[:500]


def _extract_code_changes(completion: str) -> str:
    import re

    m = re.search(r"<code_changes>(.*?)</code_changes>", completion, re.DOTALL)
    return m.group(1).strip() if m else ""


def _estimate_metadata(completion: str) -> dict:
    # Only count '+' lines within diff hunks (after '@@' markers) to avoid
    # counting non-diff content such as PR description lines starting with '+'.
    added = 0
    in_hunk = False
    for line in completion.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk and line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif (
            in_hunk
            and not line.startswith("+")
            and not line.startswith("-")
            and not line.startswith(" ")
            and not line.startswith("\\")
        ):
            # Non-diff context line signals end of hunk
            in_hunk = False
    has_tests = (
        "def test_" in completion
        or "class Test" in completion
        or "import pytest" in completion
        or "import unittest" in completion
    )
    links_issue = "#" in completion and "issue" in completion.lower()
    return {
        "lines_added": added,
        "has_tests": has_tests,
        "links_issue": links_issue,
        "files_changed": max(1, completion.count("diff --git")),
    }


def load_rl_dataset(config: GRPOTrainingConfig) -> Dataset:
    examples = []
    data_dir = Path(config.data_dir)
    for f in data_dir.glob("**/*.jsonl"):
        with f.open() as fh:
            for line in fh:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        conventions = ex.get("conventions", {})
                        examples.append(
                            {
                                "prompt": (
                                    f"Repository: {ex.get('repo', '')}\n"
                                    f"Task: {ex.get('task', '')}\n"
                                    f"Conventions: {json.dumps(conventions)}"
                                ),
                            }
                        )
                    except json.JSONDecodeError:
                        pass
    if not examples:
        raise RuntimeError(
            f"No RL training examples found in {config.data_dir!r}. "
            "Run the discovery and synthesis pipeline first."
        )
    # MC-5: shuffle so each training step sees a representative mix of repos
    random.seed(42)
    random.shuffle(examples)
    logger.info(f"RL dataset: {len(examples):,} examples")
    # MC-4: GRPOTrainer requires a datasets.Dataset, not a plain Python list
    return Dataset.from_list(examples[:50000])


def train(config: GRPOTrainingConfig) -> None:
    logger.info(f"Loading SFT checkpoint (PEFT adapter): {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    # Read adapter_config.json to find the true base model, then wrap with the
    # PEFT adapter.  Loading a PEFT adapter directory via AutoModelForCausalLM
    # silently ignores the LoRA weights — PeftModel.from_pretrained is required.
    adapter_cfg = json.load(open(Path(config.base_model) / "adapter_config.json"))
    true_base = adapter_cfg["base_model_name_or_path"]
    base_model = AutoModelForCausalLM.from_pretrained(
        true_base,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = PeftModel.from_pretrained(base_model, config.base_model)
    model.enable_input_require_grads()

    simulator = MaintainerSimulator()
    reward_fn = build_reward_fn(simulator)
    dataset = load_rl_dataset(config)

    grpo_args = GRPOConfig(
        output_dir=config.output_dir,
        num_generations=config.group_size,
        temperature=config.temperature,
        max_completion_length=config.max_new_tokens,
        beta=config.kl_coeff,
        bf16=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else [],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_args,
        train_dataset=dataset,
    )
    try:
        trainer.train()
    finally:
        trainer.save_model(config.output_dir)
    logger.success(f"GRPO complete → {config.output_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/sft")
    parser.add_argument("--output_dir", default="./checkpoints/grpo")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train(
        GRPOTrainingConfig(
            base_model=args.base_model,
            output_dir=args.output_dir,
            group_size=args.group_size,
        )
    )


if __name__ == "__main__":
    main()
