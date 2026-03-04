"""
pipeline.py — MergeCraft end-to-end pipeline.

Usage:
    python pipeline.py --stage discovery   # Collect PR outcomes, CONTRIBUTING.md files
    python pipeline.py --stage synthesis   # Label and synthesize training pairs
    python pipeline.py --stage train       # SFT → GRPO → DPO
    python pipeline.py --stage eval        # CraftBench evaluation
    python pipeline.py --contribute URL    # Generate a contribution for a repo
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger


def run_stage(name: str, cmd: list[str]) -> int:
    logger.info(f"[{name.upper()}] {' '.join(cmd)}")
    t = time.time()
    r = subprocess.run(cmd, check=False)
    logger.info(f"[{name.upper()}] Done in {time.time()-t:.1f}s (exit={r.returncode})")
    return r.returncode


def stage_discovery() -> int:
    codes = []
    codes.append(run_stage("pr_crawler", [
        sys.executable, "discovery/github_pr_outcome_crawler.py",
        "--top-repos", "1000", "--workers", "10",
    ]))
    codes.append(run_stage("contributing_md", [
        sys.executable, "discovery/contributing_md_corpus.py",
        "--min-stars", "50", "--workers", "20",
    ]))
    codes.append(run_stage("maintainer_interviews", [
        sys.executable, "discovery/maintainer_interviews.py",
    ]))
    return max(codes)


def stage_synthesis() -> int:
    codes = []
    codes.append(run_stage("contribution_synthesizer", [
        sys.executable, "synthesis/contribution_synthesizer.py",
        "--input-dir", "data/raw/prs",
        "--output-dir", "data/synthesized",
    ]))
    codes.append(run_stage("maintainer_simulator_train", [
        sys.executable, "synthesis/maintainer_simulator.py",
        "--train", "--data-dir", "data/synthesized",
    ]))
    return max(codes)


def stage_train() -> int:
    for stage_cmd in [
        ["deepspeed", "--num_gpus=18", "training/train.py",
         "--deepspeed", "training/configs/ds_config.json",
         "--output_dir", "checkpoints/sft"],
        ["deepspeed", "--num_gpus=18", "training/train_rl.py",
         "--deepspeed", "training/configs/ds_config_rl.json",
         "--base_model", "checkpoints/sft",
         "--output_dir", "checkpoints/grpo"],
        ["deepspeed", "--num_gpus=18", "training/train_dpo.py",
         "--base_model", "checkpoints/grpo",
         "--output_dir", "checkpoints/final"],
    ]:
        code = run_stage(stage_cmd[2], stage_cmd)
        if code != 0:
            return code
    return 0


def stage_eval() -> int:
    return run_stage("craftbench", [
        sys.executable, "evaluation/craftbench.py",
        "--model", "checkpoints/final",
        "--all",
        "--output", "results/craftbench_results.json",
    ])


def contribute(repo_url: str, task: str | None) -> None:
    from agents.project_analysis_agent import ProjectAnalysisAgent
    from agents.contribution_agent import ContributionAgent
    from agents.pr_description_agent import PRDescriptionAgent
    from agents.scope_agent import ScopeAgent

    model_path = "checkpoints/final"
    logger.info(f"Generating contribution for: {repo_url}")

    analysis_agent = ProjectAnalysisAgent(model_path=model_path)
    conventions = analysis_agent.analyze(repo_url)

    contribution_agent = ContributionAgent(model_path=model_path)
    code_changes = contribution_agent.generate(repo_url=repo_url, task=task, conventions=conventions)

    scope_agent = ScopeAgent(model_path=model_path)
    scoped = scope_agent.right_size(code_changes, conventions=conventions)

    desc_agent = PRDescriptionAgent(model_path=model_path)
    description = desc_agent.write(
        code_changes=scoped,
        conventions=conventions,
        task=task,
    )

    logger.success(f"Contribution ready:")
    logger.info(f"  Files changed: {len(scoped)}")
    logger.info(f"  PR title: {description['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MergeCraft pipeline")
    parser.add_argument("--stage", choices=["discovery", "synthesis", "train", "eval"])
    parser.add_argument("--contribute", metavar="REPO_URL")
    parser.add_argument("--task", help="Contribution task description")
    args = parser.parse_args()

    if args.contribute:
        contribute(args.contribute, args.task)
        return

    stage_map = {
        "discovery": stage_discovery,
        "synthesis": stage_synthesis,
        "train": stage_train,
        "eval": stage_eval,
    }

    if args.stage:
        sys.exit(stage_map[args.stage]())
    else:
        for fn in stage_map.values():
            code = fn()
            if code != 0:
                sys.exit(code)


if __name__ == "__main__":
    main()
