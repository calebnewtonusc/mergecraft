"""
synthesis/contribution_synthesizer.py — Generate (task, merged_PR) training pairs.

Converts the raw PR outcome corpus into training examples:
  - Positive: (task_description, merged_PR_code_and_description)
  - Negative: (task_description, rejected_PR) with rejection reason label

Usage:
    python synthesis/contribution_synthesizer.py \
        --input-dir data/raw/prs \
        --output-dir data/synthesized
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from synthesis.prompts import CONTRIBUTION_SYSTEM, CONTRIBUTION_USER

RAW_DIR = Path("data/raw/prs")
SYNTHESIZED_DIR = Path("data/synthesized")
# MC-31: removed unused ANTHROPIC_API_KEY constant (was imported but never referenced)
# NOTE: SYNTHESIZED_DIR.mkdir() is called lazily inside synthesize_all() to avoid
# side-effects at import time.


def pr_to_task_description(pr: dict) -> str:
    """Convert a PR record to a task description (the prompt)."""
    title = pr.get("pr_title", "")
    desc = pr.get("pr_description", "")[:500]
    return f"{title}\n\n{desc}".strip()


def pr_to_contribution(pr: dict) -> str:
    """Convert a merged PR record to a contribution string (the target)."""
    title = pr.get("pr_title", "")
    desc = pr.get("pr_description", "")
    meta = pr.get("metadata", {})
    return (
        f"PR Title: {title}\n\n"
        f"PR Description:\n{desc}\n\n"
        f"Files changed: {meta.get('files_changed', 0)}\n"
        f"Lines added: {meta.get('lines_added', 0)}\n"
        f"Has tests: {meta.get('has_tests', False)}\n"
        f"Links issue: {meta.get('links_issue', False)}"
    )


def synthesize_from_pr_file(jsonl_path: Path, output_dir: Path) -> int:
    """Process one repo's PR outcome file into training pairs."""
    pairs = []
    with jsonl_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                pr = json.loads(line)
            except json.JSONDecodeError:
                continue

            if pr.get("outcome") == "merged":
                pairs.append({
                    "id": pr["id"],
                    "repo": pr["repo"],
                    "task": pr_to_task_description(pr),
                    "contribution": pr_to_contribution(pr),
                    "outcome": "merged",
                    "rejection_reason": None,
                    "merge_probability_label": 1.0,
                    "metadata": pr.get("metadata", {}),
                })
            else:
                pairs.append({
                    "id": pr["id"],
                    "repo": pr["repo"],
                    "task": pr_to_task_description(pr),
                    "contribution": pr_to_contribution(pr),
                    "outcome": "rejected",
                    "rejection_reason": pr.get("rejection_reason"),
                    "closing_comment": pr.get("closing_comment", ""),
                    "merge_probability_label": 0.0,
                    "metadata": pr.get("metadata", {}),
                })

    if pairs:
        out_path = output_dir / f"{jsonl_path.stem}_synthesized.jsonl"
        with out_path.open("w") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair) + "\n")
        return len(pairs)
    return 0


def synthesize_all(input_dir: Path, output_dir: Path, workers: int = 8) -> int:
    """Synthesize training pairs from all collected PR outcome files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pr_files = list(input_dir.glob("*.jsonl"))
    logger.info(f"Processing {len(pr_files)} PR outcome files...")

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(synthesize_from_pr_file, f, output_dir): f for f in pr_files}
        for future in as_completed(futures):
            count = future.result()
            total += count

    logger.success(f"Synthesized {total:,} training pairs → {output_dir}")
    return total


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/prs")
    parser.add_argument("--output-dir", default="data/synthesized")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    synthesize_all(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
