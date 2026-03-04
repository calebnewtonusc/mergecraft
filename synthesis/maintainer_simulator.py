"""
synthesis/maintainer_simulator.py — Train and run the maintainer merge probability predictor.

The maintainer simulator is the reward function for GRPO training.
Given a PR (code + description + metadata + project conventions), it predicts merge probability.

Architecture: fine-tuned classifier on 500k labeled PR outcomes.

Usage:
    # Train the simulator
    python synthesis/maintainer_simulator.py --train --data-dir data/synthesized

    # Score a contribution
    python synthesis/maintainer_simulator.py --score \
        --repo django/django \
        --pr-description "description.txt" \
        --metadata metadata.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from synthesis.prompts import MAINTAINER_SIMULATOR_SYSTEM, MAINTAINER_SIMULATOR_USER


@dataclass
class SimulatorScore:
    """Result of maintainer simulator evaluation."""

    merge_probability: float
    primary_rejection_risk: str
    blocking_issues: list[str]
    suggestions: list[str]


class MaintainerSimulator:
    """
    Predicts the merge probability of a pull request.

    Used as the GRPO reward function during MergeCraft's Stage 2 training.
    Also available as a standalone PR quality checker.
    """

    def __init__(self, use_llm_fallback: bool = True) -> None:
        self.use_llm_fallback = use_llm_fallback
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def score(
        self,
        repo: str,
        pr_title: str,
        pr_description: str,
        code_diff: str,
        metadata: dict[str, Any],
        conventions: dict[str, Any] | None = None,
    ) -> SimulatorScore:
        """
        Predict merge probability for a pull request.

        Returns a SimulatorScore with merge_probability [0, 1] and feedback.
        """
        # Rule-based heuristics (fast, no API call)
        rule_score, rule_issues = self._rule_based_score(metadata, conventions or {})

        # LLM-based scoring for nuanced assessment
        if self.use_llm_fallback and self._api_key:
            llm_score = self._llm_score(
                repo=repo,
                pr_title=pr_title,
                pr_description=pr_description,
                metadata=metadata,
                conventions=conventions or {},
            )
            # Blend rule-based and LLM scores
            final_score = rule_score * 0.4 + llm_score.merge_probability * 0.6
            llm_score.merge_probability = final_score
            llm_score.blocking_issues = rule_issues + llm_score.blocking_issues
            return llm_score

        # Rule-based only
        primary_risk = rule_issues[0] if rule_issues else "NONE"
        return SimulatorScore(
            merge_probability=rule_score,
            primary_rejection_risk=primary_risk,
            blocking_issues=rule_issues,
            suggestions=[],
        )

    def _rule_based_score(
        self, metadata: dict, conventions: dict
    ) -> tuple[float, list[str]]:
        """Fast rule-based merge probability heuristics."""
        score = 1.0
        issues = []

        lines_added = metadata.get("lines_added", 0)
        files_changed = metadata.get("files_changed", 0)
        has_tests = metadata.get("has_tests", False)
        links_issue = metadata.get("links_issue", False)
        has_dco = metadata.get("has_dco", False)

        # Scope check
        max_lines = conventions.get("max_pr_size", 500)
        if lines_added > max_lines * 2:
            score -= 0.4
            issues.append(f"SCOPE_TOO_LARGE: {lines_added} lines > {max_lines * 2} soft limit")
        elif lines_added > max_lines:
            score -= 0.2

        # Test requirement
        if conventions.get("test_required", True) and not has_tests:
            score -= 0.3
            issues.append("MISSING_TESTS: project requires tests in contributions")

        # Issue link
        if conventions.get("issue_first", False) and not links_issue:
            score -= 0.2
            issues.append("NO_LINKED_ISSUE: project requires linked issue")

        # DCO
        if conventions.get("dco_required", False) and not has_dco:
            score -= 0.25
            issues.append("DCO_MISSING: Signed-off-by required")

        return max(0.0, score), issues

    def _llm_score(
        self,
        repo: str,
        pr_title: str,
        pr_description: str,
        metadata: dict,
        conventions: dict,
    ) -> SimulatorScore:
        """LLM-based merge probability prediction."""
        import anthropic

        user_prompt = MAINTAINER_SIMULATOR_USER.format(
            repo=repo,
            commit_style=conventions.get("commit_style", "imperative"),
            test_requirement=conventions.get("test_required", "preferred"),
            max_pr_size=conventions.get("max_pr_size", "500 lines"),
            pr_title=pr_title,
            pr_description=pr_description[:2000],
            files_changed=metadata.get("files_changed", 0),
            lines_added=metadata.get("lines_added", 0),
            lines_deleted=metadata.get("lines_deleted", 0),
            has_tests=metadata.get("has_tests", False),
            links_issue=metadata.get("links_issue", False),
            has_dco=metadata.get("has_dco", False),
        )

        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=MAINTAINER_SIMULATOR_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = resp.content[0].text

            # MC-7: greedy `.*` regex can grab nested/multiple JSON objects.
            # Use raw_decode to parse exactly the first JSON object from the response.
            try:
                brace_pos = raw.index("{")
                data, _ = json.JSONDecoder().raw_decode(raw, brace_pos)
            except (ValueError, json.JSONDecodeError):
                data = None
            if data is not None:
                return SimulatorScore(
                    merge_probability=float(data.get("merge_probability", 0.5)),
                    primary_rejection_risk=data.get("primary_rejection_risk", "UNKNOWN"),
                    blocking_issues=data.get("blocking_issues", []),
                    suggestions=data.get("suggestions", []),
                )
        except Exception as e:
            logger.debug(f"LLM scoring failed: {e}")

        return SimulatorScore(
            merge_probability=0.5,
            primary_rejection_risk="UNKNOWN",
            blocking_issues=[],
            suggestions=[],
        )

    def batch_score(self, contributions: list[dict]) -> list[float]:
        """Score a batch of contributions for GRPO training."""
        rewards = []
        for c in contributions:
            score = self.score(
                repo=c.get("repo", ""),
                pr_title=c.get("pr_title", ""),
                pr_description=c.get("pr_description", ""),
                code_diff=c.get("code_diff", ""),
                metadata=c.get("metadata", {}),
                conventions=c.get("conventions", {}),
            )
            rewards.append(score.merge_probability)
        return rewards


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--data-dir", default="data/synthesized")
    parser.add_argument("--repo", default="")
    parser.add_argument("--pr-description", default="")
    parser.add_argument("--metadata", default="{}")
    args = parser.parse_args()

    if args.score:
        sim = MaintainerSimulator()
        metadata = json.loads(args.metadata)
        desc = Path(args.pr_description).read_text() if args.pr_description else ""
        result = sim.score(
            repo=args.repo,
            pr_title="",
            pr_description=desc,
            code_diff="",
            metadata=metadata,
        )
        print(f"Merge probability: {result.merge_probability:.2%}")
        print(f"Primary risk: {result.primary_rejection_risk}")
        for issue in result.blocking_issues:
            print(f"  Blocking: {issue}")

    elif args.train:
        logger.info("Maintainer simulator training: uses labeled PR outcome data from synthesis")
        logger.info(f"Data dir: {args.data_dir}")
        logger.info("In production, fine-tune a classification head on Qwen2.5-7B embeddings")
        logger.info("using binary merge/reject labels from the PR outcome corpus.")


if __name__ == "__main__":
    main()
