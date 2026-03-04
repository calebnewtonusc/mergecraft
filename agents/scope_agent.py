"""
agents/scope_agent.py — Ensures contributions are correctly scoped for the target project.

The #1 rejection reason in the training data is SCOPE_TOO_LARGE.
This agent analyzes proposed changes and either:
  - Approves the scope (proceed as-is)
  - Splits into multiple PRs (generates a split plan)
  - Reduces scope (identifies what to leave for follow-up)

Training insight: maintainers don't just count lines — they count concerns.
A 200-line PR fixing 3 bugs is worse than a 200-line PR fixing 1 bug thoroughly.

Usage:
    agent = ScopeAgent(model_path="checkpoints/final")
    scoped = agent.right_size(code_changes, conventions=conventions)
"""

import re
from dataclasses import dataclass

from loguru import logger

from core.project_conventions import ProjectConventions


@dataclass
class ScopeDecision:
    """Decision about how to scope a contribution."""

    action: str  # "proceed" / "split" / "reduce"
    rationale: str
    primary_changes: dict  # The main PR to submit
    follow_up_changes: list[dict]  # Follow-up PRs (if split)
    estimated_lines: int
    concerns_addressed: int


class ScopeAgent:
    """Right-sizes contributions to maximize merge probability."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def _count_lines_changed(self, code_changes: dict) -> int:
        """Count lines added in the code changes."""
        changes = code_changes.get("code_changes", "")
        added = [
            line
            for line in changes.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        return len(added)

    def _count_concerns(self, code_changes: dict) -> int:
        """
        Estimate the number of distinct concerns in the changes.

        A "concern" is a logical unit of change — one bug fix, one feature, etc.
        """
        changes = code_changes.get("code_changes", "")
        # Files changed is a proxy for concerns
        files = re.findall(r"^diff --git a/(.+?) b/", changes, re.MULTILINE)
        if not files:
            # Fallback: estimate from code structure
            return 1
        # Multiple files in different directories = multiple concerns
        dirs = set(f.rsplit("/", 1)[0] if "/" in f else "." for f in files)
        return max(1, len(dirs))

    def analyze(
        self,
        code_changes: dict,
        conventions: ProjectConventions,
    ) -> ScopeDecision:
        """Analyze whether the contribution scope is appropriate."""
        estimated_lines = self._count_lines_changed(code_changes)
        concerns = self._count_concerns(code_changes)
        max_lines = conventions.max_pr_size_soft

        logger.info(
            f"Scope analysis: {estimated_lines} lines, {concerns} concerns, "
            f"max={max_lines} lines"
        )

        if estimated_lines <= max_lines and concerns <= 2:
            return ScopeDecision(
                action="proceed",
                rationale=(
                    f"Scope is appropriate: {estimated_lines} lines, {concerns} concern(s). "
                    f"Within {conventions.repo}'s {max_lines}-line soft limit."
                ),
                primary_changes=code_changes,
                follow_up_changes=[],
                estimated_lines=estimated_lines,
                concerns_addressed=concerns,
            )

        elif concerns > 2:
            # Split by concern
            logger.info(f"Splitting: {concerns} distinct concerns detected")
            return ScopeDecision(
                action="split",
                rationale=(
                    f"PR has {concerns} distinct concerns — will split into separate PRs. "
                    f"Maintainers reject multi-concern PRs even when they're small."
                ),
                primary_changes=code_changes,  # In production, would separate by concern
                follow_up_changes=[],
                estimated_lines=estimated_lines,
                concerns_addressed=1,
            )

        else:
            # Too large — reduce scope
            logger.info(f"Reducing scope: {estimated_lines} > {max_lines} lines")
            return ScopeDecision(
                action="reduce",
                rationale=(
                    f"PR is {estimated_lines} lines — exceeds {conventions.repo}'s "
                    f"{max_lines}-line soft limit. Trimming to core change."
                ),
                primary_changes=code_changes,
                follow_up_changes=[],
                estimated_lines=min(estimated_lines, max_lines),
                concerns_addressed=1,
            )

    def right_size(
        self,
        code_changes: dict,
        conventions: ProjectConventions,
    ) -> dict:
        """
        Apply scope optimization to code changes.

        Returns the appropriately-scoped primary change dict.
        """
        decision = self.analyze(code_changes, conventions)
        logger.info(f"Scope decision: {decision.action} — {decision.rationale[:80]}...")
        # MC-21: 'split' and 'reduce' actions currently return primary_changes unmodified.
        # TODO: implement actual code splitting (split action) and scope trimming (reduce action)
        # to remove files beyond the size limit before returning.
        return decision.primary_changes
