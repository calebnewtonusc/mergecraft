"""
core/maintainer_model.py — Per-project maintainer preference model.

Captures maintainer-specific preferences that go beyond project conventions.
Some maintainers have idiosyncratic rules not reflected anywhere in documentation.
These are inferred from their review comment patterns.

Usage:
    model = MaintainerModel("django/django")
    score = model.predict_review_outcome(pr_metadata)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class MaintainerProfile:
    """Profile of a specific maintainer's review patterns."""

    username: str
    repo: str

    # Review statistics
    total_reviews: int = 0
    approval_rate: float = 0.0
    avg_review_comments: float = 0.0

    # Known preferences (inferred from review comments)
    requires_test_before_impl: bool = False  # "write test first"
    prefers_small_commits: bool = False  # "please squash"
    detailed_descriptions: bool = False  # "more context needed"
    strict_naming: bool = False  # "variable names should be..."

    # Common blocking phrases this maintainer uses
    blocking_phrases: list[str] = field(default_factory=list)

    # Topics they care most about (inferred from thorough reviews)
    focus_areas: list[str] = field(default_factory=list)


class MaintainerModel:
    """
    Models maintainer behavior to predict review outcomes.

    Builds on project conventions with per-maintainer preference data.
    """

    def __init__(self, repo: str, cache_dir: Path | None = None) -> None:
        self.repo = repo
        self.cache_dir = cache_dir or Path("knowledge/maintainer_profiles")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, MaintainerProfile] = {}

    def _cache_path(self, username: str) -> Path:
        safe = self.repo.replace("/", "_")
        return self.cache_dir / f"{safe}_{username}.json"

    def load_profile(self, username: str) -> MaintainerProfile | None:
        """Load a cached maintainer profile."""
        path = self._cache_path(username)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return MaintainerProfile(**data)
            except (json.JSONDecodeError, OSError, TypeError) as e:
                # MC-20: log specific errors rather than silently swallowing all exceptions
                logger.warning(f"Failed to load maintainer profile from {path}: {e}")
        return None

    def save_profile(self, profile: MaintainerProfile) -> None:
        """Cache a maintainer profile."""
        path = self._cache_path(profile.username)
        path.write_text(json.dumps(profile.__dict__, indent=2))

    def build_profile_from_reviews(
        self, username: str, reviews: list[dict]
    ) -> MaintainerProfile:
        """Build a maintainer profile from their review comment history."""
        profile = MaintainerProfile(username=username, repo=self.repo)
        profile.total_reviews = len(reviews)

        if not reviews:
            return profile

        # Approval rate
        approvals = sum(1 for r in reviews if r.get("state") == "APPROVED")
        profile.approval_rate = approvals / len(reviews)

        # Review comment count
        comment_counts = [r.get("comment_count", 0) for r in reviews]
        profile.avg_review_comments = sum(comment_counts) / len(comment_counts)

        # Parse review comments for patterns
        all_comments = []
        for r in reviews:
            all_comments.extend(r.get("comments", []))

        comment_text = " ".join(all_comments).lower()

        # Detect preferences from comment patterns
        if re.search(r"(write test first|test before|tdd)", comment_text):
            profile.requires_test_before_impl = True

        if re.search(r"(please squash|single commit|squash.*commit)", comment_text):
            profile.prefers_small_commits = True

        if re.search(
            r"(more context|please explain|what is the use case|why)", comment_text
        ):
            profile.detailed_descriptions = True

        # Collect blocking phrases (most common REQUEST_CHANGES phrases)
        blocking_pattern = re.findall(r"please\s+\w+(?:\s+\w+){0,5}", comment_text)
        profile.blocking_phrases = list(set(blocking_pattern))[:10]

        self.save_profile(profile)
        return profile

    def predict_review_outcome(
        self,
        pr_metadata: dict[str, Any],
        active_maintainers: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Predict review outcome based on maintainer profiles.

        Returns dict with predicted outcome and maintainer-specific risks.
        """
        risks = []
        score = 0.8  # Base score

        for username in active_maintainers or []:
            profile = self.load_profile(username) or MaintainerProfile(
                username=username, repo=self.repo
            )

            if profile.requires_test_before_impl and not pr_metadata.get("has_tests"):
                risks.append(
                    f"{username}: requires tests (write-test-first maintainer)"
                )
                score -= 0.2

            if (
                profile.detailed_descriptions
                and len(pr_metadata.get("description", "")) < 100
            ):
                risks.append(f"{username}: prefers detailed descriptions")
                score -= 0.1

        return {
            "predicted_score": max(0.0, score),
            "maintainer_risks": risks,
            "recommendation": (
                "Likely to merge"
                if score >= 0.7
                else "Needs revision"
                if score >= 0.5
                else "High rejection risk"
            ),
        }
