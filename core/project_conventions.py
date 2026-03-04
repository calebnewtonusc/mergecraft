"""
core/project_conventions.py — Extract and model project contribution conventions.

Parses multiple signals to build a complete convention profile:
  - CONTRIBUTING.md (explicit rules)
  - PR history (implicit rules from what was merged/rejected)
  - Commit message patterns (1000 most recent commits)
  - Linter config files (.flake8, pyproject.toml, .eslintrc, etc.)
  - Review comment patterns

Usage:
    extractor = ConventionExtractor(github_token=token)
    conventions = extractor.extract("django/django")
"""

import os
import re
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ProjectConventions:
    """Complete convention profile for a GitHub project."""

    repo: str

    # Commit conventions
    commit_style: str = "imperative"  # imperative / conventional / custom / unknown
    commit_requires_issue_ref: bool = False
    commit_requires_dco: bool = False
    squash_preferred: bool = False

    # PR scope
    max_pr_size_soft: int = 500  # Lines added (advisory)
    max_pr_size_hard: int = 2000  # Lines added (enforced via bot or policy)
    single_issue_per_pr: bool = True

    # Test requirements
    test_required: bool = True
    tests_first: bool = False  # TDD projects
    min_test_coverage: float | None = None

    # Process
    issue_first: bool = False
    design_doc_required: bool = False
    min_approvals: int = 1
    cla_required: bool = False

    # Code style
    formatters: list[str] = field(default_factory=list)  # black, ruff, prettier, etc.
    linters: list[str] = field(default_factory=list)

    # Unwritten rules (inferred from rejected PR patterns)
    unwritten_rules: list[str] = field(default_factory=list)

    # Confidence scores (how sure we are about each convention)
    confidence: dict[str, float] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Human-readable summary of conventions."""
        lines = [
            f"Repository: {self.repo}",
            f"Commit style: {self.commit_style}",
            f"DCO required: {self.commit_requires_dco}",
            f"Tests required: {self.test_required}",
            f"Issue first: {self.issue_first}",
            f"Max PR size (soft): {self.max_pr_size_soft} lines",
            f"Formatters: {', '.join(self.formatters) or 'none detected'}",
        ]
        if self.unwritten_rules:
            lines.append(f"Unwritten rules: {'; '.join(self.unwritten_rules)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "commit_style": self.commit_style,
            "commit_requires_issue_ref": self.commit_requires_issue_ref,
            # Use keys that MaintainerSimulator._rule_based_score() expects:
            # "max_pr_size" (not "max_pr_size_soft") and "dco_required"
            # (not "commit_requires_dco").
            "dco_required": self.commit_requires_dco,
            "test_required": self.test_required,
            "issue_first": self.issue_first,
            "max_pr_size": self.max_pr_size_soft,
            "formatters": self.formatters,
            "linters": self.linters,
            "unwritten_rules": self.unwritten_rules,
        }


class ConventionExtractor:
    """Extracts contribution conventions from multiple signals in a GitHub repository."""

    def __init__(self, github_token: str | None = None) -> None:
        self.token = github_token or os.getenv("GITHUB_TOKEN", "")
        self._headers = {"Authorization": f"token {self.token}"} if self.token else {}

    def _get_file(self, repo: str, path: str) -> str | None:
        """Download a file from a GitHub repository."""
        import requests

        for filename in [path, path.lower(), path.upper()]:
            url = f"https://api.github.com/repos/{repo}/contents/{filename}"
            try:
                resp = requests.get(url, headers=self._headers, timeout=15)
                if resp.status_code == 200:
                    import base64

                    content = resp.json().get("content", "")
                    return base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                pass
        return None

    def _extract_from_contributing_md(self, content: str) -> dict:
        """Extract conventions from CONTRIBUTING.md content."""
        result = {}
        text = content.lower()

        # DCO
        if re.search(r"(dco|signed-off-by|developer certificate)", text):
            result["commit_requires_dco"] = True

        # Conventional commits
        if re.search(r"(conventional commit|feat:|fix:|chore:)", text):
            result["commit_style"] = "conventional"
        elif re.search(r"(imperative|present tense|not past tense)", text):
            result["commit_style"] = "imperative"

        # Tests
        if re.search(
            r"(all pr.{0,20}include test|tests? (?:must|required|should))", text
        ):
            result["test_required"] = True
        if re.search(r"(test.?first|tdd)", text):
            result["tests_first"] = True

        # Issue first
        if re.search(
            r"(open.{0,10}issue.{0,20}first|discuss.{0,20}before|issue before pr)", text
        ):
            result["issue_first"] = True

        # PR size
        size_m = re.search(
            r"(?:max|maximum|no more than)\s*(\d+)\s*(?:lines?|loc)", text
        )
        if size_m:
            result["max_pr_size_soft"] = int(size_m.group(1))

        # Formatters
        formatters = []
        for fmt in [
            "black",
            "ruff",
            "flake8",
            "isort",
            "prettier",
            "eslint",
            "gofmt",
            "rustfmt",
        ]:
            if fmt in text:
                formatters.append(fmt)
        if formatters:
            result["formatters"] = formatters

        return result

    def _extract_from_commits(self, repo: str, n: int = 100) -> dict:
        """Analyze recent commits to infer commit message conventions."""
        import requests

        result = {}
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{repo}/commits",
                headers=self._headers,
                params={"per_page": n},
                timeout=30,
            )
            if resp.status_code != 200:
                return result

            commits = resp.json()
            messages = [c["commit"]["message"].split("\n")[0] for c in commits]

            # Conventional commits
            conv_count = sum(
                1
                for m in messages
                if re.match(r"^(feat|fix|chore|docs|test|refactor):", m)
            )
            if conv_count > len(messages) * 0.5:
                result["commit_style"] = "conventional"

            # DCO
            dco_count = sum(
                1 for c in commits if "Signed-off-by:" in c["commit"]["message"]
            )
            if dco_count > len(commits) * 0.7:
                result["commit_requires_dco"] = True

            # Issue references
            issue_count = sum(1 for m in messages if re.search(r"#\d+", m))
            if issue_count > len(messages) * 0.5:
                result["commit_requires_issue_ref"] = True

        except Exception as e:
            logger.debug(f"Commit analysis failed for {repo}: {e}")

        return result

    def _infer_unwritten_rules(self, rejected_prs: list[dict]) -> list[str]:
        """Infer unwritten rules from patterns in rejected PRs."""
        rules = []
        rejection_reasons = [
            p.get("rejection_reason") for p in rejected_prs if p.get("rejection_reason")
        ]

        # Count rejection categories
        from collections import Counter

        reason_counts = Counter(rejection_reasons)

        # Identify patterns
        total_rejected = len(rejected_prs)
        for reason, count in reason_counts.most_common():
            rate = count / total_rejected if total_rejected > 0 else 0
            if rate > 0.3 and reason != "UNCLASSIFIED":
                rules.append(
                    f"{reason} is a common rejection ({rate:.0%} of closed PRs)"
                )

        return rules

    def extract(self, repo: str) -> ProjectConventions:
        """Build a complete convention profile for a repository."""
        logger.info(f"Extracting conventions for {repo}")
        conventions = ProjectConventions(repo=repo)

        # Parse CONTRIBUTING.md — track which keys were explicitly set so that
        # commit-history analysis does not silently override them.  No field is
        # ever None by default, so the old `getattr(...) is None` guard never
        # actually blocked any override.
        contributing_keys: set[str] = set()
        contributing = self._get_file(repo, "CONTRIBUTING.md")
        if contributing:
            overrides = self._extract_from_contributing_md(contributing)
            for key, val in overrides.items():
                setattr(conventions, key, val)
                contributing_keys.add(key)
            logger.debug(f"  CONTRIBUTING.md: {len(overrides)} conventions extracted")
        else:
            logger.debug(f"  No CONTRIBUTING.md found for {repo}")

        # Analyze commits — only apply values for keys not already set by
        # CONTRIBUTING.md so that explicit rules are never overridden.
        commit_conventions = self._extract_from_commits(repo)
        for key, val in commit_conventions.items():
            if key not in contributing_keys:
                setattr(conventions, key, val)

        # Check for formatter configs
        if not conventions.formatters:
            for config_file in [
                "pyproject.toml",
                ".flake8",
                "setup.cfg",
                ".eslintrc",
                ".prettierrc",
            ]:
                content = self._get_file(repo, config_file)
                if content:
                    for fmt in ["black", "ruff", "prettier", "eslint"]:
                        if fmt in content and fmt not in conventions.formatters:
                            conventions.formatters.append(fmt)

        logger.success(
            f"Convention profile for {repo}: {conventions.to_summary()[:100]}..."
        )
        return conventions
