"""
agents/project_analysis_agent.py — Learns project conventions before contributing.

Reads CONTRIBUTING.md, PR history, commit patterns, and linter configs to
build a complete understanding of what this project accepts.

Usage:
    agent = ProjectAnalysisAgent(model_path="checkpoints/final")
    conventions = agent.analyze("https://github.com/fastapi/fastapi")
"""

from loguru import logger

from core.project_conventions import ConventionExtractor, ProjectConventions


class ProjectAnalysisAgent:
    """Analyzes a GitHub repository to extract all contribution conventions."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.extractor = ConventionExtractor()
        self._cache: dict[str, ProjectConventions] = {}

    def _repo_from_url(self, url: str) -> str:
        """Extract owner/repo from GitHub URL."""
        import re

        m = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git|/|$)", url)
        if m:
            return m.group(1)
        return url  # Assume it's already owner/repo

    def analyze(self, repo_url: str) -> ProjectConventions:
        """
        Build a complete convention profile for a repository.

        Caches results to avoid repeated API calls.
        """
        repo = self._repo_from_url(repo_url)

        if repo in self._cache:
            logger.debug(f"Using cached conventions for {repo}")
            return self._cache[repo]

        logger.info(f"Analyzing project conventions: {repo}")
        conventions = self.extractor.extract(repo)

        self._cache[repo] = conventions
        logger.success(f"Conventions extracted for {repo}")
        return conventions

    def get_convention_summary(self, conventions: ProjectConventions) -> str:
        """Get a formatted summary of conventions for use in prompts."""
        return conventions.to_summary()
