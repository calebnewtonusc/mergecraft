"""
deploy/server.py — MergeCraft REST API server.

Endpoints:
  POST /contribute   — Generate a PR for a repository and task
  POST /analyze      — Analyze project conventions only
  POST /score        — Score an existing PR draft
  GET  /health       — Health check
"""

import os
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

app = FastAPI(
    title="MergeCraft API",
    description="Contributions that get merged",
    version="1.0.0",
)


class ContributeRequest(BaseModel):
    repo_url: str
    task: str
    submit: bool = False  # Actually submit the PR (requires git credentials)


class ScoreRequest(BaseModel):
    repo: str
    pr_title: str
    pr_description: str
    code_diff: str
    metadata: dict = {}
    conventions: dict = {}  # Optional project conventions; if omitted, ConventionExtractor is used.


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/contribute")
async def contribute(request: ContributeRequest):
    try:
        from agents.project_analysis_agent import ProjectAnalysisAgent
        from agents.contribution_agent import ContributionAgent
        from agents.pr_description_agent import PRDescriptionAgent
        from agents.scope_agent import ScopeAgent
        from synthesis.maintainer_simulator import MaintainerSimulator

        model_path = os.getenv("MODEL_PATH", "checkpoints/final")
        analysis_agent = ProjectAnalysisAgent(model_path=model_path)
        conventions = analysis_agent.analyze(request.repo_url)

        contribution_agent = ContributionAgent(model_path=model_path)
        changes = contribution_agent.generate(
            repo_url=request.repo_url,
            task=request.task,
            conventions=conventions,
        )

        scope_agent = ScopeAgent()
        scoped = scope_agent.right_size(changes, conventions=conventions)

        desc_agent = PRDescriptionAgent()
        description = desc_agent.write(
            code_changes=scoped,
            conventions=conventions,
            task=request.task,
        )

        sim = MaintainerSimulator()
        code_diff = scoped.get("code_changes", "")
        # Compute real metadata from the actual diff content instead of hardcoding.
        diff_lines = code_diff.splitlines()
        real_metadata = {
            "lines_added": sum(
                1 for l in diff_lines if l.startswith("+") and not l.startswith("+++")
            ),
            "lines_deleted": sum(
                1 for l in diff_lines if l.startswith("-") and not l.startswith("---")
            ),
            "files_changed": max(1, code_diff.count("diff --git")),
            "has_tests": any(
                kw in code_diff
                for kw in ["def test_", "class Test", "import pytest", "import unittest"]
            ),
            "links_issue": any(
                kw in code_diff.lower() for kw in ["closes #", "fixes #", "resolves #"]
            ),
        }
        score = sim.score(
            repo=conventions.repo,
            pr_title=description["title"],
            pr_description=description["body"],
            code_diff=code_diff,
            metadata=real_metadata,
            conventions=conventions.to_dict(),
        )

        return {
            "pr_title": description["title"],
            "pr_body": description["body"],
            "code_changes": scoped.get("code_changes", ""),
            "commit_message": changes.get("commit_message", ""),
            "simulated_merge_probability": score.merge_probability,
            "primary_rejection_risk": score.primary_rejection_risk,
            "blocking_issues": score.blocking_issues,
            "suggestions": score.suggestions,
        }
    except Exception as e:
        logger.error(f"Contribution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score")
async def score(request: ScoreRequest):
    from synthesis.maintainer_simulator import MaintainerSimulator
    from core.project_conventions import ConventionExtractor
    sim = MaintainerSimulator()
    # Use caller-supplied conventions when provided; otherwise extract them live
    # from the repository so rule-based checks (max_pr_size, test_required,
    # dco_required) use real project values instead of hardcoded defaults.
    if request.conventions:
        conventions = request.conventions
    else:
        try:
            extractor = ConventionExtractor(github_token=os.getenv("GITHUB_TOKEN"))
            conventions = extractor.extract(request.repo).to_dict()
        except Exception as e:
            logger.warning(f"Convention extraction failed for {request.repo}: {e}; using defaults")
            conventions = {}
    result = sim.score(
        repo=request.repo,
        pr_title=request.pr_title,
        pr_description=request.pr_description,
        code_diff=request.code_diff,
        metadata=request.metadata,
        conventions=conventions,
    )
    return {
        "merge_probability": result.merge_probability,
        "primary_rejection_risk": result.primary_rejection_risk,
        "blocking_issues": result.blocking_issues,
        "suggestions": result.suggestions,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")))
