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
        # MC-28: TODO: derive real metadata from the generated code diff instead of
        # hardcoding {"lines_added": 150, "has_tests": True, "links_issue": True}.
        # Use _estimate_metadata(scoped.get("code_changes", "")) from evaluation/craftbench.py.
        score = sim.score(
            repo=conventions.repo,
            pr_title=description["title"],
            pr_description=description["body"],
            code_diff=scoped.get("code_changes", ""),
            metadata={"lines_added": 150, "has_tests": True, "links_issue": True},
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
    sim = MaintainerSimulator()
    # MC-29: TODO: load real project conventions from the conventions store (core/project_conventions.py)
    # for request.repo and pass them here. Currently no conventions are passed, so rule-based
    # checks (max_pr_size, test_required, dco_required) always use their defaults.
    result = sim.score(
        repo=request.repo,
        pr_title=request.pr_title,
        pr_description=request.pr_description,
        code_diff=request.code_diff,
        metadata=request.metadata,
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
