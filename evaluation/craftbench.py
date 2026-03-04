"""
evaluation/craftbench.py — CraftBench evaluation suite.

CraftBench v1.0: 200 contribution scenarios across 50 real repositories.
Each scenario has a ground-truth label (what the project would accept).

Metrics:
  - Simulated merge rate (maintainer simulator score > 0.7)
  - Scope correctness (PR is right-sized)
  - Convention compliance (commit message, test coverage, DCO)
  - Description quality score

Usage:
    python evaluation/craftbench.py --model checkpoints/final --all
    python evaluation/craftbench.py --add-scenario --repo django/django --task "..."
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

BENCH_DIR = Path("evaluation/craftbench_data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _estimate_metadata(code_diff: str) -> dict:
    """MC-3: Estimate PR metadata from the actual code diff instead of using hardcoded values."""
    lines = code_diff.splitlines()
    lines_added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    lines_deleted = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    files_changed = max(1, code_diff.count("diff --git"))
    has_tests = any(
        kw in code_diff.lower()
        for kw in ["def test_", "class test", "pytest", "unittest", "assert ", "test_"]
    )
    links_issue = any(kw in code_diff.lower() for kw in ["closes #", "fixes #", "resolves #"])
    return {
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "files_changed": files_changed,
        "has_tests": has_tests,
        "links_issue": links_issue,
    }


@dataclass
class CraftBenchScenario:
    """A single CraftBench evaluation scenario."""

    scenario_id: str
    repo: str
    repo_url: str
    task: str
    category: str  # web_framework / ml_library / devops / system_tool / docs
    conventions: dict
    ground_truth: dict     # What the project actually accepted
    rejection_risks: list[str]  # Known rejection risks for this scenario


@dataclass
class CraftBenchResult:
    """Result for a single CraftBench scenario."""

    scenario_id: str
    repo: str
    category: str
    simulated_merge_rate: float
    scope_correct: bool
    convention_compliant: bool
    description_quality: float  # 0-1
    overall_score: float
    generation_time_s: float
    error: str | None = None


@dataclass
class CraftBenchSummary:
    """Aggregated CraftBench results."""

    total_scenarios: int
    overall_simulated_merge_rate: float
    scope_correct_rate: float
    convention_compliance_rate: float
    avg_description_quality: float
    category_scores: dict[str, float]
    results: list[CraftBenchResult]


BUILTIN_SCENARIOS: list[dict] = [
    {
        "scenario_id": "CB-DJANGO-001",
        "repo": "django/django",
        "repo_url": "https://github.com/django/django",
        "task": "Fix QuerySet.filter() with multiple Q objects using OR operator",
        "category": "web_framework",
        "conventions": {
            "commit_style": "imperative with ticket ref",
            "test_required": True,
            "max_pr_size": 200,
            "issue_first": False,
            "dco_required": False,
        },
        "ground_truth": {
            "expected_lines": 80,
            "requires_regression_test": True,
            "commit_format": "Fixed #NNNNN -- description",
        },
        "rejection_risks": [
            "Missing regression test for the specific Q() combination",
            "Too many files changed (Django reviewers prefer minimal diffs)",
        ],
    },
    {
        "scenario_id": "CB-PYTORCH-001",
        "repo": "pytorch/pytorch",
        "repo_url": "https://github.com/pytorch/pytorch",
        "task": "Add torch.nn.functional.gelu with tanh approximation option",
        "category": "ml_library",
        "conventions": {
            "commit_style": "imperative",
            "test_required": True,
            "max_pr_size": 400,
            "issue_first": True,
            "dco_required": False,
        },
        "ground_truth": {
            "expected_lines": 120,
            "requires_cuda_test": True,
            "requires_benchmark": False,
        },
        "rejection_risks": [
            "Missing CUDA implementation alongside CPU",
            "No numerical test vs reference implementation",
        ],
    },
    {
        "scenario_id": "CB-TERRAFORM-001",
        "repo": "hashicorp/terraform",
        "repo_url": "https://github.com/hashicorp/terraform",
        "task": "Add support for workspace-specific variable files (.tfvars.auto)",
        "category": "devops",
        "conventions": {
            "commit_style": "conventional",
            "test_required": True,
            "max_pr_size": 300,
            "issue_first": True,
            "dco_required": True,
        },
        "ground_truth": {
            "expected_lines": 200,
            "requires_acceptance_test": True,
            "requires_dco": True,
        },
        "rejection_risks": [
            "Missing DCO Signed-off-by",
            "No issue linked (HashiCorp requires issue-first)",
            "Missing acceptance test in testacc suite",
        ],
    },
]


class CraftBench:
    """CraftBench evaluation harness for MergeCraft."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._load_scenarios()

    def _load_scenarios(self) -> None:
        self.scenarios: list[CraftBenchScenario] = []

        for s in BUILTIN_SCENARIOS:
            self.scenarios.append(CraftBenchScenario(**s))

        if BENCH_DIR.exists():
            for f in BENCH_DIR.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    self.scenarios.append(CraftBenchScenario(**data))
                except Exception as e:
                    logger.warning(f"Failed to load scenario {f}: {e}")

        logger.info(f"Loaded {len(self.scenarios)} CraftBench scenarios")

    def _evaluate_scenario(self, scenario: CraftBenchScenario) -> CraftBenchResult:
        """Evaluate MergeCraft on one scenario."""
        from agents.project_analysis_agent import ProjectAnalysisAgent
        from agents.contribution_agent import ContributionAgent
        from agents.scope_agent import ScopeAgent
        from agents.pr_description_agent import PRDescriptionAgent
        from synthesis.maintainer_simulator import MaintainerSimulator

        start = time.time()
        try:
            # Run full contribution pipeline
            analysis_agent = ProjectAnalysisAgent(model_path=self.model_path)
            conventions = analysis_agent.analyze(scenario.repo_url)

            contribution_agent = ContributionAgent(model_path=self.model_path)
            changes = contribution_agent.generate(
                repo_url=scenario.repo_url,
                task=scenario.task,
                conventions=conventions,
            )

            scope_agent = ScopeAgent()
            scoped = scope_agent.right_size(changes, conventions=conventions)

            desc_agent = PRDescriptionAgent()
            description = desc_agent.write(
                code_changes=scoped,
                conventions=conventions,
                task=scenario.task,
            )

            # MC-3: Derive real metadata from the actual code diff instead of hardcoding
            code_diff = scoped.get("code_changes", "")
            estimated_metadata = _estimate_metadata(code_diff)

            # Evaluate with maintainer simulator
            sim = MaintainerSimulator()
            sim_result = sim.score(
                repo=scenario.repo,
                pr_title=description["title"],
                pr_description=description["body"],
                code_diff=code_diff,
                metadata=estimated_metadata,
                conventions=scenario.conventions,
            )

            # Check convention compliance
            convention_compliant = (
                sim_result.merge_probability >= 0.6
                and len(sim_result.blocking_issues) == 0
            )

            # Description quality
            desc_quality = min(1.0, len(description["body"]) / 500)  # Heuristic

            # Compute actual scope correctness: generated diff should be within
            # 2x the ground-truth expected lines.  If no expected_lines is given,
            # fall back to the per-scenario max_pr_size convention.
            expected_lines = scenario.ground_truth.get("expected_lines")
            if expected_lines is not None:
                scope_correct = estimated_metadata["lines_added"] <= expected_lines * 2
            else:
                max_pr = scenario.conventions.get("max_pr_size", 500)
                scope_correct = estimated_metadata["lines_added"] <= max_pr * 2

            elapsed = time.time() - start
            return CraftBenchResult(
                scenario_id=scenario.scenario_id,
                repo=scenario.repo,
                category=scenario.category,
                simulated_merge_rate=sim_result.merge_probability,
                scope_correct=scope_correct,
                convention_compliant=convention_compliant,
                description_quality=desc_quality,
                overall_score=(
                    sim_result.merge_probability * 0.5 +
                    float(convention_compliant) * 0.3 +
                    desc_quality * 0.2
                ),
                generation_time_s=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Scenario {scenario.scenario_id} failed: {e}")
            return CraftBenchResult(
                scenario_id=scenario.scenario_id,
                repo=scenario.repo,
                category=scenario.category,
                simulated_merge_rate=0.0,
                scope_correct=False,
                convention_compliant=False,
                description_quality=0.0,
                overall_score=0.0,
                generation_time_s=elapsed,
                error=str(e),
            )

    def run(
        self,
        category: str | None = None,
        output_path: Path | None = None,
    ) -> CraftBenchSummary:
        """Run CraftBench evaluation."""
        scenarios = self.scenarios
        if category:
            scenarios = [s for s in scenarios if s.category == category]

        logger.info(f"Running CraftBench: {len(scenarios)} scenarios")

        results = []
        for scenario in scenarios:
            logger.info(f"  [{scenario.scenario_id}] {scenario.task[:50]}...")
            result = self._evaluate_scenario(scenario)
            results.append(result)
            logger.info(f"  → merge_prob={result.simulated_merge_rate:.2f}, overall={result.overall_score:.2f}")

        # Aggregate
        overall = sum(r.simulated_merge_rate for r in results) / len(results) if results else 0
        scope_rate = sum(1 for r in results if r.scope_correct) / len(results) if results else 0
        conv_rate = sum(1 for r in results if r.convention_compliant) / len(results) if results else 0
        desc_quality = sum(r.description_quality for r in results) / len(results) if results else 0

        category_scores: dict[str, list[float]] = {}
        for r in results:
            if r.category not in category_scores:
                category_scores[r.category] = []
            category_scores[r.category].append(r.overall_score)

        category_avg = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}

        summary = CraftBenchSummary(
            total_scenarios=len(results),
            overall_simulated_merge_rate=overall,
            scope_correct_rate=scope_rate,
            convention_compliance_rate=conv_rate,
            avg_description_quality=desc_quality,
            category_scores=category_avg,
            results=results,
        )

        logger.success(f"\n=== CRAFTBENCH RESULTS ===")
        logger.success(f"Simulated merge rate: {overall:.1%}")
        logger.success(f"Scope correct: {scope_rate:.1%}")
        logger.success(f"Convention compliance: {conv_rate:.1%}")

        if output_path:
            result_data = {
                "overall_simulated_merge_rate": overall,
                "scope_correct_rate": scope_rate,
                "convention_compliance_rate": conv_rate,
                "avg_description_quality": desc_quality,
                "category_scores": category_avg,
            }
            output_path.write_text(json.dumps(result_data, indent=2))
            logger.info(f"Results saved to {output_path}")

        return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--category", help="Run specific category")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--add-scenario", action="store_true")
    parser.add_argument("--repo", help="For --add-scenario")
    parser.add_argument("--task", help="For --add-scenario")
    args = parser.parse_args()

    bench = CraftBench(model_path=args.model)
    output = Path(args.output) if args.output else RESULTS_DIR / "craftbench_results.json"
    bench.run(category=args.category, output_path=output)


if __name__ == "__main__":
    main()
