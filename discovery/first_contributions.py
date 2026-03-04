"""
discovery/first_contributions.py - Collect "good first issue" merged PRs from top repos.

These PRs are:
  - Well-scoped by maintainers (labeled "good first issue")
  - Have clear expected outcomes
  - Teach contribution patterns
  - Often have detailed feedback from maintainers

Creates (issue_description, implementation_diff, maintainer_feedback) triples.

Usage:
    python discovery/first_contributions.py --top-repos 500 --workers 8
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

RAW_DIR = Path("data/raw/first_contributions")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FIRST_ISSUE_LABELS = [
    "good first issue",
    "good-first-issue",
    "beginner",
    "starter",
    "easy",
    "first-timers-only",
    "beginner-friendly",
    "help wanted",
]


@dataclass
class FirstContributionTriple:
    """issue_description + implementation_diff + maintainer_feedback triple."""
    id: str
    repo: str
    issue_number: int
    pr_number: int
    issue_title: str
    issue_body: str
    pr_title: str
    pr_description: str
    maintainer_feedback: list[dict]   # review comments from maintainers
    outcome: str                       # "merged" | "rejected" | "pending"
    issue_labels: list[str]
    pr_metadata: dict


def _api_get(url: str, params: dict = None) -> Optional[dict]:
    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(1, reset - int(time.time()))
            time.sleep(wait)
            continue
        if resp.status_code in (404, 422, 451):
            return None
        if resp.status_code != 200:
            return None
        return resp.json()


def _get_linked_pr(repo: str, issue_number: int) -> Optional[dict]:
    """Find the PR that closes a given issue."""
    # MC-8: `closes:` is not a valid GitHub search qualifier; search for the phrase in body text
    data = _api_get(
        f"{GITHUB_API}/search/issues",
        params={
            "q": f"repo:{repo} type:pr is:merged in:body \"closes #{issue_number}\"",
            "per_page": 5,
        },
    )
    if data:
        items = data.get("items", [])
        if items:
            return items[0]

    # Fallback: search by "fixes #X" in PR body
    data = _api_get(
        f"{GITHUB_API}/search/issues",
        params={
            "q": f"repo:{repo} type:pr is:merged \"#{issue_number}\"",
            "per_page": 5,
        },
    )
    if data:
        items = data.get("items", [])
        if items:
            return items[0]

    return None


def _get_pr_reviews(repo: str, pr_number: int) -> list[dict]:
    """Get all review comments for a PR."""
    data = _api_get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews",
        params={"per_page": 50},
    )
    if not data or not isinstance(data, list):
        return []

    return [
        {
            "author": r.get("user", {}).get("login", ""),
            "body": r.get("body", "")[:500],
            "state": r.get("state", ""),
        }
        for r in data[:10]
    ]


def _get_pr_metadata(repo: str, pr_number: int) -> dict:
    """Get PR file stats."""
    data = _api_get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
        params={"per_page": 100},
    )
    if not data or not isinstance(data, list):
        return {}
    return {
        "lines_added": sum(f.get("additions", 0) for f in data),
        "lines_deleted": sum(f.get("deletions", 0) for f in data),
        "files_changed": len(data),
        "has_tests": any("test" in f.get("filename", "").lower() for f in data),
    }


def collect_repo_first_contributions(
    repo: str,
    limit: int = 200,
) -> list[FirstContributionTriple]:
    """Collect first-contribution triples for one repo."""
    triples = []

    for label in FIRST_ISSUE_LABELS[:3]:  # Check top 3 label variants
        data = _api_get(
            f"{GITHUB_API}/repos/{repo}/issues",
            params={
                "state": "closed",
                "labels": label,
                "per_page": 100,
                "sort": "updated",
                "direction": "desc",
            },
        )
        if not data or not isinstance(data, list):
            continue

        for issue in data:
            if len(triples) >= limit:
                break
            if issue.get("pull_request"):  # Skip PRs listed as issues
                continue

            issue_number = issue.get("number", 0)
            issue_labels = [l.get("name", "") for l in issue.get("labels", [])]

            # Find linked PR
            pr_data = _get_linked_pr(repo, issue_number)
            if not pr_data:
                continue

            pr_number = pr_data.get("number", 0)
            if not pr_number:
                continue

            # Get PR details
            pr_detail = _api_get(f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}")
            if not pr_detail:
                continue

            is_merged = pr_detail.get("merged_at") is not None
            outcome = "merged" if is_merged else "rejected"

            reviews = _get_pr_reviews(repo, pr_number)
            metadata = _get_pr_metadata(repo, pr_number)

            triple = FirstContributionTriple(
                id=f"{repo.replace('/', '_')}_issue{issue_number}_pr{pr_number}",
                repo=repo,
                issue_number=issue_number,
                pr_number=pr_number,
                issue_title=issue.get("title", "")[:200],
                issue_body=(issue.get("body", "") or "")[:3000],
                pr_title=pr_detail.get("title", "")[:200],
                pr_description=(pr_detail.get("body", "") or "")[:3000],
                maintainer_feedback=reviews,
                outcome=outcome,
                issue_labels=issue_labels,
                pr_metadata=metadata,
            )
            triples.append(triple)
            time.sleep(0.2)

        if triples:
            break  # Found issues with this label, don't need to check others

    return triples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect good-first-issue contribution triples")
    parser.add_argument("--top-repos", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=200, help="Triples per repo")
    parser.add_argument("--output", default="data/raw/first_contributions")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "first_contribution_triples.jsonl"

    try:
        from discovery.merged_pr_corpus import TOP_REPOS
        repos = TOP_REPOS[:args.top_repos]
    except ImportError:
        repos = [
            "django/django", "fastapi/fastapi", "pallets/flask",
            "numpy/numpy", "facebook/react", "golang/go",
        ][:args.top_repos]

    logger.info(f"Collecting first-contribution triples from {len(repos)} repos")

    total = 0
    with open(out_file, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(collect_repo_first_contributions, r, args.limit): r for r in repos}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    triples = future.result()
                    for t in triples:
                        out_f.write(json.dumps(asdict(t)) + "\n")
                    total += len(triples)
                    if triples:
                        logger.info(f"  {repo}: {len(triples)} triples")
                except Exception as e:
                    logger.debug(f"  {repo} failed: {e}")

    logger.success(f"Total first-contribution triples: {total} -> {out_file}")


if __name__ == "__main__":
    main()
