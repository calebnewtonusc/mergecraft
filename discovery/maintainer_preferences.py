"""
discovery/maintainer_preferences.py - Extract maintainer preference patterns from PR history.

Analyzes PR comments from longtime maintainers (>100 merged PRs in the repo) to extract:
  - What language signals get praised ("great addition", "excellent tests")
  - What patterns trigger rejection requests ("please split", "needs tests", "not in scope")
  - Per-repo maintainer preference profiles

Creates (repo, maintainer_preference_patterns) records.

Usage:
    python discovery/maintainer_preferences.py --top-repos 200 --workers 8
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

RAW_DIR = Path("data/raw/maintainer_prefs")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Praise signals in review comments
PRAISE_PATTERNS = [
    r"(?:great|excellent|nice|perfect|thank(?:s| you)|well done|lgtm|looks good|approved)",
    r"(?:good|solid|clean|readable|idiomatic)\s+(?:work|code|implementation|approach)",
    r"this is exactly what we needed",
    r"(?:nice|clean)\s+(?:pr|change|fix|addition)",
    r"merge(?:d|!)",
]

# Change request signals
CHANGE_REQUEST_PATTERNS = {
    "needs_tests": [
        r"(?:please\s+add|needs?|missing|lack(?:ing)?)\s+tests?",
        r"(?:no|without)\s+tests?",
        r"test\s+coverage",
        r"add.*test.*(?:case|class|file|suite)",
    ],
    "scope_issue": [
        r"(?:please\s+)?split\s+(?:this|into\s+(?:multiple|separate))",
        r"too\s+(?:large|big|broad|many\s+changes)",
        r"out\s+of\s+scope",
        r"(?:separate|different)\s+pr",
        r"one\s+(?:thing|change|feature)\s+per\s+pr",
    ],
    "style_issue": [
        r"(?:please\s+run|use)\s+(?:black|ruff|flake8|prettier|gofmt|rustfmt)",
        r"formatting\s+(?:issue|problem|fix|nit)",
        r"(?:follow|see)\s+(?:our\s+)?(?:style\s+guide|coding\s+standard)",
        r"whitespace|indentation|trailing\s+space",
    ],
    "docs_missing": [
        r"(?:please\s+)?(?:add|update|missing)\s+(?:docs?|documentation|docstring|comment)",
        r"(?:needs?|require?s?)\s+docs?",
        r"(?:explain|describe|document)\s+(?:the\s+)?(?:change|why|purpose)",
    ],
    "design_issue": [
        r"(?:not\s+(?:the\s+)?right|wrong)\s+approach",
        r"(?:consider|prefer|suggest)\s+(?:instead|a\s+different)",
        r"(?:there\s+is|there'?s)\s+a\s+better\s+way",
        r"(?:this|that)\s+(?:could|should)\s+be\s+simplified",
    ],
}


@dataclass
class MaintainerProfile:
    """A maintainer's historical preference patterns for a repo."""
    handle: str
    repo: str
    merged_prs_reviewed: int
    rejected_prs_reviewed: int
    approval_rate: float
    praise_phrases: list[str]
    common_change_requests: dict   # category -> count
    frequently_requested_reviewers: list[str]
    avg_review_turnaround_hours: float


@dataclass
class RepoMaintainerPreferences:
    """Aggregated maintainer preferences for a repo."""
    repo: str
    top_maintainers: list[MaintainerProfile]
    aggregate_preferences: dict    # What this repo consistently requires
    rejection_signal_distribution: dict  # category -> frequency
    total_prs_analyzed: int


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
        data = resp.json()
        # Handle list responses
        if isinstance(data, list):
            return {"items": data, "_list": True}
        return data


def _classify_review_comment(body: str) -> tuple[str, list[str]]:
    """
    Classify a review comment into sentiment and change_request categories.

    Returns: (sentiment, [change_request_categories])
    """
    lower = body.lower()
    sentiment = "neutral"

    if any(re.search(p, lower) for p in PRAISE_PATTERNS):
        sentiment = "positive"

    change_requests = []
    for category, patterns in CHANGE_REQUEST_PATTERNS.items():
        if any(re.search(p, lower) for p in patterns):
            change_requests.append(category)

    if change_requests:
        sentiment = "negative"

    return sentiment, change_requests


def _get_repo_contributors(repo: str) -> list[str]:
    """Get top contributors (possible maintainers) for a repo."""
    data = _api_get(
        f"{GITHUB_API}/repos/{repo}/contributors",
        params={"per_page": 30},
    )
    if not data:
        return []
    items = data.get("items", data) if not data.get("_list") else data["items"]
    if not isinstance(items, list):
        return []
    return [c.get("login", "") for c in items[:10]]


def _analyze_maintainer_reviews(
    repo: str,
    handle: str,
    max_reviews: int = 200,
) -> Optional[MaintainerProfile]:
    """Analyze a maintainer's review history for preference patterns."""
    # Get recent PRs reviewed by this person
    reviews_url = f"{GITHUB_API}/repos/{repo}/pulls"
    all_reviews_for_maintainer = []
    merged_reviewed = 0
    rejected_reviewed = 0

    page = 1
    while len(all_reviews_for_maintainer) < max_reviews and page <= 5:
        data = _api_get(reviews_url, params={"state": "closed", "per_page": 100, "page": page})
        if not data:
            break
        prs = data.get("items", []) if data.get("_list") else []
        if not prs:
            break

        for pr in prs:
            pr_number = pr.get("number", 0)
            reviews = _api_get(
                f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews",
                params={"per_page": 50},
            )
            if not reviews:
                continue
            review_items = reviews.get("items", reviews) if isinstance(reviews, dict) else []
            if not isinstance(review_items, list):
                continue

            for review in review_items:
                reviewer = review.get("user", {}).get("login", "")
                if reviewer == handle:
                    all_reviews_for_maintainer.append({
                        "pr_merged": pr.get("merged_at") is not None,
                        "state": review.get("state", ""),
                        "body": review.get("body", "")[:500],
                    })
                    if pr.get("merged_at"):
                        merged_reviewed += 1
                    else:
                        rejected_reviewed += 1

        page += 1
        time.sleep(0.5)

    if not all_reviews_for_maintainer:
        return None

    # Aggregate patterns
    praise_phrases = []
    change_request_counts: dict[str, int] = {}

    for review in all_reviews_for_maintainer:
        sentiment, changes = _classify_review_comment(review.get("body", ""))
        if sentiment == "positive":
            # Extract actual phrases
            body = review["body"][:200]
            praise_phrases.append(body)
        for cat in changes:
            change_request_counts[cat] = change_request_counts.get(cat, 0) + 1

    total = max(merged_reviewed + rejected_reviewed, 1)
    approval_rate = merged_reviewed / total

    return MaintainerProfile(
        handle=handle,
        repo=repo,
        merged_prs_reviewed=merged_reviewed,
        rejected_prs_reviewed=rejected_reviewed,
        approval_rate=round(approval_rate, 3),
        praise_phrases=praise_phrases[:5],
        common_change_requests=change_request_counts,
        frequently_requested_reviewers=[],
        avg_review_turnaround_hours=0.0,
    )


def analyze_repo_maintainers(repo: str) -> Optional[RepoMaintainerPreferences]:
    """Analyze maintainer preferences for a repo."""
    logger.info(f"  Analyzing {repo}...")

    contributors = _get_repo_contributors(repo)
    if not contributors:
        return None

    top_maintainers = []
    for handle in contributors[:5]:
        profile = _analyze_maintainer_reviews(repo, handle, max_reviews=50)
        if profile and profile.merged_prs_reviewed + profile.rejected_prs_reviewed >= 5:
            top_maintainers.append(profile)
        time.sleep(0.3)

    if not top_maintainers:
        return None

    # Aggregate across all maintainers
    aggregate: dict[str, object] = {}
    agg_change_requests: dict[str, int] = {}
    total_prs = sum(m.merged_prs_reviewed + m.rejected_prs_reviewed for m in top_maintainers)

    for m in top_maintainers:
        for cat, count in m.common_change_requests.items():
            agg_change_requests[cat] = agg_change_requests.get(cat, 0) + count

    # Sort by frequency
    rejection_distribution = dict(
        sorted(agg_change_requests.items(), key=lambda x: x[1], reverse=True)
    )

    # Derive aggregate preferences
    aggregate["consistently_requires_tests"] = agg_change_requests.get("needs_tests", 0) > 3
    aggregate["scope_sensitive"] = agg_change_requests.get("scope_issue", 0) > 2
    aggregate["style_enforced"] = agg_change_requests.get("style_issue", 0) > 2
    aggregate["docs_valued"] = agg_change_requests.get("docs_missing", 0) > 2
    avg_approval = sum(m.approval_rate for m in top_maintainers) / len(top_maintainers)
    aggregate["avg_approval_rate"] = round(avg_approval, 3)

    return RepoMaintainerPreferences(
        repo=repo,
        top_maintainers=top_maintainers,
        aggregate_preferences=aggregate,
        rejection_signal_distribution=rejection_distribution,
        total_prs_analyzed=total_prs,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect maintainer preference patterns")
    parser.add_argument("--top-repos", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="data/raw/maintainer_prefs")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    prefs_path = out_path / "maintainer_preferences.jsonl"

    # Import top repos list from merged_pr_corpus
    try:
        from discovery.merged_pr_corpus import TOP_REPOS
        repos = TOP_REPOS[:args.top_repos]
    except ImportError:
        repos = [
            "django/django", "fastapi/fastapi", "pallets/flask",
            "numpy/numpy", "facebook/react", "vercel/next.js",
        ][:args.top_repos]

    logger.info(f"Analyzing maintainer preferences for {len(repos)} repos")

    # Load repos already written to the output file to avoid duplicates.
    already_written: set[str] = set()
    if prefs_path.exists():
        with open(prefs_path) as existing_f:
            for line in existing_f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("repo"):
                            already_written.add(entry["repo"])
                    except json.JSONDecodeError:
                        pass

    total = 0
    with open(prefs_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(analyze_repo_maintainers, r): r for r in repos}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    prefs = future.result()
                    if prefs:
                        if prefs.repo in already_written:
                            logger.debug(f"  {repo}: already in output, skipping")
                            continue
                        out_f.write(json.dumps(asdict(prefs)) + "\n")
                        already_written.add(prefs.repo)
                        total += 1
                        logger.info(
                            f"  {repo}: {len(prefs.top_maintainers)} maintainers analyzed"
                        )
                except Exception as e:
                    logger.debug(f"  {repo} failed: {e}")

    logger.success(f"Total repo preference profiles: {total} -> {prefs_path}")


if __name__ == "__main__":
    main()
