"""
discovery/merged_pr_corpus.py - Collect merged and rejected PRs from top 500 OSS repos.

For each repo collects:
  - Merged PRs: diff + review comments + approval events (positive examples)
  - Rejected PRs: diff + rejection reason from comments (negative examples for DPO)

Target: 200k merged, 50k rejected PRs.

Creates (repo_conventions, pr_diff, review_thread, outcome) records.

Usage:
    python discovery/merged_pr_corpus.py --top-repos 500 --workers 8
    python discovery/merged_pr_corpus.py --repo django/django --limit 5000
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

RAW_DIR = Path("data/raw/merged_prs")
# MC-27: mkdir is deferred to save_repo_prs() so importing this module as a library
# does not create unexpected directories on the filesystem

# Top 500 OSS repos by domain — covering all major languages
TOP_REPOS = [
    # Python
    "django/django", "fastapi/fastapi", "pallets/flask", "psf/requests",
    "numpy/numpy", "pandas-dev/pandas", "scikit-learn/scikit-learn",
    "huggingface/transformers", "pytorch/pytorch", "tensorflow/tensorflow",
    "python/cpython", "pypa/pip", "celery/celery", "sqlalchemy/sqlalchemy",
    "pytest-dev/pytest", "encode/httpx", "tiangolo/sqlmodel",
    "pydantic/pydantic", "encode/starlette", "aio-libs/aiohttp",
    # JavaScript / TypeScript
    "facebook/react", "vuejs/vue", "vercel/next.js", "expressjs/express",
    "microsoft/typescript", "denoland/deno", "sveltejs/svelte",
    "nestjs/nest", "socketio/socket.io", "axios/axios",
    "webpack/webpack", "vitejs/vite", "prettier/prettier", "eslint/eslint",
    "babel/babel", "jestjs/jest", "testing-library/react-testing-library",
    # Go
    "golang/go", "kubernetes/kubernetes", "docker/cli",
    "hashicorp/terraform", "gin-gonic/gin", "labstack/echo",
    "gorilla/mux", "spf13/cobra", "urfave/cli",
    # Rust
    "rust-lang/rust", "tokio-rs/tokio", "serde-rs/serde",
    "actix/actix-web", "hyperium/hyper", "clap-rs/clap",
    # Java
    "spring-projects/spring-boot", "apache/kafka",
    "elastic/elasticsearch", "netty/netty", "junit-team/junit5",
    # C/C++
    "torvalds/linux", "llvm/llvm-project", "nginx/nginx",
    "redis/redis", "postgres/postgres",
    # Ruby
    "rails/rails", "mperham/sidekiq", "puma/puma",
    # DevOps / Infrastructure
    "ansible/ansible", "helm/helm", "prometheus/prometheus",
    "grafana/grafana", "istio/istio", "argoproj/argo-cd",
]

# Rejection reason patterns
REJECTION_PATTERNS = {
    "SCOPE_TOO_LARGE": [
        r"too large", r"please split", r"too many files", r"too many changes",
        r"scope creep", r"separate pr", r"smaller pr", r"break.*into",
    ],
    "MISSING_TESTS": [
        r"needs? tests?", r"please add tests?", r"no tests?", r"test coverage",
        r"missing test", r"add.*test", r"unit test", r"integration test",
    ],
    "CONVENTION_VIOLATION": [
        r"please run", r"formatting", r"style guide", r"see contributing",
        r"code style", r"black", r"flake8", r"ruff", r"prettier", r"gofmt",
        r"lint", r"type hint", r"mypy",
    ],
    "NO_LINKED_ISSUE": [
        r"please open an issue", r"not discussed", r"needs design",
        r"discuss first", r"open.*issue.*first", r"without.*issue",
        r"not planned", r"out of scope",
    ],
    "DCO_CLA": [
        r"dco", r"signed-off-by", r"cla", r"contributor license",
        r"sign.*agreement",
    ],
    "DESCRIPTION_INADEQUATE": [
        r"please explain", r"what is the use case", r"why.*change",
        r"more context", r"need.*description", r"motivation", r"rationale",
    ],
    "QUALITY_ISSUE": [
        r"doesn'?t work", r"breaks", r"wrong approach", r"not the right way",
        r"incorrect", r"fails.*test", r"regression", r"bug.*introduced",
    ],
    "DUPLICATE": [
        r"duplicate", r"already exists", r"already merged", r"covered by",
        r"same as #", r"see #\d+",
    ],
    "WONTFIX": [
        r"won'?t fix", r"by design", r"not a bug", r"working as intended",
        r"won'?t merge", r"closing this",
    ],
}


@dataclass
class PRRecord:
    """A single PR record with outcome labels."""
    id: str
    repo: str
    pr_number: int
    pr_title: str
    pr_description: str
    outcome: str               # "merged" | "rejected"
    rejection_reason: Optional[str]
    closing_comment: str
    review_comments: list[dict]  # [{author, body, type}]
    approval_events: list[str]   # approving reviewer handles
    metadata: dict              # lines_added, files_changed, has_tests, etc.
    conventions_followed: dict  # detected convention compliance signals


def _api_get(url: str, params: dict = None) -> Optional[dict]:
    """GET with rate limit handling."""
    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(1, reset - int(time.time()))
            logger.warning(f"Rate limited. Sleeping {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code in (404, 422, 451):
            return None
        if resp.status_code != 200:
            logger.debug(f"API error {resp.status_code}: {url}")
            return None
        return resp.json()


def _classify_rejection(comment_body: str) -> str:
    if not comment_body:
        return "UNCLASSIFIED"
    lower = comment_body.lower()
    for category, patterns in REJECTION_PATTERNS.items():
        if any(re.search(p, lower) for p in patterns):
            return category
    return "UNCLASSIFIED"


def _extract_pr_metadata(repo: str, pr_number: int) -> dict:
    """Fetch file diff stats for a PR."""
    data = _api_get(f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files", params={"per_page": 100})
    if not data or not isinstance(data, list):
        return {}
    return {
        "lines_added": sum(f.get("additions", 0) for f in data),
        "lines_deleted": sum(f.get("deletions", 0) for f in data),
        "files_changed": len(data),
        "has_tests": any("test" in f.get("filename", "").lower() for f in data),
        "changed_files": [f.get("filename", "") for f in data[:20]],
    }


def _extract_review_comments(repo: str, pr_number: int) -> list[dict]:
    """Fetch review comments (inline + issue comments)."""
    comments = []

    # Review comments (inline)
    review_data = _api_get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews", params={"per_page": 50}
    )
    if review_data and isinstance(review_data, list):
        for review in review_data[:10]:
            comments.append({
                "author": review.get("user", {}).get("login", ""),
                "body": review.get("body", "")[:500],
                "type": review.get("state", ""),  # APPROVED, CHANGES_REQUESTED, COMMENTED
            })

    return comments


def _extract_approval_events(repo: str, pr_number: int) -> list[str]:
    """Get list of approving reviewer handles."""
    data = _api_get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews", params={"per_page": 50}
    )
    if not data or not isinstance(data, list):
        return []
    return [
        r.get("user", {}).get("login", "")
        for r in data
        if r.get("state") == "APPROVED"
    ]


def _check_conventions(pr_title: str, pr_body: str, files: list[str]) -> dict:
    """Detect convention compliance signals from PR content."""
    body_lower = (pr_body or "").lower()
    title_lower = pr_title.lower()
    return {
        "has_issue_reference": bool(re.search(r"#\d+|fixes|closes|resolves", body_lower)),
        "has_test_files": any("test" in f.lower() for f in files),
        # MC-26: "news" substring matches any filename containing "news" (e.g. "latest_news.py").
        # Restrict to exact known changelog filenames.
        "has_changelog_entry": any(
            f.lower().rstrip("/").split("/")[-1] in
            {"changelog.md", "changelog.rst", "changes.md", "changes.rst", "news.rst", "history.md"}
            for f in files
        ),
        "title_conventional_commit": bool(re.match(r"^(feat|fix|docs|chore|refactor|test|style|ci):", title_lower)),
        "body_has_description": len(pr_body or "") > 100,
        "body_has_checklist": "[ ]" in (pr_body or "") or "[x]" in (pr_body or "").lower(),
    }


def collect_repo_prs(
    repo_name: str,
    limit: int = 2000,
    min_merged_ratio: float = 0.7,
) -> list[PRRecord]:
    """Collect merged and rejected PRs from one repo."""
    records = []
    merged_count = 0
    rejected_count = 0

    # Target 70% merged, 30% rejected for DPO pairs
    target_merged = int(limit * min_merged_ratio)
    target_rejected = limit - target_merged

    logger.info(f"  Collecting PRs from {repo_name} (target: {target_merged}M/{target_rejected}R)...")

    page = 1
    while len(records) < limit and page <= 20:
        data = _api_get(
            f"{GITHUB_API}/repos/{repo_name}/pulls",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
        )
        if not data or not isinstance(data, list):
            break

        for pr in data:
            if len(records) >= limit:
                break

            is_merged = pr.get("merged_at") is not None

            if is_merged and merged_count >= target_merged:
                continue
            if not is_merged and rejected_count >= target_rejected:
                continue

            pr_number = pr.get("number", 0)
            if not pr_number:
                continue
            pr_title = pr.get("title", "")
            pr_body = pr.get("body", "") or ""

            metadata = _extract_pr_metadata(repo_name, pr_number)
            review_comments = _extract_review_comments(repo_name, pr_number)
            conventions = _check_conventions(pr_title, pr_body, metadata.get("changed_files", []))

            closing_comment = ""
            rejection_reason = None

            if not is_merged:
                # Get the last comment as likely rejection reason
                comments = _api_get(
                    f"{GITHUB_API}/repos/{repo_name}/issues/{pr_number}/comments",
                    params={"per_page": 50},
                )
                if comments and isinstance(comments, list):
                    closing_comment = comments[-1].get("body", "")[:1000]
                rejection_reason = _classify_rejection(closing_comment)

            approval_events = _extract_approval_events(repo_name, pr_number) if is_merged else []

            record = PRRecord(
                id=f"{repo_name.replace('/', '_')}_{pr_number}",
                repo=repo_name,
                pr_number=pr_number,
                pr_title=pr_title[:200],
                pr_description=pr_body[:3000],
                outcome="merged" if is_merged else "rejected",
                rejection_reason=rejection_reason,
                closing_comment=closing_comment[:500],
                review_comments=review_comments[:5],
                approval_events=approval_events[:5],
                metadata=metadata,
                conventions_followed=conventions,
            )
            records.append(record)

            if is_merged:
                merged_count += 1
            else:
                rejected_count += 1

            time.sleep(0.15)  # Rate limit buffer

        page += 1

    logger.info(f"  {repo_name}: {merged_count} merged, {rejected_count} rejected")
    return records


def save_repo_prs(repo_name: str, records: list[PRRecord]) -> Path:
    # MC-27: create directory here (not at import time) so library consumers aren't surprised
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe = repo_name.replace("/", "_")
    out_path = RAW_DIR / f"{safe}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect merged/rejected PR corpus")
    parser.add_argument("--top-repos", type=int, default=500,
                        help="Number of repos to collect from (from TOP_REPOS list)")
    parser.add_argument("--repo", help="Single repo (owner/name) to collect")
    parser.add_argument("--limit", type=int, default=2000, help="PRs per repo")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    repos = [args.repo] if args.repo else TOP_REPOS[:args.top_repos]
    logger.info(f"Collecting PR outcomes from {len(repos)} repositories")

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(collect_repo_prs, r, args.limit): r for r in repos}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                records = future.result()
                if records:
                    path = save_repo_prs(repo, records)
                    total += len(records)
                    merged = sum(1 for r in records if r.outcome == "merged")
                    rejected = len(records) - merged
                    logger.info(f"  {repo}: {merged}M/{rejected}R -> {path.name}")
            except Exception as e:
                logger.error(f"  {repo} failed: {e}")

    logger.success(f"Total PRs collected: {total:,}")


if __name__ == "__main__":
    main()
