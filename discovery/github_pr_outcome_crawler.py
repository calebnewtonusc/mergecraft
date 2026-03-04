"""
discovery/github_pr_outcome_crawler.py — Collect PR outcomes from top GitHub repos.

For each repo, collects:
  - Merged PRs (positive examples)
  - Rejected PRs with rejection reason (negative examples with labels)
  - Maintainer review comments (rejection signals)

Usage:
    python discovery/github_pr_outcome_crawler.py --top-repos 1000 --workers 10
    python discovery/github_pr_outcome_crawler.py --repo django/django --limit 5000
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# MC-11: wrap PyGithub import so missing package gives a clear error message
try:
    from github import Github, GithubException
except ImportError as _pygithub_err:
    raise ImportError(
        "PyGithub is required: pip install PyGithub>=2.3.0"
    ) from _pygithub_err
from loguru import logger

RAW_DIR = Path("data/raw/prs")
RAW_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# MC-10: create the Github client once at module level rather than inside collect_repo_prs,
# which would instantiate a new HTTP session and rate-limit counter on every call
_github_client: Github = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()

# Rejection reason patterns (applied to closing comments)
REJECTION_PATTERNS = {
    "SCOPE_TOO_LARGE": [
        r"too large", r"please split", r"too many files", r"too many changes",
        r"scope", r"separate pr", r"smaller pr",
    ],
    "MISSING_TESTS": [
        r"needs tests", r"please add tests", r"no tests", r"test coverage",
        r"missing test", r"add.*test",
    ],
    "CONVENTION_VIOLATION": [
        r"please run", r"formatting", r"style guide", r"see contributing",
        r"code style", r"black", r"flake8", r"ruff", r"prettier", r"gofmt",
    ],
    "NO_LINKED_ISSUE": [
        r"please open an issue", r"not discussed", r"needs design", r"discuss first",
        r"open.*issue.*first", r"without.*issue",
    ],
    "DCO_MISSING": [
        r"dco", r"signed-off-by", r"cla", r"contributor license",
    ],
    "DESCRIPTION_INADEQUATE": [
        r"please explain", r"what is the use case", r"why.*change",
        r"more context", r"need.*description",
    ],
    "QUALITY": [
        r"doesn't work", r"breaks", r"wrong approach", r"not the right way",
        r"incorrect", r"fails.*test",
    ],
}

TOP_REPOS_BY_DOMAIN = [
    # Python
    "django/django", "fastapi/fastapi", "pallets/flask",
    "psf/requests", "numpy/numpy", "scikit-learn/scikit-learn",
    "huggingface/transformers", "pytorch/pytorch",
    # JavaScript
    "facebook/react", "vuejs/vue", "vercel/next.js",
    "expressjs/express", "microsoft/typescript",
    # Go
    "kubernetes/kubernetes", "docker/docker-ce",
    "golang/go", "gin-gonic/gin",
    # Rust
    "rust-lang/rust", "tokio-rs/tokio", "serde-rs/serde",
    # DevOps
    "hashicorp/terraform", "ansible/ansible",
]


def classify_rejection_reason(comment: str) -> str:
    """Classify a maintainer's closing comment into a rejection category."""
    if not comment:
        return "UNCLASSIFIED"

    comment_lower = comment.lower()
    for category, patterns in REJECTION_PATTERNS.items():
        if any(re.search(p, comment_lower) for p in patterns):
            return category

    return "UNCLASSIFIED"


def extract_pr_metadata(pr: Any) -> dict:
    """Extract structured metadata from a GitHub PR object."""
    try:
        files = list(pr.get_files())
        return {
            "lines_added": sum(f.additions for f in files),
            "lines_deleted": sum(f.deletions for f in files),
            "files_changed": len(files),
            "commit_count": pr.commits,
            "review_comments": pr.review_comments,
            "has_tests": any(
                "test" in f.filename.lower() for f in files
            ),
            "links_issue": pr.body is not None and (
                "#" in pr.body or "issue" in pr.body.lower()
            ),
            "has_dco": pr.body is not None and "Signed-off-by:" in pr.body,
        }
    except Exception:
        return {}


def collect_repo_prs(repo_name: str, limit: int = 2000) -> list[dict]:
    """Collect merged and closed (rejected) PRs from a repository."""
    # MC-10: reuse the module-level singleton instead of creating a new client per call
    try:
        repo = _github_client.get_repo(repo_name)
    except GithubException as e:
        logger.error(f"Cannot access repo {repo_name}: {e}")
        return []

    results = []

    # Collect merged PRs
    logger.info(f"  Collecting merged PRs from {repo_name}...")
    for pr in repo.get_pulls(state="closed", sort="updated", direction="desc"):
        if len(results) >= limit:
            break
        if pr.merged:
            record = {
                "id": f"{repo_name}_{pr.number}",
                "repo": repo_name,
                "pr_number": pr.number,
                "pr_title": pr.title,
                "pr_description": (pr.body or "")[:5000],
                "outcome": "merged",
                "rejection_reason": None,
                "metadata": extract_pr_metadata(pr),
            }
            results.append(record)

        # Also capture rejected PRs with maintainer closing comments
        elif not pr.merged:
            # Get closing comment
            closing_comment = ""
            try:
                comments = list(pr.get_issue_comments())
                if comments:
                    closing_comment = comments[-1].body[:2000]
            except Exception:
                pass

            rejection_reason = classify_rejection_reason(closing_comment)
            record = {
                "id": f"{repo_name}_{pr.number}",
                "repo": repo_name,
                "pr_number": pr.number,
                "pr_title": pr.title,
                "pr_description": (pr.body or "")[:5000],
                "outcome": "rejected",
                "rejection_reason": rejection_reason,
                "closing_comment": closing_comment,
                "metadata": extract_pr_metadata(pr),
            }
            results.append(record)

        time.sleep(0.1)  # Rate limiting

    return results


def save_repo_prs(repo_name: str, prs: list[dict]) -> Path:
    """Save collected PRs to JSONL file."""
    safe_name = repo_name.replace("/", "_")
    out_path = RAW_DIR / f"{safe_name}.jsonl"
    with out_path.open("w") as fh:
        for pr in prs:
            fh.write(json.dumps(pr) + "\n")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-repos", type=int, default=1000)
    parser.add_argument("--repo", help="Single repo to collect (owner/name)")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if args.repo:
        repos = [args.repo]
    else:
        repos = TOP_REPOS_BY_DOMAIN[: args.top_repos]

    logger.info(f"Collecting PR outcomes from {len(repos)} repositories")

    total_prs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(collect_repo_prs, r, args.limit): r for r in repos}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                prs = future.result()
                if prs:
                    path = save_repo_prs(repo, prs)
                    total_prs += len(prs)
                    merged = sum(1 for p in prs if p["outcome"] == "merged")
                    rejected = sum(1 for p in prs if p["outcome"] == "rejected")
                    logger.info(f"  {repo}: {merged} merged, {rejected} rejected → {path.name}")
            except Exception as e:
                logger.error(f"  {repo} failed: {e}")

    logger.success(f"Total PRs collected: {total_prs:,}")


if __name__ == "__main__":
    main()
