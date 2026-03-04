"""
discovery/contributing_guidelines.py - Collect CONTRIBUTING.md and PR templates from top repos.

Fetches from top 2000 GitHub repos:
  - CONTRIBUTING.md
  - DEVELOPMENT.md
  - .github/pull_request_template.md
  - .github/PULL_REQUEST_TEMPLATE/*.md
  - CODE_OF_CONDUCT.md

Extracts:
  - PR size limits ("Keep PRs under 400 lines")
  - Test requirements ("All PRs must include tests")
  - Style guides ("Run ruff before submitting")
  - Scope rules ("One feature per PR")
  - CI requirements ("All checks must pass")

Output: (repo_name, contribution_conventions) records.

Usage:
    python discovery/contributing_guidelines.py --top-repos 2000
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

RAW_DIR = Path("data/raw/contributing")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CONTRIBUTING_PATHS = [
    "CONTRIBUTING.md",
    "CONTRIBUTING.rst",
    "DEVELOPMENT.md",
    "HACKING.md",
    ".github/CONTRIBUTING.md",
    ".github/pull_request_template.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "docs/contributing.md",
    "docs/CONTRIBUTING.md",
]

# Patterns to extract structured conventions from docs
CONVENTION_PATTERNS = {
    "pr_size_limit": [
        r"(?:keep|limit|restrict)\s+prs?\s+(?:to|under|below)\s+(\d+)\s+(?:lines?|loc)",
        r"(?:no\s+more\s+than|not\s+more\s+than)\s+(\d+)\s+(?:lines?|loc)\s+(?:per\s+pr)?",
        r"prs?\s+(?:should\s+be|must\s+be|are)\s+(?:under|below|at\s+most)\s+(\d+)",
    ],
    "test_required": [
        r"(?:all|every)\s+(?:prs?|changes?|commits?)\s+must\s+include\s+tests?",
        r"tests?\s+(?:are\s+)?required",
        r"add\s+(?:unit|integration)?\s*tests?",
        r"please\s+include\s+tests?",
        r"test\s+coverage\s+(?:must|should|is\s+required)",
    ],
    "style_tools": [
        r"\b(black|ruff|flake8|pylint|mypy|isort|autopep8)\b",
        r"\b(prettier|eslint|tslint|standard|biome)\b",
        r"\b(gofmt|goimports|staticcheck)\b",
        r"\b(rustfmt|clippy)\b",
        r"\b(ktlint|checkstyle|spotless)\b",
    ],
    "ci_required": [
        r"(?:all|every)\s+(?:ci|checks?|tests?)\s+must\s+pass",
        r"green\s+(?:ci|build|checks?)",
        r"(?:ensure|make\s+sure)\s+(?:ci|tests?)\s+(?:pass|are\s+green)",
    ],
    "issue_required": [
        r"(?:open|file|create)\s+an?\s+issue\s+(?:first|before)",
        r"(?:link|reference|close?s?|fix(?:es)?|resolve?s?)\s+#\d*\s*issue",
        r"issues?\s+(?:are\s+)?required\s+before",
        r"discuss\s+(?:in\s+)?(?:an?\s+)?issue\s+first",
    ],
    "one_thing_per_pr": [
        r"(?:one|single)\s+(?:feature|change|fix|thing)\s+per\s+pr",
        r"(?:keep|limit)\s+prs?\s+(?:to\s+)?(?:one|a\s+single)\s+",
        r"(?:atomic|focused|small)\s+prs?",
        r"(?:don'?t|avoid)\s+(?:mix|combining)\s+(?:multiple|several)",
    ],
    "commit_style": [
        r"(?:conventional\s+commits?|semantic\s+commits?)",
        r"commit\s+message\s+format",
        r"(?:feat|fix|docs|chore|refactor|test|style|ci)\s*:",
        r"(?:squash|rebase)\s+before\s+(?:merging|submitting)",
    ],
    "dco_cla": [
        r"(?:sign|signed).*dco",
        r"signed-off-by",
        r"(?:contributor\s+license\s+agreement|cla)",
    ],
}


@dataclass
class ContributionConventions:
    """Extracted contribution conventions for a repo."""

    repo: str
    has_contributing_doc: bool
    contributing_content: str  # Full text of contributing doc
    pr_template_content: str  # PR template text
    extracted: dict  # Structured extracted conventions
    raw_sources: list[str]  # Which files were found


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


def _fetch_raw_file(owner: str, repo: str, path: str) -> Optional[str]:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    return resp.text if resp.status_code == 200 else None


def _extract_conventions(text: str) -> dict:
    """Extract structured conventions from contributing doc text."""
    text_lower = text.lower()
    extracted = {}

    # PR size limit
    size_mentions = []
    for pattern in CONVENTION_PATTERNS["pr_size_limit"]:
        for m in re.finditer(pattern, text_lower):
            # MC-24: only append when the capture group actually matched;
            # the previous code appended a fake 400 whenever any match had no capture group
            if m.lastindex and m.group(1):
                size_mentions.append(int(m.group(1)))
    if size_mentions:
        extracted["pr_size_limit_lines"] = min(size_mentions)

    # Test requirement
    extracted["tests_required"] = any(
        re.search(p, text_lower) for p in CONVENTION_PATTERNS["test_required"]
    )

    # Style tools
    style_tools = set()
    for pattern in CONVENTION_PATTERNS["style_tools"]:
        for m in re.finditer(pattern, text_lower):
            style_tools.add(m.group(1))
    extracted["style_tools"] = sorted(style_tools)

    # CI required
    extracted["ci_must_pass"] = any(
        re.search(p, text_lower) for p in CONVENTION_PATTERNS["ci_required"]
    )

    # Issue required
    extracted["issue_required"] = any(
        re.search(p, text_lower) for p in CONVENTION_PATTERNS["issue_required"]
    )

    # One thing per PR
    extracted["one_thing_per_pr"] = any(
        re.search(p, text_lower) for p in CONVENTION_PATTERNS["one_thing_per_pr"]
    )

    # Commit style
    extracted["conventional_commits"] = bool(
        re.search(r"conventional\s+commits?|(?:feat|fix|docs|chore)\s*:", text_lower)
    )

    # DCO/CLA
    extracted["requires_dco_or_cla"] = any(
        re.search(p, text_lower) for p in CONVENTION_PATTERNS["dco_cla"]
    )

    # Language/framework-specific checks
    extracted["code_review_guidelines"] = (
        "code review" in text_lower or "reviewer" in text_lower
    )
    extracted["documentation_required"] = bool(
        re.search(
            r"(?:update|add|include)\s+(?:docs?|documentation|changelog)", text_lower
        )
    )

    return extracted


def collect_repo_conventions(full_name: str) -> Optional[ContributionConventions]:
    """Collect and extract conventions for one repo."""
    owner, repo = full_name.split("/", 1)
    found_sources = []
    contributing_text = ""
    pr_template_text = ""

    for path in CONTRIBUTING_PATHS:
        content = _fetch_raw_file(owner, repo, path)
        if content and len(content) > 50:
            if "pull_request_template" in path.lower():
                pr_template_text = content[:5000]
            else:
                contributing_text += content[:8000] + "\n\n"
            found_sources.append(path)
        time.sleep(0.1)

    if not found_sources:
        return None

    all_text = contributing_text + "\n" + pr_template_text
    extracted = _extract_conventions(all_text)

    return ContributionConventions(
        repo=full_name,
        has_contributing_doc=bool(contributing_text.strip()),
        contributing_content=contributing_text[:10000],
        pr_template_content=pr_template_text,
        extracted=extracted,
        raw_sources=found_sources,
    )


def get_top_repos_by_stars(n: int = 2000) -> list[str]:
    """Fetch top N GitHub repos by stars (uses search API)."""
    repos = set()
    # Search across different queries to get diverse top repos
    queries = [
        "stars:>10000 language:Python",
        "stars:>10000 language:JavaScript",
        "stars:>10000 language:TypeScript",
        "stars:>10000 language:Go",
        "stars:>5000 language:Rust",
        "stars:>5000 language:Java",
        "stars:>5000 language:C",
        "stars:>3000 language:Ruby",
    ]

    for query in queries:
        if len(repos) >= n:
            break
        for page in range(1, 11):
            data = _api_get(
                f"{GITHUB_API}/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            if not data:
                break
            items = data.get("items", [])
            for item in items:
                repos.add(item["full_name"])
            if len(items) < 100:
                break
            time.sleep(1)

    return list(repos)[:n]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect contribution guidelines from top repos"
    )
    parser.add_argument("--top-repos", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", default="data/raw/contributing")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    conventions_path = out_path / "repo_conventions.jsonl"

    logger.info(f"Fetching top {args.top_repos} repos by stars...")
    repos = get_top_repos_by_stars(args.top_repos)
    logger.info(f"Processing {len(repos)} repos for contribution guidelines")

    # MC-25: append mode accumulates duplicates on re-runs; load already-written repos first
    already_written: set[str] = set()
    if conventions_path.exists():
        with open(conventions_path) as dedup_f:
            for line in dedup_f:
                try:
                    existing = json.loads(line)
                    already_written.add(existing.get("repo", ""))
                except json.JSONDecodeError:
                    pass
    repos_to_process = [r for r in repos if r not in already_written]
    logger.info(
        f"Skipping {len(repos) - len(repos_to_process)} already-collected repos"
    )

    total = 0
    with open(conventions_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(collect_repo_conventions, r): r
                for r in repos_to_process
            }
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    conventions = future.result()
                    if conventions:
                        out_f.write(json.dumps(asdict(conventions)) + "\n")
                        total += 1
                        if total % 100 == 0:
                            logger.info(f"Collected {total} convention docs")
                except Exception as e:
                    logger.debug(f"  {repo} failed: {e}")

    logger.success(f"Total convention docs: {total} -> {conventions_path}")


if __name__ == "__main__":
    main()
