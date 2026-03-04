"""
discovery/contributing_md_corpus.py — Collect CONTRIBUTING.md files from GitHub.

Builds a corpus of project contribution guidelines for convention extraction.
Cross-references with PR outcome data to validate what rules are actually enforced.

Usage:
    python discovery/contributing_md_corpus.py --min-stars 50 --workers 20
    python discovery/contributing_md_corpus.py --extract-rules --output data/raw/conventions/
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from loguru import logger

RAW_DIR = Path("data/raw/contributing_mds")
RAW_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API = "https://api.github.com"


# ─── Convention Extractor ─────────────────────────────────────────────────────

CONVENTION_PATTERNS = {
    # Commit message styles
    "commit_style": {
        "dco": r"(DCO|Signed-off-by|Developer Certificate)",
        "conventional_commits": r"(conventional commit|feat:|fix:|chore:|docs:)",
        "imperative": r"(imperative mood|imperative tense|present tense)",
        "issue_reference": r"(Fixes #|Closes #|refs? #|issue reference)",
    },
    # Test requirements
    "test_requirements": {
        "tests_required": r"(all.*PR.*include.*test|tests? (?:must|should|required)|add.*test)",
        "tests_first": r"(test.?first|TDD|tests? before)",
        "coverage_requirement": r"(\d+%?\s*(?:test\s*)?coverage)",
    },
    # Scope limits
    "scope_limits": {
        "max_lines": r"((?:max|maximum|no more than|under)\s*(\d+)\s*(?:lines?|LOC))",
        "single_issue": r"(one change per PR|single (issue|bug|feature) per PR)",
        "atomic": r"(atomic (commits?|PRs?))",
    },
    # Code style
    "code_style": {
        "black": r"\bblack\b",
        "ruff": r"\bruff\b",
        "flake8": r"\bflake8\b",
        "isort": r"\bisort\b",
        "prettier": r"\bprettier\b",
        "eslint": r"\beslint\b",
        "gofmt": r"\bgofmt\b",
    },
    # Process requirements
    "process": {
        "issue_first": r"(open an issue first|create an issue before|discuss.*before|issue.*before.*PR)",
        "design_doc": r"(design doc|RFC|proposal|design review)",
        "review_count": r"(\d+\s*(?:approvals?|reviews?)(?:\s*required)?)",
    },
}


def extract_conventions(contributing_md: str, repo: str) -> dict:
    """
    Extract structured conventions from a CONTRIBUTING.md file.

    Returns a dict of detected conventions.
    """
    text = contributing_md.lower()
    conventions: dict = {"repo": repo, "detected": {}}

    for category, patterns in CONVENTION_PATTERNS.items():
        conventions["detected"][category] = {}
        for name, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                conventions["detected"][category][name] = True
                if m.lastindex:
                    conventions["detected"][category][f"{name}_value"] = m.group(m.lastindex)

    # Heuristic: if nothing detected about tests, it's likely not enforced
    if not any(conventions["detected"].get("test_requirements", {}).values()):
        conventions["detected"]["test_requirements"]["likely_not_enforced"] = True

    return conventions


def download_contributing_md(repo_full_name: str) -> tuple[str, str] | None:
    """Download CONTRIBUTING.md for a repository."""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

    for filename in ["CONTRIBUTING.md", "contributing.md", ".github/CONTRIBUTING.md"]:
        url = f"{GITHUB_API}/repos/{repo_full_name}/contents/{filename}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "content" in data:
                    import base64
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                    return repo_full_name, content
        except Exception:
            pass

    return None


def search_and_collect(min_stars: int = 50, limit: int = 100000) -> int:
    """Search GitHub for repos with CONTRIBUTING.md files."""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

    # Search for repos with CONTRIBUTING.md
    repos_collected = 0
    page = 1
    while repos_collected < limit:
        params = {
            "q": f"filename:CONTRIBUTING.md stars:>{min_stars}",
            "type": "repositories",
            "per_page": 100,
            "page": page,
        }
        resp = requests.get(
            f"{GITHUB_API}/search/repositories",
            headers=headers, params=params, timeout=30,
        )

        if resp.status_code == 403:
            logger.warning("Rate limit — sleeping 60s")
            time.sleep(60)
            continue

        resp.raise_for_status()  # MC-22: check HTTP status before calling .json()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            repo_name = item["full_name"]
            result = download_contributing_md(repo_name)
            if result:
                repo, content = result
                conventions = extract_conventions(content, repo)

                # Save raw content
                safe_name = repo.replace("/", "_")
                raw_path = RAW_DIR / f"{safe_name}_CONTRIBUTING.md"
                raw_path.write_text(content)

                # Save extracted conventions
                conv_path = RAW_DIR / f"{safe_name}_conventions.json"
                conv_path.write_text(json.dumps(conventions, indent=2))

                repos_collected += 1
                if repos_collected % 100 == 0:
                    logger.info(f"Collected {repos_collected} CONTRIBUTING.md files...")

        page += 1
        time.sleep(1)

    return repos_collected


def main() -> None:
    import argparse
    import concurrent.futures

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-stars", type=int, default=50)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--extract-rules", action="store_true")
    parser.add_argument("--output", default="data/raw/conventions")
    args = parser.parse_args()

    # MC-23: --workers is parsed but search_and_collect iterates repos sequentially.
    # TODO: parallelize repo downloads using ThreadPoolExecutor(max_workers=args.workers)
    count = search_and_collect(min_stars=args.min_stars, limit=args.limit)
    logger.success(f"Collected {count} CONTRIBUTING.md files")

    if args.extract_rules:
        # Re-process all collected files for convention extraction
        all_conventions = []
        for conv_file in RAW_DIR.glob("*_conventions.json"):
            try:
                conv = json.loads(conv_file.read_text())
                all_conventions.append(conv)
            except Exception:
                pass

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "all_conventions.jsonl").write_text(
            "\n".join(json.dumps(c) for c in all_conventions)
        )
        logger.success(f"Extracted conventions from {len(all_conventions)} repos → {args.output}")


if __name__ == "__main__":
    main()
