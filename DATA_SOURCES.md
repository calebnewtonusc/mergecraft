# MergeCraft Data Sources

## Overview

MergeCraft's critical dataset is the PR outcome corpus — 500k+ labeled pull requests across 1000 repositories with merge/rejection labels and rejection reasons. This is the signal no other code AI trains on.

---

## Stream 1: PR Outcome Corpus (40% — ~200k labeled PRs)

### Target Repositories

Top 1000 GitHub repos by star count in target domains:

| Domain | Examples | PR Volume |
|---|---|---|
| Python web | fastapi, django, flask, starlette | High |
| Python ML | pytorch, scikit-learn, transformers, numpy | High |
| JavaScript | react, vue, next.js, express | Very high |
| Go | kubernetes, docker, gin, cobra | High |
| Rust | rust-lang/rust, tokio, serde, clap | Medium |
| DevOps | terraform, ansible, helm, k8s | Medium |

### Collection

Script: `discovery/github_pr_outcome_crawler.py`

For each target repository:
1. Fetch all closed PRs (merged + rejected) via GitHub API
2. For rejected PRs: extract the maintainer's closing comment as rejection reason
3. Classify rejection reason into 7 categories (see Architecture.md)
4. For merged PRs: extract code diff, description, review thread, final state

```python
# discovery/github_pr_outcome_crawler.py --repo django/django --limit 10000
```

### Labeling

Automated classification via regex + LLM for ambiguous cases:

| Category | Signal Patterns |
|---|---|
| SCOPE_TOO_LARGE | "too large", "please split", "too many files", "out of scope" |
| MISSING_TESTS | "needs tests", "please add tests", "no test coverage" |
| CONVENTION_VIOLATION | "please run", "formatting", "style guide", "see CONTRIBUTING" |
| NO_LINKED_ISSUE | "please open an issue", "not discussed", "needs design doc" |
| DCO_MISSING | "DCO", "Signed-off-by", "CLA" |
| DESCRIPTION_INADEQUATE | "please explain", "what is the use case", "why" |
| QUALITY | "doesn't work", "breaks tests", "wrong approach" |

---

## Stream 2: CONTRIBUTING.md Corpus (20% — ~100k files)

### Collection

Script: `discovery/contributing_md_corpus.py`

GitHub search: `filename:CONTRIBUTING.md stars:>50` → downloads all matching files.

Expected volume: ~80,000 unique CONTRIBUTING.md files across ~80,000 repos.

### Extraction

For each CONTRIBUTING.md, extract structured rules:
- Commit message format (DCO, conventional commits, custom)
- PR size limits (explicit: "PRs should not exceed X lines")
- Test requirements (explicit: "all PRs must include tests")
- Process requirements (issue-first, design-doc-first, discussion-first)
- Code style tools (black, isort, ruff, eslint, prettier, gofmt)
- Review process (who reviews, how many approvals)

Script: `discovery/contributing_md_corpus.py --extract-rules`

---

## Stream 3: Maintainer Interviews & Posts (15% — ~75k)

### Sources

**Engineering blogs and talks:**
- GitHub Blog: "What maintainers wish contributors knew"
- PyCon talks: "Sustainable open source", "How to get your PR merged"
- GitHub Universe recordings (2020-2025)
- FOSDEM contributor experience track

**Community discussions:**
- Reddit: r/programming, r/opensource ("why I close PRs" threads)
- Hacker News: "Ask HN: Maintainers, what kills contributions for you?"
- Dev.to articles: maintainer perspective on contribution quality

**Twitter/Mastodon:**
- Core maintainer threads on contribution quality
- Search: "PR closed because" OR "contributions I reject" from verified OSS maintainers

### Processing

Script: `discovery/maintainer_interviews.py`

Extracts: preference statements, rules, anti-patterns, what maintainers value.

---

## Stream 4: Review Comment Corpus (15% — ~75k)

### Collection

For each PR in Stream 1:
- Extract all review comments (blocking/non-blocking)
- Label comment type: APPROVE / REQUEST_CHANGES / COMMENT
- Classify REQUEST_CHANGES comments by category

This corpus reveals the *unwritten* rules — things maintainers request in review comments that are never documented in CONTRIBUTING.md.

### Value

Example unwritten rules discovered from review comments:
- "fastapi: always add type hints to function signatures" (not in CONTRIBUTING.md)
- "django: regression tests must use the test suite's fixture pattern" (not documented)
- "pytorch: CUDA tests must be conditional on GPU availability" (not documented)

---

## Stream 5: Domain-Specific Contribution Patterns (10% — ~50k)

Synthesized training pairs by domain:
- Web framework contributions: adding endpoints, fixing middleware, adding validators
- ML library contributions: adding metrics, fixing numerical stability, adding datasets
- DevOps tool contributions: adding CLI flags, fixing config parsing, adding providers

Each domain has distinct conventions. Web frameworks value backwards compatibility; ML libraries value numerical accuracy; DevOps tools value configurability.

---

## Data Schema

Each training example (JSONL):

```json
{
  "id": "pr_django_15234",
  "repo": "django/django",
  "pr_number": 15234,
  "pr_title": "Fixed QuerySet.update() with F() expressions on SQLite",
  "pr_description": "...",
  "code_diff": "...",
  "outcome": "merged",
  "rejection_reason": null,
  "pr_metadata": {
    "lines_added": 42,
    "lines_deleted": 8,
    "files_changed": 3,
    "has_tests": true,
    "test_lines_added": 28,
    "links_issue": true,
    "issue_number": 15100,
    "commit_count": 1,
    "commit_message": "Fixed QuerySet.update() with F() on SQLite (fixes #15100)",
    "has_dco": false,
    "has_conventional_commit": false,
    "review_comments": 4,
    "approvals": 2
  },
  "repo_conventions": {
    "commit_style": "imperative_summary_with_issue_ref",
    "test_required": true,
    "max_pr_size_soft": 200,
    "issue_first": false,
    "dco_required": false
  },
  "merge_probability_label": 0.92
}
```
