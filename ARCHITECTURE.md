# MergeCraft Architecture

## Core Thesis

Every code assistant treats open source contribution as a code generation problem. Write the code, format it nicely, open a PR. This works for toy contributions to toy projects.

Production open source is different. The top 1000 GitHub repositories by stars collectively reject 40-60% of incoming PRs — not because the code is bad, but because contributors don't understand the social contract of the project. The reviewer writes "please see CONTRIBUTING.md," "this is out of scope," "needs tests," "too many files changed," or simply closes without comment.

MergeCraft's thesis: **the gap between "good code" and "merged code" is a trainable pattern.** The social rules of open source — scope, conventions, maintainer preferences, unwritten norms — are learnable from PR outcome data. We train on 500k+ labeled PRs (merged/rejected with reason) across 1000 projects.

---

## 4-Phase Product Vision

### Phase 1 — CRAFT (v1, Q3 2026)
Core contribution pipeline. Given a GitHub repo URL and task description, generates a complete, correctly-scoped, convention-compliant PR ready for submission.

### Phase 2 — LEARN (v1.5, Q4 2026)
Adaptive convention learning. MergeCraft builds a persistent model of each project's conventions that improves with each submitted contribution and feedback received. After 5 contributions to a repo, MergeCraft knows more about that maintainer's preferences than most contributors.

### Phase 3 — NEGOTIATE (v2, Q1 2027)
Interactive PR improvement. MergeCraft processes review feedback and automatically revises contributions in response to reviewer comments, continuing until merged.

### Phase 4 — AUTOMATE (v3, 2027)
Autonomous contribution pipeline. MergeCraft discovers high-value issues across projects it has learned, generates contributions autonomously, and submits them on behalf of registered users.

---

## 7 Technical Differentiators

### 1. Trained on Merge Outcomes (Not Code Quality)
**The first model where merge rate is the primary training signal.**

Every existing code AI optimizes for code quality (does it compile? does it pass tests? is it readable?). MergeCraft optimizes for merge rate — a completely different objective. A PR can be high-quality code and still get rejected for being 500 lines when the project limit is 200. Merge rate captures all the social signal that code quality misses.

### 2. Maintainer Preference Simulator
**A model of what maintainers actually accept.**

`core/maintainer_model.py` is trained on labeled PR outcomes with maintainer review comments. Given a PR (code + description + metadata), it predicts merge probability for a specific project. This is the GRPO reward function during training.

The maintainer simulator captures:
- Scope preferences (what PR size gets accepted)
- Test requirements (tests-before or tests-with implementation)
- Commit message conventions
- Description quality signals (does it explain "why", not just "what")
- Unwritten rules inferred from rejected PR patterns

### 3. Project Convention Fingerprinting
**Reads every signal the project sends about what it wants.**

`core/project_conventions.py` extracts conventions from:
- CONTRIBUTING.md (explicit rules)
- PR history (implicit rules from what was merged/rejected)
- Commit message patterns (1000 most recent commits)
- Code style (linter config files: .flake8, pyproject.toml, .eslintrc)
- Review comment patterns ("please add tests", "this should be in a separate PR")
- Issue template requirements

### 4. Scope Optimization Agent
**The PR scope problem is the #1 rejection reason — and no one solves it.**

`agents/scope_agent.py` analyzes the intended contribution and determines:
- Whether to split it into multiple PRs
- Which files to include (what's in-scope for this change)
- Whether to address adjacent issues or leave them for follow-up PRs
- The "minimum complete change" that will be reviewed positively

Training data: PRs labeled with "too large," "out of scope," "please split" rejection reasons.

### 5. PR Description as Signal
**Maintainers read descriptions. The description is part of the contribution.**

`agents/pr_description_agent.py` writes descriptions that:
- Link the relevant issue (required by 80% of maintainer guidelines)
- Explain WHY the change is needed, not just WHAT it does
- Include before/after examples or screenshots
- List testing steps performed
- Acknowledge any tradeoffs or known limitations

MergeCraft is trained on the correlation between description quality and merge rate across 500k PRs.

### 6. 3-Stage Training (SFT → GRPO → DPO)
**Stage 2 uses simulated maintainer merge probability as reward.**

- **Stage 1 (SFT)**: Supervised learning on 500k+ (task, merged_PR) pairs
- **Stage 2 (GRPO)**: Generate candidate contributions → maintainer simulator scores merge probability → reward = merge probability. No actual PR submission required.
- **Stage 3 (DPO)**: Prefer contributions that are not only likely to merge but also high-quality code

### 7. CraftBench Defines the Standard
**No benchmark exists for open source contribution quality.**

CraftBench provides 200 real contribution scenarios with ground-truth merge outcomes. It tests:
- Does the model produce correctly-scoped PRs?
- Does it follow the project's commit message conventions?
- Does it include required tests?
- Does the maintainer simulator score it above the merge threshold?

Publishing CraftBench establishes MergeCraft as the reference for all future contribution AI.

---

## Training Data Architecture

```
Stream 1: PR Outcome Corpus (40% — ~200k labeled PRs)
├── Top 1000 GitHub repos by stars (Python, JS, Go, Rust)
├── Each PR labeled: merged/rejected + rejection reason category
├── Rejection categories:
│   ├── SCOPE_TOO_LARGE ("please split this into smaller PRs")
│   ├── MISSING_TESTS ("please add tests for this change")
│   ├── CONVENTION_VIOLATION ("please follow our formatting guide")
│   ├── NO_LINKED_ISSUE ("please open an issue to discuss first")
│   ├── DCO_MISSING ("please add Signed-off-by to commits")
│   ├── DESCRIPTION_INADEQUATE ("please explain why this change is needed")
│   └── OUT_OF_SCOPE ("this is not aligned with project goals")
└── Merged PRs: code, description, review thread, merge commit

Stream 2: CONTRIBUTING.md Corpus (20% — ~100k files)
├── Every CONTRIBUTING.md from repos with >100 stars
├── Structured extraction: rules, requirements, process
└── Cross-referenced with PR outcomes (do contributors follow the guide?)

Stream 3: Maintainer Interviews & Posts (15% — ~75k)
├── GitHub blog posts by core maintainers
├── Conference talks on OSS contribution (PyCon, GitHub Universe)
├── Reddit/HN threads: "why I closed your PR"
└── Maintainer Twitter/Mastodon threads on contribution quality

Stream 4: Review Comment Corpus (15% — ~75k)
├── Code review comments on merged and rejected PRs
├── Labeled: blocking/non-blocking, addressed/not-addressed
└── Patterns: common blocking comments that predict rejection

Stream 5: Contribution Domain Patterns (10% — ~50k)
├── Web frameworks: FastAPI, Django, Rails, Express
├── ML libraries: PyTorch, scikit-learn, Hugging Face
├── DevOps: Docker, Kubernetes, Terraform
└── System tools: Rust crates, Go libraries
```

---

## Maintainer Simulator Architecture

The maintainer simulator is the core of MergeCraft's GRPO training loop:

```
Input: {
  "repo_conventions": ProjectConventions,
  "pr_code_diff": str,         # The actual code changes
  "pr_description": str,       # PR title + body
  "pr_metadata": {
    "lines_changed": int,
    "files_changed": int,
    "has_tests": bool,
    "links_issue": bool,
    "has_dco": bool,
    "commit_count": int,
    "commit_messages": [str],
  }
}

Output: {
  "merge_probability": float,  # 0.0 - 1.0
  "primary_rejection_risk": str,  # Most likely rejection reason
  "suggestions": [str],         # What to improve
}
```

Training: Supervised on 500k labeled PR outcomes → merge probability predictor.
Fine-tuned per project type (ML library, web framework, DevOps, etc.)
