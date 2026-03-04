# MergeCraft Roadmap

## v1 — CRAFT (Q3 2026)

Core contribution pipeline. Given a GitHub repository URL and task description, generates a complete, convention-compliant contribution ready to submit.

**Goals**:
- Simulated merge rate >75% on CraftBench-200
- Correct scope in >90% of contribution scenarios
- Open weights released on HuggingFace

**Features**:
- 3-stage trained model (SFT → GRPO → DPO)
- 4-agent pipeline (Project Analysis → Contribution → PR Description → Scope)
- Maintainer simulator (merge probability predictor)
- CONTRIBUTING.md parser and convention extractor
- Commit message convention learning
- CraftBench evaluation suite (200 scenarios)
- REST API and Python SDK

**Paper Target**: ICSE 2026 — "MergeCraft: Training Open Source Contribution AI on Merge Outcomes"

---

## v1.5 — LEARN (Q4 2026)

Persistent project memory. MergeCraft builds a convention model per-project that improves with each contribution and feedback received.

**Goals**:
- Demonstrable improvement after 5+ contributions to the same repo
- Per-project convention profiles stored persistently
- Review feedback incorporation

**Features**:
- ChromaDB-backed project convention store
- Feedback loop: maintainer review comments → convention update
- "Contribution memory": knows what worked and what didn't per repo
- Maintainer preference drift detection (projects evolve their standards)
- Cross-project convention transfer ("this repo's style is similar to Django's")
- Public contribution leaderboard

---

## v2 — NEGOTIATE (Q1 2027)

Interactive PR improvement. When a reviewer leaves comments, MergeCraft revises automatically until the PR is merged.

**Goals**:
- >60% of auto-revised PRs eventually merged (vs. typical abandonment rate of 70%+)
- Review-response latency < 5 minutes

**Features**:
- Review comment parsing and response generation
- Code revision agent (applies suggested changes)
- Conversation-aware PR updates (knows what has been addressed)
- Maintainer tone modeling (adversarial vs. collaborative reviewers)
- Multi-round negotiation (handles back-and-forth)
- Automatic test fix when reviewer points out failures
- "PR health" dashboard: tracks open PRs and their review status

---

## v3 — AUTOMATE (2027)

Autonomous contribution discovery and submission pipeline.

**Goals**:
- Discovers and contributes to issues autonomously
- Operator approval flow for submission
- Tracks portfolio of submitted contributions

**Features**:
- Issue discovery engine: finds high-value "good first issue" and "help wanted" across known repos
- Autonomous contribution generation and submission
- Operator approval UI (human reviews before submission)
- Portfolio dashboard (track all contributions and their outcomes)
- Revenue model: enterprise subscription for private repo contributions
- Integration with GitHub Actions for automated PR updates

---

## Research Paper Pipeline

| Paper | Target Venue | Core Contribution |
|---|---|---|
| MergeCraft v1 | ICSE 2026 | Merge-rate reward signal for OSS contribution training |
| CraftBench | MSR 2026 | First benchmark for open source contribution quality |
| Maintainer Simulator | FSE 2026 | Predicting PR merge outcomes from project conventions |
| Convention Learning | ICSE 2027 | Adaptive per-project convention extraction |
