# MergeCraft

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Qwen2.5-7B-Coder](https://img.shields.io/badge/base_model-Qwen2.5--7B--Coder-purple.svg)](https://huggingface.co/Qwen)
[![GPUs: 18x A6000](https://img.shields.io/badge/training-18×_A6000-red.svg)](https://www.nvidia.com)
[![Market: Open Source](https://img.shields.io/badge/target-OSS_contribution_tooling-green.svg)](https://github.com/calebnewtonusc/mergecraft)

> **"Contributions that get merged."**

MergeCraft is the first AI trained on the *social and process* side of open source — the gap between "good code" and "merged code." Trained on 500k+ labeled PR outcomes (accepted/rejected with reason), thousands of CONTRIBUTING.md files, and maintainer interview transcripts, MergeCraft generates contributions that understand project conventions, right-sizes PRs, writes compelling descriptions, and passes code review. The same code gets merged in one project and rejected in another — MergeCraft knows why.

---

## Why MergeCraft Is Different

| Capability | GitHub Copilot | ChatGPT | Devin | **MergeCraft** |
|---|---|---|---|---|
| Code generation | yes | yes | yes | **yes** |
| Project convention learning | — | — | partial | **full (CONTRIBUTING.md, style guide, PR history)** |
| PR scope optimization | — | — | — | **Right-sizes PRs: knows 500-line = rejected** |
| Commit message conventions | partial | — | — | **Project-specific (DCO, conventional commits, squash)** |
| Maintainer preference modeling | — | — | — | **Trained on accepted vs. rejected PR reasons** |
| PR description quality | partial | partial | — | **Compelling narrative, links issues, explains why** |
| Test-before-implementation detection | — | — | — | **Detects projects that require tests first** |
| Merge rate reward | — | — | — | **Primary training signal = merge rate** |

---

## Architecture

```
                     ┌──────────────────────────────────────────────┐
  Target Repo URL ──►│              MERGECRAFT PIPELINE              │
                     └──────────────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼────────────────────────────┐
              ▼                          ▼                            ▼
  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
  │  Project Analysis   │   │  Contribution Agent  │   │   Scope Agent       │
  │  Agent              │   │                      │   │                     │
  │  - Read CONTRIBUTING │   │  Generate the actual │   │  Right-size the PR  │
  │  - Analyze PR hist  │   │  code changes        │   │  Split if too large │
  │  - Infer unwritten  │   │  Follow conventions  │   │  Merge if trivial   │
  │    rules            │   │  Write tests first   │   │  Max 300 lines      │
  └─────────┬───────────┘   └──────────┬──────────┘   └──────────┬──────────┘
             │                         │                           │
             └─────────────────────────┼───────────────────────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  PR Description      │
                            │  Agent               │
                            │  - Compelling title  │
                            │  - Links issue       │
                            │  - Explains why      │
                            │  - Screenshots/demos │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Maintainer Simulator│
                            │  (GRPO reward)       │
                            │  Score: 0-1          │
                            │  merge probability   │
                            └──────────┬──────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   ┌──────────┐            ┌────────────────┐
                   │ High (>  │            │  Low (<0.7)    │
                   │  0.85)   │            │  Revise or     │
                   │  Submit  │            │  Re-scope      │
                   └──────────┘            └────────────────┘
```

**Training data streams (500k+ labeled outcomes):**
- Stream 1: Merged vs. rejected PRs across top 1000 repos (labeled with reason) (40%)
- Stream 2: CONTRIBUTING.md corpus (thousands of files) (20%)
- Stream 3: Maintainer interview transcripts and blog posts (15%)
- Stream 4: Project-specific style guides and review comment corpora (15%)
- Stream 5: Successful vs. failed contribution patterns by domain (10%)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/calebnewtonusc/mergecraft
cd mergecraft
pip install -r requirements.txt
cp .env.example .env  # Fill in API keys

# Validate environment
bash scripts/check_env.sh

# Run full pipeline: data → training → eval (~36 hours on 18× A6000)
bash scripts/run_all.sh

# Or step by step:
python pipeline.py --stage discovery    # Collect PR data, CONTRIBUTING.md files
python pipeline.py --stage synthesis    # Label and synthesize training data
python pipeline.py --stage train        # SFT → GRPO → DPO
python pipeline.py --stage eval         # CraftBench evaluation
```

---

## Generate a Contribution

```bash
# Point MergeCraft at any GitHub repository
python agents/contribution_agent.py \
  --repo "https://github.com/fastapi/fastapi" \
  --task "Add response_model_exclude_unset to OpenAPI docs" \
  --output contribution/

# Or use the API
curl -X POST http://localhost:8000/contribute \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/fastapi/fastapi", "task": "..."}'
```

---

## Performance Targets (v1)

| Metric | Target | GPT-4 baseline |
|---|---|---|
| Simulated merge rate (CraftBench) | >75% | ~35% |
| PR scope correctness | >90% | ~45% |
| Commit message convention compliance | >95% | ~60% |
| Test coverage of contributions | >85% | ~55% |
| CONTRIBUTING.md adherence | >92% | ~50% |
| DCO/CLA compliance detection | >98% | ~70% |

---

## The Merge Gap

The gap between "good code" and "merged code" is entirely in the social/process layer:

**Scope violations** (most common rejection reason): A PR that fixes a bug AND refactors the module AND updates docs gets rejected for being too large. The maintainer writes "please split this." MergeCraft has learned the scope threshold of 1000+ projects.

**Convention drift**: Python projects can use Black, YAPF, flake8, ruff, or no formatter at all. Some want `snake_case`, others have 15-year-old codebases in `camelCase`. MergeCraft reads the repo history and matches.

**The tests-first rule**: Many projects (especially TDD shops like Django, pytest) require tests to be added before or with implementation. Submitting implementation without tests → rejected, regardless of code quality. MergeCraft detects this pattern from PR history.

**Commit message theology**: Some projects squash everything, some want conventional commits (feat/fix/chore), some require DCO signoffs (`Signed-off-by:`), some want issue references (`Fixes #123`). MergeCraft parses 1000 commit messages from each repo to learn the religion.

**Unwritten maintainer rules**: The scariest ones. "We don't accept any PR that touches the core module without prior discussion in an issue." Never documented, but every PR that touches core without a linked issue gets closed. MergeCraft infers these rules from rejected PR patterns.

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full technical architecture, maintainer simulator design
- [DATA_SOURCES.md](DATA_SOURCES.md) — PR outcome corpus, CONTRIBUTING.md collection
- [MODEL_CARD.md](MODEL_CARD.md) — Model specification, capabilities, limitations
- [ROADMAP.md](ROADMAP.md) — v1 through v3 feature roadmap
- [SETUP_GPU.md](SETUP_GPU.md) — 18× A6000 cluster configuration
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute to MergeCraft

---

## CraftBench

CraftBench is the first standardized evaluation suite for open source contribution quality. 200 contribution scenarios across 50 real repositories, each with:
- Target repository (real GitHub repo)
- Contribution task description
- Labeled ground truth (what was actually merged/rejected and why)
- Maintainer simulator score (proxy for merge probability)

```bash
python evaluation/craftbench.py --model checkpoints/final --all
```

---

## License

MIT License — open training pipeline, open weights after v1 release.

*The gap between "good code" and "merged code" is social. MergeCraft closes it.*
