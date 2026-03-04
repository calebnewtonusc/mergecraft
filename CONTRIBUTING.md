# Contributing to MergeCraft

MergeCraft improves with more labeled PR outcome data and richer project convention profiles. The most valuable contributions are:

1. New labeled PR outcomes (merged/rejected with reasons) for underrepresented repositories
2. CONTRIBUTING.md extractions for unusual projects
3. New CraftBench scenarios

---

## Contributing PR Outcome Data

The model learns from labeled PR outcomes. If you know a repository's conventions well, contribute labeled pairs:

```bash
python scripts/add_pr_outcome.py \
  --repo owner/repo \
  --pr-number 1234 \
  --outcome merged \
  --conventions "tests required, DCO required, max 200 LOC"
```

Alternatively, create a JSON file following the schema in `DATA_SOURCES.md` and open a PR to `data/community/`.

---

## Contributing CraftBench Scenarios

New evaluation scenarios expand CraftBench coverage. A scenario requires:
- A real GitHub repository URL
- A contribution task description (what should be contributed)
- Ground truth label (what a correct contribution looks like)
- Minimum 5 edge cases (convention requirements, scope constraints)

```bash
python evaluation/craftbench.py --add-scenario \
  --repo django/django \
  --task "Add a timeout parameter to HttpRequest.read()" \
  --label-file path/to/ground_truth.json
```

---

## Code Standards

- Python 3.11+ with full type annotations
- Black formatting, Ruff linting
- Docstrings on all public functions
- `pytest tests/` must pass

## PR Guidelines

- Keep PRs under 400 lines
- Link the relevant issue
- Include tests for new features
- Run `bash scripts/check_env.sh` before submitting
