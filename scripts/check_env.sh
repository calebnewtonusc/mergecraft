#!/usr/bin/env bash
set -e
PASS=0; FAIL=0
check() { if eval "$2" &>/dev/null; then echo "  [OK] $1"; ((PASS++)); else echo "  [FAIL] $1 — $3"; ((FAIL++)); fi; }
echo "=== MergeCraft Environment Check ==="
check "Python 3.11+"   "python3 --version | grep -E '3\.(11|12|13)'"   "Install Python 3.11+"
check "torch"          "python3 -c 'import torch'"                       "pip install torch"
check "transformers"   "python3 -c 'import transformers'"                "pip install transformers"
check "peft"           "python3 -c 'import peft'"                        "pip install peft"
check "trl"            "python3 -c 'import trl'"                         "pip install trl"
check "PyGithub"       "python3 -c 'import github'"                      "pip install PyGithub"
check "anthropic"      "python3 -c 'import anthropic'"                   "pip install anthropic"
check "chromadb"       "python3 -c 'import chromadb'"                    "pip install chromadb"
check "ANTHROPIC_KEY"  "[ -n \"\$ANTHROPIC_API_KEY\" ]"                  "Add ANTHROPIC_API_KEY to .env"
check "GITHUB_TOKEN"   "[ -n \"\$GITHUB_TOKEN\" ]"                       "Add GITHUB_TOKEN to .env (required for data collection)"
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "  GPUs: $GPU_COUNT"
echo "=== Summary: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && echo "Environment ready." || exit 1
