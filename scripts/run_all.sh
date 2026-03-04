#!/usr/bin/env bash
set -e
[ -f .env ] && set -a && source .env && set +a
bash scripts/check_env.sh
echo "[1/4] Discovery..."
python3 discovery/github_pr_outcome_crawler.py --top-repos 1000 --workers 10
python3 discovery/contributing_md_corpus.py --min-stars 50
python3 discovery/maintainer_interviews.py
echo "[2/4] Synthesis..."
python3 synthesis/contribution_synthesizer.py --input-dir data/raw/prs --output-dir data/synthesized
# Preflight check: synthesize_bulk.py requires either a vLLM endpoint or the
# Anthropic API key.  Fail fast here rather than after a long discovery run.
if [ -z "${VLLM_URLS}" ] && [ -z "${ANTHROPIC_API_KEY}" ]; then
	echo "ERROR: synthesize_bulk.py requires either VLLM_URLS or ANTHROPIC_API_KEY to be set." >&2
	exit 1
fi
python3 synthesis/synthesize_bulk.py --count 50000
echo "[3/4] Training..."
deepspeed --num_gpus=18 training/train.py --deepspeed training/configs/ds_config.json --output_dir checkpoints/sft
deepspeed --num_gpus=18 training/train_rl.py --deepspeed training/configs/ds_config_rl.json --base_model checkpoints/sft --output_dir checkpoints/grpo
deepspeed --num_gpus=18 training/train_dpo.py --base_model checkpoints/grpo --output_dir checkpoints/final
echo "[4/4] Evaluation..."
python3 evaluation/craftbench.py --model checkpoints/final --all --output results/craftbench_results.json
echo "=== MergeCraft Pipeline Complete ==="
