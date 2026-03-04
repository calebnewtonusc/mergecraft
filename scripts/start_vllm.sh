#!/usr/bin/env bash
# start_vllm.sh — Launch 4 vLLM instances across 18x A6000 GPUs (16 used for synthesis)

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
API_KEY="${VLLM_API_KEY:-synthesis}"
LOG_DIR="${VLLM_LOG_DIR:-logs/vllm}"

mkdir -p "${LOG_DIR}"

echo "Starting 4x vLLM instances of ${MODEL}"
echo "Ports: 8001-8004 | Tensor parallel: 4 | GPUs per instance: 4"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "${MODEL}" \
    --tensor-parallel-size 4 --port 8001 --api-key "${API_KEY}" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 --dtype bfloat16 \
    > "${LOG_DIR}/vllm_8001.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "${MODEL}" \
    --tensor-parallel-size 4 --port 8002 --api-key "${API_KEY}" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 --dtype bfloat16 \
    > "${LOG_DIR}/vllm_8002.log" 2>&1 &

CUDA_VISIBLE_DEVICES=8,9,10,11 vllm serve "${MODEL}" \
    --tensor-parallel-size 4 --port 8003 --api-key "${API_KEY}" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 --dtype bfloat16 \
    > "${LOG_DIR}/vllm_8003.log" 2>&1 &

CUDA_VISIBLE_DEVICES=12,13,14,15 vllm serve "${MODEL}" \
    --tensor-parallel-size 4 --port 8004 --api-key "${API_KEY}" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 --dtype bfloat16 \
    > "${LOG_DIR}/vllm_8004.log" 2>&1 &

echo "Waiting 60s for models to load..."
sleep 60 && echo "vLLM ready on ports 8001-8004"

for PORT in 8001 8002 8003 8004; do
    curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 \
        && echo "[OK] port ${PORT}" \
        || echo "[FAIL] port ${PORT} — check ${LOG_DIR}/vllm_${PORT}.log"
done
