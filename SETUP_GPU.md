# GPU Setup — 18× A6000 Cluster

## Target Configuration

| Component | Specification |
|---|---|
| GPUs | 18× NVIDIA A6000 (48GB each) |
| Total VRAM | 864GB |
| Training strategy | DeepSpeed ZeRO-3 + CPU offload |
| RAM | 512GB+ recommended |

## Training Launch Commands

### Stage 1: SFT
```bash
deepspeed --num_gpus=18 training/train.py \
  --deepspeed training/configs/ds_config.json \
  --output_dir checkpoints/sft
```

### Stage 2: GRPO (Maintainer Simulator Reward)
```bash
deepspeed --num_gpus=18 training/train_rl.py \
  --deepspeed training/configs/ds_config_rl.json \
  --base_model checkpoints/sft \
  --output_dir checkpoints/grpo \
  --group_size 8
```

### Stage 3: DPO
```bash
deepspeed --num_gpus=18 training/train_dpo.py \
  --base_model checkpoints/grpo \
  --output_dir checkpoints/final
```

## Expected Training Times

| Stage | Duration (18× A6000) |
|---|---|
| Discovery | 3-5 days (GitHub API rate limits) |
| Synthesis | 8-12 hours |
| SFT | 6 hours |
| GRPO | 4 hours |
| DPO | 2 hours |
