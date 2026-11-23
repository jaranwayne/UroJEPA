#!/usr/bin/env bash
# Fine-tune one encoder on the four outcomes with five seeds (0-4) and evaluate on the
# internal test split and the external cohort.
#
# Usage: bash scripts/run_finetune.sh <model_name> <encoder_checkpoint> [num_gpus]
#   bash scripts/run_finetune.sh urojepa        outputs/pretrain/urojepa_vitb16/latest.pth.tar
#   bash scripts/run_finetune.sh urojepa_dagger outputs/pretrain/urojepa_dagger_vitb16/latest.pth.tar
#   bash scripts/run_finetune.sh vjepa2_1       checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
#
# Override the task/seed lists with TASKS="..." SEEDS="..." if needed.
set -euo pipefail

NAME=${1:?model name}
CKPT=${2:?encoder checkpoint}
NGPU=${3:-1}
TASKS=${TASKS:-"stone_sepsis urosepsis_ursl difficult_ureter spontaneous_passage"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

for task in ${TASKS}; do
  for seed in ${SEEDS}; do
    torchrun --standalone --nproc_per_node="${NGPU}" finetune.py \
      --config "configs/finetune/tasks/${task}.yaml" \
      --opts seed="${seed}" model.pretrained="${CKPT}" folder="outputs/finetune/${task}/${NAME}"
  done
done
