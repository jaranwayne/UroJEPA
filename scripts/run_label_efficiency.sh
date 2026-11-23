#!/usr/bin/env bash
# Label-efficiency experiments (paper, Fig. 3): fine-tune on 10/25/50/75 % of the
# training patients (outcome-stratified, patient level) for the two sepsis outcomes,
# five seeds per fraction. Every model is evaluated on the full internal test split.
# The 100 % runs are those of scripts/run_finetune.sh.
#
# Usage: bash scripts/run_label_efficiency.sh <model_name> <encoder_checkpoint> [num_gpus]
#   bash scripts/run_label_efficiency.sh urojepa_dagger outputs/pretrain/urojepa_dagger_vitb16/latest.pth.tar
#   bash scripts/run_label_efficiency.sh vjepa2_1       checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
set -euo pipefail

NAME=${1:?model name}
CKPT=${2:?encoder checkpoint}
NGPU=${3:-1}
TASKS=${TASKS:-"stone_sepsis urosepsis_ursl"}
FRACTIONS=${FRACTIONS:-"0.1 0.25 0.5 0.75"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

for task in ${TASKS}; do
  for frac in ${FRACTIONS}; do
    for seed in ${SEEDS}; do
      torchrun --standalone --nproc_per_node="${NGPU}" finetune.py \
        --config "configs/finetune/tasks/${task}.yaml" \
        --opts seed="${seed}" data.train_fraction="${frac}" model.pretrained="${CKPT}" \
               folder="outputs/finetune/${task}/${NAME}" data.external="{}"
    done
  done
done
