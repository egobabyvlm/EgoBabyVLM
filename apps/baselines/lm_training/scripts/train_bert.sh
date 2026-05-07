#!/bin/bash -l
#SBATCH --job-name=bert-mlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}/bert_mlm_%j.log
#SBATCH --error=${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}/bert_mlm_%j.err

# Train BERT from scratch with masked-language modeling on a plain-text corpus.
#
# Submit with sbatch (the #SBATCH directives are honoured only by sbatch);
# pick QoS / account / partition on the command line:
#   sbatch --qos=<your_qos> --account=<your_account> \
#       apps/baselines/lm_training/scripts/train_bert.sh
#
# Required env vars:
#   TRAIN_FILE              Path to training corpus (one utterance per line)
#   VAL_FILE                Path to validation corpus (one utterance per line)
#   TOKENIZER_FOLDER        Path to the BERT tokenizer (see train_bert_tokenizer.py)
#   CONFIG_FOLDER           Path to the BERT config (see create_bert_config.py)
#
# Optional env vars (sensible defaults below):
#   MODEL_DIR               Output checkpoint dir
#                           (default: ${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}/bert_mlm)
#   LR                      Learning rate (default: 1e-4)
#   NUM_TRAIN_EPOCHS        Epochs (default: 30)
#   PER_GPU_BATCH_SIZE      Per-GPU batch size (default: 128)
#   MLM_PROBABILITY         MLM mask probability (default: 0.15)
#   SEED                    Random seed (default: 42)
#   EGOBABYVLM_CKPT_DIR     Output base (default: ./egobabyvlm_checkpoints)
#   EGOBABYVLM_LOG_DIR      SLURM %j log dir (default: ./egobabyvlm_logs)

set -euo pipefail

TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to a plain-text training corpus}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to a plain-text validation corpus}"
TOKENIZER_FOLDER="${TOKENIZER_FOLDER:?Set TOKENIZER_FOLDER to a trained BERT tokenizer (see train_bert_tokenizer.py)}"
CONFIG_FOLDER="${CONFIG_FOLDER:?Set CONFIG_FOLDER to a BERT config (see create_bert_config.py)}"

CKPT_BASE="${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}"
LOG_DIR="${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}"
mkdir -p "$LOG_DIR"

MODEL_DIR="${MODEL_DIR:-${CKPT_BASE}/bert_mlm}"
LR="${LR:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-30}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"
MLM_PROBABILITY="${MLM_PROBABILITY:-0.15}"
SEED="${SEED:-42}"

mkdir -p "$MODEL_DIR"

echo "=============================================="
echo "BERT MLM training"
echo "  Train:           ${TRAIN_FILE}"
echo "  Val:             ${VAL_FILE}"
echo "  Tokenizer:       ${TOKENIZER_FOLDER}"
echo "  Config:          ${CONFIG_FOLDER}"
echo "  Output:          ${MODEL_DIR}"
echo "  LR:              ${LR}"
echo "  Epochs:          ${NUM_TRAIN_EPOCHS}"
echo "  Per-GPU batch:   ${PER_GPU_BATCH_SIZE}"
echo "  MLM probability: ${MLM_PROBABILITY}"
echo "  Seed:            ${SEED}"
echo "=============================================="

python -m apps.baselines.lm_training.train.train_bert \
    --model_type bert \
    --config_name "$CONFIG_FOLDER" \
    --tokenizer_name "$TOKENIZER_FOLDER" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --line_by_line True \
    --per_device_train_batch_size "$PER_GPU_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_GPU_BATCH_SIZE" \
    --do_train \
    --do_eval \
    --output_dir "$MODEL_DIR" \
    --seed "$SEED" \
    --learning_rate "$LR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --mlm_probability "$MLM_PROBABILITY" \
    --load_best_model_at_end \
    --eval_strategy 'epoch' \
    --save_strategy 'epoch' \
    --save_total_limit 2 \
    --logging_steps 10 \
    --overwrite_cache \
    --report_to wandb

echo "BERT MLM training complete; checkpoint saved to ${MODEL_DIR}"
