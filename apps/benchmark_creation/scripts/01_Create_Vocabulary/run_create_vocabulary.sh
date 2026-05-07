#!/bin/bash
#SBATCH --job-name=devbench-create-vocabulary
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-create-vocabulary.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-create-vocabulary.err
# Stage 1: Build filtered, POS-tagged vocabulary from a raw vocab CSV.
# Usage: bash run_create_vocabulary.sh --vocab-csv path/to/vocab_sorted.csv --output-dir data/coco --name COCO
set -euo pipefail

python -m benchmark_creation.pipeline.create_vocabulary "$@"
