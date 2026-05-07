#!/bin/bash
#SBATCH --job-name=devbench-compute-distributions
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-compute-distributions.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-compute-distributions.err
# Stage 4: Compute SigLIP2 score distributions for lexical image-text pairs.
# Usage: bash run_compute_distributions.sh --data-dir data/coco_TIMESTAMP [--tasks nouns adjectives --styles realistic]
set -euo pipefail

python -m benchmark_creation.pipeline.filtering.compute_distributions "$@"
