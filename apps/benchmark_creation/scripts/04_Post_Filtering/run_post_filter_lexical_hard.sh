#!/bin/bash
#SBATCH --job-name=devbench-filter-lexical-hard
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-filter-lexical-hard.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-filter-lexical-hard.err
# Stage 4: Hard post-filter for lexical tasks (stricter SigLIP2 threshold).
# Usage: bash run_post_filter_lexical_hard.sh --data-dir data/coco_TIMESTAMP [--write-filtered]
set -euo pipefail

python -m benchmark_creation.pipeline.filtering.post_filter_lexical_hard "$@"
