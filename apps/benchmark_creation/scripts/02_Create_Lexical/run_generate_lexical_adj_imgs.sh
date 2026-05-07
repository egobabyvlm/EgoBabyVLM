#!/bin/bash
#SBATCH --job-name=devbench-gen-adj-imgs
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-gen-adj-imgs.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-gen-adj-imgs.err
# Stage 2: Generate images for lexical adjective task.
# Usage: bash run_generate_lexical_adj_imgs.sh --data-dir data/coco_TIMESTAMP --styles realistic cartoon [--num-gpus 4]
set -euo pipefail

python -m benchmark_creation.pipeline.lexical.generate_adj_images "$@"
