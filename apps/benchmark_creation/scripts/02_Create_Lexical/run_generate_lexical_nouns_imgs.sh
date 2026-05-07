#!/bin/bash
#SBATCH --job-name=devbench-gen-noun-imgs
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-gen-noun-imgs.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-gen-noun-imgs.err
# Stage 2: Generate images for lexical noun task.
# Usage: bash run_generate_lexical_nouns_imgs.sh --data-dir data/coco_TIMESTAMP --styles realistic cartoon [--num-gpus 4]
set -euo pipefail

python -m benchmark_creation.pipeline.lexical.generate_noun_images "$@"
