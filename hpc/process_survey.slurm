#!/bin/bash
# Geotag + classify one survey on an HPC cluster (SLURM). Submit from the repo folder:
#   mkdir -p logs
#   sbatch hpc/process_survey.slurm /data/surveys/13MAY2025 /data/surveys/13MAY2025_processed
#
# The direct pipeline needs no internet, so it runs on compute nodes without outbound access.
# (The natural-language agent needs internet for Gemini / OpenRouter: run it on a login or
#  interactive node, with its keys in ~/.config/opera/api_keys.txt, see api_keys.example.txt.)
#
# Adjust the partition / GPU lines to your cluster.
#SBATCH --job-name=opera-survey
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
VIDEOS=${1:?usage: sbatch hpc/process_survey.slurm <videos_dir> <output_dir> [extra options]}
OUTPUT=${2:?usage: sbatch hpc/process_survey.slurm <videos_dir> <output_dir> [extra options]}
shift 2

# module load miniforge     # or however conda is provided on your cluster
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opera

# Settings such as OPERA_YOLO_MODEL can live in the key file; --keys-file is optional when the file is at
# ./api_keys.txt or ~/.config/opera/api_keys.txt
python -m opera_agent process "$VIDEOS" "$OUTPUT" "$@"
