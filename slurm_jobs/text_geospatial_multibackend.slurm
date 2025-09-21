#!/bin/bash
#SBATCH --job-name=geospatial_comprehensive
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm_jobs/logs/text_geospatial_%j.out
#SBATCH --error=slurm_jobs/logs/text_geospatial_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "GEOSPATIAL COMPREHENSIVE EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Text Geospatial comprehensive evaluation..."
echo "Using: ALL available models in text_geospatial category"
echo "Samples: 10 per dataset for thorough evaluation"

# Run comprehensive evaluation with all models and datasets
crun -p ~/envs/llm_env python category_evaluation.py \
    --category text_geospatial \
    --samples 10 \
    --preset balanced

exit_code=$?