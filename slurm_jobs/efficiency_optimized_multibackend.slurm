#!/bin/bash
#SBATCH --job-name=efficiency_comprehensive
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_jobs/logs/efficiency_optimized_%j.out
#SBATCH --error=slurm_jobs/logs/efficiency_optimized_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "EFFICIENCY COMPREHENSIVE EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Efficiency Optimized comprehensive evaluation..."
echo "Using: ALL available models in efficiency_optimized category"
echo "Samples: 10 per dataset for thorough evaluation"

# Run comprehensive evaluation with all models and datasets
crun -p ~/envs/llm_env python category_evaluation.py \
    --category efficiency_optimized \
    --samples 10 \
    --preset balanced

exit_code=$?

echo "========================================="
echo "EFFICIENCY OPTIMIZED EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/efficiency_optimized_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: results/efficiency_optimized_*"
else
    echo "Evaluation failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code