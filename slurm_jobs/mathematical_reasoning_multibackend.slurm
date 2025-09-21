#!/bin/bash
#SBATCH --job-name=math_comprehensive
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm_jobs/logs/mathematical_reasoning_%j.out
#SBATCH --error=slurm_jobs/logs/mathematical_reasoning_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "MATHEMATICAL COMPREHENSIVE EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Mathematical Reasoning comprehensive evaluation..."
echo "Using: ALL available models in mathematical_reasoning category"
echo "Samples: 10 per dataset for thorough evaluation"

# Run comprehensive evaluation with all models and datasets
crun -p ~/envs/llm_env python category_evaluation.py \
    --category mathematical_reasoning \
    --samples 10 \
    --preset balanced

exit_code=$?

echo "========================================="
echo "MATHEMATICAL REASONING EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/mathematical_reasoning_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: results/mathematical_reasoning_*"
else
    echo "Evaluation failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code