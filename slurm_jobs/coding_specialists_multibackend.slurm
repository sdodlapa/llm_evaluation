#!/bin/bash
#SBATCH --job-name=coding_eval
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_jobs/logs/coding_specialists_%j.out
#SBATCH --error=slurm_jobs/logs/coding_specialists_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "CODING SPECIALISTS EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Coding Specialists evaluation..."
echo "Results will be saved to: results/coding_specialists_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support
crun -p ~/envs/llm_env python category_evaluation.py \
    --category coding_specialists \
    --samples 10 \
    --preset balanced

exit_code=$?

echo "========================================="
echo "CODING SPECIALISTS EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/coding_specialists_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: results/coding_specialists_*"
else
    echo "Evaluation failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code