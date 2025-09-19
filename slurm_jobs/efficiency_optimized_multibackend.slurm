#!/bin/bash
#SBATCH --job-name=efficiency_eval
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --time=02:30:00
#SBATCH --output=slurm_jobs/logs/efficiency_optimized_%j.out
#SBATCH --error=slurm_jobs/logs/efficiency_optimized_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "EFFICIENCY OPTIMIZED EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Efficiency Optimized evaluation..."
echo "Results will be saved to: results/efficiency_optimized_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support (supports MobileVLM, MobileLLaMA, etc.)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category efficiency_optimized \
    --samples 6 \
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