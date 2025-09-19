#!/bin/bash
#SBATCH --job-name=safety_eval
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_jobs/logs/safety_alignment_%j.out
#SBATCH --error=slurm_jobs/logs/safety_alignment_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "SAFETY ALIGNMENT EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Safety Alignment evaluation..."
echo "Results will be saved to: results/safety_alignment_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support (supports RLHF models, safety-tuned models)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category safety_alignment \
    --samples 6 \
    --preset balanced

exit_code=$?

echo "========================================="
echo "SAFETY ALIGNMENT EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/safety_alignment_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: results/safety_alignment_*"
else
    echo "Evaluation failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code