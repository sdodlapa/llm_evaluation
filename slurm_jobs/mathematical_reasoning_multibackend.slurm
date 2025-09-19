#!/bin/bash
#SBATCH --job-name=mathematical_eval
#SBATCH --partition=h100dualflex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_jobs/logs/mathematical_reasoning_%j.out
#SBATCH --error=slurm_jobs/logs/mathematical_reasoning_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "MATHEMATICAL REASONING EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Mathematical Reasoning evaluation..."
echo "Results will be saved to: results/mathematical_reasoning_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support (supports MetaMath, WizardMath, DeepSeek-Math)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category mathematical_reasoning \
    --samples 6 \
    --preset balanced \
    --exclude-models wizardmath_70b metamath_70b

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