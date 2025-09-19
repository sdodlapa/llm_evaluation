#!/bin/bash
#SBATCH --job-name=biomedical_eval
#SBATCH --partition=h100dualflex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=slurm_jobs/logs/biomedical_specialists_%j.out
#SBATCH --error=slurm_jobs/logs/biomedical_specialists_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "BIOMEDICAL SPECIALISTS EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Biomedical Specialists evaluation..."
echo "Results will be saved to: results/biomedical_specialists_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support (includes BioGPT, PubMedBERT, etc.)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category biomedical_specialists \
    --samples 8 \
    --preset balanced \
    --exclude-models clinical_camel_70b

exit_code=$?

echo "========================================="
echo "BIOMEDICAL SPECIALISTS EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/biomedical_specialists_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: results/biomedical_specialists_*"
else
    echo "Evaluation failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code