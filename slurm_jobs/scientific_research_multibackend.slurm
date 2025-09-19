#!/bin/bash
#SBATCH --job-name=scientific_eval
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --time=03:30:00
#SBATCH --output=slurm_jobs/logs/scientific_research_%j.out
#SBATCH --error=slurm_jobs/logs/scientific_research_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "SCIENTIFIC RESEARCH EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Scientific Research evaluation..."
echo "Results will be saved to: results/scientific_research_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support (supports Galactica, SciBERT, BioBERT)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category scientific_research \
    --samples 6 \
    --preset balanced

exit_code=$?

echo "========================================="
echo "SCIENTIFIC RESEARCH EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/scientific_research_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: results/scientific_research_*"
else
    echo "Evaluation failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code