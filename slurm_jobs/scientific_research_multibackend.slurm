#!/bin/bash
#SBATCH --job-name=scientific_comprehensive
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_jobs/logs/scientific_research_%j.out
#SBATCH --error=slurm_jobs/logs/scientific_research_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "SCIENTIFIC RESEARCH COMPREHENSIVE EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Scientific Research comprehensive evaluation..."
echo "Using: ALL available models in scientific_research category"
echo "Samples: 10 per dataset for thorough evaluation"

# Run comprehensive evaluation with all models and datasets
crun -p ~/envs/llm_env python category_evaluation.py \
    --category scientific_research \
    --samples 10 \
    --preset balanced

exit_code=$?

echo "Starting Scientific Research evaluation..."
echo "Results will be saved to: results/scientific_research_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support (generative models only, excluding BERT classification models)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category scientific_research \
    --samples 8 \
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