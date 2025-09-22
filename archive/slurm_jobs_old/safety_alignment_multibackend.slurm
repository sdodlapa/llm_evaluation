#!/bin/bash
#SBATCH --job-name=safety_comprehensive
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm_jobs/logs/safety_alignment_%j.out
#SBATCH --error=slurm_jobs/logs/safety_alignment_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "SAFETY ALIGNMENT COMPREHENSIVE EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Safety Alignment comprehensive evaluation..."
echo "Using: Individual model-dataset evaluations"
echo "Samples: 10 per dataset for thorough evaluation"

# Run individual evaluations for each model-dataset combination
# Models: qwen25_7b, biomistral_7b (excluding safety_bert for generative evaluation)
# Datasets: toxicity_detection, truthfulqa, hh_rlhf

echo "Evaluating qwen25_7b on safety datasets..."
crun -p ~/envs/llm_env python category_evaluation.py --model qwen25_7b --dataset toxicity_detection --samples 10 --preset balanced
crun -p ~/envs/llm_env python category_evaluation.py --model qwen25_7b --dataset truthfulqa --samples 10 --preset balanced
crun -p ~/envs/llm_env python category_evaluation.py --model qwen25_7b --dataset hh_rlhf --samples 10 --preset balanced

echo "Evaluating biomistral_7b on safety datasets..."
crun -p ~/envs/llm_env python category_evaluation.py --model biomistral_7b --dataset toxicity_detection --samples 10 --preset balanced
crun -p ~/envs/llm_env python category_evaluation.py --model biomistral_7b --dataset truthfulqa --samples 10 --preset balanced
crun -p ~/envs/llm_env python category_evaluation.py --model biomistral_7b --dataset hh_rlhf --samples 10 --preset balanced

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