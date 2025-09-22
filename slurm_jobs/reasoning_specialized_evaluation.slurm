#!/bin/bash
#SBATCH --job-name=reasoning_specialized
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4  # 4 H100 GPUs for DeepSeek-R1-Distill-Llama-70B
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_jobs/logs/reasoning_specialized_%j.out
#SBATCH --error=slurm_jobs/logs/reasoning_specialized_%j.err

# Reasoning Specialized Evaluation - DeepSeek-R1-Distill-Llama-70B
# This job evaluates reasoning-specialized models using distributed engine
# with tensor parallelism across 4 H100 GPUs

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "REASONING SPECIALIZED EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Reasoning Specialized evaluation..."
echo "Model: deepseek_r1_distill_llama_70b (140GB, 70B parameters)"
echo "Engine: Distributed (automatically selected due to size ≥15GB)"
echo "Strategy: Tensor Parallelism across 4 GPUs"
echo "Samples: 10 per dataset for comprehensive reasoning evaluation"

# Set distributed environment variables for large model
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

# Run comprehensive evaluation with performance preset for reasoning
crun -p ~/envs/llm_env python category_evaluation.py \
    --category reasoning_specialized \
    --samples 10 \
    --preset performance

exit_code=$?

echo "========================================="
echo "REASONING SPECIALIZED EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/reasoning_specialized_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
    echo "🧠 Model tested: deepseek_r1_distill_llama_70b (Reasoning-Distilled)"
    echo "🔍 Check results in: results/reasoning_specialized_*"
    echo "🎯 Datasets evaluated: gsm8k, enhanced_math_fixed, arc_challenge, mmlu, math_competition"
else
    echo "❌ Evaluation failed with exit code: $exit_code"
    echo "📝 Check logs for details"
fi

exit $exit_code