#!/bin/bash
#SBATCH --job-name=moe_evaluation
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4  # 4 H100 GPUs for Mixtral 8x7B
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_jobs/logs/mixture_of_experts_%j.out
#SBATCH --error=slurm_jobs/logs/mixture_of_experts_%j.err

# Mixture of Experts Evaluation - Mixtral 8x7B
# This job evaluates the Mixtral 8x7B model using distributed engine
# with tensor parallelism across 4 H100 GPUs

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "MIXTURE OF EXPERTS EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Mixture of Experts evaluation..."
echo "Model: mixtral_8x7b (93GB, 46.7B parameters)"
echo "Engine: Distributed (automatically selected due to size ≥15GB)"
echo "Strategy: Tensor Parallelism across 4 GPUs"
echo "Samples: 10 per dataset for comprehensive evaluation"

# Set distributed environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

# Run comprehensive evaluation with balanced preset
crun -p ~/envs/llm_env python category_evaluation.py \
    --category mixture_of_experts \
    --samples 10 \
    --preset balanced

exit_code=$?

echo "========================================="
echo "MIXTURE OF EXPERTS EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/mixture_of_experts_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
    echo "📊 Model tested: mixtral_8x7b (MoE architecture)"
    echo "🔍 Check results in: results/mixture_of_experts_*"
    echo "🎯 Datasets evaluated: mmlu, hellaswag, arc_challenge, humaneval"
else
    echo "❌ Evaluation failed with exit code: $exit_code"
    echo "📝 Check logs for details"
fi

exit $exit_code