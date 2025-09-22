#!/bin/bash
#SBATCH --job-name=test_accessible
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm_jobs/logs/test_accessible_%j.out
#SBATCH --error=slurm_jobs/logs/test_accessible_%j.err

# Test Accessible Models Only - Validate Authentication & Pipeline Fixes
# Tests the 3 models we can access to verify our fixes work

# Load environment
module load python3/2025.1-py312

# Setup HuggingFace authentication for gated models  
export HF_TOKEN="$(cat ~/.huggingface/token 2>/dev/null || echo '')"
if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    echo "HuggingFace token loaded for gated model access"
else
    echo "Warning: No HuggingFace token found - gated models may fail"
fi

echo "========================================="
echo "ACCESSIBLE MODELS TEST - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Test models that don't require gating approval
accessible_models=("starcoder2_15b" "internlm2_20b" "deepseek_r1_distill_llama_70b")

echo "Testing accessible models to validate authentication and preprocessing fixes..."

for model in "${accessible_models[@]}"; do
    echo ""
    echo "=========================================="
    echo "🧪 Testing Model: $model"
    echo "Time: $(date)"
    echo "GPU Memory Before:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    echo "=========================================="
    
    # Test with small sample to validate fixes
    echo "🧪 Running evaluation for $model..."
    crun -p ~/envs/llm_env python category_evaluation.py --model $model --samples 2
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model evaluation completed successfully"
        echo "✅ Authentication and preprocessing fixes working"
    else
        echo "❌ $model evaluation failed with exit code: $exit_code"
        echo "❌ Need to investigate further issues"
    fi
    
    echo "GPU Memory After:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    echo ""
done

echo "==========================================="
echo "ACCESSIBLE MODELS TEST COMPLETE"
echo "Finished at: $(date)"
echo "==========================================="