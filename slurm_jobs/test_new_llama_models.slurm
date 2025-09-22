#!/bin/bash
#SBATCH --job-name=test_new_llama
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm_jobs/logs/test_new_llama_%j.out
#SBATCH --error=slurm_jobs/logs/test_new_llama_%j.err

# Test New Llama Models: 3.1-8B and 3.3-70B
# Tests both the faster 8B model and the latest 3.3 70B model

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
echo "NEW LLAMA MODELS TEST - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Test new Llama models (when gating approval received)
new_models=("llama31_8b" "llama33_70b")

echo "Testing new Llama models with general purpose datasets..."

for model in "${new_models[@]}"; do
    echo ""
    echo "=========================================="
    echo "🧪 Testing Model: $model"
    echo "Time: $(date)"
    echo "GPU Memory Before:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    echo "=========================================="
    
    # Test with small sample to validate
    echo "🧪 Running evaluation for $model..."
    crun -p ~/envs/llm_env python category_evaluation.py --model $model --samples 3
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model evaluation completed successfully"
        echo "✅ New Llama model integration working"
    else
        echo "❌ $model evaluation failed with exit code: $exit_code"
        echo "❌ Check model configuration or access permissions"
    fi
    
    echo "GPU Memory After:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    echo ""
done

echo "==========================================="
echo "NEW LLAMA MODELS TEST COMPLETE"
echo "Finished at: $(date)"
echo "==========================================="