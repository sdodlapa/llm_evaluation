#!/bin/bash
#SBATCH --job-name=qwen3_next_80b
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=logs/qwen3_next_80b_%j.out
#SBATCH --error=logs/qwen3_next_80b_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Test Qwen3-Next-80B specifically with 4 GPUs
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: qwen3_next_80b"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Environment Setup
module load python3/2025.1-py312

# Change to correct working directory
cd /home/sdodl001_odu_edu/llm_evaluation

# Verify Environment
echo "Python: $(crun -p ~/envs/llm_env python --version)"
echo "Working Directory: $(pwd)"
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

available_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $available_gpus"

if [ $available_gpus -lt 4 ]; then
    echo "❌ Insufficient GPUs ($available_gpus) for Qwen3-Next-80B (requires 4)"
    exit 1
fi

echo "=========================================="
echo "Testing Qwen3-Next-80B-A3B-Instruct (80B)"
echo "4 GPUs required for tensor parallelism"
echo "Dataset: DROP reasoning, 3 samples"
echo "=========================================="

# Test configuration first
echo "Step 1: Validate model configuration..."
crun -p ~/envs/llm_env python -c "
from configs.model_configs import MODEL_CONFIGS
model = MODEL_CONFIGS.get('qwen3_next_80b')
if model:
    print(f'✅ Model found: {model.model_name}')
    print(f'   HF ID: {model.huggingface_id}')
    print(f'   Size: {model.size_gb}GB')
    print(f'   Context: {model.context_window:,}')
    print(f'   Available GPUs: $available_gpus')
else:
    print('❌ Model not found')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Model configuration validation failed"
    exit 1
fi

echo ""
echo "Step 2: Running inference test..."

# Run actual test
crun -p ~/envs/llm_env python category_evaluation.py \
    --model qwen3_next_80b \
    --dataset drop \
    --samples 3 \
    --preset performance

exit_code=$?

echo ""
echo "=========================================="
echo "TEST RESULTS - Qwen3-Next-80B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ Qwen3-Next-80B: SUCCESS"
    echo "🎉 September 2025 flagship model working correctly!"
else
    echo "❌ Qwen3-Next-80B: FAILED (exit code: $exit_code)"
    echo "Check logs for detailed error information"
fi

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "Results saved to: category_evaluation_results/"
echo "=========================================="

exit $exit_code