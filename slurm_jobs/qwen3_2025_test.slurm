#!/bin/bash
#SBATCH --job-name=qwen3_2025_test
#SBATCH --partition=h100dualflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen3_2025_test_%j.out
#SBATCH --error=logs/qwen3_2025_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Test September 2025 Qwen Models - Quick Validation
# Tests the 3 new models: qwen3_4b_2507, qwen3_30b_2507, qwen3_next_80b

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: qwen3_2025_test"
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

echo "=========================================="
echo "Testing September 2025 Qwen Models"
echo "Testing 3 models with 3 samples each"
echo "Models: qwen3_4b_2507, qwen3_30b_2507, qwen3_next_80b"
echo "=========================================="

# Run model configuration test first
echo "Step 1: Testing model configurations..."
crun -p ~/envs/llm_env python test_new_qwen_models.py

if [ $? -ne 0 ]; then
    echo "❌ Model configuration test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Testing Qwen3-4B-2507 (1 GPU)"
echo "=========================================="

crun -p ~/envs/llm_env python category_evaluation.py \
    --model qwen3_4b_2507 \
    --dataset drop \
    --samples 3 \
    --preset performance

exit_code_4b=$?

echo ""
echo "=========================================="
echo "Step 3: Testing Qwen3-30B-2507 (2 GPUs)"  
echo "=========================================="

crun -p ~/envs/llm_env python category_evaluation.py \
    --model qwen3_30b_2507 \
    --dataset drop \
    --samples 3 \
    --preset performance

exit_code_30b=$?

echo ""
echo "=========================================="
echo "Step 4: Testing Qwen3-Next-80B (4 GPUs)"
echo "Note: May need additional GPU allocation"
echo "=========================================="

# For the 80B model, we may need to check if we have enough GPUs
available_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $available_gpus"

if [ $available_gpus -ge 4 ]; then
    echo "Sufficient GPUs available for Qwen3-Next-80B test"
    
    crun -p ~/envs/llm_env python category_evaluation.py \
        --model qwen3_next_80b \
        --dataset drop \
        --samples 2 \
        --preset performance
    
    exit_code_80b=$?
else
    echo "⚠️  Insufficient GPUs ($available_gpus) for Qwen3-Next-80B (requires 4)"
    echo "Skipping 80B model test"
    exit_code_80b=0  # Skip without error
fi

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo ""

# Report results
if [ $exit_code_4b -eq 0 ]; then
    echo "✅ Qwen3-4B-2507: SUCCESS"
else
    echo "❌ Qwen3-4B-2507: FAILED (exit code: $exit_code_4b)"
fi

if [ $exit_code_30b -eq 0 ]; then
    echo "✅ Qwen3-30B-2507: SUCCESS"
else
    echo "❌ Qwen3-30B-2507: FAILED (exit code: $exit_code_30b)"
fi

if [ $available_gpus -ge 4 ]; then
    if [ $exit_code_80b -eq 0 ]; then
        echo "✅ Qwen3-Next-80B: SUCCESS"
    else
        echo "❌ Qwen3-Next-80B: FAILED (exit code: $exit_code_80b)"
    fi
else
    echo "⚠️  Qwen3-Next-80B: SKIPPED (insufficient GPUs)"
fi

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo "=========================================="
echo "September 2025 Qwen models test completed at: $(date)"
echo "Check results in: category_evaluation_results/"
echo "=========================================="

# Calculate overall exit code
overall_exit_code=0
if [ $exit_code_4b -ne 0 ] || [ $exit_code_30b -ne 0 ]; then
    overall_exit_code=1
fi

if [ $available_gpus -ge 4 ] && [ $exit_code_80b -ne 0 ]; then
    overall_exit_code=1
fi

if [ $overall_exit_code -eq 0 ]; then
    echo "🎉 All September 2025 Qwen models tested successfully!"
else
    echo "⚠️  Some models had issues - check logs for details"
fi

exit $overall_exit_code