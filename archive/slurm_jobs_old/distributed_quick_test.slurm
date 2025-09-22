#!/bin/bash
#SBATCH --job-name=distributed_quick_test
#SBATCH --partition=h100dualflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/distributed_quick_test_%j.out
#SBATCH --error=logs/distributed_quick_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Quick Distributed Engine Test - 5 Samples Only
# This job quickly verifies distributed engine functionality with minimal samples

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: distributed_quick_test"
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

# Set Distributed Environment
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

echo "=========================================="
echo "Starting Quick Distributed Engine Test"
echo "Testing: qwen3_coder_30b with 5 samples only"
echo "Dataset: drop"
echo "=========================================="

# Create results directory
mkdir -p /home/sdodl001_odu_edu/llm_evaluation/distributed_test_results/quick_test_$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/home/sdodl001_odu_edu/llm_evaluation/distributed_test_results/quick_test_$(date +%Y%m%d_%H%M%S)"

# Run distributed engine test with large model (should trigger distributed processing automatically)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category coding_specialists \
    --models qwen3_coder_30b \
    --dataset drop \
    --samples 5 \
    --preset performance

exit_code=$?

echo "=========================================="
echo "Quick Distributed Engine Test Completed"
echo "End Time: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

# Summary Report
echo ""
echo "=========================================="
echo "QUICK TEST SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test Type: Quick validation with 5 samples"
echo "Model: qwen3_coder_30b"
echo "Dataset: drop"
echo "Engine: distributed (2-GPU)"

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo "=========================================="
echo "Quick test completed at: $(date)"
echo "=========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ Distributed engine test completed successfully"
    echo "Check results in: category_evaluation_results/"
else
    echo "❌ Distributed engine test failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code