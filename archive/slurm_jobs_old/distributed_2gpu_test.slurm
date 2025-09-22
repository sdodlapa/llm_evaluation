#!/bin/bash
#SBATCH --job-name=distributed_2gpu_test
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/distributed_2gpu_test_%j.out
#SBATCH --error=logs/distributed_2gpu_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Distributed Engine Test - 2 GPUs with Tensor Parallelism
# This job tests distributed engine functionality with 2-GPU tensor parallelism

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: distributed_2gpu_test"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Environment Setup
module load python3/2025.1-py312

# Verify Environment
echo "Python: $(which python)"
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Set Distributed Environment
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

echo "=========================================="
echo "Starting 2-GPU Distributed Engine Test"
echo "Testing: qwen3_coder_30b with tensor parallelism"
echo "Dataset: drop"
echo "Samples: 5 (quick validation)"
echo "=========================================="

# Create results directory
mkdir -p distributed_test_results/2gpu_test_$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="distributed_test_results/2gpu_test_$(date +%Y%m%d_%H%M%S)"

# Run distributed engine test with 2-GPU tensor parallelism
crun -p ~/envs/llm_env python category_evaluation.py \
    --category coding_specialists \
    --models qwen3_coder_30b \
    --datasets drop \
    --samples 5 \
    --preset performance \
    --engine distributed \
    --tensor-parallel-size 2

exit_code=$?

echo "=========================================="
echo "2-GPU Distributed Engine Test Completed"
echo "End Time: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

# Summary Report
echo ""
echo "=========================================="
echo "2-GPU TEST SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test Type: 2-GPU tensor parallelism validation"
echo "Model: qwen3_coder_30b"
echo "Dataset: drop"
echo "Samples: 5"
echo "Engine: distributed (2-GPU tensor parallelism)"
echo "Tensor Parallel Size: 2"

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo "=========================================="
echo "2-GPU test completed at: $(date)"
echo "=========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ 2-GPU distributed engine test completed successfully"
    echo "Check results in: category_evaluation_results/"
else
    echo "❌ 2-GPU distributed engine test failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code