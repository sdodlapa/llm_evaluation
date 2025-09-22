#!/bin/bash
#SBATCH --job-name=distributed_4gpu_quick_test
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --output=logs/distributed_4gpu_quick_test_%j.out
#SBATCH --error=logs/distributed_4gpu_quick_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Quick 4-GPU Distributed Engine Test - 5 Samples Only
# Tests tensor parallelism with 4 GPUs

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: distributed_4gpu_quick_test"
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

# Set 4-GPU Distributed Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

echo "=========================================="
echo "Starting 4-GPU Quick Distributed Test"
echo "Testing: wizardmath_70b with 5 samples only"
echo "Dataset: drop"
echo "Configuration: 4-GPU Tensor Parallelism"
echo "=========================================="

# Run 4-GPU distributed evaluation with large model (should trigger distributed processing automatically)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category mathematical_reasoning \
    --models wizardmath_70b \
    --dataset drop \
    --samples 5 \
    --preset performance

exit_code=$?
exit_code=$?

echo "=========================================="
echo "4-GPU Quick Distributed Test Completed"
echo "End Time: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

# Summary Report
echo ""
echo "==========================================  "
echo "4-GPU QUICK TEST SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test Type: 4-GPU tensor parallelism with 5 samples"
echo "Model: wizardmath_70b"
echo "Dataset: drop"
echo "Engine: distributed (4-GPU)"

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo "=========================================="
echo "4-GPU quick test completed at: $(date)"
echo "=========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ 4-GPU Distributed engine test completed successfully"
    echo "Check results in: category_evaluation_results/"
else
    echo "❌ 4-GPU Distributed engine test failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code