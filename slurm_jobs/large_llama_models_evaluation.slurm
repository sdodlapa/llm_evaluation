#!/bin/bash
#SBATCH --job-name=large_llama_models
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4  # 4 H100 GPUs for 70B+ models
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_jobs/logs/large_llama_models_%j.out
#SBATCH --error=slurm_jobs/logs/large_llama_models_%j.err

# Large Llama Models Evaluation
# Evaluates Llama 3.1 70B and Llama 3.2 Vision 90B
# Uses distributed engine with tensor parallelism across 4 H100 GPUs

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "LARGE LLAMA MODELS EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Set distributed environment variables for large models
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

# Test Models Configuration
LARGE_LLAMA_MODELS=(
    "llama31_70b"         # 140GB - Llama 3.1 70B Instruct (general_purpose)
    "llama32_vision_90b"  # 180GB - Llama 3.2 Vision 90B (multimodal_processing)
)

echo "Testing ${#LARGE_LLAMA_MODELS[@]} large Llama models:"
echo "Models: ${LARGE_LLAMA_MODELS[*]}"
echo "All models use distributed engine (size ≥15GB)"
echo "Strategy: Tensor Parallelism across 4 GPUs"
echo "========================================="

# Test each large Llama model individually
for model in "${LARGE_LLAMA_MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing Model: $model"
    echo "Time: $(date)"
    echo "GPU Memory Before:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    echo "=========================================="
    
    # Run model-specific evaluation
    echo "Running evaluation for $model..."
    crun -p ~/envs/llm_env python category_evaluation.py \
        --model $model \
        --samples 5 \
        --preset performance
    
    model_exit_code=$?
    echo "Model $model exit code: $model_exit_code"
    
    if [ $model_exit_code -eq 0 ]; then
        echo "✅ $model evaluation completed successfully"
    else
        echo "❌ $model evaluation failed with exit code: $model_exit_code"
    fi
    
    echo "GPU Memory After:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    echo ""
    
    # Brief cooldown between large models
    sleep 60
done

echo "========================================="
echo "LARGE LLAMA MODELS EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "Results location: results/*llama*"
echo "========================================="

echo ""
echo "==========================================  "
echo "EVALUATION SUMMARY REPORT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total Models Tested: ${#LARGE_LLAMA_MODELS[@]}"
echo "Models: ${LARGE_LLAMA_MODELS[*]}"

echo ""
echo "Model Details:"
echo "  • llama31_70b: 140GB, General Purpose, 4 GPUs"
echo "  • llama32_vision_90b: 180GB, Multimodal Vision, 4 GPUs"

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="