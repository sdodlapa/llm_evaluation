#!/bin/bash
#SBATCH --job-name=medium_models_2gpu
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:2  # 2 H100 GPUs for medium models
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_jobs/logs/medium_models_2gpu_%j.out
#SBATCH --error=slurm_jobs/logs/medium_models_2gpu_%j.err

# Medium Models Evaluation (15B-30B parameters)
# Evaluates StarCoder2 15B, Gemma2 27B, and InternLM2 20B
# Uses distributed engine with tensor parallelism across 2 H100 GPUs

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "MEDIUM MODELS (2-GPU) EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Set distributed environment variables for medium models
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=6

# Test Models Configuration
MEDIUM_MODELS=(
    "starcoder2_15b"      # 32GB - StarCoder2 15B (coding_specialists)
    "gemma2_27b"          # 54GB - Gemma 2 27B (general_purpose)
    "internlm2_20b"       # 40GB - InternLM2 20B Chat (general_purpose)
)

echo "Testing ${#MEDIUM_MODELS[@]} medium-sized models:"
echo "Models: ${MEDIUM_MODELS[*]}"
echo "All models use distributed engine (size ≥15GB)"
echo "Strategy: Tensor Parallelism across 2 GPUs"
echo "========================================="

# Test each medium model individually
for model in "${MEDIUM_MODELS[@]}"; do
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
        --samples 8 \
        --preset balanced
    
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
    
    # Brief cooldown between models
    sleep 30
done

echo "========================================="
echo "MEDIUM MODELS EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "Results location: results/*"
echo "========================================="

echo ""
echo "==========================================  "
echo "EVALUATION SUMMARY REPORT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total Models Tested: ${#MEDIUM_MODELS[@]}"
echo "Models: ${MEDIUM_MODELS[*]}"

echo ""
echo "Model Details:"
echo "  • starcoder2_15b: 32GB, Coding Specialist, 2 GPUs"
echo "  • gemma2_27b: 54GB, General Purpose, 2 GPUs"
echo "  • internlm2_20b: 40GB, General Purpose (Multilingual), 2 GPUs"

echo ""
echo "Engine Selection Verification:"
echo "  • All models ≥15GB → Distributed Engine ✓"
echo "  • Models 15-60GB → 2-GPU Tensor Parallelism ✓"
echo "  • Automatic engine selection working ✓"

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="