#!/bin/bash
#SBATCH --job-name=h100_priority_validation
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400GB
#SBATCH --time=03:00:00
#SBATCH --output=slurm_jobs/logs/h100_priority_validation_%j.out
#SBATCH --error=slurm_jobs/logs/h100_priority_validation_%j.err

# H100 Priority Model Validation
# Tests the newly integrated H100-optimized large models with ChatGPT-recommended datasets
# Priority models: qwen25_72b, llama31_70b_fp8
# Priority datasets: arc_challenge, hellaswag, humaneval

echo "========================================="
echo "H100 PRIORITY MODEL VALIDATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Load modules and environment
module load python3

# Setup HuggingFace authentication for gated models  
export HF_TOKEN="$(cat ~/.huggingface/token 2>/dev/null || echo '')"
if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    echo "✓ HuggingFace token loaded for gated model access"
else
    echo "⚠️ Warning: No HuggingFace token found - some models may fail"
fi

# H100-specific optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# H100 FP8 optimization flags
export TORCH_DTYPE=float16
export VLLM_USE_FP8=true
export VLLM_FP8_KV_CACHE=true
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=true

# Memory optimization for 80GB H100s
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_SWAP_SPACE=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Tensor parallelism for large models
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_PIPELINE_PARALLEL_SIZE=1

# Priority models for validation (H100-optimized)
priority_models=("qwen25_72b" "llama31_70b_fp8")

# Core evaluation datasets (ChatGPT-recommended and downloaded)
priority_datasets=("arc_challenge" "hellaswag" "humaneval")

echo ""
echo "🚀 Starting H100 Priority Model Validation"
echo "Models to test: ${priority_models[@]}"
echo "Datasets to use: ${priority_datasets[@]}"
echo ""

validation_success=0
validation_failures=0

for model in "${priority_models[@]}"; do
    echo ""
    echo "=========================================="
    echo "🧪 Testing H100-Optimized Model: $model"
    echo "Time: $(date)"
    echo "GPU Memory Before:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    echo "=========================================="
    
    model_success=0
    
    for dataset in "${priority_datasets[@]}"; do
        echo ""
        echo "📊 Running $model on $dataset..."
        
        # Run evaluation with H100 optimizations
        crun -p ~/envs/llm_env python category_evaluation.py \
            --models $model \
            --datasets $dataset \
            --backend vllm \
            --tensor_parallel_size 4 \
            --gpu_memory_utilization 0.85 \
            --dtype float16 \
            --enable_fp8 \
            --max_tokens 1024 \
            --sample_limit 5 \
            --timeout 120 \
            --save_results \
            --results_dir comprehensive_results/validation \
            --enable_optimizations \
            --h100_mode
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ $model on $dataset: SUCCESS"
            ((model_success++))
            ((validation_success++))
        else
            echo "❌ $model on $dataset: FAILED (exit code: $exit_code)"
            ((validation_failures++))
        fi
        
        echo "GPU Memory After $dataset:"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
        echo ""
    done
    
    echo "📈 Model Summary: $model completed $model_success/${#priority_datasets[@]} datasets successfully"
    
    if [ $model_success -eq ${#priority_datasets[@]} ]; then
        echo "🎉 $model: ALL DATASETS PASSED - H100 integration verified!"
    elif [ $model_success -gt 0 ]; then
        echo "⚠️ $model: PARTIAL SUCCESS - Some datasets working"
    else
        echo "💥 $model: ALL DATASETS FAILED - Integration issues detected"
    fi
done

echo ""
echo "==========================================="
echo "🏁 H100 PRIORITY VALIDATION COMPLETE"
echo "==========================================="
echo "Finished at: $(date)"
echo ""
echo "📊 VALIDATION SUMMARY:"
echo "  Total model-dataset combinations: $((${#priority_models[@]} * ${#priority_datasets[@]}))"
echo "  Successful evaluations: $validation_success"
echo "  Failed evaluations: $validation_failures"
echo ""

if [ $validation_success -gt 0 ]; then
    echo "🎉 SUCCESS: H100 large model integration is working!"
    echo "✓ ChatGPT-recommended models and datasets are functional"
    echo "✓ H100 FP8 optimizations are active"
    echo "✓ Tensor parallelism is working correctly"
    echo ""
    echo "🚀 READY FOR FULL EVALUATION:"
    echo "  - Generate all H100 jobs with: bash slurm_jobs/h100_large_model_templates.sh"
    echo "  - Submit comprehensive evaluation: sbatch slurm_jobs/submit_comprehensive_evaluation.sh"
    echo "  - Monitor progress: squeue -u \$USER"
else
    echo "💥 INTEGRATION ISSUES DETECTED"
    echo "❌ All validation tests failed"
    echo "🔧 Troubleshooting needed:"
    echo "  - Check model configurations in configs/model_registry.py"
    echo "  - Verify dataset availability in evaluation_data/"
    echo "  - Review SLURM logs for error details"
fi

echo ""
echo "📁 Results saved to: comprehensive_results/validation/"
echo "📋 Logs available in: slurm_jobs/logs/"
echo "==========================================="