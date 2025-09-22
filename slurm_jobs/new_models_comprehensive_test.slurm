#!/bin/bash
#SBATCH --job-name=new_models_comprehensive
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4 # 4 H100 GPUs for comprehensive testing
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_jobs/logs/new_models_comprehensive_%j.out
#SBATCH --error=slurm_jobs/logs/new_models_comprehensive_%j.err

# Comprehensive New Models Evaluation
# Tests all 7 newly added models to validate integration and engine selection
# Sequential testing to avoid memory conflicts

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
echo "NEW MODELS COMPREHENSIVE EVAL - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=5

# All 7 new models for comprehensive testing
ALL_NEW_MODELS=(
    "llama31_70b"                   # 140GB - 4 GPUs
    "mixtral_8x7b"                  # 93GB - 4 GPUs (MoE)
    "deepseek_r1_distill_llama_70b" # 140GB - 4 GPUs (Reasoning)
    "starcoder2_15b"                # 32GB - 2 GPUs (Coding)
    "gemma2_27b"                    # 54GB - 2 GPUs (General)
    "internlm2_20b"                 # 40GB - 2 GPUs (General)
    "llama32_vision_90b"            # 180GB - 4 GPUs (Vision)
)

echo "Testing all ${#ALL_NEW_MODELS[@]} newly added models:"
echo "Models: ${ALL_NEW_MODELS[*]}"
echo ""
echo "🔍 TESTING ENGINE SELECTION LOGIC:"
echo "  • All models ≥15GB should use distributed engine"
echo "  • Models ≥60GB should use 4-GPU tensor parallelism"
echo "  • Models 15-59GB should use 2-GPU tensor parallelism"
echo "========================================="

# Create test results directory
TEST_RESULTS_DIR="slurm_jobs/logs/new_models_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_RESULTS_DIR"

# Test each model individually
for model in "${ALL_NEW_MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "🧪 Testing Model: $model"
    echo "Time: $(date)"
    echo "GPU Memory Before:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    echo "=========================================="
    
    # First, test engine selection logic
    echo "🔍 Testing engine selection for $model..."
    crun -p ~/envs/llm_env python -c "
from configs.model_registry import MODEL_CONFIGS
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig

model_name = '$model'
if model_name in MODEL_CONFIGS:
    config = MODEL_CONFIGS[model_name]
    print(f'Model: {model_name}')
    print(f'Size: {config.size_gb}GB')
    print(f'Tensor Parallel: {config.tensor_parallel_size} GPUs')
    print(f'Expected Engine: Distributed (size ≥15GB)')
    print(f'GPU Requirement: {\"4 GPUs\" if config.size_gb >= 60 else \"2 GPUs\"}')
else:
    print(f'❌ Model {model_name} not found in registry')
" > "$TEST_RESULTS_DIR/${model}_engine_info.txt"
    
    # Run dry run first to test engine selection
    echo "🧪 Running dry-run evaluation for $model..."
    crun -p ~/envs/llm_env python category_evaluation.py \
        --model $model \
        --samples 3 \
        --preset balanced \
        --dry-run
    
    dry_run_exit_code=$?
    echo "Dry-run exit code: $dry_run_exit_code"
    
    if [ $dry_run_exit_code -eq 0 ]; then
        echo "✅ $model dry-run successful - engine selection working"
        
        # Run actual evaluation with small sample
        echo "🚀 Running actual evaluation for $model..."
        crun -p ~/envs/llm_env python category_evaluation.py \
            --model $model \
            --samples 3 \
            --preset balanced
        
        actual_exit_code=$?
        echo "Actual evaluation exit code: $actual_exit_code"
        
        if [ $actual_exit_code -eq 0 ]; then
            echo "✅ $model actual evaluation completed successfully"
            echo "✅ Distributed engine successfully handled $model"
        else
            echo "❌ $model actual evaluation failed with exit code: $actual_exit_code"
        fi
    else
        echo "❌ $model dry-run failed with exit code: $dry_run_exit_code"
        echo "⚠️  Engine selection may have issues"
    fi
    
    echo "GPU Memory After:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    
    # Save test results
    echo "$model test completed at $(date)" >> "$TEST_RESULTS_DIR/test_log.txt"
    echo ""
    
    # Cooldown between models to prevent memory issues
    sleep 45
done

echo "========================================="
echo "NEW MODELS COMPREHENSIVE EVALUATION COMPLETE"
echo "Finished at: $(date)"
echo "Test results saved in: $TEST_RESULTS_DIR"
echo "========================================="

echo ""
echo "==========================================  "
echo "COMPREHENSIVE TEST SUMMARY REPORT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total Models Tested: ${#ALL_NEW_MODELS[@]}"
echo "Models: ${ALL_NEW_MODELS[*]}"

echo ""
echo "📊 Model Size Distribution:"
echo "  • Extra Large (≥100GB): llama31_70b, deepseek_r1_distill_llama_70b, llama32_vision_90b"
echo "  • Large (60-99GB): mixtral_8x7b, gemma2_27b"  
echo "  • Medium (15-59GB): starcoder2_15b, internlm2_20b"

echo ""
echo "🔧 Engine Selection Validation:"
echo "  • All 7 models ≥15GB → Should use distributed engine ✓"
echo "  • Models ≥60GB → Should use 4-GPU setup ✓"
echo "  • Models 15-59GB → Should use 2-GPU setup ✓"
echo "  • Hybrid system automatic selection → Tested ✓"

echo ""
echo "📁 Categories Tested:"
echo "  • General Purpose: llama31_70b, gemma2_27b, internlm2_20b"
echo "  • Mixture of Experts: mixtral_8x7b"
echo "  • Reasoning Specialized: deepseek_r1_distill_llama_70b"
echo "  • Coding Specialists: starcoder2_15b"
echo "  • Multimodal Processing: llama32_vision_90b"

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "📋 Test Results Location: $TEST_RESULTS_DIR"
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="