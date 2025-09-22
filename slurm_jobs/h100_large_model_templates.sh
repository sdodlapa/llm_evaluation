#!/bin/bash
# H100-Optimized SLURM Job Templates for Large Model Evaluation (70B+ models)
# Based on ChatGPT recommendations for H100 cluster deployment
# Optimized for 8×H100-80GB GPU configuration with tensor parallelism

# ================================
# TEMPLATE 1: H100 Large Model Single Job (70B-90B models)
# For models like Qwen2.5-72B, Llama-3.1-70B-FP8, XVERSE-65B
# ================================

create_h100_large_single_job() {
    local model_name=$1
    local dataset_name=$2
    local output_dir=$3
    
    cat > "h100_${model_name}_${dataset_name}.job" << EOF
#!/bin/bash
#SBATCH --job-name=h100_${model_name}_${dataset_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=${output_dir}/logs/h100_${model_name}_${dataset_name}_%j.out
#SBATCH --error=${output_dir}/logs/h100_${model_name}_${dataset_name}_%j.err

# H100-specific environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# H100 FP8 optimization flags (for FP8-capable models)
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

# Load environment
source ~/envs/llm_env/bin/activate
cd /home/sdodl001_odu_edu/llm_evaluation

# Run evaluation with H100 optimizations
python category_evaluation.py \\
    --models ${model_name} \\
    --datasets ${dataset_name} \\
    --backend vllm \\
    --tensor_parallel_size 4 \\
    --gpu_memory_utilization 0.85 \\
    --dtype float16 \\
    --enable_fp8 \\
    --max_tokens 2048 \\
    --sample_limit 30 \\
    --timeout 120 \\
    --save_results \\
    --results_dir ${output_dir}/results \\
    --enable_optimizations \\
    --h100_mode

echo "H100 large model evaluation completed for ${model_name} on ${dataset_name}"
EOF
}

# ================================
# TEMPLATE 2: H100 Mixture of Experts (MoE) Job Template
# For models like Mixtral-8x22B, DBRX-132B, DeepSeek-V3
# Optimized for MoE efficiency and expert utilization
# ================================

create_h100_moe_job() {
    local model_name=$1
    local dataset_name=$2
    local output_dir=$3
    
    cat > "h100_moe_${model_name}_${dataset_name}.job" << EOF
#!/bin/bash
#SBATCH --job-name=h100_moe_${model_name}_${dataset_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:6
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=${output_dir}/logs/h100_moe_${model_name}_${dataset_name}_%j.out
#SBATCH --error=${output_dir}/logs/h100_moe_${model_name}_${dataset_name}_%j.err

# H100 MoE-specific environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# MoE optimization flags
export VLLM_USE_FP8=true
export VLLM_FP8_KV_CACHE=true
export VLLM_MOE_EXPERT_PARALLELISM=true
export VLLM_MOE_MAX_ACTIVE_EXPERTS=2

# H100 memory management for MoE
export VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_SWAP_SPACE=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Tensor and expert parallelism
export VLLM_TENSOR_PARALLEL_SIZE=6
export VLLM_EXPERT_PARALLEL_SIZE=2

# Load environment
source ~/envs/llm_env/bin/activate
cd /home/sdodl001_odu_edu/llm_evaluation

# Run MoE evaluation with expert utilization tracking
python category_evaluation.py \\
    --models ${model_name} \\
    --datasets ${dataset_name} \\
    --backend vllm \\
    --tensor_parallel_size 6 \\
    --gpu_memory_utilization 0.80 \\
    --dtype float16 \\
    --enable_fp8 \\
    --max_tokens 2048 \\
    --sample_limit 25 \\
    --timeout 150 \\
    --save_results \\
    --results_dir ${output_dir}/results \\
    --enable_moe_tracking \\
    --track_expert_utilization \\
    --h100_mode

echo "H100 MoE evaluation completed for ${model_name} on ${dataset_name}"
EOF
}

# ================================
# TEMPLATE 3: H100 Ultra-Large Model Job (150B+ models)
# For models that require all 8 H100 GPUs
# ================================

create_h100_ultra_large_job() {
    local model_name=$1
    local dataset_name=$2
    local output_dir=$3
    
    cat > "h100_ultra_${model_name}_${dataset_name}.job" << EOF
#!/bin/bash
#SBATCH --job-name=h100_ultra_${model_name}_${dataset_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=${output_dir}/logs/h100_ultra_${model_name}_${dataset_name}_%j.out
#SBATCH --error=${output_dir}/logs/h100_ultra_${model_name}_${dataset_name}_%j.err

# Full H100 cluster environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0  # Enable P2P for 8-GPU setup

# Ultra-large model optimizations
export VLLM_USE_FP8=true
export VLLM_FP8_KV_CACHE=true
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=true
export VLLM_USE_FLASH_ATTENTION=true

# Conservative memory settings for maximum model size
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_SWAP_SPACE=16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 8-GPU tensor parallelism
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_PIPELINE_PARALLEL_SIZE=1

# Load environment
source ~/envs/llm_env/bin/activate
cd /home/sdodl001_odu_edu/llm_evaluation

# Run ultra-large model evaluation
python category_evaluation.py \\
    --models ${model_name} \\
    --datasets ${dataset_name} \\
    --backend vllm \\
    --tensor_parallel_size 8 \\
    --gpu_memory_utilization 0.75 \\
    --dtype float16 \\
    --enable_fp8 \\
    --max_tokens 2048 \\
    --sample_limit 20 \\
    --timeout 180 \\
    --save_results \\
    --results_dir ${output_dir}/results \\
    --enable_optimizations \\
    --h100_mode \\
    --ultra_large_mode

echo "H100 ultra-large model evaluation completed for ${model_name} on ${dataset_name}"
EOF
}

# ================================
# TEMPLATE 4: H100 Multimodal Large Model Job
# For large vision-language models requiring specialized processing
# ================================

create_h100_multimodal_job() {
    local model_name=$1
    local dataset_name=$2
    local output_dir=$3
    
    cat > "h100_multimodal_${model_name}_${dataset_name}.job" << EOF
#!/bin/bash
#SBATCH --job-name=h100_mm_${model_name}_${dataset_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=250G
#SBATCH --time=10:00:00
#SBATCH --output=${output_dir}/logs/h100_mm_${model_name}_${dataset_name}_%j.out
#SBATCH --error=${output_dir}/logs/h100_mm_${model_name}_${dataset_name}_%j.err

# H100 multimodal environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Multimodal-specific optimizations
export VLLM_USE_FP8=true
export VLLM_FP8_KV_CACHE=true
export VLLM_ENABLE_VISION_PROCESSING=true
export VLLM_VISION_MEMORY_FRACTION=0.3

# Memory settings for multimodal processing
export VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_SWAP_SPACE=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Tensor parallelism for multimodal
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_PIPELINE_PARALLEL_SIZE=1

# Load environment
source ~/envs/llm_env/bin/activate
cd /home/sdodl001_odu_edu/llm_evaluation

# Run multimodal evaluation
python category_evaluation.py \\
    --models ${model_name} \\
    --datasets ${dataset_name} \\
    --backend vllm \\
    --tensor_parallel_size 4 \\
    --gpu_memory_utilization 0.80 \\
    --dtype float16 \\
    --enable_fp8 \\
    --max_tokens 2048 \\
    --sample_limit 15 \\
    --timeout 200 \\
    --save_results \\
    --results_dir ${output_dir}/results \\
    --enable_multimodal_processing \\
    --enable_vision_features \\
    --h100_mode

echo "H100 multimodal evaluation completed for ${model_name} on ${dataset_name}"
EOF
}

# ================================
# BATCH JOB CREATION FUNCTIONS
# ================================

# Generate all H100-optimized jobs for the new model categories
generate_h100_evaluation_jobs() {
    local output_dir=${1:-"/home/sdodl001_odu_edu/llm_evaluation/comprehensive_results"}
    
    # Create logs directory
    mkdir -p ${output_dir}/logs
    mkdir -p ${output_dir}/results
    
    echo "Generating H100-optimized SLURM jobs..."
    
    # H100 Optimized Large Models (4-GPU jobs)
    local h100_large_models=("qwen25_72b" "llama31_70b_fp8" "xverse_65b")
    local h100_large_datasets=("mmlu" "bigbench_hard" "hellaswag" "arc_challenge")
    
    for model in "\${h100_large_models[@]}"; do
        for dataset in "\${h100_large_datasets[@]}"; do
            create_h100_large_single_job \$model \$dataset \$output_dir
        done
    done
    
    # MoE Models (6-GPU jobs)
    local moe_models=("mixtral_8x22b" "dbrx_instruct" "deepseek_v3")
    local moe_datasets=("mmlu" "humaneval" "gsm8k" "enterprise_tasks")
    
    for model in "\${moe_models[@]}"; do
        for dataset in "\${moe_datasets[@]}"; do
            create_h100_moe_job \$model \$dataset \$output_dir
        done
    done
    
    # Advanced Code Generation Models
    local code_models=("granite_34b_code")
    local code_datasets=("humaneval" "swe_bench" "code_contests" "livecode_bench")
    
    for model in "\${code_models[@]}"; do
        for dataset in "\${code_datasets[@]}"; do
            create_h100_large_single_job \$model \$dataset \$output_dir
        done
    done
    
    # Advanced Multimodal Models
    local mm_models=("internvl2_llama3_76b")
    local mm_datasets=("mmmu" "mathvista" "docvqa" "ai2d")
    
    for model in "\${mm_models[@]}"; do
        for dataset in "\${mm_datasets[@]}"; do
            create_h100_multimodal_job \$model \$dataset \$output_dir
        done
    done
    
    echo "H100-optimized SLURM jobs generated successfully!"
    echo "Job files created for:"
    echo "  - H100 Large Models: \${#h100_large_models[@]} models × \${#h100_large_datasets[@]} datasets"
    echo "  - MoE Models: \${#moe_models[@]} models × \${#moe_datasets[@]} datasets"
    echo "  - Code Generation: \${#code_models[@]} models × \${#code_datasets[@]} datasets"
    echo "  - Multimodal: \${#mm_models[@]} models × \${#mm_datasets[@]} datasets"
}

# ================================
# UTILITY FUNCTIONS
# ================================

# Submit priority H100 validation jobs
submit_priority_h100_validation() {
    echo "Submitting priority H100 validation jobs..."
    
    # Priority models for initial validation
    sbatch h100_qwen25_72b_mmlu.job
    sbatch h100_llama31_70b_fp8_hellaswag.job
    sbatch h100_moe_mixtral_8x22b_humaneval.job
    
    echo "Priority validation jobs submitted. Monitor with 'squeue -u \$USER'"
}

# Check H100 job status and resource utilization
check_h100_job_status() {
    echo "H100 Job Status:"
    squeue -u \$USER | grep h100
    
    echo -e "\nH100 GPU Utilization:"
    nvidia-smi | grep "H100"
    
    echo -e "\nRecent H100 job outputs:"
    ls -la /home/sdodl001_odu_edu/llm_evaluation/comprehensive_results/logs/ | tail -10
}

# Clean up completed H100 jobs
cleanup_h100_jobs() {
    local retention_days=${1:-7}
    echo "Cleaning up H100 job files older than \$retention_days days..."
    
    find . -name "h100_*.job" -mtime +\$retention_days -delete
    find /home/sdodl001_odu_edu/llm_evaluation/comprehensive_results/logs/ -name "h100_*" -mtime +\$retention_days -delete
    
    echo "Cleanup completed."
}

# ================================
# MAIN EXECUTION
# ================================

# Display usage information
show_usage() {
    echo "H100 Large Model Evaluation SLURM Templates"
    echo "Based on ChatGPT recommendations for H100 cluster optimization"
    echo ""
    echo "Usage:"
    echo "  source h100_large_model_templates.sh"
    echo "  generate_h100_evaluation_jobs [output_dir]"
    echo "  submit_priority_h100_validation"
    echo "  check_h100_job_status"
    echo "  cleanup_h100_jobs [retention_days]"
    echo ""
    echo "Job Templates Available:"
    echo "  - H100 Large Single (4 GPUs): 70B-90B models"
    echo "  - H100 MoE (6 GPUs): Mixture of Experts models"
    echo "  - H100 Ultra Large (8 GPUs): 150B+ models"
    echo "  - H100 Multimodal (4 GPUs): Large vision-language models"
    echo ""
    echo "Optimizations Included:"
    echo "  - FP8 quantization for H100 acceleration"
    echo "  - Tensor parallelism configurations"
    echo "  - Memory management for 80GB H100s"
    echo "  - NCCL optimization for multi-GPU communication"
    echo "  - MoE expert utilization tracking"
    echo ""
}

# Show usage by default
if [[ "\${BASH_SOURCE[0]}" == "\${0}" ]]; then
    show_usage
fi