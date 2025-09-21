#!/bin/bash
#SBATCH --job-name=distributed_hybrid_parallelism
#SBATCH --partition=h100octflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H100:8  # Request 8 H100 GPUs for hybrid tensor+pipeline parallelism
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=logs/distributed_hybrid_parallelism_%j.out
#SBATCH --error=logs/distributed_hybrid_parallelism_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Advanced Distributed Engine Testing - Hybrid Parallelism
# This job tests the distributed evaluation engine with the largest models
# using hybrid tensor+pipeline parallelism across 8 H100 GPUs

# Job Information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: distributed_hybrid_parallelism"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Environment Setup
module load python/3.12
source ~/envs/llm_env/bin/activate

# Verify Environment
echo "Python: $(which python)"
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Set Advanced Distributed Training Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=4

# Hybrid Parallelism Configuration
# Tensor Parallel Size: 4 (split layers across 4 GPUs)
# Pipeline Parallel Size: 2 (split model stages across 2 groups)
# Total GPUs: 4 * 2 = 8 GPUs
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=2

# Test Models for Hybrid Parallelism (largest available)
HYBRID_MODELS=(
    "wizardmath_70b"      # 70GB with AWQ - most demanding
    "metamath_70b"        # 70GB mathematical reasoning
    "clinical_camel_70b"  # 70GB biomedical specialist
)

# Comprehensive test datasets for validation
VALIDATION_DATASETS=("drop" "arc_challenge" "hellaswag" "piqa")

echo "=========================================="
echo "Starting Advanced Distributed Engine Testing"
echo "Hybrid Parallelism Configuration:"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "  Total GPUs: 8"
echo "Testing ${#HYBRID_MODELS[@]} largest models"
echo "Models: ${HYBRID_MODELS[*]}"
echo "Datasets: ${VALIDATION_DATASETS[*]}"
echo "=========================================="

# Create test results directory
mkdir -p distributed_test_results/hybrid_$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="distributed_test_results/hybrid_$(date +%Y%m%d_%H%M%S)"

# Performance monitoring function
monitor_performance() {
    local model_name=$1
    local log_file="$RESULTS_DIR/${model_name}_performance.log"
    
    echo "Starting performance monitoring for $model_name" > "$log_file"
    
    while true; do
        echo "$(date): GPU Status" >> "$log_file"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> "$log_file"
        echo "" >> "$log_file"
        sleep 30
    done &
    
    echo $!  # Return PID for cleanup
}

# Test each model with hybrid parallelism
for model in "${HYBRID_MODELS[@]}"; do
    echo ""
    echo "==========================================  "
    echo "Testing Model: $model"
    echo "Configuration: Hybrid Tensor+Pipeline Parallelism"
    echo "Time: $(date)"
    echo "GPU Memory Before:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    echo "=========================================="
    
    # Start performance monitoring
    MONITOR_PID=$(monitor_performance "$model")
    
    # Test multiple distribution strategies
    for strategy in "hybrid" "tensor_parallel" "pipeline_parallel"; do
        echo ""
        echo "Testing $model with strategy: $strategy"
        
        # Run hybrid distributed evaluation
        python -c "
import asyncio
import json
import time
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_hybrid_distributed():
    try:
        # Import after CUDA setup
        from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig
        from engines.distributed.multi_gpu_model_loader import DistributionStrategy
        from core_shared.interfaces.evaluation_interfaces import EvaluationRequest
        from configs.model_registry import MODEL_CONFIGS
        from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
        
        model_name = '$model'
        strategy = '$strategy'
        logger.info(f'Testing hybrid distributed engine: {model_name} with {strategy}')
        
        # Check if model exists in registry
        if model_name not in MODEL_CONFIGS:
            logger.error(f'Model {model_name} not found in MODEL_CONFIGS')
            return False
        
        model_config_base = MODEL_CONFIGS[model_name]
        
        # Create enhanced model config with distribution hints
        enhanced_config = EnhancedModelConfig(
            model_name=model_name,
            model_path=model_config_base.huggingface_id,
            parameters=model_config_base.size_gb * 1_000_000_000,  # Convert GB to parameters estimate
            quantization=model_config_base.quantization_method,
            precision='fp16',
            context_length=model_config_base.context_window,
            # Add distribution strategy hints
            distribution_strategy=strategy,
            tensor_parallel_size=$TENSOR_PARALLEL_SIZE if '$strategy' in ['hybrid', 'tensor_parallel'] else 1,
            pipeline_parallel_size=$PIPELINE_PARALLEL_SIZE if '$strategy' in ['hybrid', 'pipeline_parallel'] else 1
        )
        
        # Configure distributed engine for hybrid parallelism
        config = DistributedEngineConfig(
            max_concurrent_evaluations=1,
            enable_dynamic_scaling=True,
            enable_fault_tolerance=True,
            memory_optimization_level='aggressive',  # Aggressive for large models
            communication_backend='nccl',
            scheduling_strategy='load_balanced',
            performance_monitoring=True,
            automatic_model_offloading=True,
            cross_gpu_memory_sharing=True,
            pipeline_optimization=True
        )
        
        # Initialize distributed engine
        logger.info(f'Initializing distributed engine with {strategy} strategy...')
        engine = DistributedEvaluationEngine(config)
        
        # Test engine capabilities
        if hasattr(engine, 'get_capabilities'):
            capabilities = engine.get_capabilities()
            logger.info(f'Engine capabilities: max_model_size={capabilities.max_model_size_gb}GB')
            logger.info(f'Tensor parallel sizes: {capabilities.tensor_parallel_sizes}')
            logger.info(f'Pipeline parallel sizes: {capabilities.pipeline_parallel_sizes}')
        
        # Test with multiple datasets for comprehensive validation
        test_results = []
        
        for dataset in ['${VALIDATION_DATASETS[0]}', '${VALIDATION_DATASETS[1]}']:
            logger.info(f'Testing {model_name} with dataset: {dataset}')
            
            # Create evaluation request
            request = EvaluationRequest(
                request_id=f'hybrid_test_{model_name}_{strategy}_{dataset}_{int(time.time())}',
                model_config=enhanced_config,
                datasets=[dataset],
                batch_size=1,  # Conservative batch for large models
                evaluation_type='hybrid_parallelism_test'
            )
            
            # Test if engine can handle the request
            can_handle = engine.can_handle_request(request)
            logger.info(f'Can handle request ({dataset}): {can_handle}')
            
            if not can_handle:
                logger.warning(f'Distributed engine cannot handle {model_name} with {dataset}')
                continue
            
            # Perform evaluation with extended timeout
            logger.info(f'Starting hybrid evaluation: {model_name} + {dataset} with {strategy}...')
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(engine.evaluate(request), timeout=7200)  # 2 hour timeout
                end_time = time.time()
                
                test_result = {
                    'model': model_name,
                    'dataset': dataset,
                    'strategy': strategy,
                    'test_type': 'hybrid_distributed_engine',
                    'tensor_parallel_size': $TENSOR_PARALLEL_SIZE if strategy in ['hybrid', 'tensor_parallel'] else 1,
                    'pipeline_parallel_size': $PIPELINE_PARALLEL_SIZE if strategy in ['hybrid', 'pipeline_parallel'] else 1,
                    'success': True,
                    'execution_time': end_time - start_time,
                    'result_summary': {
                        'request_id': result.request_id if hasattr(result, 'request_id') else 'unknown',
                        'model_name': result.model_name if hasattr(result, 'model_name') else model_name,
                        'execution_time_seconds': result.execution_time_seconds if hasattr(result, 'execution_time_seconds') else end_time - start_time,
                        'evaluation_metrics': result.evaluation_metrics if hasattr(result, 'evaluation_metrics') else {}
                    },
                    'gpu_utilization': '8_H100_GPUs',
                    'timestamp': time.time()
                }
                
                test_results.append(test_result)
                logger.info(f'✅ Hybrid evaluation completed: {model_name} + {dataset} in {end_time - start_time:.2f}s')
                
            except asyncio.TimeoutError:
                logger.error(f'Evaluation timed out: {model_name} + {dataset}')
                test_results.append({
                    'model': model_name,
                    'dataset': dataset,
                    'strategy': strategy,
                    'success': False,
                    'error': 'timeout',
                    'timestamp': time.time()
                })
            except Exception as eval_e:
                logger.error(f'Evaluation failed: {model_name} + {dataset}: {eval_e}')
                test_results.append({
                    'model': model_name,
                    'dataset': dataset,
                    'strategy': strategy,
                    'success': False,
                    'error': str(eval_e),
                    'timestamp': time.time()
                })
        
        # Save comprehensive results
        results_file = Path('$RESULTS_DIR') / f'{model_name}_{strategy}_hybrid_test.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({
                'model': model_name,
                'strategy': strategy,
                'configuration': {
                    'tensor_parallel_size': $TENSOR_PARALLEL_SIZE if strategy in ['hybrid', 'tensor_parallel'] else 1,
                    'pipeline_parallel_size': $PIPELINE_PARALLEL_SIZE if strategy in ['hybrid', 'pipeline_parallel'] else 1,
                    'total_gpus': 8,
                    'gpu_type': 'H100'
                },
                'test_results': test_results,
                'summary': {
                    'total_tests': len(test_results),
                    'successful_tests': len([r for r in test_results if r.get('success', False)]),
                    'failed_tests': len([r for r in test_results if not r.get('success', False)])
                }
            }, f, indent=2)
        
        logger.info(f'Results saved for {model_name} with {strategy} strategy')
        
        # Cleanup
        if hasattr(engine, 'shutdown'):
            engine.shutdown()
        
        return len([r for r in test_results if r.get('success', False)]) > 0
        
    except Exception as e:
        logger.error(f'Failed to test hybrid distributed engine: {model_name} with {strategy}: {e}')
        return False

# Run the test
success = asyncio.run(test_hybrid_distributed())
exit(0 if success else 1)
"
        
        STRATEGY_STATUS=$?
        echo "Strategy $strategy status for $model: $STRATEGY_STATUS"
        
        # Brief cooldown between strategies
        sleep 60
    done
    
    # Stop performance monitoring
    kill $MONITOR_PID 2>/dev/null
    
    echo "GPU Memory After $model:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    echo ""
    
    # Extended cooldown between models for memory cleanup
    sleep 120
done

echo "=========================================="
echo "Advanced Distributed Engine Testing Completed"
echo "End Time: $(date)"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="

# Comprehensive Summary Report
echo ""
echo "==========================================  "
echo "HYBRID PARALLELISM TEST SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Configuration:"
echo "  Total GPUs: 8 H100"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "Total Models Tested: ${#HYBRID_MODELS[@]}"
echo "Models: ${HYBRID_MODELS[*]}"
echo "Strategies Tested: hybrid, tensor_parallel, pipeline_parallel"

# Analyze results
if [ -d "$RESULTS_DIR" ]; then
    echo ""
    echo "Results Directory: $RESULTS_DIR"
    echo "Result Files:"
    ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "No result files found"
    
    echo ""
    echo "Performance Logs:"
    ls -la "$RESULTS_DIR"/*_performance.log 2>/dev/null || echo "No performance logs found"
fi

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Check results in: $RESULTS_DIR"
echo "=========================================="