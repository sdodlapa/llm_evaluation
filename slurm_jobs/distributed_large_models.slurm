#!/bin/bash
#SBATCH --job-name=distributed_large_models
#SBATCH --partition=h100quadflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4  # Request 4 H100 GPUs for tensor parallelism
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/distributed_large_models_%j.out
#SBATCH --error=logs/distributed_large_models_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Distributed Engine Testing - Large Models (30B-70B parameters)
# This job tests the distributed evaluation engine with large models
# using multi-GPU tensor parallelism across 4 H100 GPUs

# Job Information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: distributed_large_models"
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

# Set Distributed Training Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

# Test Models Configuration
# Large models that require distributed processing (>15GB threshold)
LARGE_MODELS=(
    "qwen3_coder_30b"     # 32GB - Largest available, requires tensor parallelism
    "wizardmath_70b"      # 70GB with AWQ quantization
    "metamath_70b"        # 70GB mathematical reasoning
    "clinical_camel_70b"  # 70GB biomedical specialist
)

# Test Datasets (small datasets for faster distributed engine testing)
TEST_DATASETS=("drop" "arc_challenge" "hellaswag")

echo "=========================================="
echo "Starting Distributed Engine Testing"
echo "Testing ${#LARGE_MODELS[@]} large models with distributed evaluation"
echo "Models: ${LARGE_MODELS[*]}"
echo "Datasets: ${TEST_DATASETS[*]}"
echo "=========================================="

# Create test results directory
mkdir -p distributed_test_results/$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="distributed_test_results/$(date +%Y%m%d_%H%M%S)"

# Test each large model with distributed engine
for model in "${LARGE_MODELS[@]}"; do
    echo ""
    echo "==========================================  "
    echo "Testing Model: $model"
    echo "Time: $(date)"
    echo "GPU Memory Before:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    echo "=========================================="
    
    # Run distributed evaluation
    python -c "
import asyncio
import json
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_distributed_model():
    try:
        # Import after CUDA setup
        from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig
        from core_shared.interfaces.evaluation_interfaces import EvaluationRequest
        from configs.model_registry import MODEL_CONFIGS
        from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
        
        model_name = '$model'
        logger.info(f'Testing distributed engine with model: {model_name}')
        
        # Check if model exists in registry
        if model_name not in MODEL_CONFIGS:
            logger.error(f'Model {model_name} not found in MODEL_CONFIGS')
            return False
        
        model_config_base = MODEL_CONFIGS[model_name]
        
        # Create enhanced model config
        enhanced_config = EnhancedModelConfig(
            model_name=model_name,
            model_path=model_config_base.huggingface_id,
            parameters=model_config_base.size_gb * 1_000_000_000,  # Convert GB to parameters estimate
            quantization=model_config_base.quantization_method,
            precision='fp16',
            context_length=model_config_base.context_window
        )
        
        # Configure distributed engine for testing
        config = DistributedEngineConfig(
            max_concurrent_evaluations=1,
            enable_dynamic_scaling=True,
            enable_fault_tolerance=True,
            memory_optimization_level='balanced',
            communication_backend='nccl',
            performance_monitoring=True
        )
        
        # Initialize distributed engine
        logger.info('Initializing distributed evaluation engine...')
        engine = DistributedEvaluationEngine(config)
        
        # Test engine capabilities
        if hasattr(engine, 'get_capabilities'):
            capabilities = engine.get_capabilities()
            logger.info(f'Engine capabilities: {capabilities}')
        
        # Create evaluation request
        request = EvaluationRequest(
            request_id=f'distributed_test_{model_name}_{int(time.time())}',
            model_config=enhanced_config,
            datasets=['${TEST_DATASETS[0]}'],  # Use first dataset for quick test
            batch_size=2,  # Small batch for large models
            evaluation_type='performance_test'
        )
        
        # Test if engine can handle the request
        can_handle = engine.can_handle_request(request)
        logger.info(f'Can handle request: {can_handle}')
        
        if not can_handle:
            logger.warning(f'Distributed engine cannot handle model {model_name}')
            return False
        
        # Perform evaluation (with timeout for safety)
        logger.info(f'Starting distributed evaluation for {model_name}...')
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(engine.evaluate(request), timeout=3600)  # 1 hour timeout
            end_time = time.time()
            
            # Save results
            test_result = {
                'model': model_name,
                'test_type': 'distributed_engine',
                'success': True,
                'execution_time': end_time - start_time,
                'result_summary': {
                    'request_id': result.request_id if hasattr(result, 'request_id') else 'unknown',
                    'model_name': result.model_name if hasattr(result, 'model_name') else model_name,
                    'execution_time_seconds': result.execution_time_seconds if hasattr(result, 'execution_time_seconds') else end_time - start_time
                },
                'timestamp': time.time()
            }
            
            # Write results
            results_file = Path('$RESULTS_DIR') / f'{model_name}_distributed_test.json'
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(test_result, f, indent=2)
            
            logger.info(f'✅ Distributed evaluation completed for {model_name} in {end_time - start_time:.2f}s')
            return True
            
        except asyncio.TimeoutError:
            logger.error(f'Evaluation timed out for {model_name}')
            return False
        except Exception as eval_e:
            logger.error(f'Evaluation failed for {model_name}: {eval_e}')
            return False
        
        finally:
            # Cleanup
            if hasattr(engine, 'shutdown'):
                engine.shutdown()
                
    except Exception as e:
        logger.error(f'Failed to test distributed engine with {model_name}: {e}')
        return False

# Run the test
success = asyncio.run(test_distributed_model())
exit(0 if success else 1)
"
    
    TEST_STATUS=$?
    echo "Test Status for $model: $TEST_STATUS"
    
    echo "GPU Memory After:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
    echo ""
    
    # Brief cooldown between models
    sleep 30
done

echo "=========================================="
echo "Distributed Engine Testing Completed"
echo "End Time: $(date)"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="

# Summary Report
echo ""
echo "==========================================  "
echo "TEST SUMMARY REPORT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total Models Tested: ${#LARGE_MODELS[@]}"
echo "Models: ${LARGE_MODELS[*]}"

# Check results
if [ -d "$RESULTS_DIR" ]; then
    echo "Results Directory: $RESULTS_DIR"
    echo "Result Files:"
    ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "No result files found"
fi

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="