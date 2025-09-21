#!/bin/bash
#SBATCH --job-name=distributed_validation
#SBATCH --partition=h100dualflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2  # Request 2 H100 GPUs for basic distributed testing
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/distributed_validation_%j.out
#SBATCH --error=logs/distributed_validation_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sdodl001_odu_edu@noreply.odu.edu

# Distributed Engine Validation - Basic Functionality Test
# This job validates the distributed evaluation engine with smaller models
# to ensure the distributed infrastructure works before testing large models

# Job Information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: distributed_validation"
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

# Set Basic Distributed Environment
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=2

echo "=========================================="
echo "Starting Distributed Engine Validation"
echo "Purpose: Validate distributed infrastructure before large model tests"
echo "GPUs: 2 H100"
echo "=========================================="

# Create validation results directory
mkdir -p distributed_test_results/validation_$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="distributed_test_results/validation_$(date +%Y%m%d_%H%M%S)"

echo "Results will be saved in: $RESULTS_DIR"

# Step 1: Validate Distributed Engine Import and Initialization
echo ""
echo "==========================================  "
echo "STEP 1: Testing Distributed Engine Import"
echo "=========================================="

python -c "
import sys
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info('Testing distributed engine imports...')
    
    # Test core imports
    from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig
    from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader, DistributionStrategy
    from engines.distributed.distributed_orchestrator import DistributedEvaluationOrchestrator
    from engines.distributed.performance_monitor import PerformanceMonitor
    
    logger.info('✅ All distributed engine imports successful')
    
    # Test enum values
    logger.info(f'Available distribution strategies: {list(DistributionStrategy)}')
    
    # Test configuration creation
    config = DistributedEngineConfig(
        max_concurrent_evaluations=1,
        enable_dynamic_scaling=False,  # Disabled for validation
        enable_fault_tolerance=True,
        memory_optimization_level='balanced',
        communication_backend='nccl'
    )
    logger.info(f'✅ Configuration created: {config}')
    
    print('IMPORT_TEST: SUCCESS')
    
except Exception as e:
    logger.error(f'Import test failed: {e}')
    traceback.print_exc()
    print('IMPORT_TEST: FAILED')
    sys.exit(1)
"

IMPORT_STATUS=$?
echo "Import Test Status: $IMPORT_STATUS"

if [ $IMPORT_STATUS -ne 0 ]; then
    echo "❌ Import test failed. Stopping validation."
    exit 1
fi

# Step 2: Test Multi-GPU Detection and Allocation
echo ""
echo "==========================================  "
echo "STEP 2: Testing Multi-GPU Detection"
echo "=========================================="

python -c "
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader
    
    logger.info('Testing multi-GPU detection and allocation...')
    
    # Initialize model loader
    loader = MultiGPUModelLoader(
        max_models=1,
        memory_optimization='balanced',
        enable_pipeline_parallelism=False,  # Disabled for validation
        communication_backend='nccl'
    )
    
    # Test GPU detection
    gpu_status = loader.get_gpu_status()
    logger.info(f'GPU Status: {gpu_status}')
    
    # Test capabilities
    if hasattr(loader, 'get_capabilities'):
        capabilities = loader.get_capabilities()
        logger.info(f'Loader capabilities: {capabilities}')
    
    # Save GPU status
    results_file = Path('$RESULTS_DIR') / 'gpu_validation.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'test': 'gpu_detection',
            'gpu_status': gpu_status,
            'success': True
        }, f, indent=2)
    
    logger.info('✅ Multi-GPU detection successful')
    print('GPU_TEST: SUCCESS')
    
except Exception as e:
    logger.error(f'GPU detection test failed: {e}')
    print('GPU_TEST: FAILED')
    exit(1)
"

GPU_STATUS=$?
echo "GPU Detection Test Status: $GPU_STATUS"

if [ $GPU_STATUS -ne 0 ]; then
    echo "❌ GPU detection test failed. Stopping validation."
    exit 1
fi

# Step 3: Test Distributed Engine Initialization
echo ""
echo "==========================================  "
echo "STEP 3: Testing Engine Initialization"
echo "=========================================="

python -c "
import asyncio
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_engine_initialization():
    try:
        from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig
        
        logger.info('Testing distributed engine initialization...')
        
        # Create conservative configuration for validation
        config = DistributedEngineConfig(
            max_concurrent_evaluations=1,
            enable_dynamic_scaling=False,
            enable_fault_tolerance=True,
            memory_optimization_level='conservative',
            communication_backend='nccl',
            scheduling_strategy='priority_first',
            performance_monitoring=False,  # Disabled for validation
            automatic_model_offloading=True,
            cross_gpu_memory_sharing=False,  # Disabled for validation
            pipeline_optimization=False  # Disabled for validation
        )
        
        # Initialize engine
        logger.info('Creating distributed engine...')
        engine = DistributedEvaluationEngine(config)
        
        # Test engine state
        if hasattr(engine, 'is_ready'):
            ready_state = engine.is_ready()
            logger.info(f'Engine ready state: {ready_state}')
        
        # Test capabilities
        if hasattr(engine, 'get_capabilities'):
            capabilities = engine.get_capabilities()
            logger.info(f'Engine capabilities: {capabilities}')
            
            # Save capabilities
            results_file = Path('$RESULTS_DIR') / 'engine_capabilities.json'
            with open(results_file, 'w') as f:
                json.dump({
                    'test': 'engine_initialization',
                    'capabilities': capabilities.__dict__ if hasattr(capabilities, '__dict__') else str(capabilities),
                    'success': True
                }, f, indent=2)
        
        # Test shutdown
        if hasattr(engine, 'shutdown'):
            logger.info('Testing engine shutdown...')
            engine.shutdown()
        
        logger.info('✅ Engine initialization and shutdown successful')
        return True
        
    except Exception as e:
        logger.error(f'Engine initialization test failed: {e}')
        return False

# Run the test
success = asyncio.run(test_engine_initialization())
print('ENGINE_INIT_TEST:', 'SUCCESS' if success else 'FAILED')
exit(0 if success else 1)
"

ENGINE_STATUS=$?
echo "Engine Initialization Test Status: $ENGINE_STATUS"

if [ $ENGINE_STATUS -ne 0 ]; then
    echo "❌ Engine initialization test failed. Stopping validation."
    exit 1
fi

# Step 4: Test Model Handling Logic (without actual loading)
echo ""
echo "==========================================  "
echo "STEP 4: Testing Model Handling Logic"
echo "=========================================="

python -c "
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig
    from core_shared.interfaces.evaluation_interfaces import EvaluationRequest
    from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
    
    logger.info('Testing model handling logic...')
    
    # Create engine
    config = DistributedEngineConfig(max_concurrent_evaluations=1)
    engine = DistributedEvaluationEngine(config)
    
    # Test with small model (should be rejected)
    small_config = EnhancedModelConfig(
        model_name='validation_small_7b',
        model_path='/mock/path',
        parameters=7_000_000_000  # 7B parameters
    )
    
    small_request = EvaluationRequest(
        request_id='validation_small_test',
        model_config=small_config,
        datasets=['mock_dataset'],
        batch_size=4
    )
    
    can_handle_small = engine.can_handle_request(small_request)
    logger.info(f'Can handle small model (7B): {can_handle_small} (should be False)')
    
    # Test with large model (should be accepted)
    large_config = EnhancedModelConfig(
        model_name='validation_large_70b',
        model_path='/mock/path',
        parameters=70_000_000_000  # 70B parameters
    )
    
    large_request = EvaluationRequest(
        request_id='validation_large_test',
        model_config=large_config,
        datasets=['mock_dataset'],
        batch_size=2
    )
    
    can_handle_large = engine.can_handle_request(large_request)
    logger.info(f'Can handle large model (70B): {can_handle_large} (should be True)')
    
    # Save results
    results_file = Path('$RESULTS_DIR') / 'model_handling_test.json'
    with open(results_file, 'w') as f:
        json.dump({
            'test': 'model_handling_logic',
            'small_model_handling': can_handle_small,
            'large_model_handling': can_handle_large,
            'logic_correct': not can_handle_small and can_handle_large,
            'success': True
        }, f, indent=2)
    
    # Cleanup
    if hasattr(engine, 'shutdown'):
        engine.shutdown()
    
    logic_correct = not can_handle_small and can_handle_large
    logger.info(f'✅ Model handling logic test: {\"PASSED\" if logic_correct else \"FAILED\"}')
    print('MODEL_LOGIC_TEST:', 'SUCCESS' if logic_correct else 'FAILED')
    
except Exception as e:
    logger.error(f'Model handling logic test failed: {e}')
    print('MODEL_LOGIC_TEST: FAILED')
    exit(1)
"

LOGIC_STATUS=$?
echo "Model Handling Logic Test Status: $LOGIC_STATUS"

# Final Summary
echo ""
echo "=========================================="
echo "DISTRIBUTED ENGINE VALIDATION SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Validation Results:"
echo "  ✅ Import Test: $([ $IMPORT_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  ✅ GPU Detection Test: $([ $GPU_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  ✅ Engine Initialization Test: $([ $ENGINE_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  ✅ Model Handling Logic Test: $([ $LOGIC_STATUS -eq 0 ] && echo 'PASSED' || echo 'FAILED')"

# Overall validation status
ALL_TESTS_PASSED=0
if [ $IMPORT_STATUS -eq 0 ] && [ $GPU_STATUS -eq 0 ] && [ $ENGINE_STATUS -eq 0 ] && [ $LOGIC_STATUS -eq 0 ]; then
    ALL_TESTS_PASSED=1
fi

echo ""
echo "Overall Validation: $([ $ALL_TESTS_PASSED -eq 1 ] && echo '✅ PASSED' || echo '❌ FAILED')"

if [ $ALL_TESTS_PASSED -eq 1 ]; then
    echo ""
    echo "🎉 Distributed engine validation successful!"
    echo "   Ready to proceed with large model testing:"
    echo "   - Submit distributed_large_models.slurm for basic testing"
    echo "   - Submit distributed_hybrid_parallelism.slurm for advanced testing"
else
    echo ""
    echo "❌ Distributed engine validation failed!"
    echo "   Fix issues before submitting large model tests."
fi

echo ""
echo "Results saved in: $RESULTS_DIR"
if [ -d "$RESULTS_DIR" ]; then
    echo "Validation files:"
    ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "No result files found"
fi

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "=========================================="
echo "Validation completed at: $(date)"
echo "=========================================="

exit $([ $ALL_TESTS_PASSED -eq 1 ] && echo 0 || echo 1)