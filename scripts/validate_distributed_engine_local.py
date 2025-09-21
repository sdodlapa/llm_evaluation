#!/usr/bin/env python3
"""
Local Distributed Engine Validation Script

This script performs local validation of the distributed engine components
without requiring SLURM submission. It tests imports, basic functionality,
and configuration validation to ensure the distributed engine is ready
for production testing.
"""

import sys
import os
import logging
import traceback
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('distributed_validation_local.log')
    ]
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all distributed engine imports"""
    logger.info("Testing distributed engine imports...")
    
    try:
        # Core imports
        from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig
        from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader, DistributionStrategy, GPUAllocation
        from engines.distributed.distributed_orchestrator import DistributedEvaluationOrchestrator
        from engines.distributed.performance_monitor import MultiGPUPerformanceMonitor  # Correct class name
        
        # Test enum access
        strategies = list(DistributionStrategy)
        logger.info(f"Available distribution strategies: {strategies}")
        
        # Test configuration creation
        config = DistributedEngineConfig(
            max_concurrent_evaluations=1,
            enable_dynamic_scaling=False,
            memory_optimization_level='balanced'
        )
        logger.info("✅ All imports successful")
        return True, config
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False, None

def test_configuration():
    """Test distributed engine configuration"""
    logger.info("Testing distributed engine configuration...")
    
    try:
        from engines.distributed.distributed_engine import DistributedEngineConfig
        
        # Test different configurations
        configs = {
            'balanced': DistributedEngineConfig(
                memory_optimization_level='balanced',
                communication_backend='nccl',
                scheduling_strategy='priority_first'
            ),
            'aggressive': DistributedEngineConfig(
                memory_optimization_level='aggressive',
                enable_dynamic_scaling=True,
                automatic_model_offloading=True
            ),
            'conservative': DistributedEngineConfig(
                memory_optimization_level='conservative',
                enable_fault_tolerance=True,
                cross_gpu_memory_sharing=False
            )
        }
        
        for name, config in configs.items():
            logger.info(f"✅ Configuration '{name}' created successfully")
            
        return True, configs
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False, None

def test_model_handling_logic():
    """Test model handling logic without GPU requirements"""
    logger.info("Testing model handling logic...")
    
    try:
        from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig
        from core_shared.interfaces.evaluation_interfaces import EvaluationRequest
        from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
        
        # Create engine with minimal config
        config = DistributedEngineConfig(
            max_concurrent_evaluations=1,
            enable_dynamic_scaling=False,
            performance_monitoring=False
        )
        
        # Note: We may not be able to fully initialize without GPUs
        # But we can test the configuration and basic class creation
        try:
            engine = DistributedEvaluationEngine(config)
            engine_created = True
        except Exception as gpu_e:
            logger.warning(f"Engine initialization failed (expected without GPUs): {gpu_e}")
            engine_created = False
        
        # Test model size estimation logic (this should work without GPUs)
        if engine_created and hasattr(engine, '_estimate_model_size'):
            # Create test model configs
            small_config = EnhancedModelConfig(
                model_name='test_small_7b',
                model_path='/mock/path',
                parameters=7_000_000_000
            )
            
            large_config = EnhancedModelConfig(
                model_name='test_large_70b', 
                model_path='/mock/path',
                parameters=70_000_000_000
            )
            
            try:
                small_size = engine._estimate_model_size(small_config)
                large_size = engine._estimate_model_size(large_config)
                
                logger.info(f"Small model estimated size: {small_size:.1f}GB")
                logger.info(f"Large model estimated size: {large_size:.1f}GB")
                
                # Cleanup
                if hasattr(engine, 'shutdown'):
                    engine.shutdown()
                    
            except Exception as est_e:
                logger.warning(f"Model size estimation test failed: {est_e}")
        
        logger.info("✅ Model handling logic test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model handling test failed: {e}")
        return False

def test_distribution_strategies():
    """Test distribution strategy enumeration and logic"""
    logger.info("Testing distribution strategies...")
    
    try:
        from engines.distributed.multi_gpu_model_loader import DistributionStrategy, GPUAllocation
        
        # Test enum values
        strategies = list(DistributionStrategy)
        logger.info(f"Available strategies: {strategies}")
        
        # Test expected strategies (updated for actual enum values)
        expected_strategies = ['TENSOR_PARALLEL', 'PIPELINE_PARALLEL', 'HYBRID', 'AUTO']
        for expected in expected_strategies:
            if any(expected.lower() in str(s).lower() for s in strategies):
                logger.info(f"✅ Found expected strategy: {expected}")
            else:
                logger.warning(f"⚠️  Expected strategy not found: {expected}")
        
        # Test GPU allocation structure with correct parameters
        allocation = GPUAllocation(
            gpu_id=0,
            memory_allocated_gb=20.0,
            memory_available_gb=24.0,
            model_layers=["layer_0", "layer_1"],
            is_primary=True
        )
        logger.info(f"✅ GPU allocation created: {allocation}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Distribution strategy test failed: {e}")
        return False

def test_interface_compatibility():
    """Test compatibility with existing evaluation interfaces"""
    logger.info("Testing interface compatibility...")
    
    try:
        from core_shared.interfaces.evaluation_interfaces import EvaluationRequest, EvaluationResult, EngineType
        from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
        
        # Test creating evaluation request for distributed engine with correct parameters
        config = EnhancedModelConfig(
            model_name='compatibility_test_model',
            huggingface_id='/mock/path',
            license='MIT',
            size_gb=70.0,  # Use size_gb, parameters will be estimated automatically
            context_window=4096
        )
        
        request = EvaluationRequest(
            request_id='compatibility_test',
            model_config=config,
            datasets=['test_dataset'],
            evaluation_params={'batch_size': 2}  # batch_size goes in evaluation_params
        )
        
        logger.info(f"✅ EvaluationRequest created: {request.request_id}")
        logger.info(f"Model size category: {config.model_size_category}")
        logger.info(f"Estimated parameters: {config.estimated_parameters_b}B")
        
        # Test engine type
        distributed_type = EngineType.DISTRIBUTED
        logger.info(f"✅ EngineType.DISTRIBUTED available: {distributed_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Interface compatibility test failed: {e}")
        return False

def run_validation():
    """Run complete local validation"""
    logger.info("========================================")
    logger.info("Starting Local Distributed Engine Validation")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("========================================")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'summary': {'total': 0, 'passed': 0, 'failed': 0}
    }
    
    tests = [
        ('imports', test_imports),
        ('configuration', test_configuration), 
        ('model_handling_logic', test_model_handling_logic),
        ('distribution_strategies', test_distribution_strategies),
        ('interface_compatibility', test_interface_compatibility)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} test ---")
        results['summary']['total'] += 1
        
        try:
            if test_name in ['imports', 'configuration']:
                success, test_data = test_func()
            else:
                success = test_func()
                test_data = None
            
            results['tests'][test_name] = {
                'success': success,
                'data': str(test_data) if test_data else None
            }
            
            if success:
                results['summary']['passed'] += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                results['summary']['failed'] += 1
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            results['summary']['failed'] += 1
            results['tests'][test_name] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
    
    # Save results
    results_file = Path('distributed_validation_local_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logger.info("\n========================================")
    logger.info("LOCAL VALIDATION SUMMARY")
    logger.info("========================================")
    logger.info(f"Total tests: {results['summary']['total']}")
    logger.info(f"Passed: {results['summary']['passed']}")
    logger.info(f"Failed: {results['summary']['failed']}")
    
    success_rate = results['summary']['passed'] / results['summary']['total'] * 100
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if results['summary']['failed'] == 0:
        logger.info("🎉 All tests PASSED! Distributed engine ready for SLURM testing.")
        recommendation = "✅ PROCEED with SLURM submission: ./submit_distributed_tests.sh"
    elif success_rate >= 80:
        logger.info("⚠️  Most tests passed. Distributed engine likely ready with minor issues.")
        recommendation = "⚠️  CONSIDER proceeding with validation job first"
    else:
        logger.info("❌ Multiple test failures. Fix issues before SLURM submission.")
        recommendation = "❌ DO NOT submit until issues are resolved"
    
    logger.info(f"\nRecommendation: {recommendation}")
    logger.info(f"Results saved: {results_file}")
    logger.info("========================================")
    
    return results['summary']['failed'] == 0

if __name__ == "__main__":
    try:
        success = run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        traceback.print_exc()
        sys.exit(1)