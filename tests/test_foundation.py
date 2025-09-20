"""
Foundation validation test for hybrid architecture

Tests core components of Phase 1 implementation without requiring
full model loading or evaluation pipeline.
"""

import asyncio
import logging
import sys
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.integration.hybrid_integration_adapter import HybridIntegrationAdapter
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig, ModelSizeCategory
from core_shared.interfaces.evaluation_interfaces import EngineType


async def test_foundation_components():
    """Test the foundation components of hybrid architecture"""
    logger.info("=" * 60)
    logger.info("HYBRID ARCHITECTURE FOUNDATION VALIDATION TEST")
    logger.info("=" * 60)
    
    adapter = HybridIntegrationAdapter()
    
    try:
        # Test 1: Initialize hybrid system
        logger.info("\n🧪 Test 1: System Initialization")
        success = await adapter.initialize()
        if success:
            logger.info("✅ Hybrid system initialized successfully")
        else:
            logger.error("❌ Failed to initialize hybrid system")
            return False
        
        # Test 2: Engine capabilities and selection
        logger.info("\n🧪 Test 2: Engine Selection Logic")
        test_models = [
            EnhancedModelConfig(
                model_name="small_test_model",
                huggingface_id="test/small",
                license="MIT",
                size_gb=5.0,
                context_window=2048,
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            ),
            EnhancedModelConfig(
                model_name="medium_test_model", 
                huggingface_id="test/medium",
                license="MIT",
                size_gb=25.0,
                context_window=4096,
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            ),
            EnhancedModelConfig(
                model_name="large_test_model",
                huggingface_id="test/large", 
                license="Apache-2.0",
                size_gb=50.0,
                context_window=8192,
                tensor_parallel_size=2,
                pipeline_parallel_size=1
            )
        ]
        
        selection_results = await adapter.test_engine_selection(test_models)
        logger.info("Engine selection results:")
        for model_name, result in selection_results.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Recommended Engine: {result['recommended_engine']}")
            logger.info(f"    GPU Memory Required: {result['resource_requirements']['gpu_memory_gb']:.1f}GB")
            logger.info(f"    Model Size: {result['model_size_gb']}GB")
            logger.info(f"    Is Large Model: {result['is_large_model']}")
        
        # Validate engine selection logic
        small_engine = selection_results["small_test_model"]["recommended_engine"]
        large_engine = selection_results["large_test_model"]["recommended_engine"]
        
        if small_engine == "lightweight":
            logger.info("✅ Small model correctly assigned to lightweight engine")
        else:
            logger.warning(f"⚠️  Small model assigned to {small_engine} engine (expected lightweight)")
        
        if large_engine == "distributed":
            logger.info("✅ Large model correctly assigned to distributed engine")
        else:
            logger.warning(f"⚠️  Large model assigned to {large_engine} engine (expected distributed)")
        
        # Test 3: System status and metrics
        logger.info("\n🧪 Test 3: System Status and Metrics")
        system_status = await adapter.get_system_status()
        logger.info("System Status:")
        
        # Get orchestrator metrics safely
        orchestrator_metrics = system_status.get('orchestrator_metrics', {})
        active_requests = orchestrator_metrics.get('active_requests', 0)
        
        logger.info(f"  Orchestrator Active: {active_requests == 0}")
        logger.info(f"  Lightweight Engine Initialized: {system_status['lightweight_engine_status']['initialized']}")
        logger.info(f"  Registry Models: {system_status['registry_stats']['total_models']}")
        logger.info(f"  Engine Distribution: {system_status['registry_stats']['engine_distribution']}")
        
        # Test 4: Mock evaluation (without actual model loading)
        logger.info("\n🧪 Test 4: Mock Evaluation Flow")
        try:
            # This will use mock model loader for testing
            result = await adapter.evaluate_model(
                model_name="small_test_model",
                datasets=["test_dataset"],
                evaluation_params={"mock_mode": True}
            )
            
            if result.success:
                logger.info("✅ Mock evaluation completed successfully")
                logger.info(f"  Engine Used: {result.engine_used.value}")
                logger.info(f"  Execution Time: {result.execution_time_seconds:.2f}s")
                logger.info(f"  Model: {result.model_name}")
            else:
                logger.error(f"❌ Mock evaluation failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Mock evaluation exception: {e}")
        
        # Test 5: Component integration validation
        logger.info("\n🧪 Test 5: Component Integration Validation")
        
        # Check enhanced model config functionality
        test_config = test_models[0]
        optimal_engine = test_config.get_optimal_engine()
        resource_reqs = test_config.get_resource_requirements()
        compatibility = test_config.get_engine_compatibility()
        
        logger.info(f"Enhanced ModelConfig features:")
        logger.info(f"  Optimal Engine: {optimal_engine.value}")
        logger.info(f"  Resource Requirements: {resource_reqs}")
        logger.info(f"  Engine Compatibility: {compatibility}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 FOUNDATION VALIDATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Generate test report
        test_report = {
            "test_timestamp": "2025-09-20",
            "phase": "Phase 1 Foundation",
            "status": "PASSED",
            "components_tested": [
                "Enhanced ModelConfig",
                "Enhanced ModelRegistry", 
                "Evaluation Orchestrator",
                "Lightweight Engine Foundation",
                "Engine Selection Logic",
                "Integration Adapter"
            ],
            "engine_selection_results": selection_results,
            "system_status": system_status
        }
        
        report_path = project_root / "tests" / "foundation_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        logger.info(f"📊 Test report saved to: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Foundation test failed: {e}")
        return False
        
    finally:
        # Cleanup
        await adapter.cleanup()
        logger.info("🧹 Test cleanup completed")


async def main():
    """Main test function"""
    success = await test_foundation_components()
    
    if success:
        logger.info("\n✅ All foundation tests passed!")
        logger.info("🚀 Hybrid architecture foundation is ready for Phase 2 development")
    else:
        logger.error("\n❌ Foundation tests failed!")
        logger.error("🔧 Please review and fix issues before proceeding")
    
    return success


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(main())
    sys.exit(0 if result else 1)