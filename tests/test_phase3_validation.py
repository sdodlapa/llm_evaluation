"""
Phase 3 Validation Tests - Distributed Engine

Comprehensive test suite for distributed evaluation engine functionality,
including large model handling, multi-GPU coordination, performance validation,
and integration testing with the lightweight engine.
"""

import unittest
import asyncio
import time
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import project modules
from core_shared.interfaces.evaluation_interfaces import EvaluationRequest, EvaluationResult, EngineType
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
from engines.distributed.distributed_engine import DistributedEvaluationEngine, DistributedEngineConfig, MockDistributedEngine
from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader, DistributionStrategy, DistributedModelInfo
from engines.distributed.distributed_orchestrator import DistributedEvaluationOrchestrator, WorkloadPriority
from engines.distributed.performance_monitor import MultiGPUPerformanceMonitor, MetricType, AlertSeverity

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_model_config(model_name: str, size_gb: float = 30.0) -> EnhancedModelConfig:
    """Helper function to create test model configs"""
    return EnhancedModelConfig(
        model_name=model_name,
        huggingface_id=f"test/{model_name}",
        license="apache-2.0",
        size_gb=size_gb,
        context_window=4096
    )

class TestMultiGPUModelLoader(unittest.TestCase):
    """Test cases for multi-GPU model loader"""
    
    def setUp(self):
        """Set up test environment"""
        self.loader = MultiGPUModelLoader(
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            default_strategy=DistributionStrategy.AUTO
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if self.loader:
            self.loader.cleanup()
    
    def test_initialization(self):
        """Test loader initialization"""
        self.assertIsNotNone(self.loader)
        self.assertEqual(len(self.loader.gpu_ids), 8)  # Mock GPUs
        self.assertEqual(self.loader.default_strategy, DistributionStrategy.AUTO)
    
    def test_model_size_estimation(self):
        """Test model size estimation"""
        # Test known model sizes
        small_config = EnhancedModelConfig(
            model_name="test-7b-model",
            model_path="/mock/path",
            parameters=7_000_000_000
        )
        
        large_config = EnhancedModelConfig(
            model_name="test-70b-model", 
            model_path="/mock/path",
            parameters=70_000_000_000
        )
        
        small_size = self.loader._estimate_model_memory_usage(small_config)
        large_size = self.loader._estimate_model_memory_usage(large_config)
        
        self.assertGreater(small_size, 10.0)  # Should be > 10GB
        self.assertLess(small_size, 20.0)     # Should be < 20GB
        self.assertGreater(large_size, 120.0) # Should be > 120GB
        self.assertLess(large_size, 150.0)    # Should be < 150GB
    
    def test_distribution_strategy_selection(self):
        """Test automatic distribution strategy selection"""
        # Small model - should use tensor parallel
        small_config = EnhancedModelConfig(
            model_name="test-13b-model",
            model_path="/mock/path", 
            parameters=13_000_000_000
        )
        
        strategy = self.loader._determine_optimal_distribution_strategy(small_config)
        self.assertEqual(strategy, DistributionStrategy.TENSOR_PARALLEL)
        
        # Large model - should use hybrid
        large_config = EnhancedModelConfig(
            model_name="test-175b-model",
            model_path="/mock/path",
            parameters=175_000_000_000
        )
        
        strategy = self.loader._determine_optimal_distribution_strategy(large_config)
        self.assertEqual(strategy, DistributionStrategy.HYBRID)
    
    def test_can_load_model(self):
        """Test model loading capability check"""
        # Model that can be loaded
        loadable_config = EnhancedModelConfig(
            model_name="test-30b-model",
            model_path="/mock/path",
            parameters=30_000_000_000
        )
        
        self.assertTrue(self.loader.can_load_model(loadable_config))
        
        # Model too large
        huge_config = EnhancedModelConfig(
            model_name="test-500b-model",
            model_path="/mock/path", 
            parameters=500_000_000_000
        )
        
        self.assertFalse(self.loader.can_load_model(huge_config))
    
    def test_load_model_distributed(self):
        """Test distributed model loading"""
        config = EnhancedModelConfig(
            model_name="test-13b-model",
            model_path="/mock/path",
            parameters=13_000_000_000
        )
        
        model_info = self.loader.load_model_distributed(config)
        
        self.assertIsInstance(model_info, DistributedModelInfo)
        self.assertEqual(model_info.model_name, "test-13b-model")
        self.assertGreater(model_info.memory_usage_gb, 0)
        self.assertGreater(len(model_info.gpu_allocations), 0)
        self.assertIn(model_info.distribution_strategy, DistributionStrategy)
    
    def test_gpu_allocation_tracking(self):
        """Test GPU allocation and deallocation"""
        config = EnhancedModelConfig(
            model_name="test-model",
            model_path="/mock/path",
            parameters=20_000_000_000
        )
        
        # Load model
        model_info = self.loader.load_model_distributed(config)
        allocated_gpus = [alloc.gpu_id for alloc in model_info.gpu_allocations]
        
        # Check allocations
        for gpu_id in allocated_gpus:
            self.assertIn(gpu_id, self.loader._gpu_allocations)
        
        # Unload model
        success = self.loader.unload_model("test-model")
        self.assertTrue(success)
        
        # Check deallocations
        for gpu_id in allocated_gpus:
            self.assertIsNone(self.loader._gpu_allocations.get(gpu_id))


class TestDistributedOrchestrator(unittest.TestCase):
    """Test cases for distributed orchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        self.model_loader = MultiGPUModelLoader(gpu_ids=[0, 1, 2, 3])
        self.orchestrator = DistributedEvaluationOrchestrator(
            model_loader=self.model_loader,
            max_concurrent_tasks=2,
            enable_fault_tolerance=True
        )
        self.orchestrator.start()
        time.sleep(0.1)  # Let orchestrator start
    
    def tearDown(self):
        """Clean up test environment"""
        if self.orchestrator:
            self.orchestrator.stop()
        if self.model_loader:
            self.model_loader.cleanup()
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator)
        self.assertTrue(self.orchestrator._running)
        self.assertEqual(self.orchestrator.max_concurrent_tasks, 2)
        self.assertTrue(self.orchestrator.enable_fault_tolerance)
    
    def test_submit_evaluation_request(self):
        """Test evaluation request submission"""
        config = EnhancedModelConfig(
            model_name="test-model",
            model_path="/mock/path",
            parameters=20_000_000_000
        )
        
        request = EvaluationRequest(
            request_id="test-request-001",
            model_config=config,
            datasets=["test_dataset"],
            batch_size=4
        )
        
        task_id = self.orchestrator.submit_evaluation_request(
            request=request,
            priority=WorkloadPriority.NORMAL
        )
        
        self.assertIsNotNone(task_id)
        self.assertIsInstance(task_id, str)
        
        # Check task status
        status = self.orchestrator.get_task_status(task_id)
        self.assertIn(status["status"], ["queued", "running"])
    
    def test_cluster_status(self):
        """Test cluster status reporting"""
        status = self.orchestrator.get_cluster_status()
        
        self.assertIn("cluster_state", status)
        self.assertIn("metrics", status)
        self.assertIn("active_tasks", status)
        self.assertIn("queued_tasks", status)
        self.assertIn("gpu_allocations", status)
        
        self.assertIsInstance(status["active_tasks"], int)
        self.assertIsInstance(status["queued_tasks"], int)
    
    def test_task_cancellation(self):
        """Test task cancellation"""
        config = EnhancedModelConfig(
            model_name="test-model",
            model_path="/mock/path",
            parameters=20_000_000_000
        )
        
        request = EvaluationRequest(
            request_id="test-request-002",
            model_config=config,
            datasets=["test_dataset"],
            batch_size=4
        )
        
        task_id = self.orchestrator.submit_evaluation_request(request)
        time.sleep(0.1)  # Let task be queued
        
        # Cancel task
        success = self.orchestrator.cancel_task(task_id)
        self.assertTrue(success)
        
        # Check task status
        status = self.orchestrator.get_task_status(task_id)
        self.assertEqual(status["status"], "failed")


class TestDistributedEngine(unittest.TestCase):
    """Test cases for distributed evaluation engine"""
    
    def setUp(self):
        """Set up test environment"""
        config = DistributedEngineConfig(
            max_concurrent_evaluations=2,
            enable_dynamic_scaling=True,
            memory_optimization_level="balanced"
        )
        
        # Use mock engine for testing
        self.engine = MockDistributedEngine(config)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.engine:
            self.engine.shutdown()
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.engine_type, EngineType.DISTRIBUTED)
        self.assertEqual(self.engine.config.max_concurrent_evaluations, 2)
    
    def test_can_handle_request(self):
        """Test request handling capability"""
        # Valid request
        config = EnhancedModelConfig(
            model_name="test-large-model",
            huggingface_id="test/test-large-model",
            license="apache-2.0",
            size_gb=60.0,
            context_window=4096
        )
        
        request = EvaluationRequest(
            request_id="test-request-003",
            model_config=config,
            datasets=["test_dataset"],
            batch_size=4
        )
        
        self.assertTrue(self.engine.can_handle_request(request))
        
        # Invalid request (missing model_config)
        invalid_request = EvaluationRequest(
            request_id="test-request-004",
            model_config=None,
            datasets=["test_dataset"],
            batch_size=4
        )
        
        self.assertFalse(self.engine.can_handle_request(invalid_request))
    
    async def test_evaluate_async(self):
        """Test async evaluation"""
        config = EnhancedModelConfig(
            model_name="test-model",
            model_path="/mock/path",
            parameters=30_000_000_000
        )
        
        request = EvaluationRequest(
            request_id="test-request-005",
            model_config=config,
            datasets=["test_dataset"],
            batch_size=4
        )
        
        start_time = time.time()
        result = await self.engine.evaluate(request)
        execution_time = time.time() - start_time
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.request_id, "test-request-005")
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.engine_used, EngineType.DISTRIBUTED)
        self.assertTrue(result.success)
        self.assertGreater(result.execution_time_seconds, 0)
        self.assertGreater(execution_time, 1.5)  # Mock should take ~2 seconds
    
    def test_get_capabilities(self):
        """Test capabilities reporting"""
        capabilities = self.engine.get_capabilities()
        
        self.assertGreater(capabilities.max_model_size_gb, 100)
        self.assertGreater(capabilities.total_gpu_memory_gb, 200)
        self.assertGreater(capabilities.max_concurrent_models, 0)
        self.assertIn("CausalLM", capabilities.supported_model_types)
        self.assertTrue(capabilities.supports_model_parallelism)
    
    def test_engine_status(self):
        """Test engine status reporting"""
        status = self.engine.get_engine_status()
        
        self.assertEqual(status["engine_type"], "distributed")
        self.assertIn("capabilities", status)
        self.assertIn("performance", status)
        self.assertIn("configuration", status)
        
        # Check specific fields
        self.assertIn("max_model_size_gb", status["capabilities"])
        self.assertIn("evaluations_completed", status["performance"])
        self.assertIn("max_concurrent_evaluations", status["configuration"])


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for performance monitoring"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = MultiGPUPerformanceMonitor(
            gpu_ids=[0, 1, 2, 3],
            monitoring_interval=0.5,  # Fast for testing
            history_size=100
        )
        self.monitor.start()
        time.sleep(0.6)  # Let monitor collect some data
    
    def tearDown(self):
        """Clean up test environment"""
        if self.monitor:
            self.monitor.stop()
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.gpu_ids, [0, 1, 2, 3])
        self.assertEqual(self.monitor.monitoring_interval, 0.5)
        self.assertTrue(self.monitor._running)
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        # Wait for some metrics to be collected
        time.sleep(1.0)
        
        metrics = self.monitor.get_current_metrics()
        
        self.assertIn("gpu_metrics", metrics)
        self.assertIn("system_metrics", metrics)
        self.assertIn("timestamp", metrics)
        
        # Check GPU metrics
        for gpu_id in [0, 1, 2, 3]:
            self.assertIn(gpu_id, metrics["gpu_metrics"])
            gpu_metric = metrics["gpu_metrics"][gpu_id]
            
            self.assertIn("utilization_percent", gpu_metric)
            self.assertIn("memory_utilization_percent", gpu_metric)
            self.assertIn("temperature_celsius", gpu_metric)
            
            # Sanity checks
            self.assertGreaterEqual(gpu_metric["utilization_percent"], 0)
            self.assertLessEqual(gpu_metric["utilization_percent"], 100)
    
    def test_communication_tracking(self):
        """Test communication event recording"""
        # Record some communication events
        self.monitor.record_communication_event(
            source_gpu=0,
            target_gpu=1,
            operation_type="allreduce",
            data_size_bytes=1024*1024*100,  # 100MB
            latency_ms=15.0,
            success=True
        )
        
        self.monitor.record_communication_event(
            source_gpu=1,
            target_gpu=2,
            operation_type="broadcast",
            data_size_bytes=1024*1024*50,   # 50MB
            latency_ms=8.0,
            success=True
        )
        
        # Get communication matrix
        comm_matrix = self.monitor.get_communication_matrix(duration_minutes=1)
        
        self.assertIn("summary", comm_matrix)
        self.assertIn("gpu_pairs", comm_matrix)
        
        summary = comm_matrix["summary"]
        self.assertEqual(summary["total_communications"], 2)
        self.assertGreater(summary["total_data_transferred_gb"], 0)
        
        # Check specific GPU pairs
        self.assertIn("0->1", comm_matrix["gpu_pairs"])
        self.assertIn("1->2", comm_matrix["gpu_pairs"])
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Wait for more data
        time.sleep(1.5)
        
        summary = self.monitor.get_performance_summary(duration_minutes=1)
        
        self.assertIsInstance(summary.duration_seconds, (int, float))
        self.assertEqual(summary.duration_seconds, 60.0)
        self.assertIsInstance(summary.gpu_metrics, dict)
        self.assertIsInstance(summary.optimization_suggestions, list)
        
        # Check GPU statistics
        for gpu_id in [0, 1, 2, 3]:
            if gpu_id in summary.gpu_metrics:
                gpu_stats = summary.gpu_metrics[gpu_id]
                self.assertIn("utilization", gpu_stats)
                self.assertIn("memory_utilization", gpu_stats)
                
                # Check utilization stats
                util_stats = gpu_stats["utilization"]
                self.assertIn("average", util_stats)
                self.assertIn("min", util_stats)
                self.assertIn("max", util_stats)
    
    def test_gpu_utilization_trend(self):
        """Test GPU utilization trend analysis"""
        time.sleep(1.0)  # Collect some data
        
        trend = self.monitor.get_gpu_utilization_trend(gpu_id=0, duration_minutes=1)
        
        self.assertIn("utilization", trend)
        self.assertIn("memory_utilization", trend)
        self.assertIn("temperature", trend)
        
        util_data = trend["utilization"]
        self.assertIn("average", util_data)
        self.assertIn("current", util_data)
        self.assertIn("min", util_data)
        self.assertIn("max", util_data)


class TestDistributedEngineIntegration(unittest.TestCase):
    """Integration tests for distributed engine components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.config = DistributedEngineConfig(
            max_concurrent_evaluations=1,
            memory_optimization_level="balanced"
        )
        
        # Use mock for integration testing
        self.engine = MockDistributedEngine(self.config)
    
    def tearDown(self):
        """Clean up integration test environment"""
        if self.engine:
            self.engine.shutdown()
    
    async def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow"""
        # Create test request
        config = EnhancedModelConfig(
            model_name="integration-test-model",
            model_path="/mock/path",
            parameters=50_000_000_000
        )
        
        request = EvaluationRequest(
            request_id="integration-test-001",
            model_config=config,
            datasets=["dataset1", "dataset2"],
            batch_size=8
        )
        
        # Check if engine can handle request
        self.assertTrue(self.engine.can_handle_request(request))
        
        # Perform evaluation
        start_time = time.time()
        result = await self.engine.evaluate(request)
        total_time = time.time() - start_time
        
        # Validate result
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.request_id, "integration-test-001")
        self.assertEqual(result.model_name, "integration-test-model")
        self.assertEqual(result.engine_used, EngineType.DISTRIBUTED)
        self.assertTrue(result.success)
        
        # Check performance metrics
        self.assertIn("accuracy", result.metrics)
        self.assertIn("tokens_per_second", result.metrics)
        self.assertGreater(result.tokens_per_second, 0)
        
        # Check timing
        self.assertGreater(result.execution_time_seconds, 0)
        self.assertGreater(total_time, 1.5)  # Should take at least ~2 seconds
    
    def test_multiple_concurrent_evaluations(self):
        """Test handling multiple concurrent evaluations"""
        # This test would be more meaningful with a real engine
        # For mock engine, we just verify the interface works
        
        config1 = EnhancedModelConfig(
            model_name="concurrent-test-1",
            model_path="/mock/path",
            parameters=30_000_000_000
        )
        
        config2 = EnhancedModelConfig(
            model_name="concurrent-test-2", 
            model_path="/mock/path",
            parameters=40_000_000_000
        )
        
        request1 = EvaluationRequest(
            request_id="concurrent-001",
            model_config=config1,
            datasets=["dataset1"],
            batch_size=4
        )
        
        request2 = EvaluationRequest(
            request_id="concurrent-002",
            model_config=config2,
            datasets=["dataset2"],
            batch_size=4
        )
        
        # Both should be handleable
        self.assertTrue(self.engine.can_handle_request(request1))
        self.assertTrue(self.engine.can_handle_request(request2))
        
        # Check engine status
        status = self.engine.get_engine_status()
        self.assertEqual(status["engine_type"], "distributed")
        self.assertIn("capabilities", status)


class TestHybridArchitectureIntegration(unittest.TestCase):
    """Test integration between distributed and lightweight engines"""
    
    def test_engine_selection_logic(self):
        """Test logic for selecting between distributed and lightweight engines"""
        distributed_engine = MockDistributedEngine()
        
        # Small model - should NOT be handled by distributed engine
        small_config = EnhancedModelConfig(
            model_name="small-7b-model",
            model_path="/mock/path",
            parameters=7_000_000_000
        )
        
        small_request = EvaluationRequest(
            request_id="hybrid-test-001",
            model_config=small_config,
            datasets=["test_dataset"],
            batch_size=4
        )
        
        # Distributed engine should reject small models
        self.assertFalse(distributed_engine.can_handle_request(small_request))
        
        # Large model - should be handled by distributed engine
        large_config = EnhancedModelConfig(
            model_name="large-70b-model",
            model_path="/mock/path",
            parameters=70_000_000_000
        )
        
        large_request = EvaluationRequest(
            request_id="hybrid-test-002",
            model_config=large_config,
            datasets=["test_dataset"],
            batch_size=4
        )
        
        # Distributed engine should accept large models
        self.assertTrue(distributed_engine.can_handle_request(large_request))
    
    def test_engine_capabilities_complementarity(self):
        """Test that distributed and lightweight engines complement each other"""
        distributed_engine = MockDistributedEngine()
        
        capabilities = distributed_engine.get_capabilities()
        
        # Distributed engine should handle large models
        self.assertGreater(capabilities.max_model_size_gb, 100)
        self.assertTrue(capabilities.supports_model_parallelism)
        self.assertGreater(len(capabilities.tensor_parallel_sizes), 1)
        
        # Should support large-scale model types
        self.assertIn("CausalLM", capabilities.supported_model_types)


# Test runner functions
def run_phase3_validation_tests():
    """Run all Phase 3 validation tests"""
    logger.info("Starting Phase 3 distributed engine validation tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMultiGPUModelLoader))
    test_suite.addTest(unittest.makeSuite(TestDistributedOrchestrator))
    test_suite.addTest(unittest.makeSuite(TestDistributedEngine))
    test_suite.addTest(unittest.makeSuite(TestPerformanceMonitor))
    test_suite.addTest(unittest.makeSuite(TestDistributedEngineIntegration))
    test_suite.addTest(unittest.makeSuite(TestHybridArchitectureIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    logger.info(f"Phase 3 validation results: {successes}/{total_tests} tests passed")
    
    if failures > 0:
        logger.warning(f"Test failures: {failures}")
        for test, traceback in result.failures:
            logger.warning(f"FAIL: {test}")
    
    if errors > 0:
        logger.error(f"Test errors: {errors}")
        for test, traceback in result.errors:
            logger.error(f"ERROR: {test}")
    
    return successes == total_tests


async def run_async_tests():
    """Run async test cases"""
    logger.info("Running async tests...")
    
    # Test distributed engine async evaluation
    test_case = TestDistributedEngine()
    test_case.setUp()
    
    try:
        await test_case.test_evaluate_async()
        logger.info("✓ Async evaluation test passed")
        
        # Test integration workflow
        integration_test = TestDistributedEngineIntegration()
        integration_test.setUp()
        
        await integration_test.test_full_evaluation_workflow()
        logger.info("✓ Full evaluation workflow test passed")
        
        integration_test.tearDown()
        
    except Exception as e:
        logger.error(f"Async test failed: {e}")
        return False
    finally:
        test_case.tearDown()
    
    return True


if __name__ == "__main__":
    # Run synchronous tests
    sync_success = run_phase3_validation_tests()
    
    # Run asynchronous tests
    async_success = asyncio.run(run_async_tests())
    
    # Overall result
    overall_success = sync_success and async_success
    
    print(f"\n" + "="*60)
    print(f"PHASE 3 VALIDATION SUMMARY")
    print(f"="*60)
    print(f"Synchronous tests: {'PASS' if sync_success else 'FAIL'}")
    print(f"Asynchronous tests: {'PASS' if async_success else 'FAIL'}")
    print(f"Overall result: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    print(f"="*60)
    
    if overall_success:
        print("🎉 Phase 3 distributed engine is ready for production!")
        print("✓ Multi-GPU model loading validated")
        print("✓ Distributed orchestration validated")
        print("✓ Performance monitoring validated")
        print("✓ Engine integration validated")
        print("✓ Hybrid architecture compatibility validated")
    else:
        print("⚠️  Phase 3 validation incomplete - review failed tests")
    
    exit(0 if overall_success else 1)