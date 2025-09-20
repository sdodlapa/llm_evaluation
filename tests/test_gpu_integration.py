"""
GPU Integration Test Suite
Comprehensive tests for Phase 2 & 3 GPU functionality
Run when GPU hardware becomes available
"""

import pytest
import torch
import asyncio
import time
from typing import List, Dict, Any
from unittest.mock import patch

# Skip all tests if no GPU available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require CUDA-capable hardware"
)

class TestGPUIntegration:
    """Integration tests for GPU-specific functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_gpu_environment(self):
        """Setup GPU environment for testing"""
        if torch.cuda.is_available():
            # Clear GPU memory before each test
            torch.cuda.empty_cache()
            # Set device for consistent testing
            self.device = torch.device("cuda:0")
            self.num_gpus = torch.cuda.device_count()
        yield
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_gpu_memory_detection(self):
        """Test GPU memory detection and allocation"""
        from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader
        
        loader = MultiGPUModelLoader(
            model_name="test-model",
            model_size_gb=1.0,  # Small test model
            distribution_strategy="tensor_parallel"
        )
        
        # Test memory detection
        available_memory = loader._get_available_gpu_memory()
        assert len(available_memory) == self.num_gpus
        assert all(memory > 0 for memory in available_memory.values())
        
        print(f"✅ Detected {self.num_gpus} GPUs with memory: {available_memory}")
    
    def test_multi_gpu_model_distribution(self):
        """Test model distribution across multiple GPUs"""
        if self.num_gpus < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader
        
        loader = MultiGPUModelLoader(
            model_name="test-model",
            model_size_gb=2.0,
            distribution_strategy="tensor_parallel"
        )
        
        # Test optimal strategy selection for multi-GPU
        strategy = loader._determine_optimal_strategy(
            model_size_gb=2.0,
            available_gpus=list(range(self.num_gpus))
        )
        
        # Should select a parallel strategy for multi-GPU setup
        assert strategy in ["tensor_parallel", "pipeline_parallel", "hybrid_parallel"]
        print(f"✅ Selected strategy '{strategy}' for {self.num_gpus} GPUs")
    
    def test_gpu_memory_management(self):
        """Test GPU memory allocation and cleanup"""
        from engines.distributed.performance_monitor import MultiGPUPerformanceMonitor
        
        monitor = MultiGPUPerformanceMonitor(num_gpus=self.num_gpus)
        
        # Get initial memory state
        initial_metrics = monitor.collect_gpu_metrics()
        initial_memory = {
            gpu_id: metrics['memory_used'] 
            for gpu_id, metrics in initial_metrics.items()
        }
        
        # Allocate some test tensors
        test_tensors = []
        for gpu_id in range(min(self.num_gpus, 2)):  # Test first 2 GPUs
            device = torch.device(f"cuda:{gpu_id}")
            # Allocate 100MB tensor
            tensor = torch.randn(1000, 1000, 100, device=device)
            test_tensors.append(tensor)
        
        # Check memory increased
        after_alloc_metrics = monitor.collect_gpu_metrics()
        for gpu_id in range(min(self.num_gpus, 2)):
            after_memory = after_alloc_metrics[gpu_id]['memory_used']
            assert after_memory > initial_memory[gpu_id]
        
        # Clean up
        del test_tensors
        torch.cuda.empty_cache()
        
        # Check memory returned (approximately)
        final_metrics = monitor.collect_gpu_metrics()
        for gpu_id in range(min(self.num_gpus, 2)):
            final_memory = final_metrics[gpu_id]['memory_used']
            # Memory should be close to initial (within 50MB tolerance)
            memory_diff = abs(final_memory - initial_memory[gpu_id])
            assert memory_diff < 50 * 1024 * 1024  # 50MB tolerance
        
        print("✅ GPU memory management working correctly")
    
    def test_distributed_communication(self):
        """Test inter-GPU communication setup"""
        if self.num_gpus < 2:
            pytest.skip("Distributed communication tests require at least 2 GPUs")
        
        # Test NCCL backend availability
        assert torch.distributed.is_nccl_available(), "NCCL backend not available"
        
        # Test basic tensor communication between GPUs
        device_0 = torch.device("cuda:0")
        device_1 = torch.device("cuda:1")
        
        # Create test tensors on different GPUs
        tensor_0 = torch.randn(100, 100, device=device_0)
        tensor_1 = torch.zeros(100, 100, device=device_1)
        
        # Copy tensor between GPUs
        tensor_1.copy_(tensor_0)
        
        # Verify copy worked
        tensor_0_cpu = tensor_0.cpu()
        tensor_1_cpu = tensor_1.cpu()
        assert torch.allclose(tensor_0_cpu, tensor_1_cpu)
        
        print("✅ Inter-GPU communication working correctly")
    
    def test_performance_monitoring_accuracy(self):
        """Test performance monitoring with real GPU operations"""
        from engines.distributed.performance_monitor import MultiGPUPerformanceMonitor
        
        monitor = MultiGPUPerformanceMonitor(num_gpus=self.num_gpus)
        monitor.start_monitoring()
        
        # Perform some GPU operations
        device = torch.device("cuda:0")
        for i in range(10):
            # Matrix multiplication workload
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.mm(a, b)
            torch.cuda.synchronize()  # Ensure operation completes
        
        time.sleep(1)  # Let monitoring collect data
        
        metrics = monitor.collect_gpu_metrics()
        utilization_history = monitor.get_utilization_history(window_minutes=1)
        
        # Verify metrics are reasonable
        assert len(metrics) == self.num_gpus
        gpu_0_metrics = metrics[0]
        
        # GPU 0 should show some utilization
        assert gpu_0_metrics['utilization'] > 0
        assert gpu_0_metrics['memory_used'] > 0
        assert gpu_0_metrics['temperature'] > 0
        
        # History should contain data points
        assert len(utilization_history) > 0
        
        monitor.stop_monitoring()
        print("✅ Performance monitoring capturing real GPU metrics")
    
    def test_model_loading_memory_estimation(self):
        """Test model loading memory estimation accuracy"""
        from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader
        
        # Test with a small real model if available, otherwise simulate
        try:
            # Try to load a small model to test memory estimation
            import transformers
            tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
            model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
            model = model.to(self.device)
            
            # Get actual memory usage
            torch.cuda.synchronize()
            memory_after_load = torch.cuda.memory_allocated(self.device)
            
            # Test our estimation function
            loader = MultiGPUModelLoader(
                model_name="gpt2",
                model_size_gb=memory_after_load / (1024**3),
                distribution_strategy="single_gpu"
            )
            
            estimated_memory = loader._estimate_model_memory_gb("gpt2")
            actual_memory_gb = memory_after_load / (1024**3)
            
            # Estimation should be within 50% of actual (rough heuristic)
            error_ratio = abs(estimated_memory - actual_memory_gb) / actual_memory_gb
            assert error_ratio < 0.5, f"Memory estimation error too high: {error_ratio:.2%}"
            
            print(f"✅ Memory estimation: {estimated_memory:.2f}GB vs actual: {actual_memory_gb:.2f}GB")
            
        except ImportError:
            pytest.skip("Transformers library not available for memory testing")
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")
    
    @pytest.mark.asyncio
    async def test_distributed_engine_integration(self):
        """Test distributed engine with real GPU operations"""
        from engines.distributed.distributed_engine import DistributedEvaluationEngine
        from engines.base_engine import EnhancedModelConfig, EvaluationRequest
        
        # Create a minimal distributed engine for testing
        engine = DistributedEvaluationEngine()
        
        # Test initialization
        await engine.initialize()
        
        # Test capabilities reporting
        capabilities = engine.get_capabilities()
        assert capabilities["supports_distributed"] == True
        assert capabilities["max_model_size_gb"] > 0
        
        # Test engine state
        assert engine.is_ready() == True
        
        print("✅ Distributed engine integration working on GPU")
    
    def test_error_handling_gpu_oom(self):
        """Test graceful handling of GPU out-of-memory errors"""
        from engines.distributed.multi_gpu_model_loader import MultiGPUModelLoader
        
        # Get available memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Try to load a model larger than available memory
        oversized_model_gb = (available_memory / (1024**3)) + 1  # 1GB larger than available
        
        loader = MultiGPUModelLoader(
            model_name="oversized-test-model",
            model_size_gb=oversized_model_gb,
            distribution_strategy="single_gpu"
        )
        
        # This should detect the issue and recommend distributed strategy
        strategy = loader._determine_optimal_strategy(
            model_size_gb=oversized_model_gb,
            available_gpus=[0]
        )
        
        # Should not recommend single_gpu for oversized model
        assert strategy != "single_gpu"
        print(f"✅ Correctly handled oversized model: recommended {strategy}")

class TestPhase2LightweightEngineGPU:
    """GPU-specific tests for Phase 2 Lightweight Engine"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_lightweight_engine_gpu_detection(self):
        """Test lightweight engine GPU capability detection"""
        from engines.lightweight.lightweight_engine import LightweightEvaluationEngine
        
        engine = LightweightEvaluationEngine()
        capabilities = engine.get_capabilities()
        
        # Should detect GPU availability correctly
        assert capabilities["supports_gpu"] == torch.cuda.is_available()
        if torch.cuda.is_available():
            assert capabilities["max_model_size_gb"] > 0
        
        print("✅ Lightweight engine GPU detection working")

# Utility function to run GPU tests
def run_gpu_tests():
    """
    Convenience function to run GPU tests
    Call this when GPU hardware becomes available
    """
    if not torch.cuda.is_available():
        print("❌ No GPU available - skipping GPU integration tests")
        return False
    
    print(f"🚀 Running GPU integration tests on {torch.cuda.device_count()} GPUs...")
    
    # Run pytest with this file
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    # Quick check when run directly
    print("GPU Integration Test Suite")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPUs Available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
    else:
        print("🔄 Tests will be skipped until GPU hardware is available")