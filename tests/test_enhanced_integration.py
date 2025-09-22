"""
Test the clean integration of enhanced AOT compilation

This script validates that the vLLM optimizations integrate cleanly
without breaking existing functionality.
"""

import torch
import logging
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.shared import (
    create_enhanced_compiler,
    performance_monitor,
    monitor_inference,
    get_optimization_summary
)

def test_basic_integration():
    """Test basic integration without existing compiler"""
    print("Testing basic enhanced compiler integration...")
    
    # Create enhanced compiler without base compiler
    compiler = create_enhanced_compiler(
        base_compiler=None,
        enable_cuda_graphs=True,
        batch_sizes=[1, 2, 4]
    )
    
    # Create simple test model
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 32)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        example_inputs = (torch.randn(2, 64).cuda(),)
        print("✓ Using CUDA")
    else:
        example_inputs = (torch.randn(2, 64),)
        print("⚠ Using CPU (CUDA graphs disabled)")
    
    # Test compilation
    try:
        compiled_model = compiler.compile_model(
            model=model,
            example_inputs=example_inputs,
            model_id="test_model"
        )
        print("✓ Model compilation successful")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return False
    
    # Test inference
    @monitor_inference
    def run_test_inference(model, inputs):
        return model(inputs)
    
    test_input = example_inputs[0]
    
    try:
        # Run multiple inferences
        for i in range(5):
            output = run_test_inference(compiled_model, test_input)
            assert output.shape == (2, 32), f"Wrong output shape: {output.shape}"
        
        print("✓ Inference successful")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    # Test stats collection
    try:
        stats = compiler.get_optimization_stats()
        print(f"✓ Stats collection successful: {len(stats)} categories")
        
        monitor_stats = performance_monitor.get_summary()
        print(f"✓ Performance monitoring: {monitor_stats['total_calls']} calls tracked")
    except Exception as e:
        print(f"✗ Stats collection failed: {e}")
        return False
    
    return True

def test_graceful_fallbacks():
    """Test that the system handles failures gracefully"""
    print("\nTesting graceful fallback behavior...")
    
    compiler = create_enhanced_compiler()
    
    # Test with problematic model that might fail compilation
    class ProblematicModel(torch.nn.Module):
        def forward(self, x):
            # This might cause compilation issues in some cases
            return x.sum() / x.numel()
    
    model = ProblematicModel()
    example_inputs = (torch.randn(4, 8),)
    
    try:
        compiled_model = compiler.compile_model(
            model=model,
            example_inputs=example_inputs,
            model_id="problematic_model"
        )
        
        # Even if compilation has issues, we should get a working model
        output = compiled_model(example_inputs[0])
        print("✓ Graceful fallback working")
        return True
    except Exception as e:
        print(f"✗ Fallback failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring functionality"""
    print("\nTesting performance monitoring...")
    
    # Reset monitor
    performance_monitor.reset_metrics()
    
    @monitor_inference
    def dummy_operation():
        import time
        time.sleep(0.01)  # Simulate work
        return "done"
    
    # Run monitored operations
    for i in range(3):
        dummy_operation()
    
    # Check metrics
    metrics = performance_monitor.get_metrics("inference")
    
    if metrics and metrics.get("call_count") == 3:
        print("✓ Performance monitoring working")
        print(f"  Calls: {metrics['call_count']}")
        print(f"  Avg time: {metrics['avg_time']:.4f}s")
        return True
    else:
        print(f"✗ Performance monitoring failed: {metrics}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("Running Enhanced AOT Compilation Integration Tests")
    print("=" * 50)
    
    tests = [
        test_basic_integration,
        test_graceful_fallbacks, 
        test_performance_monitoring
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed! Integration is working cleanly.")
        
        # Show optimization summary
        summary = get_optimization_summary()
        print(f"\nOptimization Summary:")
        print(f"Performance monitoring: {summary['performance_monitoring']['total_operations']} operations tracked")
        print(f"CUDA graphs: {summary['cuda_graphs']}")
        
        return True
    else:
        print("✗ Some tests failed. Check the integration.")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_integration_tests()
    sys.exit(0 if success else 1)