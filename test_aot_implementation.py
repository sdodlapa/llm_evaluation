#!/usr/bin/env python3
"""
Test script for AOT (Ahead-of-Time) compilation implementation

This script tests the AOT compiler with a simple model to validate
the implementation works before testing with larger LLMs.
"""

import sys
import logging
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_aot_availability():
    """Test if AOT compilation is available in current environment"""
    logger.info("Testing AOT availability...")
    
    try:
        from engines.shared.aot_compiler import is_aot_available, get_aot_info
        
        aot_info = get_aot_info()
        logger.info(f"AOT Info: {aot_info}")
        
        if is_aot_available():
            logger.info("✅ AOT compilation is available!")
            return True
        else:
            logger.warning("❌ AOT compilation is not available")
            logger.info("Requirements:")
            logger.info(f"  - PyTorch >= 2.4.0: {aot_info['requirements_met']}")
            logger.info(f"  - torch.export: {aot_info['export_available']}")
            logger.info(f"  - torch._inductor: {aot_info['inductor_available']}")
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import AOT compiler: {e}")
        return False

def test_simple_model_compilation():
    """Test AOT compilation with a simple PyTorch model"""
    logger.info("Testing simple model compilation...")
    
    try:
        from engines.shared.aot_compiler import AOTModelCompiler
        
        # Create a simple test model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 1000)
                self.output = torch.nn.Linear(1000, 100)
            
            def forward(self, x):
                x = torch.relu(self.linear(x))
                return self.output(x)
        
        # Initialize model and compiler
        model = SimpleModel()
        if torch.cuda.is_available():
            model = model.cuda()
        
        compiler = AOTModelCompiler(
            cache_dir="test_cache/compiled",
            enable_aot=True
        )
        
        # Create example inputs
        example_input = torch.randn(1, 512)
        if torch.cuda.is_available():
            example_input = example_input.cuda()
        
        # Mock model config for testing
        class MockConfig:
            def __init__(self):
                self.model_name = "simple_test_model"
                self.size_gb = 0.1
                self.quantization_method = "none"
        
        config = MockConfig()
        
        # Test compilation
        logger.info("Starting AOT compilation of simple model...")
        compiled_model = compiler.compile_model_aot(
            model, 
            (example_input,), 
            config,
            compilation_mode="default"
        )
        
        if compiled_model is not None:
            logger.info("✅ Simple model compilation successful!")
            
            # Test inference
            with torch.no_grad():
                original_output = model(example_input)
                compiled_output = compiled_model(example_input)
                
                # Check outputs are similar
                diff = torch.max(torch.abs(original_output - compiled_output)).item()
                logger.info(f"Output difference: {diff:.6f}")
                
                if diff < 1e-3:
                    logger.info("✅ Compiled model outputs match original!")
                    return True
                else:
                    logger.warning(f"⚠️ Compiled model outputs differ by {diff}")
                    return False
        else:
            logger.error("❌ Simple model compilation failed")
            return False
            
    except Exception as e:
        logger.error(f"Simple model compilation test failed: {e}")
        logger.debug("Test error details:", exc_info=True)
        return False

def test_model_loader_integration():
    """Test AOT integration with the lightweight model loader"""
    logger.info("Testing model loader AOT integration...")
    
    try:
        from engines.lightweight.model_loader import LightweightModelLoader
        from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
        
        # Create model loader
        loader = LightweightModelLoader()
        
        # Check if AOT is enabled
        if hasattr(loader, 'aot_compiler') and loader.aot_compiler:
            logger.info("✅ AOT compiler integrated into model loader")
            
            # Check supported architectures
            supported = loader.aot_compiler.supported_architectures
            logger.info(f"Supported architectures: {supported}")
            
            return True
        else:
            logger.warning("⚠️ AOT compiler not integrated or not available")
            return False
            
    except Exception as e:
        logger.error(f"Model loader integration test failed: {e}")
        return False

def run_aot_tests():
    """Run all AOT tests"""
    logger.info("=== Starting AOT Implementation Tests ===")
    
    tests = [
        ("AOT Availability", test_aot_availability),
        ("Simple Model Compilation", test_simple_model_compilation),
        ("Model Loader Integration", test_model_loader_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All AOT tests passed! Implementation is working.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_aot_tests()
    sys.exit(0 if success else 1)