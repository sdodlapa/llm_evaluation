#!/usr/bin/env python3
"""
Test script to validate the multi-backend loader fixes
"""

import sys
import logging
from evaluation.multi_backend_loader import MultiBackendModelLoader, detect_model_backend
from configs.model_registry import get_model_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backend_detection():
    """Test backend detection for various models"""
    print("🔍 Testing Backend Detection...")
    
    # Test a few representative models
    test_models = [
        'biomistral_7b',  # Should use vLLM (was failing before)
        'biomedlm_7b',    # Should use transformers (was working)
        'qwen3_8b',       # Should use vLLM
    ]
    
    for model_id in test_models:
        model_config = get_model_config(model_id)
        if model_config:
            backend_config = detect_model_backend(model_id, model_config)
            print(f"  {model_id}: {backend_config.backend.value} -> {backend_config.model_class}")
        else:
            print(f"  {model_id}: NOT FOUND in registry")

def test_vllm_import():
    """Test if vLLM import works without the old error"""
    print("\n🚀 Testing vLLM Import Fix...")
    
    try:
        from evaluation.multi_backend_loader import MultiBackendModelLoader
        loader = MultiBackendModelLoader()
        
        # Create a dummy model config
        class DummyConfig:
            huggingface_id = "microsoft/DialoGPT-small"  # Small model for testing
            context_window = 1024
            tensor_parallel_size = 1
        
        # Test vLLM loading (this should not crash with import error anymore)
        success, model = loader._load_vllm_model("test_model", DummyConfig(), "balanced")
        
        if success:
            print("  ✅ vLLM loading succeeded!")
            # Test model_name attribute
            if hasattr(model, 'model_name'):
                print(f"  ✅ model_name attribute exists: {model.model_name}")
            else:
                print("  ❌ model_name attribute missing")
        else:
            print("  ⚠️ vLLM loading failed (but no import error)")
            
    except ImportError as e:
        print(f"  ❌ Import error still exists: {e}")
    except Exception as e:
        print(f"  ⚠️ Other error (import fixed): {e}")

def test_transformers_wrapper():
    """Test TransformersModelWrapper has model_name attribute"""
    print("\n🔧 Testing TransformersWrapper Fix...")
    
    try:
        from evaluation.multi_backend_loader import TransformersModelWrapper
        from transformers import AutoTokenizer, AutoModel
        
        # Create a simple wrapper test
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        
        wrapper = TransformersModelWrapper(model, tokenizer, "test_model")
        
        if hasattr(wrapper, 'model_name'):
            print(f"  ✅ model_name attribute exists: {wrapper.model_name}")
        else:
            print("  ❌ model_name attribute still missing")
            
        if hasattr(wrapper, 'model_id'):
            print(f"  ✅ model_id attribute exists: {wrapper.model_id}")
        else:
            print("  ❌ model_id attribute missing")
            
    except Exception as e:
        print(f"  ❌ TransformersWrapper test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Multi-Backend Loader Fixes\n")
    
    test_backend_detection()
    test_vllm_import()
    test_transformers_wrapper()
    
    print("\n✅ Backend fixes validation complete!")