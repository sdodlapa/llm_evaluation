#!/usr/bin/env python3
"""
Quick test to verify HuggingFace authentication fix
"""
import os
import sys
sys.path.append('/home/sdodl001_odu_edu/llm_evaluation')

from evaluation.multi_backend_loader import MultiBackendModelLoader

def test_auth():
    print("=== Testing HuggingFace Authentication Fix ===")
    
    # Check environment
    hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    print(f"HF Token found: {'Yes' if hf_token else 'No'}")
    if hf_token:
        print(f"Token length: {len(hf_token)} characters")
    
    # Test model loading
    loader = MultiBackendModelLoader()
    
    # Test a gated model that was failing
    model_id = "llama31_70b"
    print(f"\nTesting model loading for: {model_id}")
    
    try:
        # This should use our fixed authentication
        from transformers import AutoConfig
        
        # Get model registry to find HuggingFace ID
        from evaluation.model_registry import get_model_registry
        registry = get_model_registry()
        
        if model_id in registry:
            model_config = registry[model_id]
            huggingface_id = getattr(model_config, 'huggingface_id', model_id)
            print(f"HuggingFace ID: {huggingface_id}")
            
            # Test config loading with authentication
            auth_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
            config = AutoConfig.from_pretrained(
                huggingface_id, 
                trust_remote_code=True,
                token=auth_token
            )
            print(f"✅ Model config loaded successfully!")
            print(f"Architecture: {getattr(config, 'architectures', ['Unknown'])}")
            
        else:
            print(f"❌ Model {model_id} not found in registry")
            
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False
    
    print("✅ Authentication fix appears to be working!")
    return True

if __name__ == "__main__":
    test_auth()