#!/usr/bin/env python3
"""
Test script for September 2025 Qwen models
Tests configuration and basic functionality for the new models
"""

import sys
import os
sys.path.append('/home/sdodl001_odu_edu/llm_evaluation')

def test_new_qwen_models():
    """Test that the new September 2025 Qwen models are properly configured"""
    
    try:
        from configs.model_configs import MODEL_CONFIGS
        from configs.preset_configs import QWEN_MODEL_CONFIGS
        
        # List of new September 2025 models
        new_models = [
            "qwen3_next_80b",
            "qwen3_30b_2507", 
            "qwen3_4b_2507"
        ]
        
        print("=== Testing September 2025 Qwen Models ===")
        print()
        
        # Test 1: Check models exist in QWEN_MODEL_CONFIGS
        print("✅ Test 1: Models in QWEN_MODEL_CONFIGS")
        for model_name in new_models:
            if model_name in QWEN_MODEL_CONFIGS:
                config = QWEN_MODEL_CONFIGS[model_name]
                print(f"  ✅ {model_name}: {config.model_name}")
                print(f"     HF ID: {config.huggingface_id}")
                print(f"     Size: {config.size_gb}GB")
                print(f"     Context: {config.context_window}")
                print(f"     Priority: {config.priority}")
                if hasattr(config, 'tensor_parallel_size'):
                    print(f"     GPU Requirements: {config.tensor_parallel_size} GPUs")
                print()
            else:
                print(f"  ❌ {model_name}: NOT FOUND in QWEN_MODEL_CONFIGS")
        
        # Test 2: Check models exist in merged MODEL_CONFIGS
        print("✅ Test 2: Models in main MODEL_CONFIGS")
        for model_name in new_models:
            if model_name in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_name]
                print(f"  ✅ {model_name}: Available in main system")
            else:
                print(f"  ❌ {model_name}: NOT FOUND in main MODEL_CONFIGS")
        
        # Test 3: Validate model properties for 8-GPU constraint
        print("✅ Test 3: 8-GPU Constraint Validation")
        for model_name in new_models:
            if model_name in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_name]
                
                # Check GPU requirements
                tp_size = getattr(config, 'tensor_parallel_size', 1)
                if tp_size <= 8:
                    print(f"  ✅ {model_name}: {tp_size} GPUs (within 8-GPU limit)")
                else:
                    print(f"  ❌ {model_name}: {tp_size} GPUs (exceeds 8-GPU limit)")
                
                # Check if model is agent optimized 
                if config.agent_optimized:
                    print(f"     🤖 Agent-capable model")
                
                # Check context window
                if config.context_window >= 128000:
                    print(f"     📖 Long-context: {config.context_window} tokens")
        
        print()
        print("=== Summary ===")
        found_models = [m for m in new_models if m in MODEL_CONFIGS]
        print(f"✅ Successfully configured: {len(found_models)}/{len(new_models)} models")
        
        if len(found_models) == len(new_models):
            print("🎉 All September 2025 Qwen models ready for testing!")
            return True
        else:
            print("⚠️  Some models missing from configuration")
            return False
            
    except Exception as e:
        print(f"❌ Error testing models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """Show how to use the new models"""
    print()
    print("=== Usage Examples ===")
    print()
    print("1. Test small model first:")
    print("   python category_evaluation.py --model qwen3_4b_2507 --samples 5 --preset performance")
    print()
    print("2. Test medium model (2-4 GPUs):")
    print("   python category_evaluation.py --model qwen3_30b_2507 --samples 5 --preset performance")
    print()
    print("3. Test large model (4 GPUs):")
    print("   python category_evaluation.py --model qwen3_next_80b --samples 3 --preset performance")
    print()
    print("4. Submit SLURM job for testing:")
    print("   sbatch slurm_jobs/distributed_4gpu_quick_test.slurm")
    print()

if __name__ == "__main__":
    success = test_new_qwen_models()
    
    if success:
        show_usage_examples()
        sys.exit(0)
    else:
        print("❌ Model configuration test failed")
        sys.exit(1)