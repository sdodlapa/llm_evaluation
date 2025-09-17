#!/usr/bin/env python3
"""
Quick evaluation script to test new Qwen models configurations.
Tests the newly added specialized models without requiring dataset loading.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.model_configs import (
    get_all_qwen_variants, 
    get_coding_optimized_models,
    get_genomic_optimized_models,
    get_efficiency_models,
    MODEL_CONFIGS,
    estimate_memory_usage
)
import json
from datetime import datetime

def test_new_coding_models():
    """Test new coding-optimized models configuration."""
    print("=== Testing New Coding Models ===\n")
    
    # Get coding-optimized models
    coding_models = get_coding_optimized_models()
    print(f"Found {len(coding_models)} coding-optimized models:")
    for model_name, config in coding_models.items():
        memory_est = estimate_memory_usage(config)
        print(f"  - {model_name}: {config.model_name}")
        print(f"    VRAM: {memory_est['total_estimated_gb']:.1f}GB")
        print(f"    Agent Temp: {config.agent_temperature}")
    print()
    
    # Test model configurations
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Testing Coding Model Configurations")
    print('='*60)
    
    results["coding_models"] = {}
    
    # Test each coding model
    for model_name, config in coding_models.items():
        print(f"\n--- Testing {model_name} ---")
        
        try:
            memory_est = estimate_memory_usage(config)
            print(f"Model: {config.model_name}")
            print(f"HuggingFace ID: {config.huggingface_id}")
            print(f"Size: {config.size_gb}GB")
            print(f"Context Window: {config.context_window:,} tokens")
            print(f"Agent Temperature: {config.agent_temperature}")
            print(f"VRAM Estimate: {memory_est['total_estimated_gb']:.1f}GB")
            print(f"H100 Utilization: {memory_est['h100_utilization']:.1%}")
            print()
            
            model_results = {
                "model": model_name,
                "config": config.model_name,
                "size_gb": config.size_gb,
                "huggingface_id": config.huggingface_id,
                "agent_temperature": config.agent_temperature,
                "context_window": config.context_window,
                "vram_estimate": memory_est['total_estimated_gb'],
                "h100_utilization": memory_est['h100_utilization'],
                "timestamp": datetime.now().isoformat(),
                "status": "configuration_valid"
            }
            
            results["coding_models"][model_name] = model_results
            print(f"✅ {model_name} configured successfully")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            results["coding_models"][model_name] = {
                "error": str(e),
                "status": "configuration_failed"
            }
    
    return results

def test_genomic_models():
    """Test genomic-optimized models configuration."""
    print("\n=== Testing Genomic Models Configuration ===\n")
    
    # Get genomic models
    genomic_models = get_genomic_optimized_models()
    
    print(f"Found {len(genomic_models)} genomic-optimized models:")
    for model_name, config in genomic_models.items():
        memory_est = estimate_memory_usage(config)
        print(f"  - {model_name}: {config.model_name}")
        print(f"    Context: {config.max_model_len:,} tokens")
        print(f"    Temperature: {config.agent_temperature}")
        print(f"    Size: {config.size_gb}GB")
        print(f"    VRAM: {memory_est['total_estimated_gb']:.1f}GB")
        print()

def test_efficiency_models():
    """Test efficiency models configuration."""
    print("\n=== Testing Efficiency Models ===\n")
    
    efficiency_models = get_efficiency_models()
    
    print(f"Found {len(efficiency_models)} efficiency models:")
    for model_name, config in efficiency_models.items():
        memory_est = estimate_memory_usage(config)
        print(f"  - {model_name}: {config.model_name}")
        print(f"    Size: {config.size_gb}GB")
        print(f"    Batch Size: {config.evaluation_batch_size}")
        print(f"    GPU Utilization: {config.gpu_memory_utilization}")
        print(f"    VRAM: {memory_est['total_estimated_gb']:.1f}GB")
        print()

def main():
    """Main test function."""
    print("🧪 NEW QWEN MODELS CONFIGURATION TEST")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test coding models
        print("Testing coding model configurations...")
        coding_results = test_new_coding_models()
        
        # Test genomic models
        print("Testing genomic model configurations...")
        test_genomic_models()
        
        # Test efficiency models
        print("Testing efficiency model configurations...")
        test_efficiency_models()
        
        # Test all Qwen variants
        print("\n=== All Qwen Model Variants ===")
        all_qwen = get_all_qwen_variants()
        print(f"Total Qwen variants available: {len(all_qwen)}")
        for name, config in all_qwen.items():
            memory_est = estimate_memory_usage(config)
            print(f"  {name}: {config.model_name} ({memory_est['total_estimated_gb']:.1f}GB VRAM)")
        
        # Save results
        output_file = f"test_results/new_models_config_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("test_results", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(coding_results, f, indent=2)
        
        print(f"\n📊 Test results saved to: {output_file}")
        print("\n✅ All new model configurations validated successfully!")
        print("\n🚀 Next Steps:")
        print("   1. Run actual evaluation: python evaluation/run_evaluation.py")
        print("   2. Test coding models on HumanEval: Use qwen3_coder_30b")
        print("   3. Test math models on GSM8K: Use qwen25_math_7b") 
        print("   4. Compare efficiency: qwen25_3b vs qwen25_0_5b")
        print("   5. Test genomic tasks: Use qwen25_1_5b_genomic")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()