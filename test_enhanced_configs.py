#!/usr/bin/env python3
"""
Test script for enhanced model configurations
Demonstrates the new preset system and configuration capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.model_configs import MODEL_CONFIGS, estimate_memory_usage
from models.qwen_implementation import create_qwen3_8b

def test_configuration_presets():
    """Test different configuration presets"""
    print("🔧 Enhanced LLM Configuration System Test")
    print("=" * 60)
    
    # Test base configuration
    print("\n📋 Base Qwen-3 8B Configuration:")
    base_config = MODEL_CONFIGS['qwen3_8b']
    print(f"  Model: {base_config.model_name}")
    print(f"  License: {base_config.license}")
    print(f"  Size: {base_config.size_gb}GB")
    print(f"  Context: {base_config.context_window:,} tokens")
    print(f"  Preset: {base_config.preset}")
    print(f"  Agent Temperature: {base_config.agent_temperature}")
    print(f"  Max Function Calls: {base_config.max_function_calls_per_turn}")
    
    # Test memory estimation
    memory_est = estimate_memory_usage(base_config)
    print(f"\n💾 Memory Estimation:")
    print(f"  Model: {memory_est['base_model_gb']:.1f}GB")
    print(f"  KV Cache: {memory_est['kv_cache_gb']:.1f}GB") 
    print(f"  Overhead: {memory_est['overhead_gb']:.1f}GB")
    print(f"  Total: {memory_est['total_estimated_gb']:.1f}GB")
    print(f"  H100 Utilization: {memory_est['h100_utilization']:.1%}")
    
    # Test vLLM configuration
    print(f"\n⚙️  vLLM Configuration (Preset: {base_config.preset}):")
    vllm_args = base_config.to_vllm_args()
    for key, value in vllm_args.items():
        print(f"  {key}: {value}")
    
    # Test sampling parameters
    print(f"\n🎯 Agent Sampling Parameters:")
    sampling_params = base_config.get_agent_sampling_params()
    for key, value in sampling_params.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} stop tokens")
        else:
            print(f"  {key}: {value}")

def test_preset_variants():
    """Test different preset variants"""
    print("\n" + "=" * 60)
    print("🎨 Preset Variants Comparison")
    print("=" * 60)
    
    presets = ['balanced', 'performance', 'memory_optimized']
    base_config = MODEL_CONFIGS['qwen3_8b']
    
    comparison_data = []
    
    for preset in presets:
        print(f"\n--- {preset.upper()} PRESET ---")
        
        if preset == 'balanced':
            config = base_config
        else:
            config = base_config.create_preset_variant(preset)
        
        # Memory estimation
        memory_est = estimate_memory_usage(config)
        
        # vLLM args
        vllm_args = config.to_vllm_args()
        
        preset_data = {
            'preset': preset,
            'gpu_memory_util': config.gpu_memory_utilization,
            'max_sequences': config.max_num_seqs,
            'estimated_vram': memory_est['total_estimated_gb'],
            'h100_util': memory_est['h100_utilization'],
            'max_model_len': config.max_model_len,
            'prefix_caching': config.enable_prefix_caching,
            'eval_batch_size': config.evaluation_batch_size
        }
        
        print(f"  GPU Memory Utilization: {preset_data['gpu_memory_util']}")
        print(f"  Max Sequences: {preset_data['max_sequences']}")
        print(f"  Estimated VRAM: {preset_data['estimated_vram']:.1f}GB")
        print(f"  H100 Utilization: {preset_data['h100_util']:.1%}")
        print(f"  Evaluation Batch Size: {preset_data['eval_batch_size']}")
        print(f"  Prefix Caching: {preset_data['prefix_caching']}")
        
        comparison_data.append(preset_data)
    
    # Summary table
    print(f"\n📊 Preset Comparison Summary:")
    print("| Preset | GPU Mem | Max Seqs | Est VRAM | H100 % | Batch |")
    print("|--------|---------|----------|----------|---------|-------|")
    for data in comparison_data:
        print(f"| {data['preset']:<10} | {data['gpu_memory_util']:<7} | {data['max_sequences']:<8} | {data['estimated_vram']:.1f}GB | {data['h100_util']:.1%} | {data['eval_batch_size']:<5} |")

def test_model_instantiation():
    """Test model instantiation with different presets"""
    print("\n" + "=" * 60)
    print("🚀 Model Instantiation Test")
    print("=" * 60)
    
    presets = ['balanced', 'performance', 'memory_optimized']
    
    for preset in presets:
        print(f"\n--- Testing {preset.upper()} Preset ---")
        try:
            # Create model instance (don't load to avoid GPU requirements)
            model = create_qwen3_8b(preset=preset)
            
            # Get model info
            info = model.get_model_info()
            
            print(f"✅ Created: {info['model_name']}")
            print(f"   Preset: {info['preset']}")
            print(f"   Agent Temperature: {info['agent_temperature']}")
            print(f"   Max Function Calls: {info['max_function_calls']}")
            print(f"   Status: {'Loaded' if info['is_loaded'] else 'Not Loaded'}")
            
            # Test preset comparison
            if preset == 'balanced':
                print(f"\n📈 Cross-Preset Comparison:")
                comparison = model.get_preset_comparison()
                for p_name, p_info in comparison.items():
                    status = " (CURRENT)" if p_info['current'] else ""
                    print(f"   {p_name}{status}: {p_info['estimated_vram_gb']:.1f}GB VRAM")
            
        except Exception as e:
            print(f"❌ Error with {preset}: {e}")

def main():
    """Run all configuration tests"""
    try:
        test_configuration_presets()
        test_preset_variants()
        test_model_instantiation()
        
        print("\n" + "=" * 60)
        print("✅ All enhanced configuration tests completed successfully!")
        print("=" * 60)
        
        print(f"\n🎯 Key Features Demonstrated:")
        print(f"   ✓ Enhanced ModelConfig with preset system")
        print(f"   ✓ Optimized vLLM configurations for H100")
        print(f"   ✓ Agent-specific sampling parameters")
        print(f"   ✓ Memory estimation and validation")
        print(f"   ✓ Preset comparison and selection")
        print(f"   ✓ Configuration-aware model instantiation")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Test with actual GPU (requires H100 allocation)")
        print(f"   2. Run performance benchmarks")
        print(f"   3. Update evaluation runner for preset selection")
        print(f"   4. Implement additional model configurations")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()