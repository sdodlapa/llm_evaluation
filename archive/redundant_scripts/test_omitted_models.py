#!/usr/bin/env python3
"""
Individual Model Testing Script
==============================

Test the 2 omitted models (qwen3_coder_30b, deepseek_coder_16b) individually 
to diagnose and fix their AWQ quantization configuration issues.
"""

import sys
import os

# Add the project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_individual_model(model_name):
    """Test loading a single model to diagnose issues"""
    
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_name}")
    print(f"{'='*60}")
    
    try:
        from models.registry import create_model_from_config
        
        print(f"🔄 Attempting to create {model_name}...")
        
        # Try to create the model
        model = create_model_from_config(model_name, preset="balanced")
        
        if model and hasattr(model, 'loaded') and model.loaded:
            print(f"✅ {model_name} loaded successfully!")
            
            # Try a simple test generation
            print(f"🧪 Testing basic generation...")
            test_prompt = "def fibonacci(n):"
            result = model.generate([test_prompt], max_tokens=100)
            print(f"✅ Generation test passed!")
            print(f"Sample output: {result[0][:100]}...")
            
            return True
        else:
            print(f"❌ {model_name} failed to load properly")
            return False
            
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_awq_issues():
    """Diagnose AWQ quantization configuration issues"""
    
    print("\n🔍 DIAGNOSING AWQ QUANTIZATION ISSUES")
    print("="*60)
    
    # Test the problematic models
    models_to_test = [
        "qwen3_coder_30b",
        "deepseek_coder_16b"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n📋 Testing {model_name}...")
        success = test_individual_model(model_name)
        results[model_name] = success
        
        if not success:
            print(f"🔧 Checking model configuration for {model_name}...")
            try:
                from configs.model_configs import MODEL_CONFIGS
                config = MODEL_CONFIGS.get(model_name)
                if config:
                    print(f"   Model path: {config.get('model_path', 'Not specified')}")
                    print(f"   Quantization: {config.get('quantization', 'None')}")
                    print(f"   AWQ config: {config.get('awq_config', 'None')}")
                else:
                    print(f"   ❌ No configuration found for {model_name}")
            except Exception as e:
                print(f"   ❌ Error checking config: {e}")
    
    return results

def fix_awq_configurations():
    """Attempt to fix AWQ configuration issues"""
    
    print("\n🔧 FIXING AWQ CONFIGURATIONS")
    print("="*60)
    
    # Strategy: Try loading without AWQ quantization first
    print("Strategy: Remove AWQ quantization and use standard loading")
    
    # This would involve modifying the model configs to disable AWQ
    # and use standard model loading instead
    
    pass

def main():
    """Main execution"""
    
    print("🚀 INDIVIDUAL MODEL DIAGNOSIS AND REPAIR")
    print("="*80)
    
    # Step 1: Diagnose the issues
    results = diagnose_awq_issues()
    
    print(f"\n📊 DIAGNOSIS RESULTS:")
    print("-" * 40)
    for model, success in results.items():
        status = "✅ Working" if success else "❌ Failed"
        print(f"{model:20}: {status}")
    
    # Step 2: Suggest fixes
    failed_models = [model for model, success in results.items() if not success]
    
    if failed_models:
        print(f"\n🔧 SUGGESTED FIXES for {len(failed_models)} failed models:")
        print("-" * 50)
        print("1. Remove AWQ quantization from model configs")
        print("2. Use standard bf16/fp16 loading instead") 
        print("3. Adjust GPU memory allocation if needed")
        print("4. Test with simpler quantization methods")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Run this diagnostic script")
        print("2. Apply fixes to model configurations") 
        print("3. Re-test individual models")
        print("4. Add back to category evaluation")
        print("5. Run full 5-model evaluation")

if __name__ == "__main__":
    main()