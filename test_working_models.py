#!/usr/bin/env python3
"""
Test Working Models
===================

Quick test to identify models that can be loaded successfully.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.model_configs import MODEL_CONFIGS
from evaluation.mappings.model_categories import CATEGORY_REGISTRY
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def test_model_availability(model_id: str, config) -> bool:
    """Test if a model can be loaded without authentication issues"""
    try:
        # Extract model path/ID
        huggingface_id = getattr(config, 'huggingface_id', None)
        model_name = getattr(config, 'model_name', model_id)
        
        if not huggingface_id:
            print(f"❌ {model_id}: No huggingface_id specified")
            return False
            
        # Check for models that typically require authentication
        auth_required_patterns = [
            'meta-llama', 'mistralai', 'google/gemma', 'nvidia/Mistral-NeMo',
            '01-ai/Yi-1.5-34B-Chat-AWQ', 'allenai/OLMo-2'
        ]
        
        for pattern in auth_required_patterns:
            if pattern in huggingface_id:
                print(f"❌ {model_id}: Requires authentication ({huggingface_id})")
                return False
        
        # Check for models that are likely to work
        working_patterns = [
            'Qwen/', 'microsoft/', 'openai-community/', 'unsloth/',
            'sentence-transformers/', 'allenai/scibert'
        ]
        
        for pattern in working_patterns:
            if pattern in huggingface_id:
                print(f"✅ {model_id}: Likely to work ({huggingface_id})")
                return True
        
        print(f"⚠️  {model_id}: Unknown status ({huggingface_id})")
        return False
        
    except Exception as e:
        print(f"❌ {model_id}: Error checking - {e}")
        return False

def main():
    print("=== TESTING MODEL AVAILABILITY ===\n")
    
    working_models_by_category = {}
    
    for category_name, category in CATEGORY_REGISTRY.items():
        print(f"\n📂 {category_name.upper()}")
        print("=" * 50)
        
        models = category.get('models', [])
        working_models = []
        
        for model_id in models:
            if model_id in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_id]
                if test_model_availability(model_id, config):
                    working_models.append(model_id)
            else:
                print(f"❌ {model_id}: Not found in MODEL_CONFIGS")
        
        working_models_by_category[category_name] = working_models
        print(f"\n✅ Working models in {category_name}: {len(working_models)}/{len(models)}")
        if working_models:
            for model in working_models:
                print(f"   - {model}")
    
    print("\n" + "="*70)
    print("SUMMARY OF WORKING MODELS")
    print("="*70)
    
    total_working = 0
    for category, models in working_models_by_category.items():
        total_working += len(models)
        if models:
            print(f"\n{category}: {len(models)} working models")
            for model in models:
                print(f"  - {model}")
    
    print(f"\nTotal working models: {total_working}")
    
    # Generate a quick test script
    if total_working > 0:
        print("\n" + "="*70)
        print("SUGGESTED TEST COMMAND")
        print("="*70)
        
        # Pick one working model from each category
        for category, models in working_models_by_category.items():
            if models:
                model = models[0]  # Pick first working model
                print(f"# Test {category}")
                print(f"crun -p ~/envs/llm_env python category_evaluation.py --category {category} --samples 2 --preset balanced --dry-run")
                break

if __name__ == "__main__":
    main()