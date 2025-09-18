#!/usr/bin/env python3
"""
Model Summary Generator
Quick utility to display comprehensive model information and model-dataset mappings
"""

from configs.model_configs import MODEL_CONFIGS, get_coding_optimized_models, get_mathematical_reasoning_models
from evaluation.dataset_manager import EnhancedDatasetManager
from collections import defaultdict

def print_model_summary():
    """Print comprehensive model summary"""
    print("🤖 LLM EVALUATION FRAMEWORK - MODEL SUMMARY")
    print("=" * 60)
    
    # Overall stats
    print(f"\n🔢 OVERALL STATISTICS:")
    print(f"   • Total models: {len(MODEL_CONFIGS)}")
    
    # Group by size
    by_size = defaultdict(list)
    by_license = defaultdict(list)
    
    for name, config in MODEL_CONFIGS.items():
        # Determine size category
        if config.size_gb >= 30:
            size_cat = "Large (30B+)"
        elif config.size_gb >= 14:
            size_cat = "Medium (14-16B)"
        elif config.size_gb >= 7:
            size_cat = "Small-Medium (7-8B)"
        elif config.size_gb >= 3:
            size_cat = "Small (3-4B)"
        else:
            size_cat = "Tiny (<3B)"
            
        by_size[size_cat].append(name)
        by_license[config.license].append(name)
    
    print(f"\n📏 BY MODEL SIZE:")
    for size_cat, models in sorted(by_size.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   • {size_cat}: {len(models)} models")
        examples = sorted(models)[:2]
        print(f"     Examples: {', '.join(examples)}")
    
    print(f"\n📋 BY LICENSE TYPE:")
    for license_type, models in sorted(by_license.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   • {license_type}: {len(models)} models")
    
    # Specialization categories
    coding_models = get_coding_optimized_models()
    math_models = get_mathematical_reasoning_models()
    
    print(f"\n🎯 SPECIALIZATION CATEGORIES:")
    print(f"   • Coding optimized: {len(coding_models)} models")
    print(f"     {list(coding_models.keys())[:3]}...")
    print(f"   • Mathematical reasoning: {len(math_models)} models")
    print(f"     {list(math_models.keys())[:3]}...")
    
    print(f"\n💡 TIP: See MODEL_SUMMARY_TABLE.md for complete model catalog")

def print_model_dataset_mapping():
    """Print model to dataset category mappings"""
    print("🎯 MODEL-DATASET MAPPING ANALYSIS")
    print("=" * 50)
    
    dm = EnhancedDatasetManager()
    summary = dm.get_dataset_summary()
    
    # Get model categories
    coding_models = list(get_coding_optimized_models().keys())
    math_models = list(get_mathematical_reasoning_models().keys())
    
    # Get dataset categories
    dataset_categories = defaultdict(list)
    for name, info in summary['datasets'].items():
        dataset_categories[info['task_type']].append(name)
    
    print(f"\n💻 CODING MODELS → CODING DATASETS:")
    coding_datasets = dataset_categories.get('coding', [])
    ready_coding = [d for d in coding_datasets if summary['datasets'][d]['implemented']]
    print(f"   Models: {len(coding_models)} ({', '.join(coding_models[:2])}...)")
    print(f"   Datasets: {len(coding_datasets)} total, {len(ready_coding)} ready")
    print(f"   Ready combinations: {len(coding_models)} × {len(ready_coding)} = {len(coding_models) * len(ready_coding)}")
    
    print(f"\n🧮 MATH MODELS → MATH/REASONING DATASETS:")
    math_datasets = dataset_categories.get('mathematics', []) + dataset_categories.get('reasoning', [])
    ready_math = [d for d in math_datasets if summary['datasets'][d]['implemented']]
    print(f"   Models: {len(math_models)} ({', '.join(math_models[:2])}...)")
    print(f"   Datasets: {len(math_datasets)} total, {len(ready_math)} ready")
    print(f"   Ready combinations: {len(math_models)} × {len(ready_math)} = {len(math_models) * len(ready_math)}")
    
    print(f"\n🎯 OPTIMAL EVALUATION STRATEGIES:")
    print(f"   • Ready now: {len(ready_coding + ready_math)} specialized datasets")
    print(f"   • Full potential: {sum(len(datasets) for datasets in dataset_categories.values())} total combinations")
    print(f"   • See MODEL_DATASET_MAPPING.md for detailed strategies")

def print_category_details(category: str):
    """Print detailed info for specific model category"""
    category = category.lower()
    
    if category == "coding":
        models = get_coding_optimized_models()
        print(f"💻 CODING OPTIMIZED MODELS ({len(models)})")
        print("=" * 40)
        for name, config in models.items():
            print(f"• {name}: {config.size_gb}GB, {config.context_window} context")
            print(f"  License: {config.license}")
            print(f"  Use cases: {', '.join(config.primary_use_cases)}")
            print()
    
    elif category == "math":
        models = get_mathematical_reasoning_models()
        print(f"🧮 MATHEMATICAL REASONING MODELS ({len(models)})")
        print("=" * 40)
        for name, config in models.items():
            print(f"• {name}: {config.size_gb}GB, {config.context_window} context")
            print(f"  License: {config.license}")
            print(f"  Use cases: {', '.join(config.primary_use_cases)}")
            print()
    
    else:
        print(f"❌ Category '{category}' not recognized")
        print("Available categories: coding, math")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "mapping":
            print_model_dataset_mapping()
        elif command in ["coding", "math"]:
            print_category_details(command)
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: mapping, coding, math")
    else:
        print_model_summary()
        print(f"\n💡 ADDITIONAL COMMANDS:")
        print(f"   python show_models.py mapping    # Model-dataset mappings")
        print(f"   python show_models.py coding     # Coding model details")
        print(f"   python show_models.py math       # Math model details")