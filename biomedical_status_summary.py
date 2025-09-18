#!/usr/bin/env python3
"""
Quick Biomedical Category Status Summary
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path  
sys.path.append('/home/sdodl001_odu_edu/llm_evaluation')

from configs.model_registry import MODEL_CONFIGS
from configs.biomedical_model_dataset_mappings import BIOMEDICAL_MODEL_MAPPINGS

def main():
    print("🧬 BIOMEDICAL CATEGORY STATUS REPORT")
    print("=" * 50)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count biomedical models
    biomedical_models = {}
    for model_id, config in MODEL_CONFIGS.items():
        if (hasattr(config, 'specialization_category') and 
            config.specialization_category in ['bioinformatics', 'biomedical']) or \
           any(term in model_id.lower() for term in ['biomedlm', 'medalpa', 'biogpt', 'bio_clinical', 'biomistral']):
            biomedical_models[model_id] = config
    
    # Check datasets
    dataset_base = Path("datasets/biomedical")
    datasets = {}
    for subdir in ["pubmedqa", "medqa", "bc5cdr", "ddi"]:
        path = dataset_base / subdir
        if path.exists():
            files = list(path.glob("*.json"))
            total_size = sum(f.stat().st_size for f in files)
            datasets[subdir] = {"files": len(files), "size_mb": total_size / 1024 / 1024}
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  Models Configured: {len(biomedical_models)}")
    print(f"  Models with Mappings: {len(BIOMEDICAL_MODEL_MAPPINGS)}")
    print(f"  Datasets Available: {len(datasets)}")
    
    print(f"\n🤖 BIOMEDICAL MODELS ({len(biomedical_models)}):")
    for model_id in sorted(biomedical_models.keys()):
        config = biomedical_models[model_id]
        name = getattr(config, 'model_name', model_id)
        print(f"  ✅ {model_id} ({name})")
    
    print(f"\n📁 DATASETS ({len(datasets)}):")
    for name, info in datasets.items():
        print(f"  ✅ {name}: {info['files']} files ({info['size_mb']:.1f} MB)")
    
    print(f"\n🔗 MODEL-DATASET MAPPINGS ({len(BIOMEDICAL_MODEL_MAPPINGS)}):")
    for model_id, mapping in BIOMEDICAL_MODEL_MAPPINGS.items():
        if model_id in biomedical_models:
            print(f"  ✅ {model_id} -> {', '.join(mapping.primary_datasets)}")
    
    print(f"\n🎯 EXPECTED PERFORMANCE:")
    performance_targets = {
        "biomistral_7b": "MedQA: 55.4% | PubMedQA: 81.7%",
        "biomedlm_7b": "MedQA: 50.3% | PubMedQA: 78.2%", 
        "medalpaca_7b": "MedQA: 44.1% | Clinical tasks: 72.8%",
        "biogpt": "NER F1: 85.6% | Relation Extraction: 79.3%"
    }
    
    for model_id, perf in performance_targets.items():
        if model_id in biomedical_models:
            print(f"  📈 {model_id}: {perf}")
    
    print(f"\n🚀 STATUS: READY FOR COMPREHENSIVE BIOMEDICAL EVALUATION")
    print(f"\nNext Steps:")
    print(f"  1. Run full evaluation on all 10 biomedical models")
    print(f"  2. Compare performance against benchmark targets")
    print(f"  3. Generate comprehensive biomedical category report")
    print(f"  4. Optionally expand to genomics/proteomics models")

if __name__ == "__main__":
    main()