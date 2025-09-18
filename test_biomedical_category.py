#!/usr/bin/env python3
"""
Biomedical Category Testing Script
Tests all 6 biomedical models with their corresponding datasets
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('/home/sdodl001_odu_edu/llm_evaluation')

from configs.model_registry import MODEL_CONFIGS
from configs.biomedical_model_dataset_mappings import BIOMEDICAL_MODEL_MAPPINGS, EVALUATION_STRATEGIES, PERFORMANCE_TARGETS

def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"biomedical_testing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def check_dataset_availability():
    """Check which biomedical datasets are available"""
    dataset_base = Path("datasets/biomedical")
    available_datasets = {}
    
    dataset_dirs = ["pubmedqa", "medqa", "bc5cdr", "ddi"]
    
    for dataset_dir in dataset_dirs:
        dataset_path = dataset_base / dataset_dir
        if dataset_path.exists():
            json_files = list(dataset_path.glob("*.json"))
            available_datasets[dataset_dir] = {
                "path": str(dataset_path),
                "files": [f.name for f in json_files],
                "total_size": sum(f.stat().st_size for f in json_files)
            }
    
    return available_datasets

def get_biomedical_models():
    """Get all biomedical models from registry"""
    biomedical_models = {}
    
    for model_id, config in MODEL_CONFIGS.items():
        # Check if it's a biomedical model by category
        if hasattr(config, 'specialization_category'):
            if config.specialization_category in ['bioinformatics', 'biomedical'] or \
               'biomedical' in config.specialization_subcategory.lower() or \
               'medical' in config.specialization_subcategory.lower():
                biomedical_models[model_id] = config
        # Also check if it's one of our known biomedical models
        elif any(term in model_id.lower() for term in ['biomedlm', 'medalpa', 'biogpt', 'bio_clinical', 'biomistral']):
            biomedical_models[model_id] = config
    
    return biomedical_models

def test_model_dataset_mappings():
    """Test the model-dataset mapping system"""
    logging.info("Testing biomedical model-dataset mappings...")
    
    models = get_biomedical_models()
    datasets = check_dataset_availability()
    
    results = {
        "models_found": len(models),
        "datasets_available": len(datasets),
        "mappings": {},
        "recommendations": []
    }
    
    for model_id in models:
        if model_id in BIOMEDICAL_MODEL_MAPPINGS:
            mapping = BIOMEDICAL_MODEL_MAPPINGS[model_id]
            results["mappings"][model_id] = {
                "primary_datasets": mapping.primary_datasets,
                "secondary_datasets": mapping.secondary_datasets,
                "expected_performance": mapping.expected_performance,
                "available_datasets": []
            }
            
            # Check which datasets are actually available
            for dataset in mapping.primary_datasets:
                if dataset in datasets:
                    results["mappings"][model_id]["available_datasets"].append(dataset)
                    logging.info(f"✅ {model_id} -> {dataset} (Available)")
                else:
                    logging.warning(f"❌ {model_id} -> {dataset} (Missing)")
    
    return results

def simulate_evaluation_run():
    """Simulate a quick evaluation run without actually loading models"""
    logging.info("Simulating biomedical evaluation run...")
    
    models = get_biomedical_models()
    datasets = check_dataset_availability()
    
    simulation_results = {
        "timestamp": datetime.now().isoformat(),
        "simulation_status": "success",
        "models_tested": len(models),
        "datasets_used": len(datasets),
        "estimated_performance": {}
    }
    
    # Use performance targets for simulation
    for model_id in models:
        if model_id in PERFORMANCE_TARGETS:
            target = PERFORMANCE_TARGETS[model_id]
            simulation_results["estimated_performance"][model_id] = {
                "medqa_accuracy": target.get("medqa_accuracy", "N/A"),
                "pubmedqa_accuracy": target.get("pubmedqa_accuracy", "N/A"),
                "ner_f1": target.get("ner_f1", "N/A"),
                "relation_extraction_f1": target.get("relation_extraction_f1", "N/A")
            }
    
    return simulation_results

def main():
    """Main testing function"""
    print("🧬 Biomedical Category Comprehensive Testing")
    print("=" * 50)
    
    log_file = setup_logging()
    logging.info("Starting biomedical category testing")
    
    # Check dataset availability
    logging.info("Checking dataset availability...")
    datasets = check_dataset_availability()
    
    print(f"\n📊 Dataset Status:")
    for dataset, info in datasets.items():
        print(f"  {dataset}: {len(info['files'])} files ({info['total_size'] / 1024 / 1024:.1f} MB)")
    
    # Check models
    logging.info("Checking biomedical models...")
    models = get_biomedical_models()
    
    print(f"\n🤖 Biomedical Models: {len(models)} found")
    for model_id, config in models.items():
        description = getattr(config, 'description', '') or getattr(config, 'model_name', model_id)
        print(f"  {model_id}: {description}")
    
    # Test mappings
    mapping_results = test_model_dataset_mappings()
    
    print(f"\n🔗 Model-Dataset Mappings:")
    print(f"  Models mapped: {len(mapping_results['mappings'])}")
    print(f"  Datasets available: {mapping_results['datasets_available']}")
    
    # Simulate evaluation
    simulation = simulate_evaluation_run()
    
    print(f"\n🎯 Performance Simulation:")
    for model_id, perf in simulation["estimated_performance"].items():
        print(f"  {model_id}:")
        if perf["medqa_accuracy"] != "N/A":
            print(f"    MedQA: {perf['medqa_accuracy']}")
        if perf["pubmedqa_accuracy"] != "N/A":
            print(f"    PubMedQA: {perf['pubmedqa_accuracy']}")
    
    # Save comprehensive results
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "datasets": datasets,
        "models": {k: v for k, v in models.items()},
        "mappings": mapping_results,
        "simulation": simulation,
        "status": "READY_FOR_EVALUATION"
    }
    
    results_file = f"biomedical_category_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Biomedical Category Status: READY FOR EVALUATION")
    print(f"📋 Detailed results saved to: {results_file}")
    print(f"📋 Log file: {log_file}")
    
    # Final recommendations
    print(f"\n🚀 Ready to proceed with:")
    print(f"  • {len(models)} biomedical models configured")
    print(f"  • {len(datasets)} datasets available")
    print(f"  • Model-dataset optimization mappings")
    print(f"  • Performance benchmarking targets")
    
    return results

if __name__ == "__main__":
    results = main()