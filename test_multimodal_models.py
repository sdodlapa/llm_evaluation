#!/usr/bin/env python3
"""
Test script to validate new multimodal models integration.
Tests the 4 new models: Qwen2.5-VL-7B, MiniCPM-V-2.6, LLaVA-NeXT-Vicuna-7B, InternVL2-8B
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from configs.model_registry import get_model_config, get_all_models
    from evaluation.mappings.model_categories import MODEL_CATEGORIES
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multimodal_model_validation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_model_registry_access(logger):
    """Test if we can access the new multimodal models from the registry"""
    logger.info("Testing model registry access...")
    
    new_multimodal_models = [
        'qwen25_vl_7b',
        'minicpm_v_26', 
        'llava_next_vicuna_7b',
        'internvl2_8b'
    ]
    
    results = {}
    for model_id in new_multimodal_models:
        try:
            config = get_model_config(model_id)
            if config:
                results[model_id] = {
                    'status': 'success',
                    'model_name': config.model_name,
                    'model_type': config.model_type,
                    'max_model_len': config.max_model_len,
                    'trust_remote_code': config.trust_remote_code
                }
                logger.info(f"✓ {model_id}: Found in registry")
            else:
                results[model_id] = {'status': 'not_found'}
                logger.error(f"✗ {model_id}: Not found in registry")
        except Exception as e:
            results[model_id] = {'status': 'error', 'error': str(e)}
            logger.error(f"✗ {model_id}: Error accessing - {e}")
    
    return results

def test_category_mapping(logger):
    """Test if the new models are properly mapped in categories"""
    logger.info("Testing category mapping...")
    
    try:
        multimodal_category = MODEL_CATEGORIES.get('MULTIMODAL_PROCESSING')
        if not multimodal_category:
            logger.error("MULTIMODAL_PROCESSING category not found")
            return {'status': 'category_not_found'}
        
        models = multimodal_category.get('models', [])
        datasets = multimodal_category.get('datasets', [])
        
        logger.info(f"MULTIMODAL_PROCESSING category:")
        logger.info(f"  Models: {len(models)} - {models}")
        logger.info(f"  Datasets: {len(datasets)} - {datasets}")
        
        # Check if our new models are in the category
        new_models = ['qwen25_vl_7b', 'minicpm_v_26', 'llava_next_vicuna_7b', 'internvl2_8b']
        found_models = [model for model in new_models if model in models]
        missing_models = [model for model in new_models if model not in models]
        
        return {
            'status': 'success',
            'total_models': len(models),
            'found_new_models': found_models,
            'missing_new_models': missing_models,
            'datasets': datasets
        }
        
    except Exception as e:
        logger.error(f"Error testing category mapping: {e}")
        return {'status': 'error', 'error': str(e)}

def test_dataset_availability(logger):
    """Test if multimodal datasets are available"""
    logger.info("Testing dataset availability...")
    
    data_dir = Path(__file__).parent / "evaluation_data"
    results = {}
    
    expected_datasets = ['scienceqa', 'chartqa', 'ai2d', 'textvqa']
    
    for dataset in expected_datasets:
        dataset_dir = data_dir / dataset
        if dataset_dir.exists():
            # Count JSON files and samples
            json_files = list(dataset_dir.glob("*.json"))
            total_samples = 0
            splits = []
            
            for json_file in json_files:
                splits.append(json_file.stem)
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        total_samples += len(data)
                except:
                    pass
            
            results[dataset] = {
                'status': 'available',
                'splits': splits,
                'total_samples': total_samples,
                'files': [f.name for f in json_files]
            }
            logger.info(f"✓ {dataset}: {total_samples} samples in {len(splits)} splits")
        else:
            results[dataset] = {'status': 'missing'}
            logger.error(f"✗ {dataset}: Directory not found")
    
    return results

def create_sample_evaluation_plan(logger):
    """Create a sample evaluation plan for the new multimodal models"""
    logger.info("Creating sample evaluation plan...")
    
    plan = {
        "evaluation_name": "multimodal_model_validation",
        "description": "Validation test for newly added multimodal models",
        "models_to_test": [
            "qwen25_vl_7b",
            "minicpm_v_26", 
            "llava_next_vicuna_7b",
            "internvl2_8b"
        ],
        "datasets_to_use": [
            "scienceqa",
            "chartqa"
        ],
        "evaluation_config": {
            "max_samples_per_dataset": 5,
            "batch_size": 1,
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": 300
        },
        "expected_outputs": {
            "success_criteria": "Models load successfully and produce responses",
            "performance_baseline": "Responses should be coherent and relevant to questions",
            "error_tolerance": "Allow connection/timeout errors but not configuration errors"
        }
    }
    
    plan_file = Path(__file__).parent / "multimodal_validation_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2)
    
    logger.info(f"Evaluation plan saved to {plan_file}")
    return plan

def run_basic_model_compatibility_test(logger):
    """Run basic compatibility tests without actually loading models"""
    logger.info("Running basic model compatibility tests...")
    
    results = {}
    new_models = ['qwen25_vl_7b', 'minicpm_v_26', 'llava_next_vicuna_7b', 'internvl2_8b']
    
    for model_id in new_models:
        try:
            config = get_model_config(model_id)
            if config:
                # Check configuration validity
                checks = {
                    'has_model_name': bool(config.model_name),
                    'has_model_type': bool(config.model_type),
                    'trust_remote_code_set': hasattr(config, 'trust_remote_code'),
                    'max_model_len_valid': config.max_model_len > 0,
                    'gpu_memory_utilization_set': hasattr(config, 'gpu_memory_utilization')
                }
                
                all_checks_pass = all(checks.values())
                
                results[model_id] = {
                    'status': 'compatible' if all_checks_pass else 'issues_found',
                    'checks': checks,
                    'model_name': config.model_name,
                    'model_type': config.model_type
                }
                
                status = "✓" if all_checks_pass else "⚠"
                logger.info(f"{status} {model_id}: {'Compatible' if all_checks_pass else 'Issues found'}")
                
            else:
                results[model_id] = {'status': 'config_not_found'}
                logger.error(f"✗ {model_id}: Configuration not found")
                
        except Exception as e:
            results[model_id] = {'status': 'error', 'error': str(e)}
            logger.error(f"✗ {model_id}: Error - {e}")
    
    return results

def generate_validation_report(logger, registry_results, category_results, dataset_results, compatibility_results):
    """Generate comprehensive validation report"""
    logger.info("Generating validation report...")
    
    report = {
        "validation_timestamp": str(Path().cwd()),
        "summary": {
            "total_new_models": 4,
            "models_in_registry": len([r for r in registry_results.values() if r['status'] == 'success']),
            "models_in_category": len(category_results.get('found_new_models', [])),
            "available_datasets": len([r for r in dataset_results.values() if r['status'] == 'available']),
            "compatible_models": len([r for r in compatibility_results.values() if r['status'] == 'compatible'])
        },
        "detailed_results": {
            "model_registry": registry_results,
            "category_mapping": category_results,
            "dataset_availability": dataset_results,
            "model_compatibility": compatibility_results
        },
        "validation_status": "PASS",
        "recommendations": []
    }
    
    # Determine overall status and add recommendations
    if report["summary"]["models_in_registry"] < 4:
        report["validation_status"] = "FAIL"
        report["recommendations"].append("Some models are missing from registry")
    
    if report["summary"]["available_datasets"] < 2:
        report["validation_status"] = "WARNING"
        report["recommendations"].append("Limited datasets available for testing")
    
    if report["summary"]["compatible_models"] < 4:
        report["validation_status"] = "WARNING"
        report["recommendations"].append("Some models have compatibility issues")
    
    if not report["recommendations"]:
        report["recommendations"].append("All systems ready for multimodal evaluation")
    
    # Save report
    report_file = Path(__file__).parent / "multimodal_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report saved to {report_file}")
    
    # Print summary
    print(f"\n=== MULTIMODAL MODEL VALIDATION REPORT ===")
    print(f"Status: {report['validation_status']}")
    print(f"Models in Registry: {report['summary']['models_in_registry']}/4")
    print(f"Models in Category: {report['summary']['models_in_category']}/4")
    print(f"Available Datasets: {report['summary']['available_datasets']}/4")
    print(f"Compatible Models: {report['summary']['compatible_models']}/4")
    print(f"\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    return report

def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting multimodal model validation")
    
    try:
        # Run all validation tests
        logger.info("\n=== MODEL REGISTRY TEST ===")
        registry_results = test_model_registry_access(logger)
        
        logger.info("\n=== CATEGORY MAPPING TEST ===")
        category_results = test_category_mapping(logger)
        
        logger.info("\n=== DATASET AVAILABILITY TEST ===")
        dataset_results = test_dataset_availability(logger)
        
        logger.info("\n=== MODEL COMPATIBILITY TEST ===")
        compatibility_results = run_basic_model_compatibility_test(logger)
        
        logger.info("\n=== CREATING EVALUATION PLAN ===")
        evaluation_plan = create_sample_evaluation_plan(logger)
        
        logger.info("\n=== GENERATING REPORT ===")
        report = generate_validation_report(
            logger, registry_results, category_results, 
            dataset_results, compatibility_results
        )
        
        logger.info("\n=== VALIDATION COMPLETE ===")
        return report
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()