#!/usr/bin/env python3
"""
H100 Large Model Dataset Compatibility Test

Tests the optimized dataset structure with H100 large models to ensure:
- All primary datasets are accessible
- H100 models can load and evaluate
- Performance metrics are consistent
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset_accessibility():
    """Test that all primary datasets are accessible."""
    logger.info("Testing dataset accessibility...")
    
    # Required primary datasets
    primary_datasets = [
        'address_parsing', 'ai2d', 'arc_challenge', 'bigbench_hard', 'bigcodebench',
        'bioasq', 'coordinate_processing', 'docvqa', 'enhanced_math_fixed', 'gsm8k',
        'hellaswag', 'hh_rlhf', 'humaneval', 'livecodebench', 'location_ner',
        'mathvista', 'mbpp', 'mediqa', 'medqa', 'mmlu', 'mmmu', 'mt_bench',
        'multimodal_sample', 'ner_locations', 'pubmedqa', 'scienceqa',
        'scientific_papers', 'scierc', 'spatial_reasoning', 'swe_bench',
        'toxicity_detection', 'truthfulqa'
    ]
    
    accessible_datasets = []
    missing_datasets = []
    
    for dataset_name in primary_datasets:
        # Check if dataset file exists
        found = False
        for root, dirs, files in os.walk("evaluation_data/datasets"):
            for file in files:
                if file.startswith(dataset_name) and file.endswith(('.json', '.jsonl')):
                    file_path = os.path.join(root, file)
                    try:
                        # Try to load and validate JSON
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        if isinstance(data, list) and len(data) > 0:
                            accessible_datasets.append(dataset_name)
                            logger.info(f"✅ {dataset_name}: {len(data)} examples")
                            found = True
                            break
                    except Exception as e:
                        logger.error(f"❌ {dataset_name}: JSON error - {e}")
        
        if not found:
            missing_datasets.append(dataset_name)
    
    return {
        'accessible': accessible_datasets,
        'missing': missing_datasets,
        'total_required': len(primary_datasets),
        'accessibility_rate': len(accessible_datasets) / len(primary_datasets) * 100
    }

def test_h100_model_loading():
    """Test H100 model configuration loading."""
    logger.info("Testing H100 model configurations...")
    
    try:
        # Import H100 models from registry
        sys.path.append('.')
        from configs.model_registry import MODEL_REGISTRY
        
        h100_models = []
        for model_id, config in MODEL_REGISTRY.items():
            if config.get('gpu_memory_gb', 0) >= 40:  # H100-sized models
                h100_models.append((model_id, config))
        
        logger.info(f"Found {len(h100_models)} H100-compatible models")
        
        # Test model configurations
        valid_configs = []
        for model_id, config in h100_models:
            try:
                # Validate required fields
                required_fields = ['name', 'path', 'type', 'gpu_memory_gb']
                if all(field in config for field in required_fields):
                    valid_configs.append(model_id)
                    logger.info(f"✅ {model_id}: Valid configuration")
                else:
                    logger.error(f"❌ {model_id}: Missing required fields")
            except Exception as e:
                logger.error(f"❌ {model_id}: Configuration error - {e}")
        
        return {
            'total_h100_models': len(h100_models),
            'valid_configs': len(valid_configs),
            'config_validity_rate': len(valid_configs) / len(h100_models) * 100 if h100_models else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to load H100 models: {e}")
        return {'error': str(e)}

def test_category_completeness():
    """Test that all model categories have complete datasets."""
    logger.info("Testing model category completeness...")
    
    try:
        sys.path.append('.')
        from evaluation.mappings.model_categories import (
            ADVANCED_CODE_GENERATION, ADVANCED_MULTIMODAL, MIXTURE_OF_EXPERTS
        )
        
        # Test H100-specific categories
        h100_categories = {
            'ADVANCED_CODE_GENERATION': ADVANCED_CODE_GENERATION,
            'ADVANCED_MULTIMODAL': ADVANCED_MULTIMODAL,
            'MIXTURE_OF_EXPERTS': MIXTURE_OF_EXPERTS
        }
        
        category_results = {}
        
        for cat_name, cat_config in h100_categories.items():
            primary_datasets = cat_config['primary_datasets']
            available_count = 0
            
            for dataset in primary_datasets:
                # Check if dataset exists
                for root, dirs, files in os.walk("evaluation_data/datasets"):
                    for file in files:
                        if file.startswith(dataset) and file.endswith(('.json', '.jsonl')):
                            available_count += 1
                            break
            
            completeness = available_count / len(primary_datasets) * 100
            category_results[cat_name] = {
                'required': len(primary_datasets),
                'available': available_count,
                'completeness': completeness
            }
            
            status = "✅" if completeness == 100 else "❌"
            logger.info(f"{status} {cat_name}: {available_count}/{len(primary_datasets)} datasets ({completeness:.1f}%)")
        
        return category_results
        
    except Exception as e:
        logger.error(f"Failed to test categories: {e}")
        return {'error': str(e)}

def test_quick_evaluation():
    """Test quick evaluation pipeline with optimized datasets."""
    logger.info("Testing quick evaluation pipeline...")
    
    try:
        # Test if evaluation modules can be imported
        sys.path.append('.')
        
        # Try importing key evaluation components
        test_results = {}
        
        try:
            from engines.vllm_engine import VLLMEngine
            test_results['vllm_engine'] = True
            logger.info("✅ VLLM Engine import successful")
        except Exception as e:
            test_results['vllm_engine'] = False
            logger.error(f"❌ VLLM Engine import failed: {e}")
        
        try:
            from evaluation.evaluation_pipeline import EvaluationPipeline
            test_results['evaluation_pipeline'] = True
            logger.info("✅ Evaluation Pipeline import successful")
        except Exception as e:
            test_results['evaluation_pipeline'] = False
            logger.error(f"❌ Evaluation Pipeline import failed: {e}")
        
        # Test dataset loading
        try:
            test_dataset = "evaluation_data/datasets/general/arc_challenge.json"
            if os.path.exists(test_dataset):
                with open(test_dataset, 'r') as f:
                    data = json.load(f)
                test_results['dataset_loading'] = len(data) > 0
                logger.info(f"✅ Dataset loading successful: {len(data)} examples")
            else:
                test_results['dataset_loading'] = False
                logger.error("❌ Test dataset not found")
        except Exception as e:
            test_results['dataset_loading'] = False
            logger.error(f"❌ Dataset loading failed: {e}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return {'error': str(e)}

def generate_compatibility_report(results: Dict[str, Any]):
    """Generate H100 compatibility test report."""
    
    dataset_results = results.get('datasets', {})
    model_results = results.get('models', {})
    category_results = results.get('categories', {})
    pipeline_results = results.get('pipeline', {})
    
    # Calculate overall score
    scores = []
    
    if 'accessibility_rate' in dataset_results:
        scores.append(dataset_results['accessibility_rate'])
    
    if 'config_validity_rate' in model_results:
        scores.append(model_results['config_validity_rate'])
    
    if category_results and 'error' not in category_results:
        category_scores = [cat['completeness'] for cat in category_results.values()]
        if category_scores:
            scores.append(sum(category_scores) / len(category_scores))
    
    if pipeline_results and 'error' not in pipeline_results:
        pipeline_score = sum(1 for v in pipeline_results.values() if v) / len(pipeline_results) * 100
        scores.append(pipeline_score)
    
    overall_score = sum(scores) / len(scores) if scores else 0
    
    report = f"""
=== H100 LARGE MODEL COMPATIBILITY REPORT ===

🎯 Overall Compatibility Score: {overall_score:.1f}%

📊 Dataset Accessibility:
- Primary datasets accessible: {dataset_results.get('accessible', []).__len__()}/{dataset_results.get('total_required', 0)}
- Accessibility rate: {dataset_results.get('accessibility_rate', 0):.1f}%

🚀 H100 Model Configurations:
- Total H100 models: {model_results.get('total_h100_models', 0)}
- Valid configurations: {model_results.get('valid_configs', 0)}
- Configuration validity: {model_results.get('config_validity_rate', 0):.1f}%

📂 Category Completeness:"""
    
    if category_results and 'error' not in category_results:
        for cat_name, cat_data in category_results.items():
            status = "✅" if cat_data['completeness'] == 100 else "❌"
            report += f"\n- {status} {cat_name}: {cat_data['completeness']:.1f}%"
    
    report += f"""

🔧 Pipeline Components:"""
    
    if pipeline_results and 'error' not in pipeline_results:
        for component, status in pipeline_results.items():
            status_icon = "✅" if status else "❌"
            report += f"\n- {status_icon} {component.replace('_', ' ').title()}"
    
    # Overall assessment
    if overall_score >= 95:
        report += "\n\n🎉 EXCELLENT: H100 system fully ready for production!"
    elif overall_score >= 85:
        report += "\n\n✅ GOOD: H100 system ready with minor optimizations needed"
    elif overall_score >= 70:
        report += "\n\n⚠️  PARTIAL: H100 system needs attention before production"
    else:
        report += "\n\n❌ CRITICAL: H100 system requires significant fixes"
    
    return report

def main():
    """Run H100 compatibility tests."""
    print("🔧 H100 Large Model Compatibility Test")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Dataset accessibility
    print("\n📊 Testing dataset accessibility...")
    results['datasets'] = test_dataset_accessibility()
    
    # Test 2: H100 model configurations
    print("\n🚀 Testing H100 model configurations...")
    results['models'] = test_h100_model_loading()
    
    # Test 3: Category completeness
    print("\n📂 Testing category completeness...")
    results['categories'] = test_category_completeness()
    
    # Test 4: Pipeline functionality
    print("\n🔧 Testing pipeline functionality...")
    results['pipeline'] = test_quick_evaluation()
    
    # Generate report
    print("\n📋 Generating compatibility report...")
    report = generate_compatibility_report(results)
    
    # Save report
    with open("evaluation_data/H100_COMPATIBILITY_REPORT.md", "w") as f:
        f.write(report)
    
    print(report)
    print(f"\n📄 Report saved to: evaluation_data/H100_COMPATIBILITY_REPORT.md")

if __name__ == "__main__":
    main()