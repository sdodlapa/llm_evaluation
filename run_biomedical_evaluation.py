#!/usr/bin/env python3
"""
Comprehensive Biomedical Category Evaluation
Runs optimized evaluation across all 10 biomedical models using their mapped datasets
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
from configs.biomedical_model_dataset_mappings import BIOMEDICAL_MODEL_MAPPINGS, PERFORMANCE_TARGETS
from evaluation.category_evaluator import CategoryEvaluator

def setup_logging():
    """Setup comprehensive logging for biomedical evaluation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"biomedical_evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file, timestamp

def get_biomedical_models():
    """Get all biomedical models from registry"""
    biomedical_models = {}
    
    for model_id, config in MODEL_CONFIGS.items():
        # Check if it's a biomedical model
        if (hasattr(config, 'specialization_category') and 
            config.specialization_category in ['bioinformatics', 'biomedical']) or \
           any(term in model_id.lower() for term in ['biomedlm', 'medalpa', 'biogpt', 'bio_clinical', 'biomistral']):
            biomedical_models[model_id] = config
    
    return biomedical_models

def prepare_evaluation_plan():
    """Create evaluation plan using model-dataset mappings"""
    models = get_biomedical_models()
    
    evaluation_plan = {
        "category": "biomedical",
        "timestamp": datetime.now().isoformat(),
        "models_to_evaluate": [],
        "datasets_required": set(),
        "performance_targets": PERFORMANCE_TARGETS
    }
    
    for model_id, config in models.items():
        model_plan = {
            "model_id": model_id,
            "model_name": config.model_name,
            "datasets": [],
            "expected_performance": {},
            "priority": "high" if model_id in BIOMEDICAL_MODEL_MAPPINGS else "medium"
        }
        
        # Use mappings if available, otherwise use default datasets
        if model_id in BIOMEDICAL_MODEL_MAPPINGS:
            mapping = BIOMEDICAL_MODEL_MAPPINGS[model_id]
            model_plan["datasets"] = mapping.primary_datasets
            model_plan["expected_performance"] = mapping.expected_performance
            
            for dataset in mapping.primary_datasets:
                evaluation_plan["datasets_required"].add(dataset)
        else:
            # Default datasets for unmapped models
            default_datasets = ["pubmedqa", "medqa"]
            model_plan["datasets"] = default_datasets
            for dataset in default_datasets:
                evaluation_plan["datasets_required"].add(dataset)
        
        evaluation_plan["models_to_evaluate"].append(model_plan)
    
    evaluation_plan["datasets_required"] = list(evaluation_plan["datasets_required"])
    return evaluation_plan

def run_biomedical_evaluation():
    """Run the comprehensive biomedical evaluation"""
    log_file, timestamp = setup_logging()
    
    logging.info("🧬 Starting Comprehensive Biomedical Category Evaluation")
    logging.info("=" * 60)
    
    # Create evaluation plan
    evaluation_plan = prepare_evaluation_plan()
    
    logging.info(f"Models to evaluate: {len(evaluation_plan['models_to_evaluate'])}")
    logging.info(f"Datasets required: {evaluation_plan['datasets_required']}")
    
    # Save evaluation plan
    plan_file = f"biomedical_evaluation_plan_{timestamp}.json"
    with open(plan_file, 'w') as f:
        # Convert sets to lists for JSON serialization
        json.dump(evaluation_plan, f, indent=2, default=str)
    
    logging.info(f"Evaluation plan saved to: {plan_file}")
    
    # Initialize evaluator
    try:
        evaluator = CategoryEvaluator()
        
        # Start evaluation
        results = {
            "evaluation_id": f"biomedical_{timestamp}",
            "start_time": datetime.now().isoformat(),
            "category": "biomedical",
            "models_evaluated": [],
            "failed_models": [],
            "performance_summary": {},
            "benchmark_comparisons": {}
        }
        
        # Evaluate each model
        for model_plan in evaluation_plan["models_to_evaluate"]:
            model_id = model_plan["model_id"]
            
            logging.info(f"\n🔬 Evaluating {model_id}...")
            logging.info(f"   Datasets: {model_plan['datasets']}")
            
            try:
                # Run evaluation for this model
                model_results = evaluator.evaluate_model(
                    model_id=model_id,
                    datasets=model_plan["datasets"],
                    category="biomedical"
                )
                
                results["models_evaluated"].append({
                    "model_id": model_id,
                    "datasets_used": model_plan["datasets"],
                    "results": model_results,
                    "expected_performance": model_plan["expected_performance"]
                })
                
                logging.info(f"✅ {model_id} evaluation completed")
                
                # Compare against benchmarks if available
                if model_id in PERFORMANCE_TARGETS:
                    target = PERFORMANCE_TARGETS[model_id]
                    comparison = compare_performance(model_results, target)
                    results["benchmark_comparisons"][model_id] = comparison
                    logging.info(f"📊 Benchmark comparison: {comparison}")
                
            except Exception as e:
                logging.error(f"❌ {model_id} evaluation failed: {str(e)}")
                results["failed_models"].append({
                    "model_id": model_id,
                    "error": str(e)
                })
        
        results["end_time"] = datetime.now().isoformat()
        results["total_models_attempted"] = len(evaluation_plan["models_to_evaluate"])
        results["successful_evaluations"] = len(results["models_evaluated"])
        results["failed_evaluations"] = len(results["failed_models"])
        
        # Save comprehensive results
        results_file = f"biomedical_evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        generate_summary_report(results, timestamp)
        
        logging.info(f"\n🎯 Biomedical Evaluation Complete!")
        logging.info(f"Successfully evaluated: {results['successful_evaluations']}/{results['total_models_attempted']} models")
        logging.info(f"Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logging.error(f"Critical evaluation error: {str(e)}")
        raise

def compare_performance(actual_results, target_performance):
    """Compare actual results against benchmark targets"""
    comparison = {}
    
    for metric, target_value in target_performance.items():
        if metric in actual_results:
            actual_value = actual_results[metric]
            if isinstance(target_value, (int, float)) and isinstance(actual_value, (int, float)):
                comparison[metric] = {
                    "target": target_value,
                    "actual": actual_value,
                    "difference": actual_value - target_value,
                    "meets_target": actual_value >= target_value
                }
    
    return comparison

def generate_summary_report(results, timestamp):
    """Generate human-readable summary report"""
    report_file = f"biomedical_evaluation_summary_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# 🧬 Biomedical Category Evaluation Summary\n\n")
        f.write(f"**Evaluation ID:** {results['evaluation_id']}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Models Evaluated:** {results['successful_evaluations']}/{results['total_models_attempted']}\n\n")
        
        f.write("## 📊 Performance Summary\n\n")
        
        for model_result in results["models_evaluated"]:
            model_id = model_result["model_id"]
            f.write(f"### {model_id}\n")
            f.write(f"- **Datasets:** {', '.join(model_result['datasets_used'])}\n")
            
            if model_id in results.get("benchmark_comparisons", {}):
                comparison = results["benchmark_comparisons"][model_id]
                f.write("- **Benchmark Comparison:**\n")
                for metric, comp in comparison.items():
                    status = "✅" if comp["meets_target"] else "❌"
                    f.write(f"  - {metric}: {comp['actual']:.1f}% (target: {comp['target']:.1f}%) {status}\n")
            
            f.write("\n")
        
        if results["failed_models"]:
            f.write("## ❌ Failed Evaluations\n\n")
            for failed in results["failed_models"]:
                f.write(f"- **{failed['model_id']}:** {failed['error']}\n")
    
    logging.info(f"Summary report saved to: {report_file}")

def main():
    """Main evaluation function"""
    print("🧬 Comprehensive Biomedical Category Evaluation")
    print("=" * 50)
    
    try:
        results = run_biomedical_evaluation()
        
        print(f"\n✅ Evaluation Complete!")
        print(f"Models evaluated: {results['successful_evaluations']}/{results['total_models_attempted']}")
        
        if results["benchmark_comparisons"]:
            print(f"\n📊 Top Performers:")
            for model_id, comparison in results["benchmark_comparisons"].items():
                meets_targets = sum(1 for comp in comparison.values() if comp.get("meets_target", False))
                total_metrics = len(comparison)
                print(f"  {model_id}: {meets_targets}/{total_metrics} targets met")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        logging.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()