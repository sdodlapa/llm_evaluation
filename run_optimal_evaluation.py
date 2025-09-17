#!/usr/bin/env python3
"""
Optimal Evaluation Script for Unimplemented Datasets
Uses optimal presets per model based on performance analysis
Only evaluates the 6 unimplemented datasets to avoid repetition
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from evaluation.comprehensive_runner import ComprehensiveEvaluationRunner
from evaluation.enhanced_dataset_manager import EnhancedDatasetManager
from configs.model_configs import MODEL_CONFIGS

def setup_logging():
    """Setup logging for optimal evaluation"""
    log_dir = Path("optimal_evaluation_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'optimal_evaluation_logs/optimal_eval_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def get_optimal_configurations():
    """Get optimal model-preset configurations based on analysis"""
    
    # Based on current evaluation analysis and best practices
    optimal_configs = {
        "qwen3_8b": "balanced",      # Best overall performance from analysis
        "qwen3_14b": "performance"   # Larger model benefits from performance preset
    }
    
    return optimal_configs

def get_unimplemented_datasets():
    """Get list of datasets that haven't been tested yet"""
    return [
        "math",         # Mathematical competition problems
        "bfcl",         # Berkeley Function-Calling Leaderboard
        "toolllama",    # Tool usage and API calling benchmark  
        "mmlu",         # Massive Multitask Language Understanding
        "ifeval",       # Instruction following evaluation
        "winogrande"    # Commonsense reasoning with pronoun resolution
    ]

def create_optimal_evaluation_plan():
    """Create evaluation plan for optimal configurations on unimplemented datasets"""
    logger = logging.getLogger(__name__)
    
    dataset_manager = EnhancedDatasetManager()
    optimal_configs = get_optimal_configurations()
    unimplemented_datasets = get_unimplemented_datasets()
    
    # Verify models are available
    available_models = [model for model in optimal_configs.keys() if model in MODEL_CONFIGS]
    
    # Sample counts for unimplemented datasets (100-200 range as requested)
    sample_counts = {
        "math": 200,         # Limit to 200 from 5000 available
        "bfcl": 200,         # Limit to 200 from 2000 available
        "toolllama": 200,    # Limit to 200 from 3000 available
        "mmlu": 200,         # Limit to 200 from 14042 available
        "ifeval": 200,       # Limit to 200 from 500 available
        "winogrande": 200    # Limit to 200 from 1767 available
    }
    
    total_combinations = len(available_models) * len(unimplemented_datasets)
    total_samples = sum(sample_counts[ds] for ds in unimplemented_datasets) * len(available_models)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"optimal_results_{timestamp}"
    
    plan = {
        "evaluation_plan": {
            "timestamp": datetime.now().isoformat(),
            "strategy": "optimal_presets_unimplemented_datasets",
            "models": available_models,
            "optimal_presets": optimal_configs,
            "datasets": unimplemented_datasets,
            "sample_counts": sample_counts,
            "total_combinations": total_combinations,
            "total_samples_to_process": total_samples,
            "estimated_duration_hours": total_samples / 200 / 60,  # Rough estimate
            "results_directory": results_dir,
            "save_predictions": True,
            "save_metrics": True,
            "evaluation_focus": "Cover unimplemented datasets with optimal model configurations"
        }
    }
    
    return plan, results_dir

def handle_unimplemented_datasets():
    """Handle datasets that don't have data files yet"""
    logger = logging.getLogger(__name__)
    
    # For now, we'll create placeholder logic for missing datasets
    # In a real scenario, you would download/prepare the actual data
    
    missing_datasets = get_unimplemented_datasets()
    
    logger.warning("🚧 The following datasets need data files to be implemented:")
    for dataset in missing_datasets:
        logger.warning(f"   • {dataset}: Missing data file")
    
    logger.info("📝 Strategy: Will attempt evaluation and gracefully handle missing datasets")
    logger.info("💡 Consider implementing dataset downloaders or preparing data files")
    
    return True

def run_optimal_evaluation():
    """Run optimal evaluation on unimplemented datasets"""
    logger = logging.getLogger(__name__)
    
    # Create evaluation plan
    plan, results_dir = create_optimal_evaluation_plan()
    
    logger.info("🎯 Starting Optimal LLM Evaluation on Unimplemented Datasets")
    logger.info(f"Strategy: Use optimal presets per model")
    logger.info(f"Models: {len(plan['evaluation_plan']['models'])}")
    logger.info(f"Datasets: {len(plan['evaluation_plan']['datasets'])}")
    logger.info(f"Total combinations: {plan['evaluation_plan']['total_combinations']}")
    logger.info(f"Total samples: {plan['evaluation_plan']['total_samples_to_process']}")
    logger.info(f"Estimated duration: {plan['evaluation_plan']['estimated_duration_hours']:.1f} hours")
    
    logger.info("\n🤖 Optimal Model Configurations:")
    for model, preset in plan['evaluation_plan']['optimal_presets'].items():
        logger.info(f"   • {model}: {preset} preset")
    
    logger.info(f"\n📊 Unimplemented Datasets to Test:")
    for dataset in plan['evaluation_plan']['datasets']:
        sample_count = plan['evaluation_plan']['sample_counts'][dataset]
        logger.info(f"   • {dataset}: {sample_count} samples")
    
    # Handle missing datasets
    handle_unimplemented_datasets()
    
    # Save evaluation plan
    os.makedirs(results_dir, exist_ok=True)
    plan_file = f"{results_dir}/optimal_evaluation_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2)
    logger.info(f"📋 Evaluation plan saved to: {plan_file}")
    
    # Create runner with results directory  
    runner = ComprehensiveEvaluationRunner(str(results_dir))
    
    # Prepare model-preset combinations
    model_preset_combinations = []
    for model in plan['evaluation_plan']['models']:
        optimal_preset = plan['evaluation_plan']['optimal_presets'][model]
        model_preset_combinations.append((model, optimal_preset))
    
    logger.info(f"\n🚀 Running {len(model_preset_combinations)} model-preset combinations:")
    for model, preset in model_preset_combinations:
        logger.info(f"   • {model} with {preset} preset")
    
    # Run evaluation with optimal configurations
    try:
        result = runner.run_optimal_evaluation(
            model_preset_combinations=model_preset_combinations,
            datasets=plan['evaluation_plan']['datasets'],
            samples_per_dataset=plan['evaluation_plan']['sample_counts'],
            continue_on_failure=True
        )
        
        logger.info("✅ Optimal evaluation completed successfully")
        return result, results_dir
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        logger.info("💡 Note: Some datasets may not be implemented yet")
        return None, results_dir

def generate_optimal_performance_report(results, results_dir):
    """Generate performance report for optimal evaluation"""
    logger = logging.getLogger(__name__)
    
    if not results:
        logger.warning("⚠️ No results to generate report from")
        return
    
    logger.info("📊 Generating optimal evaluation performance report...")
    
    # Create performance summary
    summary = {
        "evaluation_strategy": "optimal_presets_unimplemented_datasets",
        "timestamp": datetime.now().isoformat(),
        "total_combinations_attempted": len(results.get('all_results', [])),
        "successful_runs": len([r for r in results.get('all_results', []) if r.get('status') == 'success']),
        "failed_runs": len([r for r in results.get('all_results', []) if r.get('status') == 'error']),
        "datasets_coverage": list(set([r.get('run_info', {}).get('dataset') for r in results.get('all_results', [])])),
        "models_tested": list(set([r.get('run_info', {}).get('model_name') for r in results.get('all_results', [])]))
    }
    
    # Save summary
    summary_file = f"{results_dir}/optimal_evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📈 Performance summary saved to: {summary_file}")
    logger.info(f"✅ Successful runs: {summary['successful_runs']}")
    logger.info(f"❌ Failed runs: {summary['failed_runs']}")
    logger.info(f"📊 Datasets covered: {summary['datasets_coverage']}")

def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Optimal LLM Evaluation - Unimplemented Datasets")
    logger.info("=" * 60)
    logger.info("Strategy: Use optimal presets per model, test only unimplemented datasets")
    logger.info("Goal: Maximize efficiency while covering missing evaluation gaps")
    
    try:
        # Run optimal evaluation
        results, results_dir = run_optimal_evaluation()
        
        # Generate performance report
        generate_optimal_performance_report(results, results_dir)
        
        logger.info(f"\n🏁 Optimal evaluation completed!")
        logger.info(f"📁 Results directory: {results_dir}")
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)