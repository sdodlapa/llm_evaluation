#!/usr/bin/env python3
"""
Run comprehensive evaluation across all models and datasets
Saves all predictions and generates detailed performance comparison
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
    """Setup comprehensive logging"""
    log_dir = Path("comprehensive_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'comprehensive_logs/full_evaluation_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def create_comprehensive_evaluation_plan():
    """Create detailed evaluation plan"""
    dataset_manager = EnhancedDatasetManager()
    
    # Get available models and datasets
    available_models = list(MODEL_CONFIGS.keys())
    available_datasets = dataset_manager.get_implemented_datasets()
    presets = ["memory_optimized", "balanced", "performance"]
    
    # Define sample counts (100-200 per dataset as requested)
    sample_counts = {
        "humaneval": 164,        # Use all samples (small dataset)
        "mbpp": 200,             # Limit to 200
        "gsm8k": 200,            # Limit to 200  
        "hellaswag": 200,        # Limit to 200
        "mt_bench": 80,          # Use all samples (small dataset)
        "arc_challenge": 200     # Limit to 200
    }
    
    total_combinations = len(available_models) * len(presets) * len(available_datasets)
    total_samples = sum(sample_counts[ds] for ds in available_datasets) * len(available_models) * len(presets)
    
    plan = {
        "evaluation_plan": {
            "timestamp": datetime.now().isoformat(),
            "models": available_models,
            "datasets": available_datasets, 
            "presets": presets,
            "sample_counts": sample_counts,
            "total_combinations": total_combinations,
            "total_samples_to_process": total_samples,
            "estimated_duration_hours": total_samples / 200 / 60,  # Rough estimate
            "save_predictions": True,
            "save_metrics": True
        }
    }
    
    return plan

def run_comprehensive_evaluation_with_predictions():
    """Run comprehensive evaluation saving all predictions and metrics"""
    logger = logging.getLogger(__name__)
    
    # Create evaluation plan
    plan = create_comprehensive_evaluation_plan()
    
    logger.info("🚀 Starting Comprehensive LLM Evaluation")
    logger.info(f"Models: {len(plan['evaluation_plan']['models'])}")
    logger.info(f"Datasets: {len(plan['evaluation_plan']['datasets'])}")
    logger.info(f"Presets: {len(plan['evaluation_plan']['presets'])}")
    logger.info(f"Total combinations: {plan['evaluation_plan']['total_combinations']}")
    logger.info(f"Total samples: {plan['evaluation_plan']['total_samples_to_process']}")
    logger.info(f"Estimated duration: {plan['evaluation_plan']['estimated_duration_hours']:.1f} hours")
    
    # Save evaluation plan
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"comprehensive_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    plan_file = results_dir / "evaluation_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2)
    
    logger.info(f"Evaluation plan saved to: {plan_file}")
    
    # Create runner with results directory
    runner = ComprehensiveEvaluationRunner(str(results_dir))
    
    # Run comprehensive evaluation
    result = runner.run_comprehensive_evaluation(
        models=plan['evaluation_plan']['models'],
        presets=plan['evaluation_plan']['presets'],
        datasets=plan['evaluation_plan']['datasets'],
        samples_per_dataset=plan['evaluation_plan']['sample_counts'],
        continue_on_failure=True
    )
    
    # Generate performance comparison report
    generate_performance_comparison_report(result, results_dir)
    
    return result, results_dir

def generate_performance_comparison_report(results, results_dir):
    """Generate detailed performance comparison report"""
    logger = logging.getLogger(__name__)
    
    logger.info("📊 Generating comprehensive performance comparison report...")
    
    # Analyze all results
    successful_runs = [r for r in results['all_results'] if r.get('status') == 'success']
    failed_runs = [r for r in results['all_results'] if r.get('status') == 'error']
    
    # Performance analysis by model
    model_performance = {}
    dataset_performance = {}
    preset_performance = {}
    
    for result in successful_runs:
        if 'run_info' not in result or 'performance_metrics' not in result:
            continue
            
        run_info = result['run_info']
        perf = result['performance_metrics']
        eval_result = result.get('evaluation_result', {})
        
        model = run_info['model_name']
        dataset = run_info['dataset']
        preset = run_info['preset']
        
        # Collect metrics
        throughput = perf.get('avg_throughput_tokens_per_second', 0)
        memory = perf.get('peak_gpu_memory_gb', 0)
        accuracy = eval_result.get('accuracy', 0)
        
        # Aggregate by model
        if model not in model_performance:
            model_performance[model] = {'throughputs': [], 'memories': [], 'accuracies': [], 'datasets': set()}
        model_performance[model]['throughputs'].append(throughput)
        model_performance[model]['memories'].append(memory)
        model_performance[model]['accuracies'].append(accuracy)
        model_performance[model]['datasets'].add(dataset)
        
        # Aggregate by dataset
        if dataset not in dataset_performance:
            dataset_performance[dataset] = {'throughputs': [], 'memories': [], 'accuracies': [], 'models': set()}
        dataset_performance[dataset]['throughputs'].append(throughput)
        dataset_performance[dataset]['memories'].append(memory)
        dataset_performance[dataset]['accuracies'].append(accuracy)
        dataset_performance[dataset]['models'].add(model)
        
        # Aggregate by preset
        if preset not in preset_performance:
            preset_performance[preset] = {'throughputs': [], 'memories': [], 'accuracies': []}
        preset_performance[preset]['throughputs'].append(throughput)
        preset_performance[preset]['memories'].append(memory)
        preset_performance[preset]['accuracies'].append(accuracy)
    
    # Generate detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"comprehensive_performance_analysis_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write(f"# Comprehensive LLM Performance Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Evaluations**: {len(results['all_results'])}\n")
        f.write(f"- **Successful**: {len(successful_runs)} ({len(successful_runs)/len(results['all_results'])*100:.1f}%)\n")
        f.write(f"- **Failed**: {len(failed_runs)} ({len(failed_runs)/len(results['all_results'])*100:.1f}%)\n")
        f.write(f"- **Models Tested**: {len(model_performance)}\n")
        f.write(f"- **Datasets Tested**: {len(dataset_performance)}\n")
        f.write(f"- **Presets Tested**: {len(preset_performance)}\n\n")
        
        # Model Performance Comparison
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model | Avg Throughput (tok/s) | Avg Peak Memory (GB) | Avg Accuracy | Datasets Tested |\n")
        f.write("|-------|----------------------|---------------------|--------------|----------------|\n")
        
        for model, stats in sorted(model_performance.items()):
            avg_throughput = sum(stats['throughputs']) / len(stats['throughputs']) if stats['throughputs'] else 0
            avg_memory = sum(stats['memories']) / len(stats['memories']) if stats['memories'] else 0
            avg_accuracy = sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else 0
            datasets_count = len(stats['datasets'])
            
            f.write(f"| {model} | {avg_throughput:.1f} | {avg_memory:.1f} | {avg_accuracy:.3f} | {datasets_count} |\n")
        
        # Dataset Performance Analysis
        f.write("\n## Dataset Performance Analysis\n\n")
        f.write("| Dataset | Avg Throughput (tok/s) | Avg Peak Memory (GB) | Avg Accuracy | Models Tested |\n")
        f.write("|---------|----------------------|---------------------|--------------|---------------|\n")
        
        for dataset, stats in sorted(dataset_performance.items()):
            avg_throughput = sum(stats['throughputs']) / len(stats['throughputs']) if stats['throughputs'] else 0
            avg_memory = sum(stats['memories']) / len(stats['memories']) if stats['memories'] else 0
            avg_accuracy = sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else 0
            models_count = len(stats['models'])
            
            f.write(f"| {dataset} | {avg_throughput:.1f} | {avg_memory:.1f} | {avg_accuracy:.3f} | {models_count} |\n")
        
        # Preset Performance Analysis
        f.write("\n## Preset Performance Analysis\n\n")
        f.write("| Preset | Avg Throughput (tok/s) | Avg Peak Memory (GB) | Avg Accuracy |\n")
        f.write("|--------|----------------------|---------------------|-------------|\n")
        
        for preset, stats in sorted(preset_performance.items()):
            avg_throughput = sum(stats['throughputs']) / len(stats['throughputs']) if stats['throughputs'] else 0
            avg_memory = sum(stats['memories']) / len(stats['memories']) if stats['memories'] else 0
            avg_accuracy = sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else 0
            
            f.write(f"| {preset} | {avg_throughput:.1f} | {avg_memory:.1f} | {avg_accuracy:.3f} |\n")
        
        # Detailed Results
        f.write("\n## Detailed Results by Combination\n\n")
        f.write("| Model | Preset | Dataset | Throughput (tok/s) | Peak Memory (GB) | Accuracy | Samples | Duration (s) |\n")
        f.write("|-------|--------|---------|-------------------|------------------|----------|---------|-------------|\n")
        
        for result in successful_runs:
            if 'run_info' not in result or 'performance_metrics' not in result:
                continue
                
            run_info = result['run_info']
            perf = result['performance_metrics']
            eval_result = result.get('evaluation_result', {})
            
            f.write(f"| {run_info['model_name']} | {run_info['preset']} | {run_info['dataset']} | ")
            f.write(f"{perf.get('avg_throughput_tokens_per_second', 0):.1f} | ")
            f.write(f"{perf.get('peak_gpu_memory_gb', 0):.1f} | ")
            f.write(f"{eval_result.get('accuracy', 0):.3f} | ")
            f.write(f"{run_info.get('samples_processed', 0)} | ")
            f.write(f"{perf.get('total_duration_seconds', 0):.1f} |\n")
        
        # Failed Runs Analysis
        if failed_runs:
            f.write("\n## Failed Runs Analysis\n\n")
            f.write("| Model | Preset | Dataset | Error |\n")
            f.write("|-------|--------|---------|-------|\n")
            
            for result in failed_runs:
                if 'run_info' in result:
                    run_info = result['run_info']
                    error = result.get('error', 'Unknown error')[:100]
                    f.write(f"| {run_info['model_name']} | {run_info['preset']} | {run_info['dataset']} | {error} |\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("### Best Performing Models\n")
        
        # Find best model by throughput
        best_throughput_model = max(model_performance.items(), 
                                  key=lambda x: sum(x[1]['throughputs']) / len(x[1]['throughputs']) if x[1]['throughputs'] else 0)
        f.write(f"- **Highest Throughput**: {best_throughput_model[0]} ({sum(best_throughput_model[1]['throughputs']) / len(best_throughput_model[1]['throughputs']):.1f} tok/s)\n")
        
        # Find best model by memory efficiency
        best_memory_model = min(model_performance.items(), 
                              key=lambda x: sum(x[1]['memories']) / len(x[1]['memories']) if x[1]['memories'] else float('inf'))
        f.write(f"- **Most Memory Efficient**: {best_memory_model[0]} ({sum(best_memory_model[1]['memories']) / len(best_memory_model[1]['memories']):.1f} GB)\n")
        
        # Find best model by accuracy
        best_accuracy_model = max(model_performance.items(), 
                                key=lambda x: sum(x[1]['accuracies']) / len(x[1]['accuracies']) if x[1]['accuracies'] else 0)
        f.write(f"- **Highest Accuracy**: {best_accuracy_model[0]} ({sum(best_accuracy_model[1]['accuracies']) / len(best_accuracy_model[1]['accuracies']):.3f})\n")
        
        f.write("\n### Next Steps\n")
        f.write("1. Investigate failed runs and resolve configuration issues\n")
        f.write("2. Focus on best performing model configurations for production\n")
        f.write("3. Optimize memory usage for large-scale deployment\n")
        f.write("4. Consider model distillation or quantization for efficiency\n")
    
    logger.info(f"📊 Comprehensive performance analysis saved to: {report_file}")
    
    # Also save raw analysis data
    analysis_data = {
        "model_performance": {k: {
            "avg_throughput": sum(v['throughputs']) / len(v['throughputs']) if v['throughputs'] else 0,
            "avg_memory": sum(v['memories']) / len(v['memories']) if v['memories'] else 0,
            "avg_accuracy": sum(v['accuracies']) / len(v['accuracies']) if v['accuracies'] else 0,
            "datasets_tested": list(v['datasets'])
        } for k, v in model_performance.items()},
        "dataset_performance": {k: {
            "avg_throughput": sum(v['throughputs']) / len(v['throughputs']) if v['throughputs'] else 0,
            "avg_memory": sum(v['memories']) / len(v['memories']) if v['memories'] else 0,
            "avg_accuracy": sum(v['accuracies']) / len(v['accuracies']) if v['accuracies'] else 0,
            "models_tested": list(v['models'])
        } for k, v in dataset_performance.items()},
        "preset_performance": {k: {
            "avg_throughput": sum(v['throughputs']) / len(v['throughputs']) if v['throughputs'] else 0,
            "avg_memory": sum(v['memories']) / len(v['memories']) if v['memories'] else 0,
            "avg_accuracy": sum(v['accuracies']) / len(v['accuracies']) if v['accuracies'] else 0
        } for k, v in preset_performance.items()}
    }
    
    analysis_file = results_dir / f"performance_analysis_data_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    logger.info(f"📊 Performance analysis data saved to: {analysis_file}")

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Run comprehensive evaluation
        result, results_dir = run_comprehensive_evaluation_with_predictions()
        
        logger.info(f"🎉 Comprehensive evaluation completed!")
        logger.info(f"Results directory: {results_dir}")
        logger.info(f"Total runs: {result['evaluation_summary']['total_runs']}")
        logger.info(f"Successful: {result['evaluation_summary']['successful_runs']}")
        logger.info(f"Failed: {result['evaluation_summary']['failed_runs']}")
        logger.info(f"Duration: {result['evaluation_summary']['total_duration_seconds']/3600:.1f} hours")
        
        print(f"\n🎉 Comprehensive Evaluation Complete!")
        print(f"📁 Results saved to: {results_dir}")
        print(f"📊 Check the performance analysis report for detailed insights")
        
    except Exception as e:
        logger.error(f"❌ Comprehensive evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()