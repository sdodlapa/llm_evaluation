#!/usr/bin/env python3
"""
Analyze current evaluation results to determine optimal presets per model
Based on GPU utilization, speed, and performance metrics
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import statistics

def load_aggregated_results(results_dir: str) -> Dict[str, Any]:
    """Load all aggregated results from metrics directory"""
    metrics_dir = Path(results_dir) / "aggregated_metrics"
    results = {}
    
    for json_file in metrics_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            run_info = data['run_info']
            key = f"{run_info['model_name']}_{run_info['preset']}_{run_info['dataset']}"
            results[key] = data
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def extract_performance_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract key performance metrics by model and preset"""
    performance_data = {}
    
    for key, data in results.items():
        model_name = data['run_info']['model_name']
        preset = data['run_info']['preset']
        dataset = data['run_info']['dataset']
        
        # Get performance metrics
        perf = data.get('performance_metrics', {})
        eval_result = data.get('evaluation_result', {})
        
        # Extract key metrics
        throughput = perf.get('avg_throughput_tokens_per_second', 0)
        gpu_util = perf.get('avg_gpu_utilization', 0)
        memory_gb = perf.get('peak_gpu_memory_gb', 0)
        memory_efficiency = perf.get('memory_efficiency', 0)
        
        # Get accuracy score (varies by task type)
        accuracy = 0
        if 'metrics' in eval_result:
            metrics = eval_result['metrics']
            if 'code_execution' in metrics:
                accuracy = metrics['code_execution']['score']
            elif 'exact_match' in metrics:
                accuracy = metrics['exact_match']['score']
            elif 'accuracy' in metrics:
                accuracy = metrics['accuracy']['score']
        
        model_preset = f"{model_name}_{preset}"
        if model_preset not in performance_data:
            performance_data[model_preset] = {
                'throughput': [],
                'gpu_utilization': [],
                'memory_gb': [],
                'memory_efficiency': [],
                'accuracy': [],
                'datasets': []
            }
        
        performance_data[model_preset]['throughput'].append(throughput)
        performance_data[model_preset]['gpu_utilization'].append(gpu_util)
        performance_data[model_preset]['memory_gb'].append(memory_gb)
        performance_data[model_preset]['memory_efficiency'].append(memory_efficiency)
        performance_data[model_preset]['accuracy'].append(accuracy)
        performance_data[model_preset]['datasets'].append(dataset)
    
    return performance_data

def calculate_optimal_presets(performance_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Any]]:
    """Calculate optimal preset per model based on weighted scoring"""
    
    model_summaries = {}
    
    # Group by model
    models = {}
    for model_preset, data in performance_data.items():
        model, preset = model_preset.rsplit('_', 1)
        if model not in models:
            models[model] = {}
        models[model][preset] = data
    
    for model, presets in models.items():
        model_summaries[model] = {'presets': {}, 'optimal_preset': None, 'reasoning': []}
        
        preset_scores = {}
        
        for preset, data in presets.items():
            # Calculate averages
            avg_throughput = statistics.mean(data['throughput']) if data['throughput'] else 0
            avg_gpu_util = statistics.mean(data['gpu_utilization']) if data['gpu_utilization'] else 0
            avg_memory_gb = statistics.mean(data['memory_gb']) if data['memory_gb'] else 0
            avg_memory_eff = statistics.mean(data['memory_efficiency']) if data['memory_efficiency'] else 0
            avg_accuracy = statistics.mean(data['accuracy']) if data['accuracy'] else 0
            
            # Weighted scoring (customize weights as needed)
            # Throughput: 30%, GPU Utilization: 20%, Memory Efficiency: 20%, Accuracy: 30%
            score = (
                (avg_throughput / 100) * 0.30 +  # Normalize throughput 
                (avg_gpu_util / 100) * 0.20 +
                (avg_memory_eff / 100) * 0.20 +
                (avg_accuracy) * 0.30
            )
            
            preset_scores[preset] = score
            model_summaries[model]['presets'][preset] = {
                'avg_throughput': avg_throughput,
                'avg_gpu_utilization': avg_gpu_util,
                'avg_memory_gb': avg_memory_gb,
                'avg_memory_efficiency': avg_memory_eff,
                'avg_accuracy': avg_accuracy,
                'composite_score': score,
                'datasets_tested': len(data['datasets'])
            }
        
        # Find optimal preset
        if preset_scores:
            optimal_preset = max(preset_scores.keys(), key=lambda x: preset_scores[x])
            model_summaries[model]['optimal_preset'] = optimal_preset
            
            # Add reasoning
            optimal_data = model_summaries[model]['presets'][optimal_preset]
            model_summaries[model]['reasoning'] = [
                f"Highest composite score: {optimal_data['composite_score']:.3f}",
                f"Throughput: {optimal_data['avg_throughput']:.1f} tokens/sec",
                f"GPU Utilization: {optimal_data['avg_gpu_utilization']:.1f}%",
                f"Memory Efficiency: {optimal_data['avg_memory_efficiency']:.1f}%",
                f"Average Accuracy: {optimal_data['avg_accuracy']:.3f}",
                f"Tested on {optimal_data['datasets_tested']} datasets"
            ]
    
    return model_summaries

def main():
    results_dir = "comprehensive_results_20250917_173559"
    
    print("🔍 Analyzing Current Evaluation Results for Optimal Presets")
    print("=" * 60)
    
    # Load results
    results = load_aggregated_results(results_dir)
    print(f"📊 Loaded {len(results)} evaluation results")
    
    # Extract performance metrics
    performance_data = extract_performance_metrics(results)
    print(f"📈 Analyzed performance for {len(performance_data)} model-preset combinations")
    
    # Calculate optimal presets
    optimal_analysis = calculate_optimal_presets(performance_data)
    
    # Print detailed analysis
    print("\n🎯 OPTIMAL PRESET ANALYSIS")
    print("=" * 60)
    
    for model, analysis in optimal_analysis.items():
        print(f"\n🤖 Model: {model}")
        print(f"✅ Optimal Preset: {analysis['optimal_preset']}")
        
        print("\n📊 All Preset Performance:")
        for preset, metrics in analysis['presets'].items():
            marker = "⭐" if preset == analysis['optimal_preset'] else "  "
            print(f"{marker} {preset}:")
            print(f"    Throughput: {metrics['avg_throughput']:.1f} tokens/sec")
            print(f"    GPU Util: {metrics['avg_gpu_utilization']:.1f}%")
            print(f"    Memory: {metrics['avg_memory_gb']:.1f}GB")
            print(f"    Memory Eff: {metrics['avg_memory_efficiency']:.1f}%")
            print(f"    Accuracy: {metrics['avg_accuracy']:.3f}")
            print(f"    Score: {metrics['composite_score']:.3f}")
        
        print(f"\n💡 Reasoning:")
        for reason in analysis['reasoning']:
            print(f"   • {reason}")
    
    # Generate next evaluation plan
    print(f"\n🚀 NEXT EVALUATION PLAN: Unimplemented Datasets")
    print("=" * 60)
    
    unimplemented_datasets = [
        "math", "bfcl", "toolllama", "mmlu", "ifeval", "winogrande"
    ]
    
    total_combinations = 0
    for model, analysis in optimal_analysis.items():
        optimal_preset = analysis['optimal_preset']
        print(f"🤖 {model} with {optimal_preset} preset")
        total_combinations += len(unimplemented_datasets)
    
    print(f"\n📈 Total combinations: {len(optimal_analysis)} models × 1 optimal preset × {len(unimplemented_datasets)} datasets = {total_combinations}")
    print(f"📅 Estimated time: {total_combinations * 8.5:.1f} minutes = {total_combinations * 8.5 / 60:.1f} hours")
    
    # Save optimal configuration
    optimal_config = {
        "optimal_presets": {model: analysis['optimal_preset'] for model, analysis in optimal_analysis.items()},
        "unimplemented_datasets": unimplemented_datasets,
        "analysis_timestamp": "2025-09-17",
        "detailed_analysis": optimal_analysis
    }
    
    with open("optimal_presets_config.json", 'w') as f:
        json.dump(optimal_config, f, indent=2)
    
    print(f"\n💾 Optimal configuration saved to: optimal_presets_config.json")

if __name__ == "__main__":
    main()