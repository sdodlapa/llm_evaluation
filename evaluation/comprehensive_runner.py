"""
Comprehensive Evaluation Runner
Orchestrates evaluation of all models across all datasets with live monitoring
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

from .performance_monitor import LivePerformanceMonitor, EvaluationMetrics
from .dataset_manager import EnhancedDatasetManager
from .json_serializer import safe_json_dump
from .run_evaluation import evaluate_model

try:
    from ..configs.model_configs import MODEL_CONFIGS
except ImportError:
    # When running as script, use absolute import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from configs.model_configs import MODEL_CONFIGS

logger = logging.getLogger(__name__)

class ResultsOrganizer:
    """Organizes and saves evaluation results in structured format"""
    
    def __init__(self, base_results_path: str = "comprehensive_results"):
        self.base_path = Path(base_results_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create organized directory structure
        self.paths = {
            'raw_results': self.base_path / "raw_results",
            'performance_data': self.base_path / "performance_data", 
            'aggregated_metrics': self.base_path / "aggregated_metrics",
            'reports': self.base_path / "reports",
            'detailed_snapshots': self.base_path / "detailed_snapshots"
        }
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def save_evaluation_result(self, model_name: str, preset: str, dataset: str, 
                             evaluation_result: Dict[str, Any], 
                             performance_metrics: EvaluationMetrics):
        """Save individual evaluation result with performance data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{model_name}_{preset}_{dataset}_{timestamp}"
        
        # Save raw evaluation result
        raw_file = self.paths['raw_results'] / f"{run_id}_result.json"
        if not safe_json_dump(evaluation_result, raw_file, indent=2):
            logger.error(f"Failed to save raw results: {raw_file}")
        
        # Save performance metrics
        perf_data = {
            'run_id': run_id,
            'model_name': model_name,
            'preset': preset,
            'dataset': dataset,
            'start_time': performance_metrics.start_time.isoformat(),
            'end_time': performance_metrics.end_time.isoformat(),
            'duration_seconds': performance_metrics.total_duration_seconds,
            
            # Performance metrics
            'gpu_metrics': {
                'avg_utilization_percent': performance_metrics.avg_gpu_utilization,
                'peak_utilization_percent': performance_metrics.peak_gpu_utilization,
                'avg_memory_gb': performance_metrics.avg_gpu_memory_gb,
                'peak_memory_gb': performance_metrics.peak_gpu_memory_gb,
                'avg_temperature_c': performance_metrics.avg_gpu_temperature,
                'peak_temperature_c': performance_metrics.peak_gpu_temperature
            },
            
            # Throughput metrics
            'throughput_metrics': {
                'total_tokens_processed': performance_metrics.total_tokens_processed,
                'avg_throughput_tokens_per_second': performance_metrics.avg_throughput_tokens_per_second,
                'peak_throughput_tokens_per_second': performance_metrics.peak_throughput_tokens_per_second,
                'avg_latency_ms': performance_metrics.avg_latency_ms
            },
            
            # Efficiency metrics
            'efficiency_metrics': {
                'memory_efficiency_percent': performance_metrics.memory_efficiency,
                'gpu_utilization_efficiency_percent': performance_metrics.gpu_utilization_efficiency,
                'tokens_per_gb_memory': performance_metrics.tokens_per_gb_memory
            },
            
            # Task metrics
            'task_metrics': {
                'dataset_samples_processed': performance_metrics.dataset_samples_processed,
                'accuracy_metrics': performance_metrics.accuracy_metrics,
                'task_specific_metrics': performance_metrics.task_specific_metrics
            }
        }
        
        perf_file = self.paths['performance_data'] / f"{run_id}_performance.json"
        if not safe_json_dump(perf_data, perf_file, indent=2):
            logger.error(f"Failed to save performance data: {perf_file}")
        
        # Save combined result
        combined_result = {
            'run_info': {
                'run_id': run_id,
                'model_name': model_name,
                'preset': preset,
                'dataset': dataset,
                'timestamp': timestamp
            },
            'evaluation_result': evaluation_result,
            'performance_metrics': perf_data
        }
        
        combined_file = self.paths['aggregated_metrics'] / f"{run_id}_combined.json"
        if not safe_json_dump(combined_result, combined_file, indent=2):
            logger.error(f"Failed to save combined results: {combined_file}")
        
        logger.info(f"Saved results for {run_id}")
        return run_id
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.paths['reports'] / f"comprehensive_evaluation_summary_{timestamp}.md"
        
        # Analyze results
        total_runs = len(all_results)
        successful_runs = [r for r in all_results if r.get('status') == 'success']
        failed_runs = [r for r in all_results if r.get('status') == 'error']
        
        models_tested = set()
        datasets_tested = set()
        presets_tested = set()
        
        for result in all_results:
            if 'run_info' in result:
                models_tested.add(result['run_info']['model_name'])
                datasets_tested.add(result['run_info']['dataset'])
                presets_tested.add(result['run_info']['preset'])
        
        # Performance statistics
        throughput_stats = []
        memory_stats = []
        
        for result in successful_runs:
            if 'run_info' in result and 'performance_metrics' in result:
                perf = result['performance_metrics']
                
                # Handle different performance metric structures
                if isinstance(perf, dict):
                    throughput = perf.get('avg_throughput_tokens_per_second', 0)
                    memory = perf.get('peak_gpu_memory_gb', 0)
                else:
                    # Handle object structure
                    throughput = getattr(perf, 'avg_throughput_tokens_per_second', 0)
                    memory = getattr(perf, 'peak_gpu_memory_gb', 0)
                
                throughput_stats.append(throughput)
                memory_stats.append(memory)
        
        avg_throughput = sum(throughput_stats) / len(throughput_stats) if throughput_stats else 0
        avg_memory = sum(memory_stats) / len(memory_stats) if memory_stats else 0
        
        # Generate report
        report_content = f"""# Comprehensive LLM Evaluation Summary
        
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Evaluation Runs**: {total_runs}
- **Successful Runs**: {len(successful_runs)} ({len(successful_runs)/total_runs*100:.1f}%)
- **Failed Runs**: {len(failed_runs)} ({len(failed_runs)/total_runs*100:.1f}%)

## Coverage
- **Models Tested**: {len(models_tested)} ({', '.join(sorted(models_tested))})
- **Datasets Tested**: {len(datasets_tested)} ({', '.join(sorted(datasets_tested))})
- **Presets Tested**: {len(presets_tested)} ({', '.join(sorted(presets_tested))})

## Performance Summary
- **Average Throughput**: {avg_throughput:.1f} tokens/second
- **Average Peak Memory**: {avg_memory:.1f} GB
- **Total Evaluation Time**: {sum(r.get('performance_metrics', {}).get('duration_seconds', 0) for r in successful_runs)/3600:.1f} hours

## Model Performance Comparison

| Model | Preset | Avg Throughput (tok/s) | Peak Memory (GB) | Success Rate |
|-------|--------|----------------------|------------------|--------------|
"""
        
        # Add performance comparison table
        model_performance = {}
        for result in successful_runs:
            if 'run_info' in result and 'performance_metrics' in result:
                run_info = result['run_info']
                perf = result['performance_metrics']
                
                key = f"{run_info['model_name']}_{run_info['preset']}"
                if key not in model_performance:
                    model_performance[key] = {
                        'throughputs': [],
                        'memories': [],
                        'total_runs': 0,
                        'successful_runs': 0
                    }
                
                model_performance[key]['throughputs'].append(
                    perf['throughput_metrics']['avg_throughput_tokens_per_second']
                )
                model_performance[key]['memories'].append(
                    perf['gpu_metrics']['peak_memory_gb']
                )
                model_performance[key]['successful_runs'] += 1
        
        # Count total runs per model-preset
        for result in all_results:
            if 'run_info' in result:
                run_info = result['run_info']
                key = f"{run_info['model_name']}_{run_info['preset']}"
                if key in model_performance:
                    model_performance[key]['total_runs'] += 1
        
        for key, stats in sorted(model_performance.items()):
            model, preset = key.split('_', 1)
            avg_throughput = sum(stats['throughputs']) / len(stats['throughputs'])
            avg_memory = sum(stats['memories']) / len(stats['memories'])
            success_rate = stats['successful_runs'] / stats['total_runs'] * 100
            
            report_content += f"| {model} | {preset} | {avg_throughput:.1f} | {avg_memory:.1f} | {success_rate:.1f}% |\n"
        
        report_content += f"""
## Dataset Performance

| Dataset | Avg Accuracy | Avg Throughput | Runs |
|---------|--------------|----------------|------|
"""
        
        # Add dataset performance
        dataset_performance = {}
        for result in successful_runs:
            if 'run_info' in result and 'evaluation_result' in result:
                dataset = result['run_info']['dataset']
                eval_result = result['evaluation_result']
                
                if dataset not in dataset_performance:
                    dataset_performance[dataset] = {
                        'accuracies': [],
                        'throughputs': [],
                        'count': 0
                    }
                
                # Extract accuracy if available
                if 'accuracy' in eval_result:
                    dataset_performance[dataset]['accuracies'].append(eval_result['accuracy'])
                
                # Extract throughput
                if 'performance_metrics' in result:
                    throughput = result['performance_metrics']['throughput_metrics']['avg_throughput_tokens_per_second']
                    dataset_performance[dataset]['throughputs'].append(throughput)
                
                dataset_performance[dataset]['count'] += 1
        
        for dataset, stats in sorted(dataset_performance.items()):
            avg_accuracy = sum(stats['accuracies']) / len(stats['accuracies']) if stats['accuracies'] else 0
            avg_throughput = sum(stats['throughputs']) / len(stats['throughputs']) if stats['throughputs'] else 0
            
            report_content += f"| {dataset} | {avg_accuracy:.3f} | {avg_throughput:.1f} | {stats['count']} |\n"
        
        # Add failure analysis
        if failed_runs:
            report_content += f"""
## Failure Analysis

Total failed runs: {len(failed_runs)}

### Common Failure Reasons:
"""
            
            failure_reasons = {}
            for result in failed_runs:
                error = result.get('error', 'Unknown error')
                reason = error.split(':')[0] if ':' in error else error
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                report_content += f"- **{reason}**: {count} occurrences\n"
        
        report_content += f"""
## Next Steps

1. **Address Failures**: Investigate and fix the {len(failed_runs)} failed evaluations
2. **Optimize Performance**: Focus on models with low throughput or high memory usage
3. **Expand Coverage**: Consider adding more datasets or evaluation scenarios
4. **Resource Planning**: Use memory and throughput data for infrastructure scaling

---
*Report generated by Comprehensive LLM Evaluation System*
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated summary report: {report_file}")
        return str(report_file)

class ComprehensiveEvaluationRunner:
    """Main runner for comprehensive evaluation across all models and datasets"""
    
    def __init__(self, base_results_path: str = "comprehensive_results"):
        self.dataset_manager = EnhancedDatasetManager()
        self.results_organizer = ResultsOrganizer(base_results_path)
        self.performance_monitor = LivePerformanceMonitor()
        
        # Load model configurations
        self.model_configs = MODEL_CONFIGS
        
        # Track progress
        self.all_results = []
        self.current_run = 0
        self.total_runs = 0
        
    def run_comprehensive_evaluation(self, 
                                   models: Optional[List[str]] = None,
                                   presets: Optional[List[str]] = None,
                                   datasets: Optional[List[str]] = None,
                                   samples_per_dataset: Optional[Dict[str, int]] = None,
                                   continue_on_failure: bool = True) -> Dict[str, Any]:
        """Run comprehensive evaluation across specified models, presets, and datasets"""
        
        # Set defaults
        if models is None:
            models = list(self.model_configs.keys())
        if presets is None:
            presets = ['memory_optimized', 'balanced', 'performance']
        if datasets is None:
            datasets = self.dataset_manager.get_implemented_datasets()
        if samples_per_dataset is None:
            samples_per_dataset = self.dataset_manager.get_recommended_sample_counts()
        
        # Calculate total runs
        self.total_runs = len(models) * len(presets) * len(datasets)
        self.current_run = 0
        
        logger.info(f"Starting comprehensive evaluation:")
        logger.info(f"  Models: {len(models)} ({models})")
        logger.info(f"  Presets: {len(presets)} ({presets})")
        logger.info(f"  Datasets: {len(datasets)} ({datasets})")
        logger.info(f"  Total combinations: {self.total_runs}")
        
        start_time = datetime.now()
        
        # Run evaluations
        for model_name in models:
            for preset in presets:
                for dataset in datasets:
                    self.current_run += 1
                    
                    logger.info(f"Running evaluation {self.current_run}/{self.total_runs}: {model_name}_{preset} on {dataset}")
                    
                    try:
                        result = self._run_single_evaluation(
                            model_name, preset, dataset, 
                            samples_per_dataset.get(dataset, 100)
                        )
                        result['status'] = 'success'
                        
                    except Exception as e:
                        logger.error(f"Evaluation failed for {model_name}_{preset} on {dataset}: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        result = {
                            'run_info': {
                                'model_name': model_name,
                                'preset': preset,
                                'dataset': dataset,
                                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                            },
                            'status': 'error',
                            'error': str(e),
                            'error_traceback': traceback.format_exc()
                        }
                        
                        if not continue_on_failure:
                            break
                    
                    self.all_results.append(result)
                    
                    # Save intermediate results
                    intermediate_file = self.results_organizer.base_path / "intermediate_results.json"
                    if not safe_json_dump(self.all_results, intermediate_file, indent=2):
                        logger.error(f"Failed to save intermediate results: {intermediate_file}")
                
                if not continue_on_failure and any(r.get('status') == 'error' for r in self.all_results[-len(datasets):]):
                    break
            
            if not continue_on_failure and any(r.get('status') == 'error' for r in self.all_results[-len(presets)*len(datasets):]):
                break
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate final summary
        summary = {
            'evaluation_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'total_runs': len(self.all_results),
                'successful_runs': len([r for r in self.all_results if r.get('status') == 'success']),
                'failed_runs': len([r for r in self.all_results if r.get('status') == 'error']),
                'models_tested': models,
                'presets_tested': presets,
                'datasets_tested': datasets
            },
            'all_results': self.all_results
        }
        
        # Save final results
        final_file = self.results_organizer.base_path / f"final_comprehensive_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        if not safe_json_dump(summary, final_file, indent=2):
            logger.error(f"Failed to save final results: {final_file}")
        else:
            logger.info(f"Results saved to: {final_file}")
        logger.info(f"Report generated: {report_file}")
        
        return summary
    
    def _run_single_evaluation(self, model_name: str, preset: str, dataset: str, num_samples: int) -> Dict[str, Any]:
        """Run single model-preset-dataset evaluation with performance monitoring"""
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(model_name, preset, dataset)
        
        try:
            # Load dataset
            samples = self.dataset_manager.load_dataset(dataset, num_samples)
            
            # Get model configuration
            if model_name not in self.model_configs:
                raise ValueError(f"Model configuration not found: {model_name}")
            
            base_config = self.model_configs[model_name]
            model_config = base_config.create_preset_variant(preset)
            
            # Prepare evaluation parameters
            eval_params = {
                'model_name': model_name,
                'config': model_config,
                'dataset_name': dataset,
                'samples': samples,
                'performance_monitor': self.performance_monitor
            }
            
            # Run evaluation
            evaluation_result = evaluate_model(**eval_params)
            
            # Extract accuracy metrics for performance tracking
            accuracy_metrics = {}
            if 'accuracy' in evaluation_result:
                accuracy_metrics['accuracy'] = evaluation_result['accuracy']
            if 'pass_rate' in evaluation_result:
                accuracy_metrics['pass_rate'] = evaluation_result['pass_rate']
            
            # Stop monitoring and get metrics
            performance_metrics = self.performance_monitor.stop_monitoring(
                dataset_samples_processed=len(samples),
                accuracy_metrics=accuracy_metrics,
                task_specific_metrics=evaluation_result.get('metrics', {})
            )
            
            # Save detailed snapshots
            snapshot_file = self.results_organizer.paths['detailed_snapshots'] / f"{model_name}_{preset}_{dataset}_snapshots.json"
            self.performance_monitor.save_detailed_snapshots(str(snapshot_file))
            
            # Save results
            run_id = self.results_organizer.save_evaluation_result(
                model_name, preset, dataset, evaluation_result, performance_metrics
            )
            
            return {
                'run_info': {
                    'run_id': run_id,
                    'model_name': model_name,
                    'preset': preset,
                    'dataset': dataset,
                    'samples_processed': len(samples),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                },
                'evaluation_result': evaluation_result,
                'performance_metrics': performance_metrics.__dict__
            }
            
        except Exception as e:
            # Stop monitoring even on failure
            try:
                self.performance_monitor.stop_monitoring(0, {}, {})
            except:
                pass
            raise e
    
    def run_quick_validation(self, model_name: str = "qwen3_8b", 
                           preset: str = "memory_optimized", 
                           dataset: str = "humaneval",
                           num_samples: int = 5) -> Dict[str, Any]:
        """Run quick validation test with live monitoring"""
        logger.info(f"Running quick validation: {model_name}_{preset} on {dataset} ({num_samples} samples)")
        
        result = self._run_single_evaluation(model_name, preset, dataset, num_samples)
        
        logger.info("Quick validation completed successfully!")
        logger.info(f"Throughput: {result['performance_metrics']['avg_throughput_tokens_per_second']:.1f} tokens/sec")
        logger.info(f"Peak Memory: {result['performance_metrics']['peak_gpu_memory_gb']:.1f} GB")
        
        return result
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress status"""
        if self.total_runs == 0:
            return {'status': 'not_started', 'progress': 0.0}
        
        progress = self.current_run / self.total_runs
        successful_runs = len([r for r in self.all_results if r.get('status') == 'success'])
        failed_runs = len([r for r in self.all_results if r.get('status') == 'error'])
        
        return {
            'status': 'running' if self.current_run < self.total_runs else 'completed',
            'progress': progress,
            'current_run': self.current_run,
            'total_runs': self.total_runs,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'success_rate': successful_runs / len(self.all_results) if self.all_results else 0.0
        }
    
    def run_optimal_evaluation(self, 
                             model_preset_combinations: List[tuple],
                             datasets: List[str],
                             samples_per_dataset: Dict[str, int],
                             continue_on_failure: bool = True) -> Dict[str, Any]:
        """
        Run evaluation with optimal model-preset combinations on specific datasets
        
        Args:
            model_preset_combinations: List of (model_name, preset) tuples
            datasets: List of dataset names to evaluate
            samples_per_dataset: Number of samples per dataset
            continue_on_failure: Whether to continue if a model fails
            
        Returns:
            Dictionary with evaluation results and summary
        """
        logger.info(f"🎯 Starting optimal evaluation")
        logger.info(f"Models: {len(model_preset_combinations)} optimal combinations")
        logger.info(f"Datasets: {len(datasets)} datasets")
        
        # Initialize evaluation state
        self.all_results = []
        self.current_run = 0
        self.total_runs = len(model_preset_combinations) * len(datasets)
        
        logger.info(f"📊 Total evaluations: {self.total_runs}")
        
        dataset_manager = EnhancedDatasetManager()
        
        evaluation_start_time = time.time()
        
        # Run evaluations
        for model_name, preset in model_preset_combinations:
            logger.info(f"\n🤖 Processing {model_name} with {preset} preset")
            
            for dataset_name in datasets:
                self.current_run += 1
                
                logger.info(f"\n📋 Evaluation {self.current_run}/{self.total_runs}: "
                           f"{model_name}_{preset} on {dataset_name}")
                
                try:
                    # Check if dataset is implemented
                    if dataset_name not in dataset_manager.get_implemented_datasets():
                        logger.warning(f"⚠️ Dataset {dataset_name} not implemented - skipping")
                        
                        # Add placeholder result for unimplemented dataset
                        result = {
                            'run_info': {
                                'run_id': f"{model_name}_{preset}_{dataset_name}_placeholder",
                                'model_name': model_name,
                                'preset': preset,
                                'dataset': dataset_name,
                                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                            },
                            'status': 'skipped',
                            'reason': 'dataset_not_implemented',
                            'evaluation_result': None,
                            'performance_metrics': None
                        }
                        self.all_results.append(result)
                        continue
                    
                    # Load dataset with sampling
                    num_samples = samples_per_dataset.get(dataset_name, 100)
                    logger.info(f"📊 Loading {num_samples} samples from {dataset_name}")
                    
                    try:
                        dataset_samples = dataset_manager.load_dataset(dataset_name, num_samples)
                        logger.info(f"✅ Loaded {len(dataset_samples)} samples")
                    except Exception as e:
                        logger.error(f"❌ Failed to load dataset {dataset_name}: {e}")
                        
                        result = {
                            'run_info': {
                                'run_id': f"{model_name}_{preset}_{dataset_name}_failed",
                                'model_name': model_name,
                                'preset': preset,
                                'dataset': dataset_name,
                                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                            },
                            'status': 'error',
                            'reason': f'dataset_load_failed: {str(e)}',
                            'evaluation_result': None,
                            'performance_metrics': None
                        }
                        self.all_results.append(result)
                        continue
                    
                    # Run evaluation with performance monitoring
                    evaluation_result, performance_metrics = self._run_single_evaluation(
                        model_name, preset, dataset_name, dataset_samples
                    )
                    
                    # Save results
                    run_id = self.results_organizer.save_evaluation_result(
                        model_name, preset, dataset_name, 
                        evaluation_result, performance_metrics
                    )
                    
                    # Store in results list
                    result = {
                        'run_info': {
                            'run_id': run_id,
                            'model_name': model_name,
                            'preset': preset,
                            'dataset': dataset_name,
                            'samples_processed': len(dataset_samples),
                            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                        },
                        'status': 'success',
                        'evaluation_result': evaluation_result,
                        'performance_metrics': performance_metrics.to_dict() if performance_metrics else None
                    }
                    self.all_results.append(result)
                    
                    logger.info(f"✅ Completed {model_name}_{preset} on {dataset_name}")
                    
                except Exception as e:
                    logger.error(f"💥 Evaluation failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    result = {
                        'run_info': {
                            'run_id': f"{model_name}_{preset}_{dataset_name}_error",
                            'model_name': model_name,
                            'preset': preset,
                            'dataset': dataset_name,
                            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                        },
                        'status': 'error',
                        'reason': str(e),
                        'evaluation_result': None,
                        'performance_metrics': None
                    }
                    self.all_results.append(result)
                    
                    if not continue_on_failure:
                        logger.error("🛑 Stopping evaluation due to failure")
                        break
        
        evaluation_end_time = time.time()
        total_evaluation_time = evaluation_end_time - evaluation_start_time
        
        # Generate summary
        successful_runs = [r for r in self.all_results if r.get('status') == 'success']
        failed_runs = [r for r in self.all_results if r.get('status') == 'error']
        skipped_runs = [r for r in self.all_results if r.get('status') == 'skipped']
        
        summary = {
            'evaluation_type': 'optimal_evaluation',
            'total_combinations': self.total_runs,
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs),
            'skipped_runs': len(skipped_runs),
            'success_rate': len(successful_runs) / self.total_runs if self.total_runs > 0 else 0,
            'total_time_seconds': total_evaluation_time,
            'total_time_hours': total_evaluation_time / 3600,
            'avg_time_per_evaluation': total_evaluation_time / self.total_runs if self.total_runs > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'all_results': self.all_results
        }
        
        # Save summary
        summary_file = self.results_organizer.base_path / "optimal_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n🏁 Optimal evaluation completed!")
        logger.info(f"✅ Successful: {len(successful_runs)}")
        logger.info(f"❌ Failed: {len(failed_runs)}")
        logger.info(f"⏭️ Skipped: {len(skipped_runs)}")
        logger.info(f"⏱️ Total time: {total_evaluation_time/3600:.2f} hours")
        logger.info(f"📁 Results saved to: {self.results_organizer.base_path}")
        
        return summary