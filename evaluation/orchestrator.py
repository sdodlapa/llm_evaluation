"""
Evaluation Orchestrator - High-level coordination of evaluation processes
Manages evaluation workflows, result aggregation, and error handling
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from models.base_model import BaseModelImplementation, ModelPerformanceMetrics, AgentEvaluationResult
from configs.model_configs import ModelConfig, get_model_config

try:
    from .evaluation_engine import EvaluationEngine
    from .result_processor import ResultProcessor
    from .dataset_manager import EnhancedDatasetManager
    from .performance_monitor import LivePerformanceMonitor
except ImportError:
    # Handle when running as script
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from evaluation.evaluation_engine import EvaluationEngine
    from evaluation.result_processor import ResultProcessor
    from evaluation.dataset_manager import EnhancedDatasetManager
    from evaluation.performance_monitor import LivePerformanceMonitor

logger = logging.getLogger(__name__)

class EvaluationOrchestrator:
    """High-level orchestrator for LLM evaluation workflows"""
    
    def __init__(self, cache_dir: Optional[str] = None, data_cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or "model_cache"
        self.data_cache_dir = data_cache_dir or "evaluation_data"
        
        # Initialize components
        self.evaluation_engine = EvaluationEngine(cache_dir, data_cache_dir)
        self.result_processor = ResultProcessor()
        self.dataset_manager = EnhancedDatasetManager(self.data_cache_dir)
        self.performance_monitor = LivePerformanceMonitor()
        
        # Create cache directories
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def run_comprehensive_evaluation(self, 
                                   models: List[str],
                                   datasets: List[str],
                                   presets: List[str] = None,
                                   sample_limit: Optional[int] = None,
                                   save_results: bool = True,
                                   continue_on_error: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across multiple models and datasets
        
        Args:
            models: List of model names to evaluate
            datasets: List of dataset names to evaluate on
            presets: List of presets to test (default: ["balanced"])
            sample_limit: Maximum samples per dataset
            save_results: Whether to save results to files
            continue_on_error: Whether to continue evaluation on errors
        
        Returns:
            Dict containing all evaluation results
        """
        presets = presets or ["balanced"]
        start_time = time.time()
        
        logger.info(f"Starting comprehensive evaluation: {len(models)} models × {len(datasets)} datasets × {len(presets)} presets")
        
        # Initialize results structure
        results = {
            'evaluation_metadata': {
                'start_time': datetime.now().isoformat(),
                'models': models,
                'datasets': datasets,
                'presets': presets,
                'sample_limit': sample_limit,
                'continue_on_error': continue_on_error
            },
            'results': {},
            'summary': {},
            'errors': []
        }
        
        # Track progress
        total_evaluations = len(models) * len(datasets) * len(presets)
        completed_evaluations = 0
        
        # Evaluate each model-dataset-preset combination
        for model_name in models:
            model_results = {}
            
            try:
                logger.info(f"Processing model: {model_name}")
                
                # Get model configuration
                model_config = get_model_config(model_name)
                if not model_config:
                    error_msg = f"Model configuration not found for {model_name}"
                    logger.error(error_msg)
                    results['errors'].append({
                        'model': model_name,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    if not continue_on_error:
                        raise ValueError(error_msg)
                    continue
                
                # Evaluate with each preset
                for preset in presets:
                    preset_results = {}
                    
                    try:
                        logger.info(f"Creating model instance: {model_name} with preset {preset}")
                        
                        # Create model instance
                        model = self.evaluation_engine.create_model_instance(model_name, model_config, preset)
                        if not model:
                            error_msg = f"Failed to create model instance for {model_name}"
                            logger.error(error_msg)
                            results['errors'].append({
                                'model': model_name,
                                'preset': preset,
                                'error': error_msg,
                                'timestamp': datetime.now().isoformat()
                            })
                            if not continue_on_error:
                                raise ValueError(error_msg)
                            continue
                        
                        # Run performance benchmark
                        performance_metrics = self.evaluation_engine.run_performance_benchmark(model, preset, model_config)
                        preset_results['performance_metrics'] = performance_metrics
                        
                        # Evaluate on each dataset
                        dataset_results = {}
                        for dataset_name in datasets:
                            try:
                                logger.info(f"Evaluating {model_name}({preset}) on {dataset_name}")
                                
                                # Run dataset evaluation
                                dataset_result = self.evaluation_engine.evaluate_on_single_dataset(
                                    model, dataset_name, sample_limit, continue_on_error
                                )
                                dataset_results[dataset_name] = dataset_result
                                
                                completed_evaluations += 1
                                progress = (completed_evaluations / total_evaluations) * 100
                                logger.info(f"Progress: {completed_evaluations}/{total_evaluations} ({progress:.1f}%)")
                                
                            except Exception as e:
                                error_msg = f"Dataset evaluation failed: {model_name}({preset}) on {dataset_name}: {e}"
                                logger.error(error_msg)
                                results['errors'].append({
                                    'model': model_name,
                                    'preset': preset,
                                    'dataset': dataset_name,
                                    'error': str(e),
                                    'timestamp': datetime.now().isoformat()
                                })
                                
                                if not continue_on_error:
                                    raise
                                
                                # Add error placeholder
                                dataset_results[dataset_name] = {
                                    'error': str(e),
                                    'dataset': dataset_name
                                }
                                completed_evaluations += 1
                        
                        preset_results['dataset_results'] = dataset_results
                        
                    except Exception as e:
                        error_msg = f"Preset evaluation failed: {model_name}({preset}): {e}"
                        logger.error(error_msg)
                        results['errors'].append({
                            'model': model_name,
                            'preset': preset,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        if not continue_on_error:
                            raise
                    
                    model_results[preset] = preset_results
                
            except Exception as e:
                error_msg = f"Model evaluation failed: {model_name}: {e}"
                logger.error(error_msg)
                results['errors'].append({
                    'model': model_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                if not continue_on_error:
                    raise
            
            results['results'][model_name] = model_results
        
        # Calculate evaluation summary
        total_time = time.time() - start_time
        results['evaluation_metadata']['end_time'] = datetime.now().isoformat()
        results['evaluation_metadata']['total_time_seconds'] = total_time
        results['evaluation_metadata']['completed_evaluations'] = completed_evaluations
        results['evaluation_metadata']['total_planned_evaluations'] = total_evaluations
        
        # Generate summary statistics
        results['summary'] = self.result_processor.generate_evaluation_summary(results['results'])
        
        logger.info(f"Comprehensive evaluation completed in {total_time:.2f}s")
        logger.info(f"Completed {completed_evaluations}/{total_evaluations} evaluations")
        logger.info(f"Encountered {len(results['errors'])} errors")
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(results)
        
        return results
    
    def run_model_comparison(self, 
                           model_names: List[str],
                           preset: str = "balanced",
                           datasets: List[str] = None,
                           sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a focused comparison between specific models
        
        Args:
            model_names: List of models to compare
            preset: Preset configuration to use
            datasets: List of datasets (default: common evaluation datasets)
            sample_limit: Maximum samples per dataset
        
        Returns:
            Comparison results with detailed analysis
        """
        if not datasets:
            # Use common evaluation datasets
            datasets = ["humaneval", "mbpp", "gsm8k"]
        
        logger.info(f"Running model comparison: {model_names} on preset '{preset}'")
        
        # Run evaluation
        results = self.run_comprehensive_evaluation(
            models=model_names,
            datasets=datasets,
            presets=[preset],
            sample_limit=sample_limit,
            save_results=True,
            continue_on_error=True
        )
        
        # Generate comparison analysis
        comparison_analysis = self.result_processor.generate_model_comparison(
            results['results'], preset, datasets
        )
        
        # Add comparison analysis to results
        results['comparison_analysis'] = comparison_analysis
        
        return results
    
    def run_preset_comparison(self, 
                            model_name: str,
                            presets: List[str] = None,
                            datasets: List[str] = None,
                            sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare different presets for a single model
        
        Args:
            model_name: Model to evaluate
            presets: List of presets to compare
            datasets: List of datasets to evaluate on
            sample_limit: Maximum samples per dataset
        
        Returns:
            Preset comparison results
        """
        if not presets:
            presets = ["speed", "balanced", "quality"]
        
        if not datasets:
            datasets = ["humaneval", "mbpp", "gsm8k"]
        
        logger.info(f"Running preset comparison for {model_name}: {presets}")
        
        # Run evaluation
        results = self.run_comprehensive_evaluation(
            models=[model_name],
            datasets=datasets,
            presets=presets,
            sample_limit=sample_limit,
            save_results=True,
            continue_on_error=True
        )
        
        # Generate preset comparison analysis
        preset_analysis = self.result_processor.generate_preset_comparison(
            results['results'][model_name], datasets
        )
        
        # Add preset analysis to results
        results['preset_analysis'] = preset_analysis
        
        return results
    
    def run_single_evaluation(self,
                            model_name: str,
                            dataset_name: str,
                            preset: str = "balanced",
                            sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run evaluation for a single model-dataset combination
        
        Args:
            model_name: Model to evaluate
            dataset_name: Dataset to evaluate on
            preset: Preset configuration
            sample_limit: Maximum samples to process
        
        Returns:
            Single evaluation result
        """
        logger.info(f"Running single evaluation: {model_name} on {dataset_name} with preset {preset}")
        
        try:
            # Get model configuration
            model_config = get_model_config(model_name)
            if not model_config:
                raise ValueError(f"Model configuration not found for {model_name}")
            
            # Create model instance
            model = self.evaluation_engine.create_model_instance(model_name, model_config, preset)
            if not model:
                raise ValueError(f"Failed to create model instance for {model_name}")
            
            # Run performance benchmark
            performance_metrics = self.evaluation_engine.run_performance_benchmark(model, preset, model_config)
            
            # Run dataset evaluation
            dataset_result = self.evaluation_engine.evaluate_on_single_dataset(
                model, dataset_name, sample_limit, continue_on_error=False
            )
            
            # Compile results
            result = {
                'model': model_name,
                'dataset': dataset_name,
                'preset': preset,
                'performance_metrics': performance_metrics,
                'evaluation_result': dataset_result,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Single evaluation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Single evaluation failed: {e}")
            return {
                'model': model_name,
                'dataset': dataset_name,
                'preset': preset,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to files"""
        try:
            # Create results directory
            results_dir = Path("test_results")
            results_dir.mkdir(exist_ok=True)
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save comprehensive results
            results_file = results_dir / f"evaluation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            if 'summary' in results:
                summary_file = results_dir / f"evaluation_summary_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(results['summary'], f, indent=2, default=str)
            
            # Save comparison analysis if present
            if 'comparison_analysis' in results:
                comparison_file = results_dir / f"model_comparison_{timestamp}.json"
                with open(comparison_file, 'w') as f:
                    json.dump(results['comparison_analysis'], f, indent=2, default=str)
            
            if 'preset_analysis' in results:
                preset_file = results_dir / f"preset_comparison_{timestamp}.json"
                with open(preset_file, 'w') as f:
                    json.dump(results['preset_analysis'], f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.evaluation_engine.model_registry.list_models()
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        return self.dataset_manager.get_available_datasets()
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset"""
        return self.dataset_manager.get_dataset_info(dataset_name)
    
    def validate_configuration(self, models: List[str], datasets: List[str]) -> Dict[str, Any]:
        """Validate that models and datasets are available"""
        validation_result = {
            'valid': True,
            'issues': []
        }
        
        # Check models
        available_models = self.get_available_models()
        for model in models:
            if model not in available_models:
                validation_result['valid'] = False
                validation_result['issues'].append(f"Model '{model}' not found. Available: {available_models}")
        
        # Check datasets
        available_datasets = self.get_available_datasets()
        for dataset in datasets:
            if dataset not in available_datasets:
                validation_result['valid'] = False
                validation_result['issues'].append(f"Dataset '{dataset}' not found. Available: {available_datasets}")
        
        return validation_result