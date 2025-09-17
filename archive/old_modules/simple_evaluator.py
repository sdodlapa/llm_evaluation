"""
Simple evaluation function for comprehensive runner integration
"""

import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from .run_evaluation import LLMEvaluationRunner
    from .enhanced_dataset_manager import EnhancedDatasetManager
except ImportError:
    from run_evaluation import LLMEvaluationRunner
    from enhanced_dataset_manager import EnhancedDatasetManager

logger = logging.getLogger(__name__)

def evaluate_model(model_name: str, config, dataset_name: str, 
                  samples: List[Dict[str, Any]], performance_monitor=None) -> Dict[str, Any]:
    """
    Simple evaluation function for comprehensive runner
    
    Args:
        model_name: Name of the model to evaluate
        config: Model configuration (ModelConfig object)
        dataset_name: Name of the dataset
        samples: List of dataset samples
        performance_monitor: Optional performance monitor instance
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Create a temporary runner instance
        runner = LLMEvaluationRunner(output_dir="temp_results")
        
        # Create model instance using the ModelConfig directly
        model = runner._create_model_instance(model_name, config, "custom")
        if model is None:
            raise ValueError(f"Failed to create model instance for {model_name}")
        
        # Load the model
        logger.info(f"Loading model {model_name}...")
        if not model.load_model():
            raise ValueError(f"Failed to load model {model_name}")
        
        # Prepare dataset in expected format
        dataset = {
            "name": dataset_name,
            "task_type": _get_task_type(dataset_name),
            "samples": samples
        }
        
        logger.info(f"Starting evaluation: {model_name} on {dataset_name} ({len(samples)} samples)")
        
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Run evaluation
        result = runner._evaluate_on_single_dataset(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            save_predictions=False,
            sample_limit=len(samples)
        )
        
        end_time = time.time()
        
        # Record tokens processed for performance monitoring
        if performance_monitor:
            # Estimate tokens processed (rough calculation)
            estimated_tokens = len(samples) * 200  # Average tokens per sample
            performance_monitor.record_tokens_processed(estimated_tokens)
            performance_monitor.record_request_timing(start_time, end_time)
        
        # Add timing information
        result["evaluation_time_seconds"] = end_time - start_time
        result["samples_processed"] = len(samples)
        result["model_name"] = model_name
        result["dataset_name"] = dataset_name
        
        logger.info(f"Evaluation completed: {model_name} on {dataset_name} in {end_time - start_time:.1f}s")
        
        # Clean up model resources
        try:
            if hasattr(model, 'unload_model'):
                model.unload_model()
            elif hasattr(model, 'cleanup'):
                model.cleanup()
        except Exception as e:
            logger.warning(f"Failed to cleanup model resources: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Evaluation failed for {model_name} on {dataset_name}: {str(e)}")
        raise e

def _get_task_type(dataset_name: str) -> str:
    """Get task type for dataset"""
    task_mapping = {
        'humaneval': 'coding',
        'mbpp': 'coding',
        'gsm8k': 'reasoning',
        'math': 'reasoning',
        'hellaswag': 'reasoning',
        'winogrande': 'reasoning',
        'mmlu': 'qa',
        'arc_challenge': 'qa',
        'mt_bench': 'instruction_following',
        'ifeval': 'instruction_following',
        'bfcl': 'function_calling',
        'toolllama': 'function_calling'
    }
    return task_mapping.get(dataset_name, 'general')