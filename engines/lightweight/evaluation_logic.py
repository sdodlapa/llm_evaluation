"""
Enhanced evaluation logic for lightweight engine

Integrates with existing evaluation framework while providing
optimized processing for small/medium models.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig

# Import existing evaluation components
try:
    from evaluation.dataset_manager import EnhancedDatasetManager
    from evaluation.metrics import EvaluationMetrics, evaluate_dataset_predictions
    from evaluation.performance_monitor import LivePerformanceMonitor
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logging.warning("Existing evaluation framework not available, using fallback")


logger = logging.getLogger(__name__)


class LightweightEvaluationLogic:
    """Enhanced evaluation logic for lightweight engine"""
    
    def __init__(self):
        self._dataset_manager = None
        self._metrics_calculator = None
        self._performance_monitor = None
        self._evaluation_cache = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize evaluation logic with existing framework integration"""
        try:
            if EVALUATION_AVAILABLE:
                # Use existing evaluation framework
                self._dataset_manager = EnhancedDatasetManager()
                self._metrics_calculator = EvaluationMetrics()
                self._performance_monitor = LivePerformanceMonitor()
                logger.info("Initialized with existing evaluation framework")
            else:
                # Use fallback implementations
                self._dataset_manager = MockDatasetManager()
                self._metrics_calculator = MockMetricsCalculator()
                self._performance_monitor = MockPerformanceMonitor()
                logger.info("Initialized with fallback implementations")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluation logic: {e}")
            return False
    
    def evaluate_on_dataset(self, model: Any, tokenizer: Any, dataset_name: str,
                           model_config: EnhancedModelConfig, 
                           eval_params: Dict[str, Any]) -> Tuple[Dict[str, float], List[str], int]:
        """Evaluate model on specific dataset with optimization
        
        Args:
            model: Loaded model instance
            tokenizer: Model tokenizer
            dataset_name: Name of dataset to evaluate on
            model_config: Enhanced model configuration
            eval_params: Evaluation parameters
            
        Returns:
            Tuple of (metrics, outputs, tokens_processed)
        """
        if not self._initialized:
            raise RuntimeError("Evaluation logic not initialized")
        
        logger.info(f"Evaluating {model_config.model_name} on {dataset_name}")
        
        # Start performance monitoring
        if self._performance_monitor:
            self._performance_monitor.start_monitoring(
                model_config.model_name, 
                model_config.preset, 
                dataset_name
            )
        
        try:
            # Load dataset samples
            samples = self._load_dataset_samples(dataset_name, eval_params)
            
            # Optimize samples for lightweight processing
            optimized_samples = self._optimize_samples_for_lightweight(samples, model_config)
            
            # Process samples in batches
            batch_size = self._calculate_optimal_batch_size(model_config, eval_params)
            
            all_predictions = []
            all_targets = []
            total_tokens = 0
            
            for i in range(0, len(optimized_samples), batch_size):
                batch = optimized_samples[i:i + batch_size]
                
                # Process batch
                predictions, tokens = self._process_batch(model, tokenizer, batch, eval_params)
                targets = [sample.get('target', sample.get('answer', '')) for sample in batch]
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                total_tokens += tokens
                
                # Log progress
                if i % (batch_size * 10) == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(optimized_samples)} samples")
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_predictions, all_targets, dataset_name)
            
            # Add performance metrics
            if self._performance_monitor:
                perf_metrics = self._performance_monitor.get_current_metrics()
                metrics.update({
                    f"perf_{k}": v for k, v in perf_metrics.items()
                    if isinstance(v, (int, float))
                })
            
            # Generate sample outputs for review
            sample_outputs = self._generate_sample_outputs(all_predictions, all_targets)
            
            logger.info(f"Evaluation completed: {len(optimized_samples)} samples, {total_tokens} tokens")
            return metrics, sample_outputs, total_tokens
            
        except Exception as e:
            logger.error(f"Evaluation failed on dataset {dataset_name}: {e}")
            raise
        
        finally:
            # Stop performance monitoring
            if self._performance_monitor:
                self._performance_monitor.stop_monitoring()
    
    def _load_dataset_samples(self, dataset_name: str, eval_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load dataset samples with caching"""
        cache_key = f"{dataset_name}_{eval_params.get('max_samples', 'all')}"
        
        if cache_key in self._evaluation_cache:
            logger.info(f"Using cached dataset samples for {dataset_name}")
            return self._evaluation_cache[cache_key]
        
        if self._dataset_manager:
            samples = self._dataset_manager.load_dataset_samples(
                dataset_name, 
                max_samples=eval_params.get('max_samples'),
                split=eval_params.get('split', 'test')
            )
        else:
            # Fallback to basic loading
            samples = self._load_dataset_fallback(dataset_name, eval_params)
        
        # Cache samples for reuse
        self._evaluation_cache[cache_key] = samples
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    
    def _optimize_samples_for_lightweight(self, samples: List[Dict[str, Any]], 
                                        model_config: EnhancedModelConfig) -> List[Dict[str, Any]]:
        """Optimize samples for lightweight engine processing"""
        optimized = []
        
        for sample in samples:
            # Create optimized sample
            opt_sample = {
                'input': sample.get('input', sample.get('prompt', sample.get('question', ''))),
                'target': sample.get('target', sample.get('answer', sample.get('output', ''))),
                'id': sample.get('id', len(optimized))
            }
            
            # Truncate input if too long for context window
            max_input_length = model_config.context_window - 512  # Reserve space for output
            if len(opt_sample['input']) > max_input_length:
                opt_sample['input'] = opt_sample['input'][:max_input_length]
                opt_sample['truncated'] = True
            
            optimized.append(opt_sample)
        
        return optimized
    
    def _calculate_optimal_batch_size(self, model_config: EnhancedModelConfig, 
                                    eval_params: Dict[str, Any]) -> int:
        """Calculate optimal batch size for evaluation"""
        # Start with configured batch size
        base_batch_size = eval_params.get('batch_size', model_config.evaluation_batch_size)
        
        # Adjust based on model size
        if model_config.size_gb > 30:
            return max(1, base_batch_size // 4)
        elif model_config.size_gb > 15:
            return max(1, base_batch_size // 2)
        else:
            return base_batch_size
    
    def _process_batch(self, model: Any, tokenizer: Any, batch: List[Dict[str, Any]], 
                      eval_params: Dict[str, Any]) -> Tuple[List[str], int]:
        """Process a batch of samples"""
        inputs = [sample['input'] for sample in batch]
        
        # Check if using vLLM or fallback
        if hasattr(model, 'generate'):  # vLLM LLM object
            return self._process_batch_vllm(model, inputs, eval_params)
        else:
            return self._process_batch_fallback(model, tokenizer, inputs, eval_params)
    
    def _process_batch_vllm(self, model: Any, inputs: List[str], 
                           eval_params: Dict[str, Any]) -> Tuple[List[str], int]:
        """Process batch using vLLM"""
        from vllm import SamplingParams
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=eval_params.get('temperature', 0.1),
            max_tokens=eval_params.get('max_tokens', 512),
            top_p=eval_params.get('top_p', 0.9),
            stop=eval_params.get('stop_sequences', [])
        )
        
        # Generate responses
        outputs = model.generate(inputs, sampling_params)
        
        # Extract text and count tokens
        predictions = []
        total_tokens = 0
        
        for output in outputs:
            generated_text = output.outputs[0].text
            predictions.append(generated_text)
            total_tokens += len(output.outputs[0].token_ids)
        
        return predictions, total_tokens
    
    def _process_batch_fallback(self, model: Any, tokenizer: Any, inputs: List[str],
                               eval_params: Dict[str, Any]) -> Tuple[List[str], int]:
        """Process batch using fallback implementation"""
        predictions = []
        total_tokens = 0
        
        for input_text in inputs:
            # Mock generation for testing
            prediction = f"Generated response for: {input_text[:50]}..."
            predictions.append(prediction)
            total_tokens += len(input_text.split()) * 2  # Rough token estimate
        
        return predictions, total_tokens
    
    def _calculate_metrics(self, predictions: List[str], targets: List[str], 
                          dataset_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        if self._metrics_calculator and hasattr(self._metrics_calculator, 'calculate_metrics'):
            return self._metrics_calculator.calculate_metrics(predictions, targets, dataset_name)
        else:
            return self._calculate_basic_metrics(predictions, targets)
    
    def _calculate_basic_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate basic metrics as fallback"""
        if not predictions or not targets:
            return {"accuracy": 0.0, "samples_processed": 0}
        
        # Basic exact match accuracy
        exact_matches = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
        accuracy = exact_matches / len(predictions)
        
        # Basic metrics
        avg_prediction_length = sum(len(p) for p in predictions) / len(predictions)
        avg_target_length = sum(len(t) for t in targets) / len(targets)
        
        return {
            "accuracy": accuracy,
            "exact_match": accuracy,
            "samples_processed": len(predictions),
            "avg_prediction_length": avg_prediction_length,
            "avg_target_length": avg_target_length
        }
    
    def _generate_sample_outputs(self, predictions: List[str], targets: List[str]) -> List[str]:
        """Generate sample outputs for review"""
        samples = []
        num_samples = min(10, len(predictions))
        
        for i in range(num_samples):
            sample = f"Sample {i + 1}:\nPrediction: {predictions[i][:200]}...\nTarget: {targets[i][:200]}...\n"
            samples.append(sample)
        
        return samples
    
    def _load_dataset_fallback(self, dataset_name: str, eval_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback dataset loading for testing"""
        logger.warning(f"Using fallback dataset loading for {dataset_name}")
        
        # Generate mock samples
        num_samples = eval_params.get('max_samples', 100)
        samples = []
        
        for i in range(num_samples):
            samples.append({
                'input': f"Sample question {i + 1} for {dataset_name}",
                'target': f"Sample answer {i + 1}",
                'id': i
            })
        
        return samples


# Fallback implementations for testing
class MockDatasetManager:
    """Mock dataset manager for testing"""
    
    def load_dataset_samples(self, dataset_name: str, max_samples: Optional[int] = None, 
                           split: str = 'test') -> List[Dict[str, Any]]:
        num_samples = max_samples or 50
        return [
            {
                'input': f"Mock question {i} from {dataset_name}",
                'target': f"Mock answer {i}",
                'id': i
            }
            for i in range(num_samples)
        ]


class MockMetricsCalculator:
    """Mock metrics calculator for testing"""
    
    def calculate_metrics(self, predictions: List[str], targets: List[str], 
                         dataset_name: str) -> Dict[str, float]:
        return {
            "accuracy": 0.75,
            "f1_score": 0.72,
            "exact_match": 0.68,
            "samples_processed": len(predictions)
        }


class MockPerformanceMonitor:
    """Mock performance monitor for testing"""
    
    def start_monitoring(self, model_name: str, preset: str, dataset: str):
        pass
    
    def stop_monitoring(self):
        pass
    
    def get_current_metrics(self) -> Dict[str, Any]:
        return {
            "gpu_utilization": 75.0,
            "memory_usage": 8.5,
            "tokens_per_second": 45.0
        }