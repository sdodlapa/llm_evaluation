"""
Lightweight evaluation engine implementation

Optimized for single-GPU evaluation of small to medium models (≤30B)
with emphasis on fast loading and efficient memory usage.
"""

import logging
import time
from typing import Dict, List, Optional, Any
import torch
import gc

from core_shared.interfaces.evaluation_interfaces import (
    EvaluationEngine, EvaluationRequest, EvaluationResult, 
    EngineCapabilities, EngineType, ResourceRequirements
)
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
from .model_loader import LightweightModelLoader
from .performance_optimizer import LightweightPerformanceOptimizer


logger = logging.getLogger(__name__)


class LightweightEvaluationEngine(EvaluationEngine):
    """Lightweight evaluation engine for small/medium models"""
    
    def __init__(self, engine_id: str = "lightweight_engine"):
        # Define capabilities
        capabilities = EngineCapabilities(
            engine_type=EngineType.LIGHTWEIGHT,
            max_model_size_gb=60.0,  # Up to ~30B parameters
            supports_tensor_parallel=False,  # Single GPU only
            supports_pipeline_parallel=False,
            max_gpu_count=1,
            supported_dtypes=["float16", "bfloat16", "float32"],
            supported_quantization=["none", "awq", "gptq", "int8"],
            avg_tokens_per_second=50.0,  # Conservative estimate
            memory_efficiency_score=0.9,  # High efficiency for small models
            startup_time_seconds=30.0,  # Fast startup
            requires_persistent_service=False,
            supports_batch_processing=True,
            max_concurrent_evaluations=3  # Can handle multiple small models
        )
        
        super().__init__(engine_id, capabilities)
        
        # Engine components
        self.model_loader = LightweightModelLoader()
        self.performance_optimizer = LightweightPerformanceOptimizer()
        
        # State management
        self._loaded_models: Dict[str, Any] = {}
        self._model_configs: Dict[str, EnhancedModelConfig] = {}
        self._evaluation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_cached_models = 2  # Keep 2 models in memory
        self.memory_threshold = 0.85  # Trigger cleanup at 85% GPU memory
        
    def initialize(self) -> bool:
        """Initialize the lightweight engine"""
        try:
            logger.info(f"Initializing {self.engine_id}")
            
            # Check GPU availability (allow CPU-only for testing)
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, running in CPU-only mode for testing")
                # Continue with CPU-only initialization for testing purposes
            
            # Initialize components
            if not self.model_loader.initialize():
                logger.error("Failed to initialize model loader")
                return False
            
            if not self.performance_optimizer.initialize():
                logger.error("Failed to initialize performance optimizer")
                return False
            
            self._is_initialized = True
            logger.info(f"Lightweight engine {self.engine_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize lightweight engine: {e}")
            return False
    
    def can_handle(self, request: EvaluationRequest) -> bool:
        """Check if this engine can handle the evaluation request"""
        try:
            model_config = request.model_config
            
            # Check model size
            if model_config.size_gb > self.capabilities.max_model_size_gb:
                return False
            
            # Check if requires multi-GPU
            if (model_config.tensor_parallel_size > 1 or 
                model_config.pipeline_parallel_size > 1):
                return False
            
            # Check resource constraints
            if request.resource_constraints:
                if request.resource_constraints.min_gpu_count > 1:
                    return False
            
            # Check if model is lightweight-compatible
            if hasattr(model_config, 'is_large_model') and model_config.is_large_model():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if can handle request: {e}")
            return False
    
    def get_resource_requirements(self, request: EvaluationRequest) -> ResourceRequirements:
        """Calculate resource requirements for the evaluation"""
        model_config = request.model_config
        
        # Base requirements
        gpu_memory_gb = model_config.size_gb * 2.0  # Model + activation memory
        
        # Add evaluation overhead
        gpu_memory_gb += 2.0  # Additional overhead for evaluation
        
        # Adjust for optimizations
        if hasattr(model_config, 'lightweight_optimizations'):
            opts = model_config.lightweight_optimizations
            if opts.memory_mapping:
                gpu_memory_gb *= 0.9  # Memory mapping reduces memory usage
            if opts.use_flash_attention:
                gpu_memory_gb *= 0.85  # Flash attention reduces memory
        
        return ResourceRequirements(
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=4,
            system_memory_gb=8.0,
            disk_space_gb=model_config.size_gb * 1.2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            min_gpu_count=1,
            max_evaluation_time_minutes=60,
            priority="medium"
        )
    
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Execute the evaluation request"""
        request_id = request.request_id
        model_config = request.model_config
        start_time = time.time()
        
        logger.info(f"Starting evaluation {request_id} for model {model_config.model_name}")
        
        try:
            # Load model if not cached
            model_key = self._get_model_key(model_config)
            if model_key not in self._loaded_models:
                self._load_model(model_config)
            
            # Get loaded model
            model_info = self._loaded_models[model_key]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Execute evaluations on datasets
            all_metrics = {}
            all_outputs = []
            total_tokens = 0
            
            for dataset_name in request.datasets:
                dataset_metrics, dataset_outputs, dataset_tokens = self._evaluate_on_dataset(
                    model, tokenizer, dataset_name, request.evaluation_params
                )
                all_metrics[dataset_name] = dataset_metrics
                all_outputs.extend(dataset_outputs)
                total_tokens += dataset_tokens
            
            # Calculate performance metrics
            end_time = time.time()
            execution_time = end_time - start_time
            tokens_per_second = total_tokens / execution_time if execution_time > 0 else 0
            
            # Create result
            result = EvaluationResult(
                request_id=request_id,
                model_name=model_config.model_name,
                dataset_name=",".join(request.datasets),
                metrics=self._aggregate_metrics(all_metrics),
                raw_outputs=all_outputs[:100],  # Limit output size
                performance_data={
                    "execution_time_seconds": execution_time,
                    "tokens_processed": total_tokens,
                    "tokens_per_second": tokens_per_second,
                    "gpu_memory_used_gb": self._get_gpu_memory_usage(),
                    "model_loading_time": model_info.get("load_time", 0)
                },
                resource_usage={
                    "gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "gpu_utilization": self._get_gpu_utilization(),
                    "cpu_usage": self._get_cpu_usage()
                },
                engine_used=EngineType.LIGHTWEIGHT,
                execution_time_seconds=execution_time,
                tokens_processed=total_tokens,
                tokens_per_second=tokens_per_second,
                success=True,
                started_at=start_time,
                completed_at=end_time
            )
            
            logger.info(f"Completed evaluation {request_id} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for request {request_id}: {e}")
            return EvaluationResult(
                request_id=request_id,
                model_name=model_config.model_name,
                dataset_name=",".join(request.datasets),
                metrics={},
                engine_used=EngineType.LIGHTWEIGHT,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=time.time()
            )
    
    def cleanup(self):
        """Clean up engine resources"""
        logger.info(f"Cleaning up {self.engine_id}")
        
        # Unload all models
        for model_key in list(self._loaded_models.keys()):
            self._unload_model(model_key)
        
        # Clear caches
        self._loaded_models.clear()
        self._model_configs.clear()
        self._evaluation_stats.clear()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _get_model_key(self, model_config: EnhancedModelConfig) -> str:
        """Generate unique key for model configuration"""
        return f"{model_config.model_name}_{model_config.preset}_{model_config.quantization_method}"
    
    def _load_model(self, model_config: EnhancedModelConfig):
        """Load model into memory"""
        model_key = self._get_model_key(model_config)
        
        # Check if we need to free up space
        if len(self._loaded_models) >= self.max_cached_models:
            self._evict_oldest_model()
        
        # Check GPU memory
        if self._get_gpu_memory_usage() > self.memory_threshold:
            self._evict_oldest_model()
        
        logger.info(f"Loading model {model_config.model_name}")
        load_start = time.time()
        
        try:
            # Load model using model loader
            model_info = self.model_loader.load_model(model_config)
            model_info["load_time"] = time.time() - load_start
            model_info["last_used"] = time.time()
            
            self._loaded_models[model_key] = model_info
            self._model_configs[model_key] = model_config
            
            logger.info(f"Loaded model {model_config.model_name} in {model_info['load_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_config.model_name}: {e}")
            raise
    
    def _unload_model(self, model_key: str):
        """Unload model from memory"""
        if model_key in self._loaded_models:
            logger.info(f"Unloading model {model_key}")
            
            # Clean up model resources
            model_info = self._loaded_models[model_key]
            del model_info["model"]
            if "tokenizer" in model_info:
                del model_info["tokenizer"]
            
            del self._loaded_models[model_key]
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _evict_oldest_model(self):
        """Evict the least recently used model"""
        if not self._loaded_models:
            return
        
        # Find oldest model
        oldest_key = min(
            self._loaded_models.keys(),
            key=lambda k: self._loaded_models[k].get("last_used", 0)
        )
        
        self._unload_model(oldest_key)
    
    def _evaluate_on_dataset(self, model: Any, tokenizer: Any, dataset_name: str, 
                           eval_params: Dict[str, Any]) -> tuple:
        """Evaluate model on a specific dataset"""
        # This is a simplified implementation
        # In practice, this would integrate with the existing evaluation framework
        
        logger.info(f"Evaluating on dataset {dataset_name}")
        
        # Placeholder implementation
        metrics = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "latency_ms": 50.0,
            "samples_evaluated": 100
        }
        
        outputs = [f"Sample output {i}" for i in range(10)]
        tokens_processed = 1000
        
        return metrics, outputs, tokens_processed
    
    def _aggregate_metrics(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across datasets"""
        aggregated = {}
        
        if not all_metrics:
            return aggregated
        
        # Get all metric names
        all_metric_names = set()
        for dataset_metrics in all_metrics.values():
            all_metric_names.update(dataset_metrics.keys())
        
        # Calculate averages
        for metric_name in all_metric_names:
            values = []
            for dataset_metrics in all_metrics.values():
                if metric_name in dataset_metrics:
                    values.append(dataset_metrics[metric_name])
            
            if values:
                aggregated[metric_name] = sum(values) / len(values)
        
        return aggregated
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction"""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return allocated / total
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        # Placeholder - would use nvidia-ml-py or similar
        return 75.0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        # Placeholder - would use psutil
        return 25.0