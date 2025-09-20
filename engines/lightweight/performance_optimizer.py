"""
Performance optimizer for lightweight evaluation engine

Provides optimizations for single-GPU model evaluation
with focus on memory efficiency and speed.
"""

import logging
import torch
import gc
from typing import Dict, Any, Optional
import psutil
import time

from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig


logger = logging.getLogger(__name__)


class LightweightPerformanceOptimizer:
    """Performance optimizer for lightweight engine"""
    
    def __init__(self):
        self._optimization_cache: Dict[str, Dict[str, Any]] = {}
        self._monitoring_enabled = True
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the performance optimizer"""
        try:
            # Check system capabilities
            if torch.cuda.is_available():
                self._gpu_count = torch.cuda.device_count()
                self._gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU detected: {self._gpu_count} devices, {self._gpu_memory:.1f}GB memory")
            else:
                logger.warning("No GPU detected, running in CPU-only mode for testing")
                self._gpu_count = 0
                self._gpu_memory = 0
            
            # Check system memory
            self._system_memory = psutil.virtual_memory().total / 1024**3
            logger.info(f"System memory: {self._system_memory:.1f}GB")
            
            self._initialized = True
            logger.info("Performance optimizer initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            return False
    
    def optimize_model_config(self, model_config: EnhancedModelConfig) -> EnhancedModelConfig:
        """Apply optimizations to model configuration
        
        Args:
            model_config: Original model configuration
            
        Returns:
            Optimized model configuration
        """
        if not self._initialized:
            logger.warning("Optimizer not initialized, returning original config")
            return model_config
        
        # Create optimized copy
        optimized_config = model_config.clone_with_overrides()
        
        # Apply memory optimizations
        optimized_config = self._optimize_memory_usage(optimized_config)
        
        # Apply performance optimizations
        optimized_config = self._optimize_performance_settings(optimized_config)
        
        # Cache optimization results
        self._cache_optimization_results(model_config.model_name, optimized_config)
        
        logger.info(f"Applied optimizations to {model_config.model_name}")
        return optimized_config
    
    def _optimize_memory_usage(self, config: EnhancedModelConfig) -> EnhancedModelConfig:
        """Optimize memory usage settings"""
        # Adjust GPU memory utilization based on model size
        if config.size_gb > 40:
            config.gpu_memory_utilization = 0.85  # Conservative for large models
        elif config.size_gb > 20:
            config.gpu_memory_utilization = 0.90  # Moderate for medium models
        else:
            config.gpu_memory_utilization = 0.95  # Aggressive for small models
        
        # Enable memory optimizations
        if hasattr(config, 'lightweight_optimizations'):
            config.lightweight_optimizations.memory_mapping = True
            config.lightweight_optimizations.use_flash_attention = True
            config.lightweight_optimizations.enable_kv_cache = True
        
        # Adjust sequence settings for memory efficiency
        if config.size_gb > 30:
            config.max_num_seqs = min(config.max_num_seqs, 32)  # Reduce batch size
        else:
            config.max_num_seqs = min(config.max_num_seqs, 64)
        
        return config
    
    def _optimize_performance_settings(self, config: EnhancedModelConfig) -> EnhancedModelConfig:
        """Optimize performance settings"""
        # Enable performance features
        config.enable_prefix_caching = True
        config.use_v2_block_manager = True
        
        # Adjust based on model size
        if config.size_gb <= 15:  # Small models
            config.enforce_eager = False  # Allow CUDA graphs
            if hasattr(config, 'lightweight_optimizations'):
                config.lightweight_optimizations.fast_loading = True
                config.lightweight_optimizations.batch_size_optimization = True
        else:  # Larger models
            config.enforce_eager = True  # More stable for larger models
        
        # Optimize tensor parallel for single GPU
        config.tensor_parallel_size = 1
        config.pipeline_parallel_size = 1
        
        return config
    
    def optimize_evaluation_parameters(self, model_config: EnhancedModelConfig, 
                                     eval_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize evaluation parameters for performance"""
        optimized_params = eval_params.copy()
        
        # Adjust batch size based on model size and available memory
        recommended_batch_size = self._calculate_optimal_batch_size(model_config)
        if "batch_size" in optimized_params:
            optimized_params["batch_size"] = min(optimized_params["batch_size"], recommended_batch_size)
        else:
            optimized_params["batch_size"] = recommended_batch_size
        
        # Set timeout based on model size
        if "timeout_seconds" not in optimized_params:
            optimized_params["timeout_seconds"] = max(300, model_config.size_gb * 10)
        
        # Enable optimizations
        optimized_params["use_cache"] = True
        optimized_params["low_memory_mode"] = model_config.size_gb > 40
        
        return optimized_params
    
    def _calculate_optimal_batch_size(self, model_config: EnhancedModelConfig) -> int:
        """Calculate optimal batch size based on model and system constraints"""
        # Base batch size on model size
        if model_config.size_gb <= 7:
            base_batch_size = 16
        elif model_config.size_gb <= 15:
            base_batch_size = 8
        elif model_config.size_gb <= 30:
            base_batch_size = 4
        else:
            base_batch_size = 2
        
        # Adjust based on available GPU memory
        available_memory = self._get_available_gpu_memory()
        memory_factor = available_memory / (model_config.size_gb * 2.0)  # Model + activations
        
        if memory_factor < 1.2:
            base_batch_size = max(1, base_batch_size // 2)
        elif memory_factor > 3.0:
            base_batch_size = min(32, base_batch_size * 2)
        
        return base_batch_size
    
    def monitor_performance(self, model_name: str) -> Dict[str, Any]:
        """Monitor current performance metrics"""
        if not self._monitoring_enabled:
            return {}
        
        metrics = {
            "timestamp": time.time(),
            "model_name": model_name
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            metrics.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_memory_utilization": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory,
                "gpu_temperature": self._get_gpu_temperature()
            })
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics.update({
            "cpu_utilization": cpu_percent,
            "system_memory_used_gb": memory.used / 1024**3,
            "system_memory_utilization": memory.percent / 100.0
        })
        
        return metrics
    
    def cleanup_memory(self):
        """Clean up GPU and system memory"""
        logger.info("Cleaning up memory")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear optimization cache
        self._optimization_cache.clear()
    
    def get_optimization_recommendations(self, model_config: EnhancedModelConfig) -> Dict[str, str]:
        """Get optimization recommendations for a model"""
        recommendations = []
        
        # Memory recommendations
        if model_config.size_gb > 40:
            recommendations.append("Consider using distributed engine for better performance")
        
        if model_config.gpu_memory_utilization > 0.95:
            recommendations.append("Reduce GPU memory utilization to avoid OOM errors")
        
        if model_config.max_num_seqs > 64 and model_config.size_gb > 20:
            recommendations.append("Reduce max_num_seqs for large models")
        
        # Performance recommendations
        if not config.enable_prefix_caching:
            recommendations.append("Enable prefix caching for repeated evaluations")
        
        if model_config.size_gb <= 15 and model_config.enforce_eager:
            recommendations.append("Disable enforce_eager for small models to enable CUDA graphs")
        
        return {
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score(model_config),
            "estimated_memory_gb": model_config.size_gb * 2.0,
            "recommended_batch_size": self._calculate_optimal_batch_size(model_config)
        }
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        return total_memory - allocated_memory
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature (placeholder)"""
        # This would use nvidia-ml-py or similar library
        return None
    
    def _calculate_optimization_score(self, model_config: EnhancedModelConfig) -> float:
        """Calculate optimization score (0-1, higher is better)"""
        score = 1.0
        
        # Penalize high memory utilization
        if model_config.gpu_memory_utilization > 0.9:
            score -= 0.2
        
        # Reward optimizations
        if hasattr(model_config, 'lightweight_optimizations'):
            opts = model_config.lightweight_optimizations
            if opts.use_flash_attention:
                score += 0.1
            if opts.enable_kv_cache:
                score += 0.1
            if opts.memory_mapping:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _cache_optimization_results(self, model_name: str, config: EnhancedModelConfig):
        """Cache optimization results for future use"""
        self._optimization_cache[model_name] = {
            "config": config,
            "timestamp": time.time(),
            "optimization_score": self._calculate_optimization_score(config)
        }