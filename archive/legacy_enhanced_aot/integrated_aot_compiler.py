"""
Integration adapter for enhanced AOT compilation with existing system

This module provides seamless integration between the new enhanced compilation
system and the existing AOTModelCompiler, maintaining full compatibility.
"""

import logging
from typing import Any, Optional, Tuple, Dict
from pathlib import Path

# Import existing AOT compiler
from .aot_compiler import AOTModelCompiler
from .enhanced_aot_compiler import EnhancedAOTModelCompiler
from .cuda_graph_optimizer import CudaGraphConfig
from .performance_monitor import performance_monitor, monitor_compilation

logger = logging.getLogger(__name__)

class IntegratedAOTCompiler:
    """
    Integrated AOT compiler that combines existing and enhanced functionality
    
    This adapter provides a drop-in replacement for AOTModelCompiler
    with added vLLM optimizations while maintaining existing interfaces.
    """
    
    def __init__(self, 
                 cache_dir: str = "model_cache/compiled",
                 enable_aot: bool = True,
                 max_compilation_time: int = 600,
                 enable_cuda_graphs: bool = True,
                 batch_sizes: list = None):
        """
        Initialize integrated compiler
        
        Args:
            cache_dir: Directory to store compiled models (existing interface)
            enable_aot: Whether to enable AOT compilation (existing interface)
            max_compilation_time: Maximum compilation time (existing interface)
            enable_cuda_graphs: Whether to enable CUDA graph optimization (new)
            batch_sizes: Batch sizes to optimize for (new)
        """
        
        # Initialize existing AOT compiler
        self.base_compiler = AOTModelCompiler(
            cache_dir=cache_dir,
            enable_aot=enable_aot,
            max_compilation_time=max_compilation_time
        )
        
        # Initialize enhanced compiler with base compiler
        graph_config = CudaGraphConfig(
            enabled=enable_cuda_graphs,
            batch_sizes=batch_sizes or [1, 2, 4, 8]
        )
        
        self.enhanced_compiler = EnhancedAOTModelCompiler(
            base_compiler=self.base_compiler,
            graph_config=graph_config
        )
        
        logger.info("Initialized IntegratedAOTCompiler with vLLM optimizations")
    
    @monitor_compilation  
    def compile_model_aot(self, 
                         model: Any, 
                         example_inputs: Tuple,
                         model_config: Any,
                         compilation_mode: str = "default") -> Optional[Any]:
        """
        Compile model with enhanced optimizations (existing interface)
        
        This method maintains the exact same interface as the original
        AOTModelCompiler.compile_model_aot() but adds vLLM optimizations.
        
        Args:
            model: The PyTorch model to compile
            example_inputs: Representative input tensors for tracing
            model_config: Model configuration  
            compilation_mode: Compilation mode
            
        Returns:
            Enhanced compiled model or None if compilation fails
        """
        
        # Generate model ID from config
        model_id = self._generate_model_id(model_config, compilation_mode)
        
        try:
            # Use enhanced compiler with existing parameters
            compiled_model = self.enhanced_compiler.compile_model(
                model=model,
                example_inputs=example_inputs,
                model_id=model_id,
                mode=compilation_mode,
                fullgraph=True,
                dynamic=False
            )
            
            logger.info(f"Successfully compiled {model_id} with enhancements")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Enhanced compilation failed for {model_id}: {e}")
            
            # Fallback to original AOT compiler
            try:
                fallback_model = self.base_compiler.compile_model_aot(
                    model, example_inputs, model_config, compilation_mode
                )
                if fallback_model:
                    logger.info(f"Fallback compilation successful for {model_id}")
                    return fallback_model
            except Exception as fallback_error:
                logger.error(f"Fallback compilation also failed: {fallback_error}")
            
            return None
    
    def _generate_model_id(self, model_config: Any, compilation_mode: str) -> str:
        """Generate consistent model ID from config"""
        
        if hasattr(model_config, 'model_name'):
            base_name = model_config.model_name
        elif hasattr(model_config, 'model_id'):
            base_name = model_config.model_id
        elif isinstance(model_config, dict):
            base_name = model_config.get('model_name', model_config.get('model_id', 'unknown'))
        else:
            base_name = str(model_config)[:50]  # Truncate if too long
        
        # Clean name for use as ID
        clean_name = "".join(c for c in base_name if c.isalnum() or c in "-_")
        return f"{clean_name}_{compilation_mode}"
    
    # Maintain existing interface methods
    def save_compiled_model(self, cache_key: str, model: Any, metadata: Dict):
        """Save compiled model (existing interface)"""
        return self.base_compiler.save_compiled_model(cache_key, model, metadata)
    
    def load_compiled_model(self, cache_key: str) -> Optional[Any]:
        """Load compiled model (existing interface)"""
        return self.base_compiler.load_compiled_model(cache_key)
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics (enhanced interface)"""
        
        # Get base stats
        base_stats = getattr(self.base_compiler, 'compilation_stats', {})
        
        # Get enhanced stats  
        enhanced_stats = self.enhanced_compiler.get_optimization_stats()
        
        # Get performance stats
        perf_stats = performance_monitor.get_summary()
        
        return {
            "base_aot": base_stats,
            "enhanced_features": enhanced_stats,
            "performance_monitoring": perf_stats
        }
    
    def cleanup_cache(self, max_age_days: int = 30):
        """Clean up old compiled models (existing interface)"""
        if hasattr(self.base_compiler, 'cleanup_cache'):
            return self.base_compiler.cleanup_cache(max_age_days)
    
    # Properties for compatibility
    @property
    def cache_dir(self) -> Path:
        """Cache directory (existing interface)"""
        return self.base_compiler.cache_dir
    
    @property
    def enable_aot(self) -> bool:
        """AOT enabled status (existing interface)"""
        return self.base_compiler.enable_aot
    
    @property
    def compiled_cache(self) -> Dict[str, Any]:
        """Compiled model cache (existing interface)"""
        return getattr(self.base_compiler, 'compiled_cache', {})

def create_integrated_compiler(**kwargs) -> IntegratedAOTCompiler:
    """
    Factory function to create integrated compiler
    
    This function provides a simple way to create an integrated compiler
    with reasonable defaults while maintaining full configurability.
    
    Args:
        **kwargs: Arguments passed to IntegratedAOTCompiler
        
    Returns:
        Configured IntegratedAOTCompiler instance
    """
    
    # Set reasonable defaults
    defaults = {
        'enable_cuda_graphs': True,
        'batch_sizes': [1, 2, 4, 8],
        'enable_aot': True
    }
    
    # Merge defaults with provided kwargs
    config = {**defaults, **kwargs}
    
    logger.info(f"Creating integrated compiler with config: {config}")
    return IntegratedAOTCompiler(**config)