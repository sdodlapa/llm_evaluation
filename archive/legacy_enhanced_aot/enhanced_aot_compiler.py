"""
Enhanced AOT Model Compiler with CUDA Graph Integration
Clean extension of existing AOTModelCompiler functionality

This module extends the existing AOT compilation system with CUDA graph
optimization while maintaining full backward compatibility.
"""

import torch
import logging
from typing import Optional, Dict, Any, Tuple, Union
from .cuda_graph_optimizer import CudaGraphOptimizer, CudaGraphConfig

logger = logging.getLogger(__name__)

class EnhancedAOTModelCompiler:
    """
    Enhanced AOT compiler with CUDA graph optimization
    
    This class wraps and extends the existing AOTModelCompiler
    without modifying its core functionality.
    """
    
    def __init__(self, base_compiler=None, graph_config: CudaGraphConfig = None):
        """
        Initialize enhanced compiler
        
        Args:
            base_compiler: Existing AOTModelCompiler instance
            graph_config: CUDA graph configuration
        """
        self.base_compiler = base_compiler
        self.graph_optimizer = CudaGraphOptimizer(graph_config)
        self.enhanced_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized EnhancedAOTModelCompiler")
    
    def compile_model(self, 
                     model: torch.nn.Module,
                     example_inputs: Tuple[torch.Tensor, ...],
                     model_id: str,
                     **compile_kwargs) -> Any:
        """
        Compile model with optional CUDA graph enhancement
        
        Args:
            model: PyTorch model to compile
            example_inputs: Example input tensors
            model_id: Unique model identifier
            **compile_kwargs: Additional compilation arguments
            
        Returns:
            Enhanced compiled model
        """
        try:
            # First, use existing AOT compilation
            if self.base_compiler:
                # Call the correct method on the base compiler
                compiled_model = self.base_compiler.compile_model_aot(
                    model, example_inputs, 
                    model_id,  # Use model_id as config substitute
                    compile_kwargs.get('mode', 'default')
                )
            else:
                # Fallback to basic torch.compile
                compiled_model = torch.compile(model, **compile_kwargs)
            
            # Store basic compiled model
            self.enhanced_models[model_id] = compiled_model
            self.model_metadata[model_id] = {
                "has_cuda_graphs": False,
                "compilation_success": True,
                "input_shapes": [tuple(t.shape) for t in example_inputs]
            }
            
            # Attempt CUDA graph enhancement
            if self.graph_optimizer.config.enabled:
                graph_success = self.graph_optimizer.enhance_compiled_model(
                    compiled_model, example_inputs, model_id
                )
                self.model_metadata[model_id]["has_cuda_graphs"] = graph_success
                
                if graph_success:
                    logger.info(f"Enhanced model {model_id} with CUDA graphs")
            
            return EnhancedCompiledModel(
                compiled_model, 
                self.graph_optimizer,
                model_id,
                self.model_metadata[model_id]
            )
            
        except Exception as e:
            logger.error(f"Enhanced compilation failed for {model_id}: {e}")
            # Graceful fallback to original model
            self.enhanced_models[model_id] = model
            self.model_metadata[model_id] = {
                "has_cuda_graphs": False,
                "compilation_success": False,
                "error": str(e)
            }
            return model
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about compiled model"""
        return self.model_metadata.get(model_id, {})
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get overall optimization statistics"""
        base_stats = {}
        if hasattr(self.base_compiler, 'get_stats'):
            base_stats = self.base_compiler.get_stats()
        
        graph_stats = self.graph_optimizer.get_statistics()
        
        return {
            "base_compiler": base_stats,
            "cuda_graphs": graph_stats,
            "enhanced_models": len(self.enhanced_models),
            "models_with_graphs": sum(
                1 for meta in self.model_metadata.values()
                if meta.get("has_cuda_graphs", False)
            )
        }

class EnhancedCompiledModel:
    """
    Wrapper for compiled models with graph optimization
    
    This class provides a seamless interface that automatically
    chooses between graph execution and fallback compilation.
    """
    
    def __init__(self, 
                 compiled_model: Any,
                 graph_optimizer: CudaGraphOptimizer,
                 model_id: str,
                 metadata: Dict[str, Any]):
        self.compiled_model = compiled_model
        self.graph_optimizer = graph_optimizer
        self.model_id = model_id
        self.metadata = metadata
        self.execution_stats = {
            "graph_hits": 0,
            "graph_misses": 0,
            "total_calls": 0
        }
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Execute model with automatic graph optimization
        
        Attempts graph execution first, falls back to compiled model
        """
        self.execution_stats["total_calls"] += 1
        
        # Try graph execution for single tensor input
        if (len(args) == 1 and 
            isinstance(args[0], torch.Tensor) and 
            len(kwargs) == 0):
            
            graph_result = self.graph_optimizer.try_graph_execution(
                self.model_id, args[0]
            )
            
            if graph_result is not None:
                self.execution_stats["graph_hits"] += 1
                return graph_result
        
        # Fallback to compiled model
        self.execution_stats["graph_misses"] += 1
        return self.compiled_model(*args, **kwargs)
    
    def eval(self):
        """Set model to evaluation mode"""
        if hasattr(self.compiled_model, 'eval'):
            return self.compiled_model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode"""
        if hasattr(self.compiled_model, 'train'):
            return self.compiled_model.train(mode)
        return self
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = self.execution_stats["total_calls"]
        return {
            **self.execution_stats,
            "graph_hit_rate": (
                self.execution_stats["graph_hits"] / total 
                if total > 0 else 0.0
            ),
            "metadata": self.metadata
        }
    
    def __getattr__(self, name):
        """Delegate attribute access to compiled model"""
        return getattr(self.compiled_model, name)