"""
CUDA Graph Optimization Module
Enhances existing AOT compilation with graph capture/replay

This module provides CUDA graph optimization that integrates cleanly
with the existing AOTModelCompiler without modifying core functionality.
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CudaGraphConfig:
    """Configuration for CUDA graph optimization"""
    enabled: bool = True
    batch_sizes: List[int] = None
    warmup_steps: int = 3
    max_graphs: int = 10
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]

class CudaGraphOptimizer:
    """
    CUDA Graph optimizer that integrates with existing AOT compilation
    
    This class enhances the existing AOTModelCompiler without modifying
    its core functionality, following a clean integration pattern.
    """
    
    def __init__(self, config: CudaGraphConfig = None):
        self.config = config or CudaGraphConfig()
        self.captured_graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.graph_inputs: Dict[str, List[torch.Tensor]] = {}
        self.graph_outputs: Dict[str, List[torch.Tensor]] = {}
        self.stats = {
            "graphs_captured": 0,
            "graph_replays": 0,
            "capture_failures": 0
        }
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, disabling graph optimization")
            self.config.enabled = False
    
    def enhance_compiled_model(self, 
                             compiled_model: Any,
                             example_inputs: Tuple[torch.Tensor, ...],
                             model_id: str) -> bool:
        """
        Enhance a compiled model with CUDA graph capture
        
        Args:
            compiled_model: AOT compiled model
            example_inputs: Representative input tensors
            model_id: Unique identifier for this model
            
        Returns:
            bool: True if enhancement successful
        """
        if not self.config.enabled:
            return False
        
        try:
            success_count = 0
            
            for batch_size in self.config.batch_sizes:
                graph_key = f"{model_id}_batch_{batch_size}"
                
                if self._capture_graph(compiled_model, example_inputs, 
                                     batch_size, graph_key):
                    success_count += 1
            
            logger.info(f"Enhanced model {model_id} with {success_count} CUDA graphs")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to enhance model {model_id}: {e}")
            self.stats["capture_failures"] += 1
            return False
    
    def _capture_graph(self, 
                      model: Any,
                      example_inputs: Tuple[torch.Tensor, ...],
                      batch_size: int,
                      graph_key: str) -> bool:
        """Capture CUDA graph for specific batch size"""
        
        try:
            # Prepare inputs for this batch size
            graph_inputs = self._prepare_batch_inputs(example_inputs, batch_size)
            
            # Warmup runs
            model.eval()
            with torch.no_grad():
                for _ in range(self.config.warmup_steps):
                    _ = model(*graph_inputs)
                    torch.cuda.synchronize()
            
            # Capture graph
            graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(graph):
                graph_outputs = model(*graph_inputs)
            
            # Store graph components
            self.captured_graphs[graph_key] = graph
            self.graph_inputs[graph_key] = graph_inputs
            self.graph_outputs[graph_key] = (
                graph_outputs if isinstance(graph_outputs, (list, tuple))
                else [graph_outputs]
            )
            
            self.stats["graphs_captured"] += 1
            logger.debug(f"Captured graph: {graph_key}")
            return True
            
        except Exception as e:
            logger.debug(f"Graph capture failed for {graph_key}: {e}")
            return False
    
    def _prepare_batch_inputs(self, 
                            example_inputs: Tuple[torch.Tensor, ...],
                            batch_size: int) -> List[torch.Tensor]:
        """Prepare input tensors with target batch size"""
        
        batch_inputs = []
        
        for input_tensor in example_inputs:
            # Create tensor with target batch size
            target_shape = list(input_tensor.shape)
            target_shape[0] = batch_size
            
            batch_input = torch.empty(
                target_shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device
            )
            
            # Fill with example data
            if batch_size <= input_tensor.shape[0]:
                batch_input.copy_(input_tensor[:batch_size])
            else:
                # Repeat to fill larger batch
                repeat_factor = (batch_size + input_tensor.shape[0] - 1) // input_tensor.shape[0]
                repeated = input_tensor.repeat(repeat_factor, *([1] * (len(input_tensor.shape) - 1)))
                batch_input.copy_(repeated[:batch_size])
            
            batch_inputs.append(batch_input)
        
        return batch_inputs
    
    def try_graph_execution(self, 
                          model_id: str,
                          input_batch: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Attempt to execute using captured graph
        
        Args:
            model_id: Model identifier
            input_batch: Input tensor batch
            
        Returns:
            Model output if graph execution successful, None otherwise
        """
        if not self.config.enabled:
            return None
        
        batch_size = input_batch.shape[0]
        graph_key = f"{model_id}_batch_{batch_size}"
        
        if graph_key not in self.captured_graphs:
            return None
        
        try:
            # Copy input data to graph tensors
            graph_inputs = self.graph_inputs[graph_key]
            graph_inputs[0].copy_(input_batch)
            
            # Execute graph
            self.captured_graphs[graph_key].replay()
            
            # Return copied output
            graph_outputs = self.graph_outputs[graph_key]
            output = graph_outputs[0].clone()
            
            self.stats["graph_replays"] += 1
            return output
            
        except Exception as e:
            logger.debug(f"Graph execution failed for {graph_key}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.stats,
            "total_graphs": len(self.captured_graphs),
            "enabled": self.config.enabled
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.captured_graphs.clear()
        self.graph_inputs.clear()
        self.graph_outputs.clear()