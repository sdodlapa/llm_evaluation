"""
Shared components for evaluation engines

This module contains shared utilities and components used across
lightweight and distributed evaluation engines, including vLLM
optimizations and performance monitoring.
"""

from .cuda_graph_optimizer import CudaGraphOptimizer, CudaGraphConfig
from .enhanced_aot_compiler import EnhancedAOTModelCompiler, EnhancedCompiledModel
from .integrated_aot_compiler import IntegratedAOTCompiler, create_integrated_compiler
from .performance_monitor import (
    SimplePerformanceMonitor, 
    performance_monitor,
    monitor_inference,
    monitor_compilation,
    monitor_evaluation
)

def create_enhanced_compiler(base_compiler=None, 
                           enable_cuda_graphs: bool = True,
                           batch_sizes: list = None) -> EnhancedAOTModelCompiler:
    """
    Create enhanced AOT compiler with vLLM optimizations
    
    Args:
        base_compiler: Existing AOT compiler to enhance
        enable_cuda_graphs: Whether to enable CUDA graph optimization
        batch_sizes: Batch sizes to optimize for
    
    Returns:
        Enhanced compiler instance
    """
    graph_config = CudaGraphConfig(
        enabled=enable_cuda_graphs,
        batch_sizes=batch_sizes or [1, 2, 4, 8]
    )
    
    return EnhancedAOTModelCompiler(
        base_compiler=base_compiler,
        graph_config=graph_config
    )

def get_optimization_summary() -> dict:
    """Get summary of all optimizations"""
    return {
        "performance_monitoring": performance_monitor.get_summary(),
        "cuda_graphs": "Available" if CudaGraphOptimizer().config.enabled else "Disabled"
    }

__all__ = [
    'CudaGraphOptimizer',
    'CudaGraphConfig', 
    'EnhancedAOTModelCompiler',
    'EnhancedCompiledModel',
    'IntegratedAOTCompiler',
    'SimplePerformanceMonitor',
    'performance_monitor',
    'monitor_inference',
    'monitor_compilation', 
    'monitor_evaluation',
    'create_enhanced_compiler',
    'create_integrated_compiler',
    'get_optimization_summary'
]