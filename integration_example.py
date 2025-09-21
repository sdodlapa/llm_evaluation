"""
Example integration of enhanced AOT compilation with existing evaluation pipeline

This script demonstrates how to cleanly integrate the new vLLM optimizations
with the existing evaluation system without breaking existing functionality.
"""

import torch
import logging
from typing import Any, Dict

# Example imports from existing system
# from engines.lightweight.aot_compiler import AOTModelCompiler
# from engines.lightweight.hybrid_system import run_evaluation

# New optimized imports
from engines.shared import (
    create_enhanced_compiler,
    monitor_inference,
    monitor_compilation,
    performance_monitor
)

logger = logging.getLogger(__name__)

class OptimizedEvaluationExample:
    """
    Example of integrating enhanced compilation with existing evaluation
    
    This shows how to add optimizations without changing core evaluation logic.
    """
    
    def __init__(self, existing_compiler=None):
        """Initialize with optional existing compiler"""
        
        # Create enhanced compiler that wraps existing one
        self.compiler = create_enhanced_compiler(
            base_compiler=existing_compiler,
            enable_cuda_graphs=True,
            batch_sizes=[1, 2, 4, 8]  # Common evaluation batch sizes
        )
        
        logger.info("Initialized optimized evaluation system")
    
    @monitor_compilation
    def compile_model_optimized(self, model: torch.nn.Module, 
                               example_inputs: tuple,
                               model_id: str) -> Any:
        """
        Compile model with all optimizations
        
        This method replaces the original compile_model call
        but maintains the same interface.
        """
        try:
            # Enhanced compilation with automatic fallbacks
            compiled_model = self.compiler.compile_model(
                model=model,
                example_inputs=example_inputs,
                model_id=model_id,
                # Additional torch.compile options
                mode="reduce-overhead",  # Good for inference
                dynamic=False,
                fullgraph=True
            )
            
            logger.info(f"Successfully compiled {model_id} with optimizations")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Compilation failed for {model_id}: {e}")
            # Graceful fallback to original model
            return model
    
    @monitor_inference
    def run_inference_optimized(self, compiled_model: Any, 
                               inputs: torch.Tensor) -> torch.Tensor:
        """
        Run inference with automatic optimization selection
        
        The compiled model automatically chooses between CUDA graphs
        and regular compilation based on input characteristics.
        """
        with performance_monitor.time_operation("inference_total"):
            # Model automatically uses CUDA graphs if available
            outputs = compiled_model(inputs)
            
            # Optional: synchronize for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            return outputs
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        # Get base compilation stats
        compiler_stats = self.compiler.get_optimization_stats()
        
        # Get performance monitoring stats
        monitor_stats = performance_monitor.get_summary()
        
        return {
            "optimization_summary": compiler_stats,
            "performance_metrics": monitor_stats,
            "recommendations": self._generate_recommendations(compiler_stats)
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> list:
        """Generate optimization recommendations based on stats"""
        recommendations = []
        
        cuda_stats = stats.get("cuda_graphs", {})
        total_graphs = cuda_stats.get("total_graphs", 0)
        
        if total_graphs == 0:
            recommendations.append(
                "No CUDA graphs captured. Check CUDA availability and input consistency."
            )
        elif total_graphs < 3:
            recommendations.append(
                "Few CUDA graphs captured. Consider adding more common batch sizes."
            )
        
        graph_replays = cuda_stats.get("graph_replays", 0)
        if graph_replays > 0:
            recommendations.append(
                f"CUDA graphs working well: {graph_replays} replays executed."
            )
        
        return recommendations

# Example usage function
def demonstrate_integration():
    """
    Demonstrate how to integrate with existing evaluation pipeline
    
    This shows the minimal changes needed to add optimizations.
    """
    
    # Initialize optimized evaluation (replaces original)
    evaluator = OptimizedEvaluationExample()
    
    # Example model and data (replace with actual)
    example_model = torch.nn.Linear(512, 768).cuda()
    example_inputs = (torch.randn(4, 512).cuda(),)
    
    # Compile with optimizations (same interface as before)
    compiled_model = evaluator.compile_model_optimized(
        model=example_model,
        example_inputs=example_inputs,
        model_id="example_model"
    )
    
    # Run inference (same interface, automatic optimization)
    test_input = torch.randn(4, 512).cuda()
    
    # First run (may trigger graph capture)
    output1 = evaluator.run_inference_optimized(compiled_model, test_input)
    
    # Subsequent runs (may use CUDA graphs)
    for i in range(10):
        output = evaluator.run_inference_optimized(compiled_model, test_input)
    
    # Get performance report
    report = evaluator.get_performance_report()
    
    print("Performance Report:")
    print(f"Models compiled: {report['optimization_summary']['enhanced_models']}")
    print(f"Models with CUDA graphs: {report['optimization_summary']['models_with_graphs']}")
    print(f"Total inference calls: {report['performance_metrics']['total_calls']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
    
    return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = demonstrate_integration()
    print(f"\nDetailed report: {report}")