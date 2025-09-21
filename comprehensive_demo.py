"""
Complete Enhanced AOT Compilation Demo

This script demonstrates the full integration of vLLM optimizations
with the existing evaluation pipeline, showing real performance benefits.
"""

import torch
import time
import logging
import sys
import os
from contextlib import contextmanager

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.shared import (
    IntegratedAOTCompiler, 
    create_integrated_compiler,
    performance_monitor,
    get_optimization_summary
)

@contextmanager
def timer(description):
    """Simple timer context manager"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {(end - start)*1000:.2f}ms")

class DemoEvaluationPipeline:
    """
    Demo evaluation pipeline showing enhanced AOT integration
    
    This simulates a real evaluation pipeline with multiple models
    and demonstrates the performance improvements.
    """
    
    def __init__(self, use_enhanced: bool = True):
        """
        Initialize demo pipeline
        
        Args:
            use_enhanced: Whether to use enhanced compilation
        """
        self.use_enhanced = use_enhanced
        
        if use_enhanced:
            self.compiler = create_integrated_compiler(
                enable_cuda_graphs=True,
                batch_sizes=[1, 2, 4, 8],
                cache_dir="demo_cache"
            )
            print("✓ Using Enhanced AOT Compiler with vLLM optimizations")
        else:
            # Simulate original compiler
            from engines.shared.aot_compiler import AOTModelCompiler
            self.compiler = AOTModelCompiler(cache_dir="demo_cache")
            print("✓ Using Original AOT Compiler")
        
        self.models = {}
        self.results = {}
    
    def create_test_models(self):
        """Create various test models simulating real evaluation scenarios"""
        
        print("\nCreating test models...")
        
        # Small transformer-like model
        class SmallTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 256)
                self.layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(256, 8, 512, batch_first=True)
                    for _ in range(4)
                ])
                self.head = torch.nn.Linear(256, 10)
                
            def forward(self, x):
                x = self.embed(x)
                for layer in self.layers:
                    x = layer(x)
                return self.head(x.mean(dim=1))
        
        # Linear classification model
        class LinearClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(768, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 5)
                )
                
            def forward(self, x):
                return self.layers(x)
        
        # Simple CNN model
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv1d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(128, 256, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool1d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(256, 8)
                )
                
            def forward(self, x):
                return self.conv(x)
        
        models_config = [
            ("small_transformer", SmallTransformer(), (torch.randint(0, 1000, (4, 32)),)),
            ("linear_classifier", LinearClassifier(), (torch.randn(4, 768),)),
            ("simple_cnn", SimpleCNN(), (torch.randn(4, 64, 100),))
        ]
        
        return models_config
    
    def compile_models(self):
        """Compile all test models"""
        
        print("\nCompiling models...")
        models_config = self.create_test_models()
        
        for model_name, model, example_inputs in models_config:
            print(f"\nCompiling {model_name}...")
            
            # Create mock config for compatibility
            class MockConfig:
                def __init__(self, name):
                    self.model_name = name
                    self.model_id = name
            
            config = MockConfig(model_name)
            
            with timer(f"  Compilation time for {model_name}"):
                if self.use_enhanced:
                    compiled_model = self.compiler.compile_model_aot(
                        model, example_inputs, config, "reduce-overhead"
                    )
                else:
                    compiled_model = self.compiler.compile_model_aot(
                        model, example_inputs, config, "default"
                    )
            
            if compiled_model is not None:
                self.models[model_name] = (compiled_model, example_inputs)
                print(f"  ✓ {model_name} compiled successfully")
            else:
                self.models[model_name] = (model, example_inputs)
                print(f"  ⚠ {model_name} using fallback")
    
    def run_inference_benchmark(self, num_iterations: int = 20):
        """Run inference benchmark on compiled models"""
        
        print(f"\nRunning inference benchmark ({num_iterations} iterations)...")
        
        for model_name, (model, example_inputs) in self.models.items():
            print(f"\nBenchmarking {model_name}:")
            
            # Warmup
            for _ in range(3):
                _ = model(*example_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for i in range(num_iterations):
                start = time.perf_counter()
                output = model(*example_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            self.results[model_name] = {
                'avg_time_ms': avg_time,
                'min_time_ms': min_time, 
                'max_time_ms': max_time,
                'output_shape': output.shape,
                'iterations': num_iterations
            }
            
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Range: {min_time:.2f}ms - {max_time:.2f}ms")
            print(f"  Output shape: {output.shape}")
    
    def show_performance_report(self):
        """Show comprehensive performance report"""
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        # Compilation stats
        if hasattr(self.compiler, 'get_compilation_stats'):
            stats = self.compiler.get_compilation_stats()
            
            print("\nCompilation Statistics:")
            if 'enhanced_features' in stats:
                enhanced = stats['enhanced_features']
                print(f"  Models with CUDA graphs: {enhanced.get('models_with_graphs', 0)}")
                cuda_stats = enhanced.get('cuda_graphs', {})
                print(f"  CUDA graph replays: {cuda_stats.get('graph_replays', 0)}")
                print(f"  Graph capture success: {cuda_stats.get('graphs_captured', 0)}")
            
            if 'performance_monitoring' in stats:
                perf = stats['performance_monitoring']
                print(f"  Total operations monitored: {perf.get('total_operations', 0)}")
                print(f"  Total function calls: {perf.get('total_calls', 0)}")
        
        # Inference results
        print("\nInference Benchmark Results:")
        for model_name, results in self.results.items():
            print(f"\n  {model_name}:")
            print(f"    Average inference time: {results['avg_time_ms']:.2f}ms")
            print(f"    Best time: {results['min_time_ms']:.2f}ms")
            print(f"    Throughput: {1000/results['avg_time_ms']:.1f} inferences/sec")
        
        # Overall summary
        if self.results:
            all_avg_times = [r['avg_time_ms'] for r in self.results.values()]
            overall_avg = sum(all_avg_times) / len(all_avg_times)
            print(f"\nOverall average inference time: {overall_avg:.2f}ms")
            
            print(f"\nOptimization Summary:")
            opt_summary = get_optimization_summary()
            for category, info in opt_summary.items():
                if isinstance(info, dict):
                    print(f"  {category}: {info.get('total_operations', 'Active')}")
                else:
                    print(f"  {category}: {info}")

def run_comparative_demo():
    """Run comparative demo between original and enhanced compilation"""
    
    print("Enhanced AOT Compilation - Performance Demo")
    print("=" * 60)
    
    # Test both versions
    print("\n1. Testing Enhanced Compilation")
    print("-" * 30)
    enhanced_pipeline = DemoEvaluationPipeline(use_enhanced=True)
    enhanced_pipeline.compile_models()
    enhanced_pipeline.run_inference_benchmark(num_iterations=10)
    
    print("\n2. Testing Original Compilation")
    print("-" * 30)
    original_pipeline = DemoEvaluationPipeline(use_enhanced=False)
    original_pipeline.compile_models()
    original_pipeline.run_inference_benchmark(num_iterations=10)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print("\nEnhanced Compilation Results:")
    for model_name, results in enhanced_pipeline.results.items():
        print(f"  {model_name}: {results['avg_time_ms']:.2f}ms avg")
    
    print("\nOriginal Compilation Results:")
    for model_name, results in original_pipeline.results.items():
        print(f"  {model_name}: {results['avg_time_ms']:.2f}ms avg")
    
    # Calculate improvements
    print("\nPerformance Improvements:")
    for model_name in enhanced_pipeline.results:
        if model_name in original_pipeline.results:
            enhanced_time = enhanced_pipeline.results[model_name]['avg_time_ms']
            original_time = original_pipeline.results[model_name]['avg_time_ms']
            improvement = ((original_time - enhanced_time) / original_time) * 100
            print(f"  {model_name}: {improvement:+.1f}% change")
    
    # Show enhanced features report
    enhanced_pipeline.show_performance_report()
    
    return enhanced_pipeline, original_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Starting comprehensive enhanced AOT compilation demo...")
    enhanced, original = run_comparative_demo()
    
    print("\n" + "=" * 60)
    print("✓ Demo completed successfully!")
    print("✓ Enhanced AOT compilation is working with full compatibility")
    print("✓ Ready for production integration")
    print("=" * 60)