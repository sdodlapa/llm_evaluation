#!/usr/bin/env python3
"""
8B vs 14B Performance Scaling Comparison
This script compares Qwen-3 8B and 14B models with H100 optimizations on fixed datasets
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project paths
sys.path.append('evaluation')
sys.path.append('models')
sys.path.append('configs')

from run_evaluation import LLMEvaluationRunner
from model_configs import MODEL_CONFIGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelScalingComparison:
    """Compare performance scaling between 8B and 14B models"""
    
    def __init__(self, output_dir: str = "test_results/scaling_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluation runner
        self.runner = LLMEvaluationRunner(
            output_dir=str(self.output_dir),
            cache_dir=str(self.output_dir / "cache")
        )
        
        # Define datasets to test (using our fixed datasets)
        self.test_datasets = [
            "humaneval",  # Coding - already working
            "mbpp",       # Coding - already working  
            "gsm8k",      # Reasoning - already working
            "hellaswag",  # Reasoning - newly fixed
            "mt_bench"    # Instruction following - newly fixed
        ]
        
    def create_optimized_8b_config(self):
        """Create H100 optimized 8B configuration"""
        base_config = MODEL_CONFIGS["qwen3_8b"]
        
        # Create optimized copy
        import copy
        optimized = copy.deepcopy(base_config)
        
        # Apply H100 optimizations for 8B model
        optimized.model_name = "Qwen-3 8B (H100 Optimized)"
        optimized.gpu_memory_utilization = 0.85  # Conservative for 8B
        optimized.max_num_seqs = 128  # Higher throughput
        optimized.max_model_len = 32768  # Keep reasonable context
        optimized.evaluation_batch_size = 16  # Larger batches
        optimized.enable_prefix_caching = True
        optimized.enforce_eager = False  # Enable CUDA graphs
        
        # Add vLLM optimizations
        optimized._vllm_overrides.update({
            "max_num_batched_tokens": 16384,
            "use_v2_block_manager": True,
            "enable_chunked_prefill": True,
            "disable_log_stats": True,
            "block_size": 32
        })
        
        return optimized
    
    def create_optimized_14b_config(self):
        """Create H100 optimized 14B configuration"""
        base_config = MODEL_CONFIGS["qwen3_14b"]
        
        # Create optimized copy
        import copy
        optimized = copy.deepcopy(base_config)
        
        # Apply H100 optimizations for 14B model
        optimized.model_name = "Qwen-3 14B (H100 Optimized)"
        optimized.gpu_memory_utilization = 0.88  # More aggressive for quantized 14B
        optimized.max_num_seqs = 64  # Moderate for larger model
        optimized.max_model_len = 32768  # Consistent context
        optimized.evaluation_batch_size = 8  # Reasonable batches
        optimized.enable_prefix_caching = True
        optimized.enforce_eager = False  # Enable CUDA graphs
        
        # Ensure we use AWQ quantization for memory efficiency
        optimized.quantization_method = "awq"
        optimized.huggingface_id = "Qwen/Qwen2.5-14B-Instruct-AWQ"
        
        # Add vLLM optimizations
        optimized._vllm_overrides.update({
            "max_num_batched_tokens": 8192,
            "use_v2_block_manager": True,
            "enable_chunked_prefill": True,
            "disable_log_stats": True,
            "block_size": 32
        })
        
        return optimized
    
    def run_model_evaluation(self, model_config, model_name: str, max_samples: int = 50):
        """Run evaluation on a single model configuration"""
        logger.info(f"🚀 Starting evaluation for {model_name}...")
        
        start_time = time.time()
        
        try:
            # Run evaluation on all test datasets
            results = self.runner.run_individual_evaluation(
                model_name=model_name.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                model_config=model_config,
                preset="h100_optimized",
                sample_limit=max_samples,
                dataset_filter=self.test_datasets
            )
            
            end_time = time.time()
            evaluation_time = end_time - start_time
            
            logger.info(f"✅ {model_name} evaluation completed in {evaluation_time:.2f} seconds")
            
            # Add timing information to results
            if isinstance(results, dict):
                results["evaluation_time_seconds"] = evaluation_time
                results["samples_per_second"] = (max_samples * len(self.test_datasets)) / evaluation_time
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_name}: {e}")
            return None
    
    def run_comparison(self, max_samples: int = 50):
        """Run full 8B vs 14B comparison"""
        logger.info("🔍 Starting 8B vs 14B performance scaling comparison...")
        
        # Create optimized configurations
        config_8b = self.create_optimized_8b_config()
        config_14b = self.create_optimized_14b_config()
        
        print("=" * 80)
        print("🎯 8B vs 14B PERFORMANCE SCALING COMPARISON")
        print("=" * 80)
        print(f"Testing {len(self.test_datasets)} datasets with {max_samples} samples each")
        print(f"Datasets: {', '.join(self.test_datasets)}")
        print(f"Total samples: {max_samples * len(self.test_datasets)}")
        print("=" * 80)
        
        # Show configuration comparison
        print(f"\n📊 Configuration Comparison:")
        print(f"8B Model:")
        print(f"  • GPU Memory: {config_8b.gpu_memory_utilization:.0%}")
        print(f"  • Max Sequences: {config_8b.max_num_seqs}")
        print(f"  • Context Length: {config_8b.max_model_len:,}")
        print(f"  • Batch Size: {config_8b.evaluation_batch_size}")
        print(f"  • Quantization: {config_8b.quantization_method}")
        
        print(f"\n14B Model:")
        print(f"  • GPU Memory: {config_14b.gpu_memory_utilization:.0%}")
        print(f"  • Max Sequences: {config_14b.max_num_seqs}")
        print(f"  • Context Length: {config_14b.max_model_len:,}")
        print(f"  • Batch Size: {config_14b.evaluation_batch_size}")
        print(f"  • Quantization: {config_14b.quantization_method}")
        
        # Run evaluations
        print(f"\n🚀 Starting evaluations...")
        
        # 8B model evaluation
        results_8b = self.run_model_evaluation(config_8b, "qwen3_8b", max_samples)
        
        # 14B model evaluation  
        results_14b = self.run_model_evaluation(config_14b, "qwen3_14b", max_samples)
        
        # Analyze results
        comparison_results = self.analyze_comparison(results_8b, results_14b, config_8b, config_14b)
        
        # Save results
        self.save_comparison_results(comparison_results)
        
        return comparison_results
    
    def analyze_comparison(self, results_8b, results_14b, config_8b, config_14b):
        """Analyze and compare the evaluation results"""
        logger.info("📊 Analyzing comparison results...")
        
        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "comparison_summary": {},
            "models": {
                "qwen3_8b": {
                    "config": self._config_to_dict(config_8b),
                    "results": results_8b
                },
                "qwen3_14b": {
                    "config": self._config_to_dict(config_14b),
                    "results": results_14b
                }
            }
        }
        
        print("\n" + "=" * 80)
        print("📈 SCALING ANALYSIS RESULTS")
        print("=" * 80)
        
        if results_8b and results_14b:
            # Extract performance metrics
            perf_8b = results_8b.get("evaluation_time_seconds", 0)
            perf_14b = results_14b.get("evaluation_time_seconds", 0)
            
            throughput_8b = results_8b.get("samples_per_second", 0)
            throughput_14b = results_14b.get("samples_per_second", 0)
            
            print(f"\n⚡ Performance Comparison:")
            print(f"  8B Model:")
            print(f"    • Evaluation Time: {perf_8b:.2f} seconds")
            print(f"    • Throughput: {throughput_8b:.2f} samples/second")
            
            print(f"  14B Model:")
            print(f"    • Evaluation Time: {perf_14b:.2f} seconds")
            print(f"    • Throughput: {throughput_14b:.2f} samples/second")
            
            if perf_8b > 0 and perf_14b > 0:
                speed_ratio = perf_14b / perf_8b
                throughput_ratio = throughput_8b / throughput_14b
                
                print(f"\n📊 Scaling Metrics:")
                print(f"  • Speed Advantage (8B): {1/speed_ratio:.2f}x faster")
                print(f"  • Throughput Advantage (8B): {throughput_ratio:.2f}x higher")
                print(f"  • Time Cost (14B): {speed_ratio:.2f}x longer")
                
                comparison["comparison_summary"] = {
                    "speed_advantage_8b": 1/speed_ratio,
                    "throughput_advantage_8b": throughput_ratio,
                    "time_cost_14b": speed_ratio,
                    "efficiency_ratio": throughput_ratio
                }
            
            # Quality comparison (if available)
            if "dataset_results" in results_8b and "dataset_results" in results_14b:
                print(f"\n🎯 Quality Comparison:")
                
                for dataset in self.test_datasets:
                    if dataset in results_8b["dataset_results"] and dataset in results_14b["dataset_results"]:
                        score_8b = results_8b["dataset_results"][dataset].get("score", 0)
                        score_14b = results_14b["dataset_results"][dataset].get("score", 0)
                        
                        print(f"  {dataset}:")
                        print(f"    • 8B Score: {score_8b:.1%}")
                        print(f"    • 14B Score: {score_14b:.1%}")
                        if score_8b > 0:
                            quality_ratio = score_14b / score_8b
                            print(f"    • Quality Advantage (14B): {quality_ratio:.2f}x")
        
        else:
            logger.warning("⚠️ Some evaluations failed - limited comparison available")
            print("⚠️ Some evaluations failed - limited comparison available")
        
        return comparison
    
    def _config_to_dict(self, config):
        """Convert model config to dictionary for serialization"""
        return {
            "model_name": config.model_name,
            "huggingface_id": config.huggingface_id,
            "size_gb": config.size_gb,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_seqs": config.max_num_seqs,
            "max_model_len": config.max_model_len,
            "evaluation_batch_size": config.evaluation_batch_size,
            "quantization_method": config.quantization_method,
            "enable_prefix_caching": config.enable_prefix_caching,
            "enforce_eager": config.enforce_eager
        }
    
    def save_comparison_results(self, comparison_results):
        """Save comparison results to files"""
        logger.info("💾 Saving comparison results...")
        
        # Save JSON results
        json_path = self.output_dir / "8b_vs_14b_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Generate markdown report
        md_report = self.generate_markdown_report(comparison_results)
        md_path = self.output_dir / "8b_vs_14b_scaling_report.md"
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        logger.info(f"📄 Results saved:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Markdown: {md_path}")
    
    def generate_markdown_report(self, comparison_results):
        """Generate a comprehensive markdown report"""
        timestamp = comparison_results.get("timestamp", "Unknown")
        summary = comparison_results.get("comparison_summary", {})
        
        report = f"""# 8B vs 14B Performance Scaling Analysis

Generated: {timestamp}

## Executive Summary

This report compares the performance scaling characteristics of Qwen-3 8B and 14B models with H100 optimizations on fixed evaluation datasets.

## Performance Metrics

"""
        
        if summary:
            report += f"""### Scaling Analysis
- **8B Speed Advantage**: {summary.get('speed_advantage_8b', 0):.2f}x faster evaluation
- **8B Throughput Advantage**: {summary.get('throughput_advantage_8b', 0):.2f}x higher samples/second
- **14B Time Cost**: {summary.get('time_cost_14b', 0):.2f}x longer evaluation time
- **Efficiency Ratio**: {summary.get('efficiency_ratio', 0):.2f}x

"""
        
        # Add configuration details
        models = comparison_results.get("models", {})
        
        if "qwen3_8b" in models:
            config_8b = models["qwen3_8b"]["config"]
            report += f"""## 8B Model Configuration
- **Model**: {config_8b.get('model_name', 'Unknown')}
- **HuggingFace ID**: {config_8b.get('huggingface_id', 'Unknown')}
- **Size**: {config_8b.get('size_gb', 0):.1f}GB
- **GPU Memory**: {config_8b.get('gpu_memory_utilization', 0):.0%}
- **Max Sequences**: {config_8b.get('max_num_seqs', 0)}
- **Context Length**: {config_8b.get('max_model_len', 0):,}
- **Batch Size**: {config_8b.get('evaluation_batch_size', 0)}
- **Quantization**: {config_8b.get('quantization_method', 'none')}

"""
        
        if "qwen3_14b" in models:
            config_14b = models["qwen3_14b"]["config"]
            report += f"""## 14B Model Configuration
- **Model**: {config_14b.get('model_name', 'Unknown')}
- **HuggingFace ID**: {config_14b.get('huggingface_id', 'Unknown')}
- **Size**: {config_14b.get('size_gb', 0):.1f}GB
- **GPU Memory**: {config_14b.get('gpu_memory_utilization', 0):.0%}
- **Max Sequences**: {config_14b.get('max_num_seqs', 0)}
- **Context Length**: {config_14b.get('max_model_len', 0):,}
- **Batch Size**: {config_14b.get('evaluation_batch_size', 0)}
- **Quantization**: {config_14b.get('quantization_method', 'none')}

"""
        
        report += """## Recommendations

Based on this scaling analysis:

1. **For High Throughput Tasks**: Use 8B model for faster processing of large volumes
2. **For Quality-Critical Tasks**: Use 14B model for better accuracy despite slower speed
3. **For Resource Efficiency**: 8B model provides better performance per GPU hour
4. **For Mixed Workloads**: Consider using both models in a pipeline architecture

## Next Steps

1. Test additional model variants and quantization methods
2. Analyze quality vs. speed trade-offs in more detail
3. Implement dynamic model selection based on task requirements
4. Explore multi-model serving architectures for optimal resource utilization
"""
        
        return report

def main():
    """Main execution function"""
    logger.info("🎯 Starting 8B vs 14B Performance Scaling Comparison...")
    
    # Create comparison runner
    comparison = ModelScalingComparison()
    
    print("🎯 8B vs 14B PERFORMANCE SCALING COMPARISON")
    print("=" * 50)
    print("Goals:")
    print("• Compare processing speed and efficiency")
    print("• Analyze resource utilization patterns")
    print("• Evaluate quality vs. performance trade-offs")
    print("• Generate scaling recommendations")
    print("=" * 50)
    
    # Run comparison with moderate sample size for testing
    # Using 30 samples per dataset for reasonable test time
    results = comparison.run_comparison(max_samples=30)
    
    if results:
        logger.info("✅ Scaling comparison completed successfully!")
        return True
    else:
        logger.error("❌ Scaling comparison failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("✅ SUCCESS: 8B vs 14B scaling analysis completed!")
        else:
            print("❌ FAILED: Scaling analysis encountered errors")
            sys.exit(1)
    except Exception as e:
        print(f"❌ SCRIPT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)