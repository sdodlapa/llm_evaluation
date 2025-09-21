#!/usr/bin/env python3
"""
Comprehensive Model Coverage Expansion
Implements the final priority from SESSION_STATUS_2025_09_17.md: expanding model coverage
Tests all Qwen variants across different presets and quantization methods
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project paths
sys.path.append('evaluation')
sys.path.append('models')
sys.path.append('configs')

from run_evaluation import LLMEvaluationRunner
from model_configs import MODEL_CONFIGS, get_all_qwen_variants
from models.registry import list_available_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveModelTester:
    """Test comprehensive model coverage across variants, presets, and configurations"""
    
    def __init__(self, output_dir: str = "test_results/model_coverage"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluation runner
        self.runner = LLMEvaluationRunner(
            output_dir=str(self.output_dir),
            cache_dir=str(self.output_dir / "cache")
        )
        
        # Define core test datasets (using our fixed/optimized datasets)
        self.core_datasets = [
            "humaneval",   # Coding - strong performance
            "gsm8k",       # Reasoning - strong performance  
            "hellaswag",   # Common sense - newly fixed
            "mt_bench"     # Instruction following - newly fixed
        ]
        
        # Define preset configurations to test
        self.presets_to_test = [
            "balanced",           # Default optimization
            "performance",        # Maximum performance
            "memory_optimized"    # Maximum efficiency
        ]
        
        # Define model variants to test
        self.model_variants = self._get_available_variants()
        
        print(f"🚀 Comprehensive Model Coverage Testing")
        print(f"📊 Models to test: {len(self.model_variants)}")
        print(f"⚙️  Presets per model: {len(self.presets_to_test)}")
        print(f"📋 Datasets per test: {len(self.core_datasets)}")
        print(f"🔢 Total configurations: {len(self.model_variants) * len(self.presets_to_test)}")
        print()
        
    def _get_available_variants(self) -> List[str]:
        """Get all available Qwen model variants from registry"""
        available_models = list_available_models()
        
        # Filter for Qwen models
        qwen_models = [model for model in available_models if 'qwen' in model.lower()]
        
        # Sort by size for logical testing order
        def sort_key(model_name):
            if '8b' in model_name.lower():
                return 1
            elif '14b' in model_name.lower():
                return 2
            else:
                return 0
        
        qwen_models.sort(key=sort_key)
        
        logger.info(f"Available Qwen variants: {qwen_models}")
        return qwen_models
    
    def get_model_size_category(self, model_name: str) -> str:
        """Categorize model by size"""
        model_lower = model_name.lower()
        if '8b' in model_lower:
            return "8B"
        elif '14b' in model_lower:
            return "14B"
        else:
            return "Unknown"
    
    def run_single_configuration_test(self, model_name: str, preset: str, max_samples: int = 25) -> Dict[str, Any]:
        """Run evaluation for a single model/preset configuration"""
        logger.info(f"🧪 Testing {model_name} with {preset} preset...")
        
        start_time = time.time()
        
        try:
            # Get base model config
            config_key = model_name
            if model_name not in MODEL_CONFIGS:
                # Try alternative mappings
                if model_name == "qwen_8b":
                    config_key = "qwen3_8b"
                elif model_name == "qwen_14b":
                    config_key = "qwen3_14b"
                elif model_name == "qwen2.5_8b":
                    config_key = "qwen3_8b"
                elif model_name == "qwen2.5_14b":
                    config_key = "qwen3_14b"
                else:
                    logger.warning(f"No config found for {model_name}, skipping...")
                    return {"status": "skipped", "reason": "no_config"}
            
            base_config = MODEL_CONFIGS[config_key]
            
            # Apply preset configuration
            if preset != "balanced":
                model_config = base_config.create_preset_variant(preset)
            else:
                model_config = base_config
            
            # Run evaluation with limited samples for speed
            results = self.runner.run_individual_evaluation(
                model_name=model_name,
                model_config=model_config,
                preset=preset,
                sample_limit=max_samples,
                dataset_filter=self.core_datasets,
                save_predictions=False  # Skip saving for bulk testing
            )
            
            end_time = time.time()
            test_time = end_time - start_time
            
            # Extract key metrics
            test_result = {
                "model_name": model_name,
                "preset": preset,
                "status": "completed",
                "test_time_seconds": test_time,
                "model_size_category": self.get_model_size_category(model_name),
                "config_summary": {
                    "gpu_memory_utilization": model_config.gpu_memory_utilization,
                    "max_model_len": model_config.max_model_len,
                    "max_num_seqs": getattr(model_config, 'max_num_seqs', None),
                    "quantization": model_config.quantization_method,
                },
                "performance_summary": {},
                "dataset_results": {}
            }
            
            # Extract performance metrics
            if results and "performance" in results and results["performance"]:
                perf = results["performance"]
                test_result["performance_summary"] = {
                    "throughput_tokens_per_second": perf.get("throughput_tokens_per_second", 0),
                    "memory_used_gb": perf.get("memory_used_gb", 0),
                    "memory_efficiency": perf.get("memory_efficiency_percent", 0),
                    "inference_latency_ms": perf.get("inference_latency_ms", 0)
                }
            
            # Extract dataset evaluation results
            if results and "dataset_evaluation" in results and results["dataset_evaluation"]:
                dataset_eval = results["dataset_evaluation"]
                for dataset_name in self.core_datasets:
                    if dataset_name in dataset_eval:
                        dataset_result = dataset_eval[dataset_name]
                        if isinstance(dataset_result, dict) and "metrics" in dataset_result:
                            metrics = dataset_result["metrics"]
                            test_result["dataset_results"][dataset_name] = {
                                "accuracy": metrics.get("accuracy", 0),
                                "samples_evaluated": dataset_result.get("samples_evaluated", 0),
                                "evaluation_time": dataset_result.get("evaluation_time_seconds", 0)
                            }
            
            logger.info(f"✅ {model_name}/{preset} completed in {test_time:.2f}s")
            return test_result
            
        except Exception as e:
            end_time = time.time()
            test_time = end_time - start_time
            
            logger.error(f"❌ {model_name}/{preset} failed: {e}")
            return {
                "model_name": model_name,
                "preset": preset, 
                "status": "failed",
                "error": str(e),
                "test_time_seconds": test_time,
                "model_size_category": self.get_model_size_category(model_name)
            }
    
    def run_comprehensive_testing(self, max_samples: int = 25) -> Dict[str, Any]:
        """Run comprehensive testing across all variants and presets"""
        logger.info("🚀 Starting comprehensive model coverage testing...")
        
        start_time = time.time()
        all_results = []
        success_count = 0
        failure_count = 0
        
        print("=" * 80)
        print("🎯 COMPREHENSIVE MODEL COVERAGE TESTING")
        print("=" * 80)
        print(f"Testing {len(self.model_variants)} models × {len(self.presets_to_test)} presets")
        print(f"Datasets: {', '.join(self.core_datasets)}")
        print(f"Samples per dataset: {max_samples}")
        print("=" * 80)
        
        total_tests = len(self.model_variants) * len(self.presets_to_test)
        current_test = 0
        
        for model_name in self.model_variants:
            print(f"\n📍 Testing model: {model_name} ({self.get_model_size_category(model_name)})")
            
            for preset in self.presets_to_test:
                current_test += 1
                print(f"  ⚙️  [{current_test}/{total_tests}] {preset} preset...")
                
                result = self.run_single_configuration_test(model_name, preset, max_samples)
                all_results.append(result)
                
                if result["status"] == "completed":
                    success_count += 1
                    # Brief summary
                    perf = result.get("performance_summary", {})
                    throughput = perf.get("throughput_tokens_per_second", 0)
                    memory = perf.get("memory_used_gb", 0)
                    print(f"      ✅ Success - {throughput:.1f} tok/s, {memory:.1f}GB VRAM")
                else:
                    failure_count += 1
                    print(f"      ❌ Failed - {result.get('error', 'Unknown error')}")
                
                # Pause between tests for cleanup
                time.sleep(2)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "total_configurations_tested": total_tests,
                "successful_tests": success_count,
                "failed_tests": failure_count,
                "success_rate": success_count / total_tests if total_tests > 0 else 0,
                "models_tested": self.model_variants,
                "presets_tested": self.presets_to_test,
                "datasets_used": self.core_datasets,
                "samples_per_dataset": max_samples
            },
            "individual_results": all_results,
            "analysis": self._analyze_comprehensive_results(all_results)
        }
        
        # Save results
        self._save_comprehensive_results(comprehensive_results)
        
        # Print summary
        self._print_comprehensive_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _analyze_comprehensive_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comprehensive testing results"""
        logger.info("📊 Analyzing comprehensive results...")
        
        # Successful results only
        successful_results = [r for r in results if r["status"] == "completed"]
        
        if not successful_results:
            return {"error": "No successful tests to analyze"}
        
        analysis = {
            "performance_by_size": {},
            "performance_by_preset": {},
            "best_configurations": {},
            "scaling_analysis": {},
            "recommendations": {}
        }
        
        # Group by model size
        size_groups = {}
        for result in successful_results:
            size = result["model_size_category"]
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(result)
        
        # Analyze by size category
        for size, size_results in size_groups.items():
            if not size_results:
                continue
                
            throughputs = [r.get("performance_summary", {}).get("throughput_tokens_per_second", 0) for r in size_results]
            memories = [r.get("performance_summary", {}).get("memory_used_gb", 0) for r in size_results]
            
            analysis["performance_by_size"][size] = {
                "count": len(size_results),
                "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
                "avg_memory": sum(memories) / len(memories) if memories else 0,
                "min_memory": min(memories) if memories else 0,
                "configurations": [f"{r['model_name']}_{r['preset']}" for r in size_results]
            }
        
        # Group by preset
        preset_groups = {}
        for result in successful_results:
            preset = result["preset"]
            if preset not in preset_groups:
                preset_groups[preset] = []
            preset_groups[preset].append(result)
        
        # Analyze by preset
        for preset, preset_results in preset_groups.items():
            if not preset_results:
                continue
                
            throughputs = [r.get("performance_summary", {}).get("throughput_tokens_per_second", 0) for r in preset_results]
            memories = [r.get("performance_summary", {}).get("memory_used_gb", 0) for r in preset_results]
            
            analysis["performance_by_preset"][preset] = {
                "count": len(preset_results),
                "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
                "avg_memory": sum(memories) / len(memories) if memories else 0,
                "models_tested": list(set(r["model_name"] for r in preset_results))
            }
        
        # Find best configurations
        if successful_results:
            # Best throughput
            best_throughput = max(successful_results, 
                                key=lambda x: x.get("performance_summary", {}).get("throughput_tokens_per_second", 0))
            
            # Most memory efficient
            memory_results = [r for r in successful_results if r.get("performance_summary", {}).get("memory_used_gb", 0) > 0]
            best_memory = min(memory_results, 
                            key=lambda x: x.get("performance_summary", {}).get("memory_used_gb", float('inf'))) if memory_results else None
            
            analysis["best_configurations"] = {
                "highest_throughput": {
                    "config": f"{best_throughput['model_name']}_{best_throughput['preset']}",
                    "throughput": best_throughput.get("performance_summary", {}).get("throughput_tokens_per_second", 0),
                    "memory": best_throughput.get("performance_summary", {}).get("memory_used_gb", 0)
                }
            }
            
            if best_memory:
                analysis["best_configurations"]["most_memory_efficient"] = {
                    "config": f"{best_memory['model_name']}_{best_memory['preset']}",
                    "memory": best_memory.get("performance_summary", {}).get("memory_used_gb", 0),
                    "throughput": best_memory.get("performance_summary", {}).get("throughput_tokens_per_second", 0)
                }
        
        # Scaling analysis between 8B and 14B
        if "8B" in size_groups and "14B" in size_groups:
            size_8b = analysis["performance_by_size"]["8B"]
            size_14b = analysis["performance_by_size"]["14B"]
            
            throughput_ratio = size_8b["avg_throughput"] / size_14b["avg_throughput"] if size_14b["avg_throughput"] > 0 else 0
            memory_ratio = size_14b["avg_memory"] / size_8b["avg_memory"] if size_8b["avg_memory"] > 0 else 0
            
            analysis["scaling_analysis"] = {
                "throughput_advantage_8b": throughput_ratio,
                "memory_cost_14b": memory_ratio,
                "size_efficiency_score": throughput_ratio / memory_ratio if memory_ratio > 0 else 0
            }
        
        # Generate recommendations
        recommendations = []
        
        if analysis["best_configurations"]:
            best_config = analysis["best_configurations"].get("highest_throughput", {})
            recommendations.append(f"For maximum performance: {best_config.get('config', 'N/A')}")
            
            efficient_config = analysis["best_configurations"].get("most_memory_efficient", {})
            recommendations.append(f"For memory efficiency: {efficient_config.get('config', 'N/A')}")
        
        if "scaling_analysis" in analysis:
            scaling = analysis["scaling_analysis"]
            if scaling.get("throughput_advantage_8b", 0) > 1.5:
                recommendations.append("8B models show significant speed advantage - consider for latency-critical applications")
            if scaling.get("memory_cost_14b", 0) < 2.0:
                recommendations.append("14B models have reasonable memory overhead - worth considering for quality-critical applications")
        
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive testing results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        results_file = self.output_dir / f"comprehensive_model_coverage_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Comprehensive results saved to: {results_file}")
        
        # Save summary report
        report_file = self.output_dir / f"model_coverage_summary_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(self._generate_markdown_report(results))
        
        logger.info(f"📄 Summary report saved to: {report_file}")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown summary report"""
        metadata = results["test_metadata"]
        analysis = results.get("analysis", {})
        
        report = f"""# Comprehensive Model Coverage Test Results

## Test Overview
- **Timestamp**: {metadata["timestamp"]}
- **Total Configurations**: {metadata["total_configurations_tested"]}
- **Success Rate**: {metadata["success_rate"]:.1%} ({metadata["successful_tests"]}/{metadata["total_configurations_tested"]})
- **Total Test Time**: {metadata["total_time_seconds"]:.1f} seconds

## Models Tested
{', '.join(metadata["models_tested"])}

## Presets Tested
{', '.join(metadata["presets_tested"])}

## Datasets Used
{', '.join(metadata["datasets_used"])} ({metadata["samples_per_dataset"]} samples each)

"""
        
        if "performance_by_size" in analysis:
            report += "## Performance by Model Size\n\n"
            for size, perf in analysis["performance_by_size"].items():
                report += f"### {size} Models\n"
                report += f"- **Count**: {perf['count']} configurations\n"
                report += f"- **Average Throughput**: {perf['avg_throughput']:.1f} tokens/second\n"
                report += f"- **Maximum Throughput**: {perf['max_throughput']:.1f} tokens/second\n"
                report += f"- **Average Memory**: {perf['avg_memory']:.1f} GB\n"
                report += f"- **Minimum Memory**: {perf['min_memory']:.1f} GB\n\n"
        
        if "performance_by_preset" in analysis:
            report += "## Performance by Preset\n\n"
            for preset, perf in analysis["performance_by_preset"].items():
                report += f"### {preset.title()} Preset\n"
                report += f"- **Models Tested**: {perf['count']}\n"
                report += f"- **Average Throughput**: {perf['avg_throughput']:.1f} tokens/second\n"
                report += f"- **Average Memory**: {perf['avg_memory']:.1f} GB\n\n"
        
        if "best_configurations" in analysis:
            report += "## Best Configurations\n\n"
            best = analysis["best_configurations"]
            
            if "highest_throughput" in best:
                ht = best["highest_throughput"]
                report += f"### Highest Throughput\n"
                report += f"- **Configuration**: {ht['config']}\n"
                report += f"- **Throughput**: {ht['throughput']:.1f} tokens/second\n"
                report += f"- **Memory Usage**: {ht['memory']:.1f} GB\n\n"
            
            if "most_memory_efficient" in best:
                me = best["most_memory_efficient"]
                report += f"### Most Memory Efficient\n"
                report += f"- **Configuration**: {me['config']}\n"
                report += f"- **Memory Usage**: {me['memory']:.1f} GB\n"
                report += f"- **Throughput**: {me['throughput']:.1f} tokens/second\n\n"
        
        if "scaling_analysis" in analysis:
            report += "## Scaling Analysis (8B vs 14B)\n\n"
            scaling = analysis["scaling_analysis"]
            report += f"- **8B Throughput Advantage**: {scaling.get('throughput_advantage_8b', 0):.2f}x\n"
            report += f"- **14B Memory Cost**: {scaling.get('memory_cost_14b', 0):.2f}x\n"
            report += f"- **Size Efficiency Score**: {scaling.get('size_efficiency_score', 0):.2f}\n\n"
        
        if "recommendations" in analysis and analysis["recommendations"]:
            report += "## Recommendations\n\n"
            for i, rec in enumerate(analysis["recommendations"], 1):
                report += f"{i}. {rec}\n"
            report += "\n"
        
        report += "---\n"
        report += f"*Report generated by Comprehensive Model Coverage Tester*\n"
        
        return report
    
    def _print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print comprehensive summary to console"""
        metadata = results["test_metadata"]
        analysis = results.get("analysis", {})
        
        print("\n" + "=" * 80)
        print("📈 COMPREHENSIVE MODEL COVERAGE RESULTS")
        print("=" * 80)
        
        print(f"\n🎯 Test Summary:")
        print(f"  • Total Configurations: {metadata['total_configurations_tested']}")
        print(f"  • Success Rate: {metadata['success_rate']:.1%} ({metadata['successful_tests']}/{metadata['total_configurations_tested']})")
        print(f"  • Total Test Time: {metadata['total_time_seconds']:.1f} seconds")
        print(f"  • Average Time per Test: {metadata['total_time_seconds']/metadata['total_configurations_tested']:.1f} seconds")
        
        if "performance_by_size" in analysis:
            print(f"\n⚡ Performance by Size:")
            for size, perf in analysis["performance_by_size"].items():
                print(f"  {size} Models:")
                print(f"    • Configurations: {perf['count']}")
                print(f"    • Avg Throughput: {perf['avg_throughput']:.1f} tok/s")
                print(f"    • Max Throughput: {perf['max_throughput']:.1f} tok/s")
                print(f"    • Avg Memory: {perf['avg_memory']:.1f} GB")
        
        if "best_configurations" in analysis:
            print(f"\n🏆 Best Configurations:")
            best = analysis["best_configurations"]
            
            if "highest_throughput" in best:
                ht = best["highest_throughput"]
                print(f"  Highest Throughput: {ht['config']}")
                print(f"    • {ht['throughput']:.1f} tok/s, {ht['memory']:.1f} GB")
            
            if "most_memory_efficient" in best:
                me = best["most_memory_efficient"]
                print(f"  Most Memory Efficient: {me['config']}")
                print(f"    • {me['memory']:.1f} GB, {me['throughput']:.1f} tok/s")
        
        if "scaling_analysis" in analysis:
            print(f"\n📊 Scaling Analysis:")
            scaling = analysis["scaling_analysis"]
            print(f"  • 8B Speed Advantage: {scaling.get('throughput_advantage_8b', 0):.2f}x")
            print(f"  • 14B Memory Cost: {scaling.get('memory_cost_14b', 0):.2f}x")
            print(f"  • Efficiency Score: {scaling.get('size_efficiency_score', 0):.2f}")
        
        if "recommendations" in analysis and analysis["recommendations"]:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Model Coverage Expansion")
    print("This implements the final priority from SESSION_STATUS_2025_09_17.md")
    print()
    
    # Create tester instance
    tester = ComprehensiveModelTester()
    
    # Run comprehensive testing with reasonable sample size
    results = tester.run_comprehensive_testing(max_samples=25)
    
    print(f"\n✅ Comprehensive model coverage testing completed!")
    print(f"📁 Results saved to: {tester.output_dir}")
    
    return results

if __name__ == "__main__":
    main()