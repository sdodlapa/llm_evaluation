#!/usr/bin/env python3
"""
Main evaluation runner for individual LLM model testing
Coordinates systematic evaluation of all models under 16B parameters
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.model_configs import MODEL_CONFIGS, get_high_priority_models, get_agent_optimized_models
from models.qwen_implementation import create_qwen3_8b, create_qwen3_14b
from models.base_model import BaseModelImplementation, ModelPerformanceMetrics, AgentEvaluationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMEvaluationRunner:
    """Main class for running comprehensive LLM evaluations"""
    
    def __init__(self, output_dir: str = "results", cache_dir: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "performance").mkdir(exist_ok=True)
        (self.output_dir / "agent_tests").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.evaluation_results = {}
        self.test_suite = self._load_test_suite()
        
    def _load_test_suite(self) -> Dict[str, List]:
        """Load agent evaluation test suite"""
        # This would normally load from files, but for now we'll create inline
        return {
            "function_calling": [
                {
                    "prompt": "Calculate the tip for a $67.50 bill at 18%",
                    "expected_function": "calculate",
                    "expected_args": {"expression": "67.50 * 0.18"}
                },
                {
                    "prompt": "What's the weather like in Tokyo right now?",
                    "expected_function": "get_weather",
                    "expected_args": {"location": "Tokyo"}
                },
                {
                    "prompt": "Send an email to team@company.com about tomorrow's meeting",
                    "expected_function": "send_email", 
                    "expected_args": {"to": "team@company.com", "subject": "meeting"}
                }
            ],
            "instructions": [
                {
                    "prompt": "Write a Python function that calculates fibonacci numbers. Include docstring and type hints.",
                    "requirements": ["def", "fibonacci", "docstring", "type hints", "return"]
                },
                {
                    "prompt": "Explain machine learning in exactly 3 sentences. Use simple language.",
                    "requirements": ["machine learning", "3 sentences", "simple"]
                },
                {
                    "prompt": "Create a JSON object with user data: name=John, age=25, city=Boston",
                    "requirements": ["json", "name", "age", "city", "John", "25", "Boston"]
                }
            ],
            "conversations": [
                [
                    {"user": "I'm building a web app and need help with database design", "expected_themes": ["database", "web app"]},
                    {"user": "It's an e-commerce site with products and users", "expected_themes": ["e-commerce", "products", "users"]},
                    {"user": "What tables should I create?", "expected_themes": ["tables", "schema", "structure"]}
                ],
                [
                    {"user": "I'm having trouble with my Python code", "expected_themes": ["python", "code", "debug"]},
                    {"user": "The function returns None instead of the expected value", "expected_themes": ["function", "None", "return"]},
                    {"user": "Here's the code: def calc(x): x * 2", "expected_themes": ["missing return", "syntax"]}
                ]
            ],
            "tool_use": [
                {
                    "prompt": "I need to search for information about climate change",
                    "expected_tool": "search"
                },
                {
                    "prompt": "Create a new file called 'data.txt' with some sample content",
                    "expected_tool": "file_operations"
                }
            ],
            "reasoning": [
                {
                    "prompt": "If a train travels 60 mph for 2.5 hours, then 80 mph for 1 hour, what's the total distance?",
                    "expected_steps": ["60 * 2.5", "80 * 1", "add results", "150", "80", "230"]
                },
                {
                    "prompt": "A company has 100 employees. 60% work remotely. Of remote workers, 25% are managers. How many remote managers?",
                    "expected_steps": ["100 * 0.6", "60", "60 * 0.25", "15"]
                }
            ],
            "json_tasks": [
                {
                    "prompt": "Create a JSON response with status='success', data=['item1', 'item2'], count=2"
                },
                {
                    "prompt": "Format this as JSON: User John Smith, age 30, email john@email.com, active user"
                }
            ]
        }
    
    def run_individual_evaluation(self, model_name: str, model_config: Dict) -> Dict[str, Any]:
        """Run complete evaluation for a single model"""
        logger.info(f"🚀 Starting evaluation for {model_name}")
        
        results = {
            "model_name": model_name,
            "config": model_config,
            "timestamp": datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            # Create model instance
            model = self._create_model_instance(model_name, model_config)
            if not model:
                results["status"] = "failed"
                results["error"] = "Failed to create model instance"
                return results
            
            # Load model
            logger.info(f"Loading {model_name}...")
            if not model.load_model():
                results["status"] = "failed"
                results["error"] = "Failed to load model"
                return results
            
            # Performance benchmarking
            logger.info(f"Running performance benchmark for {model_name}...")
            performance_metrics = self._run_performance_benchmark(model)
            results["performance"] = vars(performance_metrics) if performance_metrics else None
            
            # Agent capability evaluation
            logger.info(f"Running agent evaluation for {model_name}...")
            agent_results = model.evaluate_agent_capabilities(self.test_suite)
            results["agent_evaluation"] = vars(agent_results) if agent_results else None
            
            # Memory usage analysis
            results["memory_analysis"] = model.get_memory_usage()
            
            # Model-specific tests
            if hasattr(model, 'test_function_calling_capability'):
                logger.info(f"Running function calling tests for {model_name}...")
                function_results = model.test_function_calling_capability()
                results["function_calling_test"] = function_results
            
            # Save individual results
            self._save_individual_results(model_name, results)
            
            # Cleanup
            model.unload_model()
            
            results["status"] = "completed"
            logger.info(f"✅ Evaluation completed for {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_name}: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            
            # Try to cleanup on error
            try:
                if 'model' in locals() and model:
                    model.unload_model()
            except:
                pass
        
        return results
    
    def _create_model_instance(self, model_name: str, config: Dict) -> Optional[BaseModelImplementation]:
        """Create appropriate model instance based on model name"""
        try:
            if "qwen3_8b" in model_name.lower() or "qwen" in model_name.lower() and "8b" in model_name.lower():
                return create_qwen3_8b(self.cache_dir)
            elif "qwen3_14b" in model_name.lower() or "qwen" in model_name.lower() and "14b" in model_name.lower():
                return create_qwen3_14b(self.cache_dir)
            else:
                # For other models, we'd need to implement their specific loaders
                logger.warning(f"No specific implementation for {model_name}, skipping")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create model instance for {model_name}: {e}")
            return None
    
    def _run_performance_benchmark(self, model: BaseModelImplementation) -> Optional[ModelPerformanceMetrics]:
        """Run performance benchmark with standard test prompts"""
        test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a Python function to calculate the factorial of a number.",
            "What are the main differences between supervised and unsupervised learning?",
            "Describe the process of photosynthesis step by step.",
            "How would you design a simple recommendation system?"
        ]
        
        try:
            return model.benchmark_performance(test_prompts)
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return None
    
    def _save_individual_results(self, model_name: str, results: Dict):
        """Save individual model results"""
        # Clean model name for filename
        safe_name = model_name.replace(" ", "_").replace("/", "_").lower()
        
        # Save detailed results
        results_file = self.output_dir / "performance" / f"{safe_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def run_comparison_analysis(self):
        """Run comparative analysis across all evaluated models"""
        logger.info("🔍 Running comparison analysis...")
        
        # Load all individual results
        all_results = []
        results_dir = self.output_dir / "performance"
        
        for results_file in results_dir.glob("*_results.json"):
            try:
                with open(results_file) as f:
                    result = json.load(f)
                    if result.get("status") == "completed":
                        all_results.append(result)
            except Exception as e:
                logger.warning(f"Could not load {results_file}: {e}")
        
        if not all_results:
            logger.warning("No completed results found for comparison")
            return
        
        # Create comparison report
        comparison = self._create_comparison_report(all_results)
        
        # Save comparison
        comparison_file = self.output_dir / "comparisons" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(comparison)
        
        logger.info(f"Comparison analysis saved to {comparison_file}")
    
    def _create_comparison_report(self, results: List[Dict]) -> Dict:
        """Create detailed comparison report"""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(results),
            "models": []
        }
        
        for result in results:
            model_summary = {
                "name": result["model_name"],
                "license": result["config"].get("license", "Unknown"),
                "size_gb": result["config"].get("size_gb", 0),
                "context_window": result["config"].get("context_window", 0),
                "agent_optimized": result["config"].get("agent_optimized", False)
            }
            
            # Add performance metrics
            if result.get("performance"):
                perf = result["performance"]
                model_summary.update({
                    "tokens_per_second": perf.get("tokens_per_second", 0),
                    "memory_usage_gb": perf.get("memory_usage_gb", 0),
                    "latency_first_token_ms": perf.get("latency_first_token_ms", 0)
                })
            
            # Add agent evaluation scores
            if result.get("agent_evaluation"):
                agent = result["agent_evaluation"]
                model_summary.update({
                    "function_calling_accuracy": agent.get("function_calling_accuracy", 0),
                    "instruction_following_score": agent.get("instruction_following_score", 0),
                    "multi_turn_coherence": agent.get("multi_turn_coherence", 0),
                    "reasoning_quality_score": agent.get("reasoning_quality_score", 0)
                })
            
            # Calculate overall agent score
            agent_scores = [
                model_summary.get("function_calling_accuracy", 0),
                model_summary.get("instruction_following_score", 0), 
                model_summary.get("multi_turn_coherence", 0),
                model_summary.get("reasoning_quality_score", 0)
            ]
            model_summary["overall_agent_score"] = sum(agent_scores) / len(agent_scores) if agent_scores else 0
            
            comparison["models"].append(model_summary)
        
        # Sort by overall agent score
        comparison["models"].sort(key=lambda x: x.get("overall_agent_score", 0), reverse=True)
        
        return comparison
    
    def _create_summary_report(self, comparison: Dict):
        """Create human-readable summary report"""
        report_lines = [
            "# LLM Evaluation Summary Report",
            f"Generated: {comparison['timestamp']}",
            f"Total models evaluated: {comparison['total_models']}",
            "",
            "## Top Performing Models (by Overall Agent Score)",
            ""
        ]
        
        for i, model in enumerate(comparison["models"][:5], 1):
            report_lines.extend([
                f"### {i}. {model['name']}",
                f"- **License**: {model['license']}",
                f"- **Size**: {model['size_gb']}GB",
                f"- **Overall Agent Score**: {model.get('overall_agent_score', 0):.3f}",
                f"- **Function Calling**: {model.get('function_calling_accuracy', 0):.3f}",
                f"- **Instruction Following**: {model.get('instruction_following_score', 0):.3f}",
                f"- **Memory Usage**: {model.get('memory_usage_gb', 0):.1f}GB",
                f"- **Performance**: {model.get('tokens_per_second', 0):.1f} tokens/sec",
                ""
            ])
        
        # Performance comparison
        report_lines.extend([
            "## Performance Comparison",
            "",
            "| Model | Memory (GB) | Speed (tok/s) | Agent Score |",
            "|-------|-------------|---------------|-------------|"
        ])
        
        for model in comparison["models"]:
            memory = model.get('memory_usage_gb', 0)
            speed = model.get('tokens_per_second', 0)
            score = model.get('overall_agent_score', 0)
            report_lines.append(f"| {model['name']} | {memory:.1f} | {speed:.1f} | {score:.3f} |")
        
        # Recommendations
        if comparison["models"]:
            best_overall = comparison["models"][0]
            best_apache = next((m for m in comparison["models"] if "Apache" in m.get("license", "")), None)
            most_efficient = min(comparison["models"], key=lambda x: x.get("memory_usage_gb", float('inf')))
            
            report_lines.extend([
                "",
                "## Recommendations",
                "",
                f"**Best Overall**: {best_overall['name']} (Score: {best_overall.get('overall_agent_score', 0):.3f})",
                f"**Best Apache Licensed**: {best_apache['name'] if best_apache else 'None'} (Safe for commercial use)",
                f"**Most Memory Efficient**: {most_efficient['name']} ({most_efficient.get('memory_usage_gb', 0):.1f}GB)",
                ""
            ])
        
        # Save report
        report_file = self.output_dir / "reports" / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Summary report saved to {report_file}")
    
    def run_full_evaluation(self, models_to_test: Optional[List[str]] = None):
        """Run complete evaluation suite"""
        logger.info("🚀 Starting full LLM evaluation suite")
        
        # Determine which models to test
        if models_to_test:
            configs_to_test = {k: v for k, v in MODEL_CONFIGS.items() if k in models_to_test}
        else:
            # Default: test high priority models
            configs_to_test = get_high_priority_models()
        
        logger.info(f"Will evaluate {len(configs_to_test)} models: {list(configs_to_test.keys())}")
        
        # Run individual evaluations
        for model_name, config in configs_to_test.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'='*60}")
            
            result = self.run_individual_evaluation(model_name, vars(config))
            self.evaluation_results[model_name] = result
            
            # Brief pause between models to allow cleanup
            time.sleep(5)
        
        # Run comparison analysis
        self.run_comparison_analysis()
        
        logger.info("🎉 Full evaluation completed!")
        return self.evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation suite")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--cache-dir", help="Model cache directory")
    parser.add_argument("--priority-only", action="store_true", help="Test only high priority models")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    # Create runner
    runner = LLMEvaluationRunner(args.output_dir, args.cache_dir)
    
    if args.quick_test:
        # Quick test with just one model
        logger.info("Running quick test with Qwen-3 8B")
        runner.run_individual_evaluation("qwen3_8b", vars(MODEL_CONFIGS["qwen3_8b"]))
    else:
        # Full evaluation
        models_to_test = None
        if args.models:
            models_to_test = args.models
        elif args.priority_only:
            models_to_test = list(get_high_priority_models().keys())
        
        runner.run_full_evaluation(models_to_test)

if __name__ == "__main__":
    main()