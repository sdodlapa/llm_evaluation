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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.model_configs import MODEL_CONFIGS, get_high_priority_models, get_agent_optimized_models, ModelConfig, estimate_memory_usage
from models.base_model import BaseModelImplementation, ModelPerformanceMetrics, AgentEvaluationResult
try:
    from .dataset_manager import EnhancedDatasetManager
    from .metrics import EvaluationMetrics, evaluate_dataset_predictions, print_evaluation_summary
    from .performance_monitor import LivePerformanceMonitor
    from .dataset_evaluation import DatasetEvaluator
    from .reporting import ResultsManager
except ImportError:
    # When running as script, use absolute imports
    from dataset_manager import EnhancedDatasetManager
    from metrics import EvaluationMetrics, evaluate_dataset_predictions, print_evaluation_summary
    from performance_monitor import LivePerformanceMonitor
    from dataset_evaluation import DatasetEvaluator
    from reporting import ResultsManager

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
    
    def __init__(self, output_dir: str = "results", cache_dir: Optional[str] = None, 
                 data_cache_dir: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "performance").mkdir(exist_ok=True)
        (self.output_dir / "agent_tests").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "dataset_results").mkdir(exist_ok=True)
        
        # Initialize dataset manager, performance monitor, and dataset evaluator
        self.dataset_manager = EnhancedDatasetManager(
            base_data_path=data_cache_dir or "evaluation_data"
        )
        self.performance_monitor = LivePerformanceMonitor()
        self.dataset_evaluator = DatasetEvaluator(self.dataset_manager)
        self.results_manager = ResultsManager(str(self.output_dir))
        self.evaluation_results = {}
        
        # Load both synthetic and real test suites
        self.test_suite = self._load_test_suite()
        self.real_datasets = {}
        
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
    
    def run_individual_evaluation(self, model_name: str, model_config: ModelConfig, preset: str = "balanced",
                                 save_predictions: bool = False, prediction_count: Optional[int] = None,
                                 dataset_filter: Optional[List[str]] = None,
                                 sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Run complete evaluation with enhanced configuration support"""
        logger.info(f"🚀 Starting evaluation for {model_name} with preset: {preset}")
        
        results = {
            "model_name": model_name,
            "preset": preset,
            "config": {
                "model_config": vars(model_config),
                "vllm_args": model_config.to_vllm_args(),
                "sampling_params": model_config.get_agent_sampling_params(),
                "preset": preset
            },
            "timestamp": datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            # Create model instance with specific preset
            model = self._create_model_instance(model_name, model_config, preset)
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
            
            # Start performance monitoring
            logger.info(f"Running performance benchmark for {model_name}...")
            self.performance_monitor.start_monitoring(model_name, preset, "performance_test")
            
            # Run a simple performance test
            try:
                test_prompt = "Write a simple function to calculate the factorial of a number."
                start_time = time.time()
                response = model.generate_response(test_prompt, max_tokens=100, temperature=0.1)
                end_time = time.time()
                
                # Record performance metrics
                self.performance_monitor.record_tokens_processed(len(test_prompt.split()) + 100)  # Approximate
                self.performance_monitor.record_request_timing(start_time, end_time)
                
                # Stop monitoring and get metrics
                performance_metrics = self.performance_monitor.stop_monitoring(
                    dataset_samples_processed=1,
                    accuracy_metrics={"completion_success": 1.0 if response else 0.0}
                )
                results["performance"] = vars(performance_metrics) if performance_metrics else None
                
            except Exception as e:
                logger.warning(f"Performance benchmark failed: {e}")
                results["performance"] = {"error": str(e)}
            
            # Real dataset evaluation (unless synthetic-only) using new module
            if not getattr(self, '_synthetic_only', False):
                logger.info(f"Running real dataset evaluation for {model_name}...")
                dataset_results = self.dataset_evaluator.evaluate_datasets(
                    model, preset, save_predictions, prediction_count, 
                    dataset_filter, sample_limit
                )
                results["dataset_evaluation"] = dataset_results
            
            # Agent capability evaluation (synthetic tests, unless datasets-only)
            if not getattr(self, '_datasets_only', False):
                logger.info(f"Running synthetic agent evaluation for {model_name}...")
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
            self.results_manager.save_individual_results(model_name, results, preset)
            
            # Cleanup
            model.unload_model()
            
            results["status"] = "completed"
            logger.info(f"✅ Evaluation completed for {model_name} with preset: {preset}")
            
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
    
    def _create_model_instance(self, model_name: str, model_config: ModelConfig, preset: str = "balanced") -> Optional[BaseModelImplementation]:
        """Create model instance using the model registry"""
        from models.registry import create_model, list_available_models
        
        try:
            logger.info(f"Creating {model_name} instance with preset: {preset}")
            
            model = create_model(model_name, preset, self.cache_dir)
            if model is None:
                available = list_available_models()
                logger.warning(f"Model {model_name} not found in registry. "
                             f"Available models: {', '.join(available)}")
                return None
            
            return model
                
        except Exception as e:
            logger.error(f"Failed to create model instance for {model_name} with preset {preset}: {e}")
            return None

    
    def _evaluate_on_single_dataset(self, model: BaseModelImplementation, 
                                   dataset: Dict, dataset_name: str, 
                                   save_predictions: bool = False,
                                   prediction_count: Optional[int] = None,
                                   sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate model on a single dataset"""
        samples = dataset.get("samples", [])
        if not samples:
            return {"error": "No samples in dataset", "samples_evaluated": 0}   
        
        # Apply sample limit if specified
        if sample_limit:
            max_samples = sample_limit
        else:
            max_samples = 100  # Default limit for efficiency
            
        if len(samples) > max_samples:
            samples = samples[:max_samples]
            logger.info(f"Limited {dataset_name} to {max_samples} samples")
        
        predictions = []
        evaluation_errors = 0
        
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            try:
                # Generate prediction based on sample type
                prompt = self._create_prompt_from_sample(sample, dataset["task_type"])
                prediction = model.generate_response(prompt)
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(samples)} samples for {dataset_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate prediction for sample {i}: {e}")
                predictions.append("")  # Empty prediction for failed cases
                evaluation_errors += 1
        
        evaluation_time = time.time() - start_time
        
        # Calculate metrics using the metrics module
        try:
            metrics_results = evaluate_dataset_predictions(
                dataset["task_type"], predictions, samples
            )
            
            # Convert EvaluationResult objects to dictionaries
            metrics_dict = {}
            for metric_name, result in metrics_results.items():
                metrics_dict[metric_name] = {
                    "score": result.score,
                    "total_samples": result.total_samples,
                    "successful_samples": result.successful_samples,
                    "metric_name": result.metric_name,
                    "details": result.details
                }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed for {dataset_name}: {e}")
            metrics_dict = {"error": str(e)}
        
        # Determine how many predictions to save
        if save_predictions:
            if prediction_count is not None:
                saved_predictions = predictions[:prediction_count]
            else:
                saved_predictions = predictions  # Save all
        else:
            saved_predictions = predictions[:5]  # Default: first 5 for debugging
        
        # Save predictions to file if requested
        prediction_file = None
        if save_predictions:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            pred_dir = self.output_dir / "dataset_results"
            pred_dir.mkdir(exist_ok=True)
            
            prediction_file = pred_dir / f"{dataset_name.lower()}_predictions_{timestamp}.json"
            
            prediction_data = {
                "model": model.config.model_name,
                "dataset": dataset_name,
                "timestamp": timestamp,
                "samples_count": len(samples),
                "successful_predictions": len([p for p in predictions if p.strip()]),
                "evaluation_time": evaluation_time,
                "samples": [
                    {
                        "id": sample.get("id", i),
                        "prompt": sample.get("prompt", ""),
                        "prediction": pred,
                        "test_cases": sample.get("test_cases", ""),
                        "expected_output": sample.get("expected_output", "")
                    }
                    for i, (sample, pred) in enumerate(zip(samples, saved_predictions))
                ]
            }
            
            try:
                with open(prediction_file, 'w') as f:
                    json.dump(prediction_data, f, indent=2)
                logger.info(f"Predictions saved to {prediction_file}")
            except Exception as e:
                logger.warning(f"Failed to save predictions: {e}")
                prediction_file = None

        return {
            "dataset_name": dataset_name,
            "task_type": dataset["task_type"],
            "samples_evaluated": len(samples),
            "evaluation_errors": evaluation_errors,
            "evaluation_time_seconds": evaluation_time,
            "avg_time_per_sample": evaluation_time / len(samples) if samples else 0,
            "metrics": metrics_dict,
            "sample_predictions": saved_predictions,
            "prediction_file": str(prediction_file) if prediction_file else None,
        }
    
    def _create_prompt_from_sample(self, sample: Dict, task_type: str) -> str:
        """Create appropriate prompt from dataset sample"""
        if task_type == "coding":
            prompt = sample.get("prompt", "")
            if not prompt and "question" in sample:
                prompt = f"Write Python code to solve: {sample['question']}"
            return prompt
        
        elif task_type == "function_calling":
            prompt = sample.get("prompt", "")
            functions = sample.get("functions", [])
            
            if functions:
                # Add function definitions to prompt
                func_desc = "Available functions:\n"
                for func in functions:
                    func_desc += f"- {func.get('name', 'unknown')}: {func.get('description', '')}\n"
                prompt = f"{func_desc}\nTask: {prompt}"
            
            return prompt
        
        elif task_type == "reasoning":
            question = sample.get("question", "")
            choices = sample.get("choices", [])
            
            if choices:
                # Multiple choice format
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                prompt = f"{question}\n\n{choices_text}\n\nAnswer:"
            else:
                prompt = f"Solve this problem step by step: {question}"
            
            return prompt
        
        elif task_type == "instruction_following":
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            
            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}\n\nOutput:"
            else:
                prompt = instruction
            
            return prompt
        
        elif task_type == "qa":
            question = sample.get("question", "")
            choices = sample.get("choices", [])
            
            if choices:
                # Multiple choice format
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                prompt = f"{question}\n\n{choices_text}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            return prompt
        
        else:
            # Generic format
            return sample.get("input", sample.get("prompt", sample.get("question", "")))
    
    
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
        
        # Create comparison report using ResultsManager
        comparison = self.results_manager.create_comparison_report(all_results)
        
        # Save comparison
        comparison_file = self.output_dir / "comparisons" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        comparison_file.parent.mkdir(exist_ok=True)
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Create summary report using ResultsManager
        self.results_manager.create_summary_report(comparison)
        
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
    
    def run_full_evaluation(self, models_to_test: Optional[List[str]] = None, preset: str = "balanced",
                           synthetic_only: bool = False, datasets_only: bool = False,
                           save_predictions: bool = False, prediction_count: Optional[int] = None,
                           dataset_filter: Optional[List[str]] = None,
                           sample_limit: Optional[int] = None):
        """Run complete evaluation suite with enhanced configuration support"""
        logger.info(f"🚀 Starting full LLM evaluation suite with preset: {preset}")
        
        # Set evaluation mode flags
        self._synthetic_only = synthetic_only
        self._datasets_only = datasets_only
        
        if synthetic_only:
            logger.info("📝 Running synthetic tests only")
        elif datasets_only:
            logger.info("📊 Running real datasets only")
        else:
            logger.info("🔄 Running both synthetic and real dataset evaluation")
        
        # Determine which models to test
        if models_to_test:
            configs_to_test = {k: v for k, v in MODEL_CONFIGS.items() if k in models_to_test}
        else:
            # Default: test high priority models
            configs_to_test = get_high_priority_models()
        
        logger.info(f"Will evaluate {len(configs_to_test)} models with {preset} preset: {list(configs_to_test.keys())}")
        
        # Apply preset to configurations
        enhanced_configs = {}
        for name, base_config in configs_to_test.items():
            if preset != "balanced":
                enhanced_configs[name] = base_config.create_preset_variant(preset)
            else:
                enhanced_configs[name] = base_config
        
        # Run individual evaluations
        for model_name, model_config in enhanced_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_name} (preset: {preset})")
            logger.info(f"{'='*60}")
            
            result = self.run_individual_evaluation(
                model_name, model_config, preset,
                save_predictions=save_predictions,
                prediction_count=prediction_count,
                dataset_filter=dataset_filter,
                sample_limit=sample_limit
            )
            # Use preset-aware key for results
            result_key = f"{model_name}_{preset}" if preset != "balanced" else model_name
            self.evaluation_results[result_key] = result
            
            # Brief pause between models to allow cleanup
            time.sleep(5)
        
        # Run comparison analysis
        self.run_comparison_analysis()
        
        logger.info("🎉 Full evaluation completed!")
        return self.evaluation_results
    
    def run_preset_comparison(self, model_name: str, model_config: ModelConfig) -> Dict[str, Any]:
        """Compare different presets for a single model"""
        logger.info(f"🔍 Running preset comparison for {model_name}")
        
        presets = ["balanced", "performance", "memory_optimized"]
        comparison_results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "preset_results": {}
        }
        
        for preset in presets:
            logger.info(f"Testing {model_name} with {preset} preset...")
            
            try:
                # Create preset variant
                if preset != "balanced":
                    preset_config = model_config.create_preset_variant(preset)
                else:
                    preset_config = model_config
                
                # Create model instance with specific preset
                model = self._create_model_instance(model_name, preset_config, preset)
                if not model:
                    comparison_results["preset_results"][preset] = {
                        "status": "failed",
                        "error": "Failed to create model instance"
                    }
                    continue
                
                # Run lightweight evaluation (no full model loading)
                preset_result = self._run_preset_evaluation(model, preset, preset_config)
                comparison_results["preset_results"][preset] = preset_result
                
                # Cleanup
                model.unload_model()
                time.sleep(2)  # Brief pause between presets
                
            except Exception as e:
                logger.warning(f"Preset {preset} failed for {model_name}: {e}")
                comparison_results["preset_results"][preset] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Save comparison results
        self._save_preset_comparison(model_name, comparison_results)
        
        return comparison_results
    
    def _run_preset_evaluation(self, model: BaseModelImplementation, preset: str, config: ModelConfig) -> Dict[str, Any]:
        """Run lightweight evaluation for preset comparison (without model loading)"""
        try:
            # Get configuration analysis
            memory_est = estimate_memory_usage(config)
            vllm_args = config.to_vllm_args()
            sampling_params = config.get_agent_sampling_params()
            
            # Get model info
            model_info = model.get_model_info()
            
            result = {
                "status": "completed",
                "preset": preset,
                "configuration": {
                    "gpu_memory_utilization": config.gpu_memory_utilization,
                    "max_num_seqs": config.max_num_seqs,
                    "max_model_len": config.max_model_len,
                    "enable_prefix_caching": config.enable_prefix_caching,
                    "use_v2_block_manager": config.use_v2_block_manager,
                    "evaluation_batch_size": config.evaluation_batch_size,
                    "quantization_method": config.quantization_method
                },
                "memory_estimation": memory_est,
                "sampling_parameters": sampling_params,
                "model_info": model_info,
                "optimization_score": self._calculate_optimization_score(config, memory_est)
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "failed",
                "preset": preset,
                "error": str(e)
            }
    
    def _calculate_optimization_score(self, config: ModelConfig, memory_est: Dict) -> Dict[str, float]:
        """Calculate optimization scores for different aspects"""
        # Performance score (higher is better)
        perf_score = (
            config.gpu_memory_utilization * 0.3 +  # Memory utilization
            (config.max_num_seqs / 128) * 0.3 +     # Batch efficiency
            (1.0 if config.enable_prefix_caching else 0.0) * 0.2 +  # Caching
            (1.0 if config.use_v2_block_manager else 0.0) * 0.2     # Block manager
        )
        
        # Memory efficiency score (lower memory usage is better, but normalized)
        memory_efficiency = max(0, 1.0 - (memory_est["h100_utilization"] / 0.5))  # Target <50% for efficiency
        
        # Agent suitability score
        agent_score = (
            (1.0 if config.agent_optimized else 0.0) * 0.4 +
            (min(config.max_function_calls_per_turn / 5, 1.0)) * 0.3 +
            (1.0 - config.agent_temperature) * 0.3  # Lower temp better for agents
        )
        
        return {
            "performance_score": round(perf_score, 3),
            "memory_efficiency": round(memory_efficiency, 3),
            "agent_suitability": round(agent_score, 3),
            "overall_score": round((perf_score + memory_efficiency + agent_score) / 3, 3)
        }
    
    def _save_preset_comparison(self, model_name: str, comparison_results: Dict):
        """Save preset comparison results"""
        safe_name = model_name.replace(" ", "_").replace("/", "_").lower()
        comparison_file = self.output_dir / "comparisons" / f"{safe_name}_preset_comparison.json"
        
        with open(comparison_file, "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        logger.info(f"Preset comparison saved to {comparison_file}")
        
        # Also create a summary report
        self._create_preset_comparison_report(model_name, comparison_results)
    
    def _create_preset_comparison_report(self, model_name: str, comparison_results: Dict):
        """Create human-readable preset comparison report"""
        safe_name = model_name.replace(" ", "_").replace("/", "_").lower()
        report_file = self.output_dir / "reports" / f"{safe_name}_preset_comparison.md"
        
        report_lines = [
            f"# Preset Comparison Report: {model_name}",
            f"Generated: {comparison_results['timestamp']}",
            "",
            "## Configuration Comparison",
            "",
            "| Preset | GPU Mem | Max Seqs | Est VRAM | H100 % | Batch | Caching |",
            "|--------|---------|----------|----------|---------|-------|---------|"
        ]
        
        preset_order = ["balanced", "performance", "memory_optimized"]
        best_scores = {"performance": 0, "memory": 1, "agent": 0, "overall": 0}
        best_presets = {"performance": "", "memory": "", "agent": "", "overall": ""}
        
        for preset in preset_order:
            if preset in comparison_results["preset_results"]:
                result = comparison_results["preset_results"][preset]
                if result["status"] == "completed":
                    config = result["configuration"]
                    memory = result["memory_estimation"]
                    scores = result["optimization_score"]
                    
                    # Track best scores
                    if scores["performance_score"] > best_scores["performance"]:
                        best_scores["performance"] = scores["performance_score"]
                        best_presets["performance"] = preset
                    if scores["memory_efficiency"] > best_scores["memory"]:
                        best_scores["memory"] = scores["memory_efficiency"]
                        best_presets["memory"] = preset
                    if scores["agent_suitability"] > best_scores["agent"]:
                        best_scores["agent"] = scores["agent_suitability"]
                        best_presets["agent"] = preset
                    if scores["overall_score"] > best_scores["overall"]:
                        best_scores["overall"] = scores["overall_score"]
                        best_presets["overall"] = preset
                    
                    report_lines.append(
                        f"| {preset} | {config['gpu_memory_utilization']:.2f} | "
                        f"{config['max_num_seqs']} | {memory['total_estimated_gb']:.1f}GB | "
                        f"{memory['h100_utilization']:.1%} | {config['evaluation_batch_size']} | "
                        f"{'Yes' if config['enable_prefix_caching'] else 'No'} |"
                    )
        
        # Add optimization scores section
        report_lines.extend([
            "",
            "## Optimization Scores",
            "",
            "| Preset | Performance | Memory Efficiency | Agent Suitability | Overall |",
            "|--------|-------------|-------------------|-------------------|---------|"
        ])
        
        for preset in preset_order:
            if preset in comparison_results["preset_results"]:
                result = comparison_results["preset_results"][preset]
                if result["status"] == "completed":
                    scores = result["optimization_score"]
                    report_lines.append(
                        f"| {preset} | {scores['performance_score']:.3f} | "
                        f"{scores['memory_efficiency']:.3f} | {scores['agent_suitability']:.3f} | "
                        f"{scores['overall_score']:.3f} |"
                    )
        
        # Add recommendations
        report_lines.extend([
            "",
            "## Recommendations",
            "",
            f"**Best for Performance**: {best_presets['performance']} (score: {best_scores['performance']:.3f})",
            f"**Best for Memory Efficiency**: {best_presets['memory']} (score: {best_scores['memory']:.3f})",
            f"**Best for Agent Tasks**: {best_presets['agent']} (score: {best_scores['agent']:.3f})",
            f"**Best Overall**: {best_presets['overall']} (score: {best_scores['overall']:.3f})",
            "",
            "## Usage Recommendations",
            "",
            "- **Performance Preset**: Use for maximum throughput when memory is abundant",
            "- **Memory Optimized Preset**: Use when running multiple models or limited VRAM",
            "- **Balanced Preset**: Good general-purpose choice for most scenarios",
        ])
        
        with open(report_file, "w") as f:
            f.write("\\n".join(report_lines))
        
        logger.info(f"Preset comparison report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation suite")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--cache-dir", help="Model cache directory")
    parser.add_argument("--data-cache-dir", help="Dataset cache directory")
    parser.add_argument("--priority-only", action="store_true", help="Test only high priority models")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test only")
    
    # Evaluation options
    parser.add_argument("--synthetic-only", action="store_true",
                       help="Use only synthetic test cases (no real datasets)")
    parser.add_argument("--datasets-only", action="store_true",
                       help="Use only real datasets (no synthetic tests)")
    
    # NEW: Enhanced configuration arguments
    parser.add_argument("--preset", default="balanced", 
                       choices=["balanced", "performance", "memory_optimized"],
                       help="Configuration preset to use (default: balanced)")
    parser.add_argument("--compare-presets", action="store_true",
                       help="Compare all presets for selected models")
    parser.add_argument("--memory-budget", type=int, default=80,
                       help="Available GPU memory in GB (default: 80 for H100)")
    
    # Prediction saving options
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save all model predictions to files for debugging")
    parser.add_argument("--save-prediction-count", type=int, default=None,
                       help="Number of predictions to save (default: all if --save-predictions)")
    parser.add_argument("--datasets", nargs="+", 
                       help="Specific datasets to evaluate (e.g., HumanEval MBPP)")
    parser.add_argument("--sample-limit", type=int, default=None,
                       help="Limit number of samples per dataset for quick testing")
    
    args = parser.parse_args()
    
    # Create runner
    runner = LLMEvaluationRunner(args.output_dir, args.cache_dir, args.data_cache_dir)
    
    if args.compare_presets:
        # Run preset comparison mode
        logger.info(f"🔍 Running preset comparison mode")
        models_to_compare = args.models if args.models else ["qwen3_8b"]
        
        for model_name in models_to_compare:
            if model_name in MODEL_CONFIGS:
                model_config = MODEL_CONFIGS[model_name]
                logger.info(f"\\nComparing presets for {model_name}...")
                comparison_result = runner.run_preset_comparison(model_name, model_config)
                
                # Print quick summary
                print(f"\\n📊 Quick Summary for {model_name}:")
                for preset, result in comparison_result["preset_results"].items():
                    if result["status"] == "completed":
                        scores = result["optimization_score"]
                        print(f"  {preset}: Overall Score {scores['overall_score']:.3f}")
            else:
                logger.warning(f"Model {model_name} not found in configurations")
                
    elif args.quick_test:
        # Quick test with specified preset
        logger.info(f"Running quick test with Qwen-3 8B using {args.preset} preset")
        qwen_config = MODEL_CONFIGS["qwen3_8b"]
        if args.preset != "balanced":
            qwen_config = qwen_config.create_preset_variant(args.preset)
        
        # Set evaluation mode for quick test
        runner._synthetic_only = args.synthetic_only
        runner._datasets_only = args.datasets_only
        
        runner.run_individual_evaluation(
            "qwen3_8b", qwen_config, args.preset,
            save_predictions=args.save_predictions,
            prediction_count=args.save_prediction_count,
            dataset_filter=args.datasets,
            sample_limit=args.sample_limit
        )
    else:
        # Full evaluation with preset
        models_to_test = None
        if args.models:
            models_to_test = args.models
        elif args.priority_only:
            models_to_test = list(get_high_priority_models().keys())
        
        runner.run_full_evaluation(
            models_to_test, args.preset, args.synthetic_only, args.datasets_only,
            save_predictions=args.save_predictions,
            prediction_count=args.save_prediction_count,
            dataset_filter=args.datasets,
            sample_limit=args.sample_limit
        )

def evaluate_model(model_name: str, config, dataset_name: str, 
                  samples: List[Dict[str, Any]], performance_monitor=None) -> Dict[str, Any]:
    """
    Simple evaluation function for comprehensive runner
    
    Args:
        model_name: Name of the model to evaluate
        config: Model configuration (ModelConfig object)
        dataset_name: Name of the dataset
        samples: List of dataset samples
        performance_monitor: Optional performance monitor instance
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Create a temporary runner instance
        runner = LLMEvaluationRunner(output_dir="temp_results")
        
        # Create model instance using the ModelConfig directly
        model = runner._create_model_instance(model_name, config, "custom")
        if model is None:
            raise ValueError(f"Failed to create model instance for {model_name}")
        
        # Load the model
        logger.info(f"Loading model {model_name}...")
        if not model.load_model():
            raise ValueError(f"Failed to load model {model_name}")
        
        # Prepare dataset in expected format
        dataset = {
            "name": dataset_name,
            "task_type": _get_task_type(dataset_name),
            "samples": samples
        }
        
        logger.info(f"Starting evaluation: {model_name} on {dataset_name} ({len(samples)} samples)")
        
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Run evaluation
        result = runner._evaluate_on_single_dataset(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            save_predictions=False,
            sample_limit=len(samples)
        )
        
        end_time = time.time()
        
        # Record tokens processed for performance monitoring
        if performance_monitor:
            # Estimate tokens processed (rough calculation)
            estimated_tokens = len(samples) * 200  # Average tokens per sample
            performance_monitor.record_tokens_processed(estimated_tokens)
            performance_monitor.record_request_timing(start_time, end_time)
        
        # Add timing information
        result["evaluation_time_seconds"] = end_time - start_time
        result["samples_processed"] = len(samples)
        result["model_name"] = model_name
        result["dataset_name"] = dataset_name
        
        logger.info(f"Evaluation completed: {model_name} on {dataset_name} in {end_time - start_time:.1f}s")
        
        # Clean up model resources
        try:
            if hasattr(model, 'unload_model'):
                model.unload_model()
            elif hasattr(model, 'cleanup'):
                model.cleanup()
        except Exception as e:
            logger.warning(f"Failed to cleanup model resources: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Evaluation failed for {model_name} on {dataset_name}: {str(e)}")
        raise e

def _get_task_type(dataset_name: str) -> str:
    """Get task type for dataset"""
    task_mapping = {
        'humaneval': 'coding',
        'mbpp': 'coding',
        'gsm8k': 'reasoning',
        'math': 'reasoning',
        'hellaswag': 'reasoning',
        'winogrande': 'reasoning',
        'mmlu': 'qa',
        'arc_challenge': 'qa',
        'mt_bench': 'instruction_following',
        'ifeval': 'instruction_following',
        'bfcl': 'function_calling',
        'toolllama': 'function_calling'
    }
    return task_mapping.get(dataset_name, 'general')

if __name__ == "__main__":
    main()