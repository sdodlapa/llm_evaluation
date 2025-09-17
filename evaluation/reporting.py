"""
Results Management and Reporting Module

Handles results aggregation, scoring calculations, report generation, and file I/O
for the LLM evaluation pipeline.

Extracted from monolithic run_evaluation.py to improve modularity and maintainability.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from configs.model_configs import ModelConfig

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages results aggregation, scoring, and report generation"""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize the results manager with output directory"""
        self.output_dir = output_dir
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all necessary output directories exist"""
        directories = [
            self.output_dir,
            f"{self.output_dir}/performance",
            f"{self.output_dir}/comparisons",
            f"{self.output_dir}/reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def calculate_summary_scores(self, dataset_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate summary scores across all evaluated datasets"""
        if not dataset_results or not dataset_results.get("results_by_dataset"):
            return {"overall_score": 0.0, "average_score": 0.0}
        
        # Aggregate scores by task type
        scores_by_type = {}
        all_scores = []
        
        for dataset_name, result in dataset_results["results_by_dataset"].items():
            if not result or not isinstance(result, dict):
                continue
                
            # Extract score from result
            dataset_scores = result.get("scores", {})
            if isinstance(dataset_scores, dict):
                avg_score = dataset_scores.get("average", 0.0)
            else:
                avg_score = float(dataset_scores) if dataset_scores else 0.0
            
            # Track by task type
            task_type = result.get("task_type", "unknown")
            if task_type not in scores_by_type:
                scores_by_type[task_type] = []
            
            if dataset_scores and task_type in scores_by_type:
                scores_by_type[task_type].append(avg_score)
            
            all_scores.append(avg_score)
        
        # Calculate summary metrics
        summary = {}
        if all_scores:
            summary["overall_score"] = sum(all_scores) / len(all_scores)
            summary["average_score"] = summary["overall_score"]
        
        # Add task-specific averages
        for task_type, scores in scores_by_type.items():
            if scores:
                summary[f"{task_type}_average"] = sum(scores) / len(scores)
        
        return summary
    
    def save_individual_results(self, model_name: str, results: Dict, preset: str = "balanced"):
        """Save individual model results to JSON file"""
        # Clean model name for filename
        clean_name = model_name.replace("/", "_").replace(" ", "_")
        filename = f"{clean_name}_{preset}_results.json"
        filepath = os.path.join(self.output_dir, "performance", filename)
        
        # Add metadata
        results["metadata"] = {
            "model_name": model_name,
            "preset": preset,
            "timestamp": datetime.now().isoformat(),
            "evaluation_version": "2.0"
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results to {filepath}: {e}")
    
    def save_individual_results_legacy(self, model_name: str, results: Dict):
        """Save results in legacy format for backward compatibility"""
        # Clean model name for filename
        clean_name = model_name.replace("/", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        filename = f"{clean_name}_evaluation_results_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Legacy results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save legacy results to {filepath}: {e}")
        
        # Save simplified summary
        summary_filename = f"{clean_name}_summary_{timestamp}.json"
        summary_filepath = os.path.join(self.output_dir, summary_filename)
        
        summary = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "performance": results.get("performance", {}),
            "summary_scores": results.get("summary_scores", {}),
            "datasets_evaluated": results.get("datasets_evaluated", [])
        }
        
        try:
            with open(summary_filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save summary to {summary_filepath}: {e}")
    
    def create_comparison_report(self, results: List[Dict]) -> Dict:
        """Create comparison report across multiple model results"""
        if not results:
            logger.warning("No results provided for comparison")
            return {"models": [], "comparison": {}, "summary": {}}
        
        comparison = {
            "models": [],
            "performance_comparison": {},
            "dataset_comparison": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract model information
        for result in results:
            model_info = {
                "name": result.get("model_name", "Unknown"),
                "preset": result.get("preset", "balanced"),
                "performance": result.get("performance", {}),
                "summary_scores": result.get("summary_scores", {})
            }
            comparison["models"].append(model_info)
        
        # Performance comparison
        perf_metrics = {}
        for result in results:
            model_name = result.get("model_name", "Unknown")
            performance = result.get("performance", {})
            
            if isinstance(performance, dict):
                for metric, value in performance.items():
                    if metric not in perf_metrics:
                        perf_metrics[metric] = {}
                    perf_metrics[metric][model_name] = value
        
        comparison["performance_comparison"] = perf_metrics
        
        # Dataset comparison
        dataset_metrics = {}
        for result in results:
            model_name = result.get("model_name", "Unknown")
            dataset_results = result.get("dataset_results", {}).get("results_by_dataset", {})
            
            for dataset_name, dataset_result in dataset_results.items():
                if dataset_name not in dataset_metrics:
                    dataset_metrics[dataset_name] = {}
                
                scores = dataset_result.get("scores", {})
                if isinstance(scores, dict):
                    dataset_metrics[dataset_name][model_name] = scores.get("average", 0.0)
                else:
                    dataset_metrics[dataset_name][model_name] = float(scores) if scores else 0.0
        
        comparison["dataset_comparison"] = dataset_metrics
        
        return comparison
    
    def create_summary_report(self, comparison: Dict):
        """Create and save markdown summary report"""
        if not comparison or not comparison.get("models"):
            logger.warning("No comparison data available for summary report")
            return
        
        report_lines = [
            "# LLM Evaluation Summary Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Models Evaluated",
            ""
        ]
        
        # Models table
        for i, model in enumerate(comparison["models"], 1):
            report_lines.extend([
                f"### {i}. {model['name']}",
                f"- **Preset**: {model.get('preset', 'N/A')}",
                f"- **Performance**: {model.get('performance', {})}",
                f"- **Summary Scores**: {model.get('summary_scores', {})}",
                ""
            ])
        
        # Performance comparison
        perf_comparison = comparison.get("performance_comparison", {})
        if perf_comparison:
            report_lines.extend([
                "## Performance Comparison",
                ""
            ])
            
            for metric, values in perf_comparison.items():
                report_lines.append(f"### {metric}")
                for model, value in values.items():
                    report_lines.append(f"- **{model}**: {value}")
                report_lines.append("")
        
        # Dataset comparison
        dataset_comparison = comparison.get("dataset_comparison", {})
        if dataset_comparison:
            report_lines.extend([
                "## Dataset Performance Comparison",
                ""
            ])
            
            for dataset, scores in dataset_comparison.items():
                report_lines.append(f"### {dataset}")
                for model, score in scores.items():
                    report_lines.append(f"- **{model}**: {score:.3f}")
                report_lines.append("")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evaluation_summary_{timestamp}.md"
        report_filepath = os.path.join(self.output_dir, "reports", report_filename)
        
        try:
            with open(report_filepath, 'w') as f:
                f.write('\n'.join(report_lines))
            logger.info(f"Summary report saved to {report_filepath}")
        except Exception as e:
            logger.error(f"Failed to save summary report: {e}")
    
    def calculate_optimization_score(self, config: ModelConfig, memory_est: Dict) -> Dict[str, float]:
        """Calculate optimization scores for model configuration"""
        scores = {
            "efficiency_score": 0.0,
            "memory_score": 0.0,
            "speed_score": 0.0,
            "overall_optimization": 0.0
        }
        
        try:
            # Memory efficiency (lower is better, normalized)
            total_memory = memory_est.get("total_memory_gb", 16.0)
            memory_score = max(0.0, min(1.0, (32.0 - total_memory) / 32.0))
            scores["memory_score"] = memory_score
            
            # Speed efficiency (based on quantization and precision)
            speed_score = 0.5  # baseline
            if hasattr(config, 'quantization') and config.quantization:
                speed_score += 0.3
            if hasattr(config, 'precision') and config.precision == "fp16":
                speed_score += 0.2
            scores["speed_score"] = min(1.0, speed_score)
            
            # Overall efficiency
            scores["efficiency_score"] = (memory_score + scores["speed_score"]) / 2
            scores["overall_optimization"] = scores["efficiency_score"]
            
        except Exception as e:
            logger.warning(f"Error calculating optimization scores: {e}")
        
        return scores
    
    def save_preset_comparison(self, model_name: str, comparison_results: Dict):
        """Save preset comparison results"""
        clean_name = model_name.replace("/", "_").replace(" ", "_")
        filename = f"{clean_name}_preset_comparison.json"
        filepath = os.path.join(self.output_dir, "comparisons", filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(comparison_results, f, indent=2, default=str)
            logger.info(f"Preset comparison saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save preset comparison: {e}")
    
    def create_preset_comparison_report(self, model_name: str, comparison_results: Dict):
        """Create markdown report for preset comparison"""
        if not comparison_results:
            logger.warning("No comparison results to report")
            return
        
        clean_name = model_name.replace("/", "_").replace(" ", "_")
        report_filename = f"{clean_name}_preset_comparison.md"
        report_filepath = os.path.join(self.output_dir, "reports", report_filename)
        
        report_lines = [
            f"# Preset Comparison Report: {model_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance by Preset",
            ""
        ]
        
        # Add preset performance data
        for preset, results in comparison_results.items():
            if preset == "metadata":
                continue
                
            report_lines.extend([
                f"### {preset.title()} Preset",
                ""
            ])
            
            # Performance metrics
            performance = results.get("performance", {})
            if performance:
                report_lines.append("#### Performance Metrics")
                for metric, value in performance.items():
                    report_lines.append(f"- **{metric}**: {value}")
                report_lines.append("")
            
            # Dataset results
            dataset_results = results.get("dataset_results", {})
            if dataset_results:
                report_lines.append("#### Dataset Results")
                for dataset, result in dataset_results.items():
                    if isinstance(result, dict):
                        score = result.get("average", "N/A")
                    else:
                        score = result
                    report_lines.append(f"- **{dataset}**: {score}")
                report_lines.append("")
        
        try:
            with open(report_filepath, 'w') as f:
                f.write('\n'.join(report_lines))
            logger.info(f"Preset comparison report saved to {report_filepath}")
        except Exception as e:
            logger.error(f"Failed to save preset comparison report: {e}")