#!/usr/bin/env python3
"""
Dataset Evaluation Module
Extracted from run_evaluation.py for better modularity
"""

import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from models.base_model import BaseModelImplementation

logger = logging.getLogger(__name__)

class DatasetEvaluator:
    """Handles evaluation on real datasets"""
    
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
    
    def evaluate_datasets(self, model: BaseModelImplementation, preset: str = "balanced",
                         save_predictions: bool = False, prediction_count: Optional[int] = None,
                         dataset_filter: Optional[List[str]] = None,
                         sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run evaluation on real datasets
        
        Args:
            model: Model instance to evaluate
            preset: Configuration preset used
            save_predictions: Whether to save predictions
            prediction_count: Number of predictions to save
            dataset_filter: List of specific datasets to evaluate
            sample_limit: Limit samples per dataset
            
        Returns:
            Dictionary containing evaluation results
        """
        dataset_results = {
            "datasets_evaluated": [],
            "total_samples": 0,
            "evaluation_time": 0,
            "results_by_dataset": {},
            "summary_scores": {}
        }

        start_time = time.time()
        
        # Determine which datasets to evaluate
        available_datasets = self.dataset_manager.get_recommended_datasets()
        datasets_to_eval = dataset_filter if dataset_filter else available_datasets
        
        logger.info(f"Available datasets: {available_datasets}")
        logger.info(f"Will evaluate datasets: {datasets_to_eval}")
        
        total_samples = 0
        for dataset_name in datasets_to_eval:
            try:
                logger.info(f"Evaluating on dataset: {dataset_name}")
                result = self._evaluate_on_single_dataset(
                    model, dataset_name, preset, save_predictions, 
                    prediction_count, sample_limit
                )
                
                if result:
                    dataset_results["results_by_dataset"][dataset_name] = result
                    dataset_results["datasets_evaluated"].append(dataset_name)
                    total_samples += result.get("samples_evaluated", 0)
                    
            except Exception as e:
                logger.error(f"Failed to evaluate on {dataset_name}: {e}")
                dataset_results["results_by_dataset"][dataset_name] = {
                    "error": str(e),
                    "samples_evaluated": 0
                }
        
        dataset_results["total_samples"] = total_samples
        dataset_results["evaluation_time"] = time.time() - start_time
        
        # Calculate summary scores
        if dataset_results["results_by_dataset"]:
            dataset_results["summary_scores"] = self._calculate_summary_scores(
                dataset_results["results_by_dataset"]
            )
        
        return dataset_results
    
    def _evaluate_on_single_dataset(self, model: BaseModelImplementation, 
                                   dataset_name: str, preset: str = "balanced",
                                   save_predictions: bool = False,
                                   prediction_count: Optional[int] = None,
                                   sample_limit: Optional[int] = None) -> Optional[Dict]:
        """Evaluate model on a single dataset"""
        try:
            # Load dataset - try cached first, then download
            dataset = self.dataset_manager.load_cached_dataset(dataset_name)
            if not dataset:
                logger.info(f"Dataset {dataset_name} not cached, downloading...")
                dataset = self.dataset_manager.download_dataset(dataset_name)
            if not dataset:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Extract samples from dataset 
            samples = dataset.get("samples", [])
            if not samples:
                logger.error(f"No samples found in dataset: {dataset_name}")
                return {"error": "No samples in dataset", "samples_evaluated": 0}
            
            # Apply sample limit if specified
            if sample_limit:
                max_samples = sample_limit
            else:
                max_samples = 100  # Default limit for efficiency
                
            if len(samples) > max_samples:
                samples = samples[:max_samples]
                logger.info(f"Limited {dataset_name} to {max_samples} samples")
            
            # Determine task type
            task_type = dataset.get("task_type", "unknown")
            
            # Generate predictions
            predictions = []
            for i, sample in enumerate(samples):
                try:
                    prompt = self._create_prompt_from_sample(sample, task_type)
                    response = model.generate_response(prompt)
                    
                    predictions.append({
                        "sample_id": i,
                        "prompt": prompt,
                        "prediction": response,
                        "expected": sample.get("expected_output", ""),
                        "metadata": sample.get("metadata", {})
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to generate prediction for sample {i}: {e}")
                    predictions.append({
                        "sample_id": i,
                        "error": str(e),
                        "prediction": "",
                        "expected": sample.get("expected_output", "")
                    })
            
            # Save predictions if requested
            if save_predictions and predictions:
                self._save_predictions(dataset_name, model.model_name, preset, 
                                     predictions, prediction_count)
            
            # Evaluate predictions using metrics
            try:
                from metrics import evaluate_dataset_predictions
                evaluation_result = evaluate_dataset_predictions(
                    predictions, dataset_name, task_type
                )
                
                return {
                    "samples_evaluated": len(predictions),
                    "task_type": task_type,
                    "metrics": evaluation_result,
                    "predictions_saved": save_predictions
                }
                
            except Exception as e:
                logger.error(f"Failed to evaluate predictions for {dataset_name}: {e}")
                return {
                    "samples_evaluated": len(predictions),
                    "task_type": task_type,
                    "error": f"Evaluation failed: {e}",
                    "predictions_saved": save_predictions
                }
                
        except Exception as e:
            logger.error(f"Dataset evaluation failed for {dataset_name}: {e}")
            return None
    
    def _create_prompt_from_sample(self, sample: Dict, task_type: str) -> str:
        """Create appropriate prompt from dataset sample"""
        if task_type == "coding":
            if "problem" in sample:
                return f"Complete the following Python function:\n\n{sample['problem']}\n\nYour solution:"
            elif "prompt" in sample:
                return sample["prompt"]
            else:
                return str(sample.get("input", ""))
        
        elif task_type == "reasoning":
            if "problem" in sample:
                return f"Solve this step by step:\n\n{sample['problem']}\n\nSolution:"
            elif "question" in sample:
                return f"Question: {sample['question']}\nAnswer:"
            else:
                return str(sample.get("input", ""))
        
        elif task_type == "qa":
            if "question" in sample:
                context = sample.get("context", "")
                if context:
                    return f"Context: {context}\n\nQuestion: {sample['question']}\nAnswer:"
                else:
                    return f"Question: {sample['question']}\nAnswer:"
            else:
                return str(sample.get("input", ""))
        
        else:
            # Generic prompt creation
            if "prompt" in sample:
                return sample["prompt"]
            elif "input" in sample:
                return str(sample["input"])
            else:
                return str(sample)
    
    def _calculate_summary_scores(self, dataset_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate summary scores across all datasets"""
        summary = {
            "average_score": 0.0,
            "total_samples": 0,
            "successful_datasets": 0,
            "failed_datasets": 0
        }
        
        scores = []
        total_samples = 0
        
        for dataset_name, result in dataset_results.items():
            if "error" in result:
                summary["failed_datasets"] += 1
                continue
                
            summary["successful_datasets"] += 1
            total_samples += result.get("samples_evaluated", 0)
            
            # Extract score from metrics
            metrics = result.get("metrics", {})
            if isinstance(metrics, dict) and "overall_score" in metrics:
                scores.append(metrics["overall_score"])
            elif isinstance(metrics, dict) and "accuracy" in metrics:
                scores.append(metrics["accuracy"])
        
        summary["total_samples"] = total_samples
        if scores:
            summary["average_score"] = sum(scores) / len(scores)
        
        return summary
    
    def _save_predictions(self, dataset_name: str, model_name: str, preset: str,
                         predictions: List[Dict], prediction_count: Optional[int] = None):
        """Save predictions to file"""
        try:
            # Limit predictions if requested
            if prediction_count:
                predictions = predictions[:prediction_count]
            
            # Create output directory
            output_dir = Path("test_results") / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            filename = f"{model_name}_{preset}_{dataset_name}_predictions.json"
            filepath = output_dir / filename
            
            import json
            with open(filepath, 'w') as f:
                json.dump({
                    "model_name": model_name,
                    "preset": preset,
                    "dataset_name": dataset_name,
                    "predictions": predictions,
                    "total_predictions": len(predictions)
                }, f, indent=2)
            
            logger.info(f"Saved {len(predictions)} predictions to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")