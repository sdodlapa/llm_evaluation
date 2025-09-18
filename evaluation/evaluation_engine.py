"""
Evaluation Engine - Core evaluation logic
Handles model creation, dataset evaluation, and prompt processing
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from models.base_model import BaseModelImplementation, ModelPerformanceMetrics, AgentEvaluationResult
from models.registry import get_model_registry
from configs.model_configs import ModelConfig, estimate_memory_usage

try:
    from .dataset_manager import EnhancedDatasetManager
    from .metrics import EvaluationMetrics, evaluate_dataset_predictions
    from .performance_monitor import LivePerformanceMonitor
except ImportError:
    # Handle when running as script
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from evaluation.dataset_manager import EnhancedDatasetManager
    from evaluation.metrics import EvaluationMetrics, evaluate_dataset_predictions
    from evaluation.performance_monitor import LivePerformanceMonitor

logger = logging.getLogger(__name__)

class EvaluationEngine:
    """Core evaluation engine for processing models and datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None, data_cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.data_cache_dir = data_cache_dir
        self.dataset_manager = EnhancedDatasetManager(data_cache_dir or "evaluation_data")
        self.performance_monitor = LivePerformanceMonitor()
        self.model_registry = get_model_registry()
        
        # Load test suite for evaluation
        self.test_suite = self._load_test_suite()
    
    def _load_test_suite(self) -> Dict[str, List]:
        """Load comprehensive test suite for evaluation"""
        return {
            "coding": [
                {"user": "Write a Python function to calculate factorial", "expected_themes": ["def", "factorial", "recursion or loop"]},
                {"user": "Create a class for a simple calculator", "expected_themes": ["class", "Calculator", "methods"]},
                {"user": "Write a function to find prime numbers", "expected_themes": ["def", "prime", "mathematical logic"]},
                {"user": "Implement bubble sort algorithm", "expected_themes": ["def", "sort", "nested loops", "swap"]},
                {"user": "Create a binary tree traversal function", "expected_themes": ["def", "traverse", "tree", "recursion"]}
            ],
            "reasoning": [
                {"user": "If all roses are flowers and some flowers are red, can we conclude that some roses are red?", "expected_themes": ["logical reasoning", "cannot conclude", "insufficient information"]},
                {"user": "A farmer has chickens and cows. If there are 20 heads and 56 legs total, how many of each animal?", "expected_themes": ["system of equations", "8 cows", "12 chickens", "mathematical reasoning"]},
                {"user": "What comes next in the sequence: 2, 6, 12, 20, 30, ?", "expected_themes": ["pattern recognition", "42", "n(n+1)", "quadratic sequence"]},
                {"user": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?", "expected_themes": ["5 minutes", "parallel processing", "rate analysis"]},
                {"user": "Three friends split a bill. Alice pays twice what Bob pays, and Carol pays $5 more than Bob. If the total is $65, how much does each pay?", "expected_themes": ["algebraic reasoning", "Bob $15", "Alice $30", "Carol $20"]}
            ],
            "agent_capabilities": [
                {"user": "Help me plan a 3-day itinerary for visiting Tokyo", "expected_themes": ["structured planning", "day-by-day breakdown", "specific locations", "practical advice"]},
                {"user": "Analyze this data and provide insights: [1,5,3,8,2,9,4,6,7]", "expected_themes": ["data analysis", "statistics", "patterns", "insights"]},
                {"user": "Debug this code and explain the issue: def calc(x): x * 2", "expected_themes": ["missing return", "syntax issue", "debugging explanation"]},
                {"user": "Compare and contrast renewable vs fossil fuels", "expected_themes": ["balanced comparison", "environmental impact", "economic factors", "structured analysis"]},
                {"user": "Create a study plan for learning machine learning in 3 months", "expected_themes": ["structured timeline", "progressive learning", "specific resources", "milestones"]}
            ],
            "function_calling": [
                {"user": "Calculate the area of a circle with radius 5", "expected_themes": ["math calculation", "π * r²", "78.54"]},
                {"user": "What's the weather like in New York?", "expected_themes": ["function call", "weather API", "location-specific"]},
                {"user": "Send an email to john@example.com about the meeting", "expected_themes": ["email function", "recipient", "meeting context"]},
                {"user": "Search for recent papers on transformers in AI", "expected_themes": ["search function", "academic papers", "transformers", "AI"]},
                {"user": "Set a reminder for tomorrow at 2 PM", "expected_themes": ["calendar function", "reminder", "time specification"]}
            ],
            "code_review": [
                {"user": "Review this Python code for best practices: def process_data(data): for i in data: print(i)", "expected_themes": ["code review", "best practices", "improvements", "pythonic"]},
                {"user": "Check this function for potential bugs: def divide(a, b): return a / b", "expected_themes": ["division by zero", "error handling", "type checking"]},
                {"user": "Optimize this algorithm for better performance: [nested loop example]", "expected_themes": ["performance optimization", "algorithm improvement", "complexity analysis"]},
                {"user": "Suggest improvements for code readability: [messy code example]", "expected_themes": ["readability", "naming conventions", "structure", "comments"]},
                {"user": "Here's the code: def calc(x): x * 2", "expected_themes": ["missing return", "syntax"]}
            ]
        }
    
    def create_model_instance(self, model_name: str, model_config: ModelConfig, preset: str = "balanced") -> Optional[BaseModelImplementation]:
        """Create model instance using the registry"""
        try:
            logger.info(f"Creating model instance: {model_name} with preset {preset}")
            
            # Apply preset to configuration
            model_config.apply_preset(preset)
            
            # Create model using registry
            model = self.model_registry.create_model(model_name, preset, self.cache_dir)
            
            if model is None:
                logger.error(f"Failed to create model {model_name}")
                return None
            
            logger.info(f"Successfully created model {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            return None
    
    def evaluate_on_single_dataset(self, model: BaseModelImplementation, 
                                 dataset_name: str, 
                                 sample_limit: Optional[int] = None,
                                 continue_on_error: bool = False) -> Dict[str, Any]:
        """Evaluate model on a single dataset"""
        logger.info(f"Evaluating {model.model_name} on {dataset_name}")
        
        try:
            # Load dataset samples
            samples = self.dataset_manager.load_dataset_samples(dataset_name, sample_limit)
            if not samples:
                logger.warning(f"No samples loaded for dataset {dataset_name}")
                return {"error": "No samples loaded", "dataset": dataset_name}
            
            # Get dataset info for task type
            dataset_info = self.dataset_manager.get_dataset_info(dataset_name)
            task_type = dataset_info.get('task_type', 'general') if dataset_info else 'general'
            
            # Evaluate samples
            predictions = []
            ground_truth = []
            evaluation_details = []
            
            for i, sample in enumerate(samples):
                try:
                    # Create prompt from sample
                    prompt = self._create_prompt_from_sample(sample, task_type)
                    
                    # Get model prediction
                    start_time = time.time()
                    if hasattr(model, 'generate_response'):
                        response = model.generate_response(prompt)
                    else:
                        # Fallback for older model interface
                        response = model.generate(prompt)
                    
                    execution_time = time.time() - start_time
                    
                    # Extract expected answer
                    expected = self._extract_expected_answer(sample, task_type)
                    
                    predictions.append(response)
                    ground_truth.append(expected)
                    
                    evaluation_details.append({
                        'sample_id': i,
                        'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                        'prediction': response,
                        'expected': expected,
                        'execution_time': execution_time
                    })
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(samples)} samples for {dataset_name}")
                
                except Exception as e:
                    logger.error(f"Error processing sample {i} in {dataset_name}: {e}")
                    if not continue_on_error:
                        raise
                    # Continue with placeholder values
                    predictions.append("ERROR")
                    ground_truth.append("UNKNOWN")
                    evaluation_details.append({
                        'sample_id': i,
                        'error': str(e),
                        'execution_time': 0
                    })
            
            # Evaluate predictions
            evaluation_metrics = evaluate_dataset_predictions(
                predictions, ground_truth, dataset_name, task_type
            )
            
            result = {
                'dataset': dataset_name,
                'task_type': task_type,
                'samples_processed': len(samples),
                'evaluation_metrics': evaluation_metrics,
                'execution_details': evaluation_details[:5],  # Sample of details
                'average_execution_time': sum(d.get('execution_time', 0) for d in evaluation_details) / len(evaluation_details) if evaluation_details else 0
            }
            
            logger.info(f"Completed evaluation on {dataset_name}: {evaluation_metrics}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate on dataset {dataset_name}: {e}")
            return {
                'dataset': dataset_name,
                'error': str(e),
                'samples_processed': 0
            }
    
    def _create_prompt_from_sample(self, sample: Dict, task_type: str) -> str:
        """Create prompt from dataset sample based on task type"""
        if task_type == "coding":
            if 'prompt' in sample:
                return f"Write a Python function to solve this problem:\n\n{sample['prompt']}\n\nProvide only the function definition:"
            elif 'instruction' in sample:
                return f"Complete this coding task:\n\n{sample['instruction']}\n\nProvide clean, well-commented code:"
            elif 'problem' in sample:
                return f"Solve this programming problem:\n\n{sample['problem']}\n\nCode:"
        
        elif task_type == "reasoning" or task_type == "mathematics":
            if 'question' in sample:
                return f"Solve this step by step:\n\n{sample['question']}\n\nExplain your reasoning:"
            elif 'problem' in sample:
                return f"Problem: {sample['problem']}\n\nSolution:"
        
        elif task_type == "qa":
            if 'question' in sample and 'context' in sample:
                return f"Context: {sample['context']}\n\nQuestion: {sample['question']}\n\nAnswer:"
            elif 'question' in sample:
                return f"Question: {sample['question']}\n\nAnswer:"
        
        elif task_type == "biomedical_qa" or task_type == "clinical_qa":
            if 'question' in sample and 'context' in sample:
                return f"Medical Context: {sample['context']}\n\nQuestion: {sample['question']}\n\nAnswer:"
            elif 'question' in sample:
                return f"Medical Question: {sample['question']}\n\nAnswer:"
        
        elif task_type == "scientific_summarization":
            if 'abstract' in sample and 'title' in sample:
                return f"Title: {sample['title']}\n\nAbstract: {sample['abstract']}\n\nProvide a concise summary:"
            elif 'text' in sample:
                return f"Summarize this scientific text:\n\n{sample['text']}\n\nSummary:"
        
        elif task_type == "document_vqa":
            if 'question' in sample and 'document_text' in sample:
                return f"Document: {sample['document_text']}\n\nQuestion: {sample['question']}\n\nAnswer:"
        
        # Default handling
        if 'prompt' in sample:
            return sample['prompt']
        elif 'instruction' in sample:
            return sample['instruction']
        elif 'question' in sample:
            return sample['question']
        elif 'text' in sample:
            return sample['text']
        else:
            # Try to construct from available fields
            return str(sample)
    
    def _extract_expected_answer(self, sample: Dict, task_type: str) -> str:
        """Extract expected answer from sample"""
        # Try common answer fields
        for field in ['answer', 'expected', 'target', 'solution', 'canonical_solution']:
            if field in sample:
                return str(sample[field])
        
        # Task-specific extraction
        if task_type == "coding":
            if 'test' in sample and isinstance(sample['test'], str):
                return sample['test']
        
        # Default to empty string
        return ""
    
    def run_performance_benchmark(self, model: BaseModelImplementation, preset: str, config: ModelConfig) -> Dict[str, Any]:
        """Run performance benchmark on model"""
        logger.info(f"Running performance benchmark for {model.model_name}")
        
        try:
            # Calculate memory estimation
            memory_est = estimate_memory_usage(config)
            
            # Run test suite evaluation
            test_results = {}
            total_tests = 0
            total_time = 0
            
            for category, tests in self.test_suite.items():
                category_results = []
                category_time = 0
                
                for test in tests[:3]:  # Limit to 3 tests per category for performance
                    start_time = time.time()
                    try:
                        if hasattr(model, 'generate_response'):
                            response = model.generate_response(test['user'])
                        else:
                            response = model.generate(test['user'])
                        
                        execution_time = time.time() - start_time
                        category_time += execution_time
                        total_time += execution_time
                        total_tests += 1
                        
                        category_results.append({
                            'test': test['user'][:100] + "...",
                            'response': response[:200] + "..." if len(response) > 200 else response,
                            'execution_time': execution_time,
                            'expected_themes': test.get('expected_themes', [])
                        })
                    
                    except Exception as e:
                        logger.error(f"Error in performance test: {e}")
                        category_results.append({
                            'test': test['user'][:100] + "...",
                            'error': str(e),
                            'execution_time': 0
                        })
                
                test_results[category] = {
                    'results': category_results,
                    'category_time': category_time,
                    'avg_time_per_test': category_time / len(category_results) if category_results else 0
                }
            
            # Calculate performance metrics
            performance_metrics = {
                'total_tests': total_tests,
                'total_execution_time': total_time,
                'average_response_time': total_time / total_tests if total_tests > 0 else 0,
                'memory_estimation': memory_est,
                'preset': preset,
                'test_results': test_results
            }
            
            logger.info(f"Performance benchmark completed: avg response time {performance_metrics['average_response_time']:.3f}s")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return {
                'error': str(e),
                'preset': preset,
                'total_tests': 0,
                'total_execution_time': 0
            }