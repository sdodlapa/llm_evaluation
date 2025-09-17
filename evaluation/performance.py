#!/usr/bin/env python3
"""
Performance Benchmarking Module
Extracted from run_evaluation.py for better modularity
"""

import logging
from typing import Optional
from models.base_model import BaseModelImplementation, ModelPerformanceMetrics

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Handles performance benchmarking for LLM models"""
    
    def __init__(self):
        self.test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a Python function to calculate the factorial of a number.",
            "What are the main differences between supervised and unsupervised learning?",
            "Describe the process of photosynthesis step by step.",
            "How would you design a simple recommendation system?"
        ]
    
    def run_benchmark(self, model: BaseModelImplementation) -> Optional[ModelPerformanceMetrics]:
        """
        Run performance benchmark with standard test prompts
        
        Args:
            model: Model instance to benchmark
            
        Returns:
            ModelPerformanceMetrics object or None if failed
        """
        try:
            logger.info(f"Running performance benchmark for {model.model_name}")
            return model.benchmark_performance(self.test_prompts)
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return None
    
    def run_custom_benchmark(self, model: BaseModelImplementation, custom_prompts: list) -> Optional[ModelPerformanceMetrics]:
        """
        Run performance benchmark with custom prompts
        
        Args:
            model: Model instance to benchmark
            custom_prompts: List of custom prompts to test
            
        Returns:
            ModelPerformanceMetrics object or None if failed
        """
        try:
            logger.info(f"Running custom performance benchmark for {model.model_name}")
            return model.benchmark_performance(custom_prompts)
        except Exception as e:
            logger.error(f"Custom performance benchmark failed: {e}")
            return None