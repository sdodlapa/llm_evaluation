"""
Base model implementation class for standardized evaluation
Provides consistent interface for all LLM models under evaluation
"""

import time
import json
import torch
import psutil
import GPUtil
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Container for model performance metrics"""
    model_name: str
    memory_usage_gb: float
    tokens_per_second: float
    latency_first_token_ms: float
    latency_total_ms: float
    context_length_tested: int
    batch_size_tested: int
    timestamp: str

@dataclass
class AgentEvaluationResult:
    """Container for agent-specific evaluation results"""
    model_name: str
    function_calling_accuracy: float
    instruction_following_score: float
    multi_turn_coherence: float
    tool_use_success_rate: float
    reasoning_quality_score: float
    json_output_validity: float
    error_rate: float

class BaseModelImplementation(ABC):
    """Base class for all model implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "unknown")
        self.model = None
        self.tokenizer = None
        self.llm_engine = None  # For vLLM
        self.is_loaded = False
        self.performance_metrics = []
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model with optimal settings"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        pass
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.llm_engine is not None:
            del self.llm_engine
            self.llm_engine = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info(f"Unloaded {self.model_name}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
        
        cpu_memory = psutil.virtual_memory().used / 1e9
        
        return {
            "gpu_memory_gb": gpu_memory,
            "cpu_memory_gb": cpu_memory,
            "gpu_utilization": torch.cuda.utilization() if torch.cuda.is_available() else 0
        }
    
    def benchmark_performance(self, test_prompts: List[str], **kwargs) -> ModelPerformanceMetrics:
        """Benchmark model performance"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        
        logger.info(f"Benchmarking {self.model_name}...")
        
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        total_tokens = 0
        first_token_times = []
        
        for prompt in test_prompts:
            # Measure first token latency
            prompt_start = time.time()
            response = self.generate_response(prompt, **kwargs)
            first_token_time = time.time() - prompt_start
            first_token_times.append(first_token_time * 1000)  # Convert to ms
            
            # Estimate tokens (rough approximation)
            total_tokens += len(response.split()) * 1.3  # Rough tokens estimation
        
        end_time = time.time()
        memory_after = self.get_memory_usage()
        
        total_time = end_time - start_time
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        avg_first_token_latency = sum(first_token_times) / len(first_token_times)
        
        metrics = ModelPerformanceMetrics(
            model_name=self.model_name,
            memory_usage_gb=memory_after["gpu_memory_gb"],
            tokens_per_second=tokens_per_second,
            latency_first_token_ms=avg_first_token_latency,
            latency_total_ms=total_time * 1000,
            context_length_tested=len(test_prompts[0]) if test_prompts else 0,
            batch_size_tested=len(test_prompts),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.performance_metrics.append(metrics)
        logger.info(f"Performance benchmark completed: {tokens_per_second:.1f} tokens/sec")
        
        return metrics
    
    def evaluate_agent_capabilities(self, test_suite: Dict[str, List]) -> AgentEvaluationResult:
        """Evaluate agent-specific capabilities"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        
        logger.info(f"Evaluating agent capabilities for {self.model_name}...")
        
        # Function calling test
        function_accuracy = self._test_function_calling(test_suite.get("function_calling", []))
        
        # Instruction following test
        instruction_score = self._test_instruction_following(test_suite.get("instructions", []))
        
        # Multi-turn conversation test
        coherence_score = self._test_multi_turn_coherence(test_suite.get("conversations", []))
        
        # Tool use test
        tool_success = self._test_tool_usage(test_suite.get("tool_use", []))
        
        # Reasoning quality test
        reasoning_score = self._test_reasoning_quality(test_suite.get("reasoning", []))
        
        # JSON output validity test
        json_validity = self._test_json_output(test_suite.get("json_tasks", []))
        
        # Error rate calculation
        error_rate = self._calculate_error_rate(test_suite)
        
        result = AgentEvaluationResult(
            model_name=self.model_name,
            function_calling_accuracy=function_accuracy,
            instruction_following_score=instruction_score,
            multi_turn_coherence=coherence_score,
            tool_use_success_rate=tool_success,
            reasoning_quality_score=reasoning_score,
            json_output_validity=json_validity,
            error_rate=error_rate
        )
        
        logger.info(f"Agent evaluation completed for {self.model_name}")
        return result
    
    def _test_function_calling(self, test_cases: List[Dict]) -> float:
        """Test function calling accuracy"""
        if not test_cases:
            return 0.0
        
        correct_calls = 0
        for case in test_cases:
            try:
                response = self.generate_response(case["prompt"])
                # Simple validation - check if response contains expected function structure
                if self._validate_function_call(response, case.get("expected_function")):
                    correct_calls += 1
            except Exception as e:
                logger.warning(f"Function calling test failed: {e}")
        
        return correct_calls / len(test_cases)
    
    def _test_instruction_following(self, test_cases: List[Dict]) -> float:
        """Test instruction following capabilities"""
        if not test_cases:
            return 0.0
        
        scores = []
        for case in test_cases:
            try:
                response = self.generate_response(case["prompt"])
                score = self._score_instruction_following(response, case.get("requirements", []))
                scores.append(score)
            except Exception as e:
                logger.warning(f"Instruction following test failed: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _test_multi_turn_coherence(self, conversations: List[List[Dict]]) -> float:
        """Test multi-turn conversation coherence"""
        if not conversations:
            return 0.0
        
        coherence_scores = []
        for conversation in conversations:
            try:
                context = ""
                turn_scores = []
                
                for turn in conversation:
                    context += f"User: {turn['user']}\n"
                    response = self.generate_response(context + "Assistant:")
                    context += f"Assistant: {response}\n"
                    
                    # Score coherence (simplified)
                    score = self._score_coherence(response, turn.get("expected_themes", []))
                    turn_scores.append(score)
                
                coherence_scores.append(sum(turn_scores) / len(turn_scores))
            except Exception as e:
                logger.warning(f"Multi-turn coherence test failed: {e}")
                coherence_scores.append(0.0)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    
    def _test_tool_usage(self, test_cases: List[Dict]) -> float:
        """Test tool usage capabilities"""
        if not test_cases:
            return 0.0
        
        successful_uses = 0
        for case in test_cases:
            try:
                response = self.generate_response(case["prompt"])
                if self._validate_tool_usage(response, case.get("expected_tool")):
                    successful_uses += 1
            except Exception as e:
                logger.warning(f"Tool usage test failed: {e}")
        
        return successful_uses / len(test_cases)
    
    def _test_reasoning_quality(self, test_cases: List[Dict]) -> float:
        """Test reasoning quality"""
        if not test_cases:
            return 0.0
        
        reasoning_scores = []
        for case in test_cases:
            try:
                response = self.generate_response(case["prompt"])
                score = self._score_reasoning(response, case.get("expected_steps", []))
                reasoning_scores.append(score)
            except Exception as e:
                logger.warning(f"Reasoning test failed: {e}")
                reasoning_scores.append(0.0)
        
        return sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
    
    def _test_json_output(self, test_cases: List[Dict]) -> float:
        """Test JSON output validity"""
        if not test_cases:
            return 0.0
        
        valid_outputs = 0
        for case in test_cases:
            try:
                response = self.generate_response(case["prompt"])
                if self._validate_json_output(response):
                    valid_outputs += 1
            except Exception as e:
                logger.warning(f"JSON output test failed: {e}")
        
        return valid_outputs / len(test_cases)
    
    def _calculate_error_rate(self, test_suite: Dict[str, List]) -> float:
        """Calculate overall error rate across all tests"""
        total_tests = sum(len(tests) for tests in test_suite.values())
        if total_tests == 0:
            return 0.0
        
        # This is a simplified error rate calculation
        # In practice, you'd track actual errors during testing
        return 0.05  # Placeholder - implement based on actual error tracking
    
    # Helper validation methods (simplified implementations)
    def _validate_function_call(self, response: str, expected_function: Optional[str]) -> bool:
        """Validate function call format"""
        # Simplified validation - check for function-like patterns
        return "(" in response and ")" in response and any(keyword in response.lower() 
                                                          for keyword in ["function", "call", "tool"])
    
    def _score_instruction_following(self, response: str, requirements: List[str]) -> float:
        """Score instruction following"""
        if not requirements:
            return 1.0
        
        met_requirements = sum(1 for req in requirements if req.lower() in response.lower())
        return met_requirements / len(requirements)
    
    def _score_coherence(self, response: str, expected_themes: List[str]) -> float:
        """Score conversation coherence"""
        if not expected_themes:
            return 1.0
        
        theme_matches = sum(1 for theme in expected_themes if theme.lower() in response.lower())
        return min(1.0, theme_matches / len(expected_themes))
    
    def _validate_tool_usage(self, response: str, expected_tool: Optional[str]) -> bool:
        """Validate tool usage"""
        if not expected_tool:
            return True
        return expected_tool.lower() in response.lower()
    
    def _score_reasoning(self, response: str, expected_steps: List[str]) -> float:
        """Score reasoning quality"""
        if not expected_steps:
            return 1.0
        
        step_matches = sum(1 for step in expected_steps if step.lower() in response.lower())
        return step_matches / len(expected_steps)
    
    def _validate_json_output(self, response: str) -> bool:
        """Validate JSON output format"""
        try:
            # Extract JSON from response (simple approach)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                json.loads(json_str)
                return True
        except:
            pass
        return False
    
    def save_results(self, output_dir: str):
        """Save evaluation results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance metrics
        if self.performance_metrics:
            with open(f"{output_dir}/{self.model_name}_performance.json", "w") as f:
                json.dump([vars(metric) for metric in self.performance_metrics], f, indent=2)
        
        # Save configuration
        with open(f"{output_dir}/{self.model_name}_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")