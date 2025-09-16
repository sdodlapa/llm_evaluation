"""
Qwen-3 model implementation for evaluation
Optimized for H100 GPU and agentic system testing
"""

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
from typing import Dict, Any, Optional

from .base_model import BaseModelImplementation

logger = logging.getLogger(__name__)

class Qwen3Implementation(BaseModelImplementation):
    """Qwen-3 model implementation with vLLM backend"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sampling_params = None
    
    def load_model(self) -> bool:
        """Load Qwen-3 model with optimal settings for H100"""
        try:
            logger.info(f"Loading {self.model_name} with vLLM...")
            
            # Get vLLM arguments from config
            vllm_args = self.config.get("vllm_args", {})
            
            # Default vLLM settings optimized for Qwen-3
            default_args = {
                "model": self.config["huggingface_id"],
                "trust_remote_code": True,
                "dtype": "auto",
                "max_model_len": self.config.get("max_model_len", 32768),
                "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.85),
                "quantization": self.config.get("quantization_method") if self.config.get("quantization_method") != "none" else None,
                "tensor_parallel_size": 1,  # Single H100
                "enforce_eager": False,  # Use CUDA graphs for speed
                "disable_log_stats": True,  # Reduce logging overhead
                "max_num_seqs": 64,  # Reasonable batch size for agents
            }
            
            # Merge with provided args
            final_args = {**default_args, **vllm_args}
            
            # Load the model
            self.llm_engine = LLM(**final_args)
            
            # Load tokenizer separately for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["huggingface_id"],
                trust_remote_code=True
            )
            
            # Set up sampling parameters optimized for agents
            self.sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for consistent agent behavior
                top_p=0.9,
                max_tokens=2048,  # Reasonable for agent responses
                frequency_penalty=0.1,  # Slight penalty to avoid repetition
                presence_penalty=0.1,
                stop=["<|endoftext|>", "<|im_end|>", "\n\nUser:", "\n\nHuman:"],  # Qwen-specific stops
            )
            
            self.is_loaded = True
            logger.info(f"✅ {self.model_name} loaded successfully")
            
            # Log memory usage
            memory_info = self.get_memory_usage()
            logger.info(f"Memory usage: {memory_info['gpu_memory_gb']:.1f}GB GPU")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using vLLM"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        
        try:
            # Override sampling parameters if provided
            sampling_params = self.sampling_params
            if kwargs:
                custom_params = SamplingParams(
                    temperature=kwargs.get("temperature", self.sampling_params.temperature),
                    top_p=kwargs.get("top_p", self.sampling_params.top_p),
                    max_tokens=kwargs.get("max_tokens", self.sampling_params.max_tokens),
                    frequency_penalty=kwargs.get("frequency_penalty", self.sampling_params.frequency_penalty),
                    presence_penalty=kwargs.get("presence_penalty", self.sampling_params.presence_penalty),
                    stop=kwargs.get("stop", self.sampling_params.stop)
                )
                sampling_params = custom_params
            
            # Format prompt for Qwen-3 chat format
            formatted_prompt = self._format_prompt_for_qwen(prompt)
            
            # Generate response
            outputs = self.llm_engine.generate([formatted_prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                return generated_text.strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            return f"Error: {str(e)}"
    
    def generate_with_function_calling(self, prompt: str, functions: list, **kwargs) -> Dict[str, Any]:
        """Generate response with function calling capability"""
        # Format prompt with function definitions
        function_prompt = self._format_function_calling_prompt(prompt, functions)
        
        # Use higher max_tokens for function calling
        kwargs["max_tokens"] = kwargs.get("max_tokens", 3072)
        kwargs["temperature"] = kwargs.get("temperature", 0.05)  # More deterministic for function calls
        
        response = self.generate_response(function_prompt, **kwargs)
        
        # Parse function calls from response
        function_calls = self._parse_function_calls(response)
        
        return {
            "response": response,
            "function_calls": function_calls,
            "raw_output": response
        }
    
    def _format_prompt_for_qwen(self, prompt: str) -> str:
        """Format prompt using Qwen-3 chat template"""
        # Qwen-3 uses ChatML format
        if "You are" in prompt or "System:" in prompt:
            # Already formatted
            return prompt
        
        # Standard agent system prompt
        system_prompt = """You are a helpful AI assistant with access to various tools and functions. You can:
1. Answer questions and provide information
2. Use tools when appropriate to complete tasks
3. Maintain context across conversations
4. Provide structured outputs when requested

Always be helpful, accurate, and follow instructions carefully."""
        
        formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        return formatted
    
    def _format_function_calling_prompt(self, prompt: str, functions: list) -> str:
        """Format prompt with function definitions for function calling"""
        if not functions:
            return self._format_prompt_for_qwen(prompt)
        
        # Format functions as JSON schema
        functions_text = "Available functions:\n"
        for func in functions:
            functions_text += f"- {func.get('name', 'unknown')}: {func.get('description', 'No description')}\n"
            if 'parameters' in func:
                functions_text += f"  Parameters: {func['parameters']}\n"
        
        function_calling_system = f"""You are a helpful AI assistant with access to the following functions:

{functions_text}

When you need to use a function, respond with a JSON object in this format:
{{
    "function_call": {{
        "name": "function_name",
        "arguments": {{
            "param1": "value1",
            "param2": "value2"
        }}
    }},
    "reasoning": "Why you chose this function and these parameters"
}}

If you don't need to use a function, respond normally with helpful information."""
        
        formatted = f"""<|im_start|>system
{function_calling_system}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        return formatted
    
    def _parse_function_calls(self, response: str) -> list:
        """Parse function calls from model response"""
        import json
        import re
        
        function_calls = []
        
        # Look for JSON function call patterns
        json_pattern = r'\{[^}]*"function_call"[^}]*\{[^}]*\}[^}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if "function_call" in parsed:
                    function_calls.append(parsed["function_call"])
            except json.JSONDecodeError:
                continue
        
        return function_calls
    
    def test_function_calling_capability(self) -> Dict[str, float]:
        """Test function calling with standard examples"""
        test_functions = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_weather", 
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name or coordinates"}
                    },
                    "required": ["location"]
                }
            }
        ]
        
        test_cases = [
            "What is 15% of 250?",
            "What's the weather like in New York?",
            "Calculate the area of a circle with radius 5",
            "Get the current temperature in San Francisco"
        ]
        
        results = []
        for prompt in test_cases:
            result = self.generate_with_function_calling(prompt, test_functions)
            function_calls = result.get("function_calls", [])
            success = len(function_calls) > 0
            results.append(success)
        
        accuracy = sum(results) / len(results) if results else 0.0
        
        return {
            "function_calling_accuracy": accuracy,
            "total_tests": len(test_cases),
            "successful_calls": sum(results)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = {
            "model_name": self.model_name,
            "huggingface_id": self.config["huggingface_id"],
            "license": self.config.get("license", "Unknown"),
            "context_window": self.config.get("context_window", 0),
            "quantization": self.config.get("quantization_method", "none"),
            "max_model_len": self.config.get("max_model_len", 0),
            "is_loaded": self.is_loaded,
            "agent_optimized": self.config.get("agent_optimized", False)
        }
        
        if self.is_loaded:
            memory_info = self.get_memory_usage()
            info.update(memory_info)
        
        return info

# Factory function for easy instantiation
def create_qwen3_8b(cache_dir: Optional[str] = None) -> Qwen3Implementation:
    """Create Qwen-3 8B instance with default config"""
    config = {
        "model_name": "Qwen-3 8B Instruct",
        "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
        "license": "Apache 2.0",
        "size_gb": 7.5,
        "context_window": 128000,
        "quantization_method": "awq",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.85,
        "agent_optimized": True
    }
    
    if cache_dir:
        config["cache_dir"] = cache_dir
    
    return Qwen3Implementation(config)

def create_qwen3_14b(cache_dir: Optional[str] = None) -> Qwen3Implementation:
    """Create Qwen-3 14B instance with default config"""
    config = {
        "model_name": "Qwen-3 14B Instruct",
        "huggingface_id": "Qwen/Qwen2.5-14B-Instruct",
        "license": "Apache 2.0", 
        "size_gb": 14.0,
        "context_window": 128000,
        "quantization_method": "awq",
        "max_model_len": 24576,  # Reduced for 14B model
        "gpu_memory_utilization": 0.80,
        "agent_optimized": True
    }
    
    if cache_dir:
        config["cache_dir"] = cache_dir
    
    return Qwen3Implementation(config)