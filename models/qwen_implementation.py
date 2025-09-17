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
from configs.model_configs import ModelConfig

logger = logging.getLogger(__name__)

class Qwen3Implementation(BaseModelImplementation):
    """Qwen-3 model implementation with enhanced configuration support"""
    
    def __init__(self, config: ModelConfig):
        # Convert ModelConfig to dict for base class compatibility
        config_dict = {
            "model_name": config.model_name,
            "huggingface_id": config.huggingface_id,
            "license": config.license,
            "size_gb": config.size_gb,
            "context_window": config.context_window,
            "preset": config.preset,
            "agent_optimized": config.agent_optimized,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
        }
        super().__init__(config_dict)
        
        # Store the enhanced ModelConfig
        self.model_config = config
        self.sampling_params = None
    
    def load_model(self) -> bool:
        """Load Qwen-3 model with enhanced configuration system"""
        try:
            logger.info(f"Loading {self.model_name} with preset: {self.model_config.preset}")
            
            # Get optimized vLLM arguments from enhanced config
            vllm_args = self.model_config.to_vllm_args()
            
            logger.info(f"vLLM Configuration:")
            logger.info(f"  Model: {vllm_args['model']}")
            logger.info(f"  Max Length: {vllm_args['max_model_len']}")
            logger.info(f"  GPU Memory: {vllm_args['gpu_memory_utilization']}")
            logger.info(f"  Max Sequences: {vllm_args['max_num_seqs']}")
            logger.info(f"  Prefix Caching: {vllm_args['enable_prefix_caching']}")
            logger.info(f"  Quantization: {vllm_args.get('quantization', 'None')}")
            
            # Load the model with enhanced configuration
            self.llm_engine = LLM(**vllm_args)
            
            # Load tokenizer separately for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.huggingface_id,
                trust_remote_code=True
            )
            
            # Get optimized sampling parameters from config
            sampling_config = self.model_config.get_agent_sampling_params()
            self.sampling_params = SamplingParams(**sampling_config)
            
            logger.info(f"Sampling Configuration:")
            logger.info(f"  Temperature: {self.sampling_params.temperature}")
            logger.info(f"  Top-p: {self.sampling_params.top_p}")
            logger.info(f"  Max Tokens: {self.sampling_params.max_tokens}")
            logger.info(f"  Stop Tokens: {len(self.sampling_params.stop)} configured")
            
            self.is_loaded = True
            logger.info(f"✅ {self.model_name} loaded successfully with {self.model_config.preset} preset")
            
            # Log memory usage
            memory_info = self.get_memory_usage()
            logger.info(f"Memory usage: {memory_info['gpu_memory_gb']:.1f}GB GPU")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_name}: {e}")
            logger.error(f"Configuration used: {self.model_config.preset} preset")
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
        """Get detailed model information including enhanced configuration"""
        info = {
            "model_name": self.model_name,
            "huggingface_id": self.model_config.huggingface_id,
            "license": self.model_config.license,
            "size_gb": self.model_config.size_gb,
            "context_window": self.model_config.context_window,
            "preset": self.model_config.preset,
            "quantization": self.model_config.quantization_method,
            "max_model_len": self.model_config.max_model_len,
            "agent_optimized": self.model_config.agent_optimized,
            "agent_temperature": self.model_config.agent_temperature,
            "max_function_calls": self.model_config.max_function_calls_per_turn,
            "is_loaded": self.is_loaded,
        }
        
        if self.is_loaded:
            memory_info = self.get_memory_usage()
            info.update(memory_info)
            
            # Add configuration-specific runtime info
            info.update({
                "gpu_memory_utilization": self.model_config.gpu_memory_utilization,
                "max_num_seqs": self.model_config.max_num_seqs,
                "enable_prefix_caching": self.model_config.enable_prefix_caching,
                "use_v2_block_manager": self.model_config.use_v2_block_manager,
            })
        
        return info
    
    def get_preset_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare this model across different presets"""
        from configs.model_configs import estimate_memory_usage
        
        presets = ["balanced", "performance", "memory_optimized"]
        comparison = {}
        
        for preset in presets:
            if preset == self.model_config.preset:
                # Current configuration
                config = self.model_config
            else:
                # Create variant
                config = self.model_config.create_preset_variant(preset)
            
            memory_est = estimate_memory_usage(config)
            vllm_args = config.to_vllm_args()
            
            comparison[preset] = {
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_num_seqs": config.max_num_seqs,
                "estimated_vram_gb": memory_est["total_estimated_gb"],
                "h100_utilization": memory_est["h100_utilization"],
                "evaluation_batch_size": config.evaluation_batch_size,
                "max_model_len": config.max_model_len,
                "enable_prefix_caching": config.enable_prefix_caching,
                "current": preset == self.model_config.preset
            }
        
        return comparison
    
    def cleanup(self) -> None:
        """Clean up resources and properly shutdown the model"""
        try:
            if hasattr(self, 'llm_engine') and self.llm_engine is not None:
                # Clean up vLLM engine
                del self.llm_engine
                self.llm_engine = None
                logger.info(f"✅ {self.model_name} engine cleaned up")
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Clean up tokenizer
                del self.tokenizer
                self.tokenizer = None
            
            self.is_loaded = False
            
            # Force garbage collection to free GPU memory
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ GPU memory cache cleared")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# Enhanced factory functions using ModelConfig
def create_qwen3_8b(preset: str = "balanced", cache_dir: Optional[str] = None) -> Qwen3Implementation:
    """Create Qwen-3 8B instance with specified preset"""
    from configs.model_configs import MODEL_CONFIGS
    
    base_config = MODEL_CONFIGS["qwen3_8b"]
    if preset != "balanced":
        config = base_config.create_preset_variant(preset)
    else:
        config = base_config
    
    if cache_dir:
        # Add cache directory to vLLM overrides
        config._vllm_overrides["download_dir"] = cache_dir
    
    return Qwen3Implementation(config)

def create_qwen3_14b(preset: str = "balanced", cache_dir: Optional[str] = None) -> Qwen3Implementation:
    """Create Qwen-3 14B instance with specified preset"""
    from configs.model_configs import MODEL_CONFIGS
    
    base_config = MODEL_CONFIGS["qwen3_14b"]
    if preset != "balanced":
        config = base_config.create_preset_variant(preset)
    else:
        config = base_config
    
    if cache_dir:
        # Add cache directory to vLLM overrides
        config._vllm_overrides["download_dir"] = cache_dir
    
    return Qwen3Implementation(config)

def create_qwen3_from_config(config: ModelConfig) -> Qwen3Implementation:
    """Create Qwen-3 instance from custom ModelConfig"""
    return Qwen3Implementation(config)

def create_recommended_qwen(task_type: str = "agent_development", 
                          available_memory_gb: int = 80,
                          cache_dir: Optional[str] = None) -> Optional[Qwen3Implementation]:
    """Create recommended Qwen configuration based on task and constraints"""
    from configs.model_configs import recommend_config_for_task
    
    # Get recommendation from enhanced config system
    recommended_config = recommend_config_for_task(task_type, available_memory_gb)
    
    if not recommended_config:
        logger.warning(f"No suitable Qwen configuration found for task: {task_type}, memory: {available_memory_gb}GB")
        return None
    
    if "qwen" not in recommended_config.huggingface_id.lower():
        logger.warning(f"Recommended config is not Qwen model: {recommended_config.model_name}")
        return None
    
    if cache_dir:
        recommended_config._vllm_overrides["download_dir"] = cache_dir
    
    logger.info(f"Creating recommended Qwen for {task_type}: {recommended_config.model_name} ({recommended_config.preset})")
    return Qwen3Implementation(recommended_config)