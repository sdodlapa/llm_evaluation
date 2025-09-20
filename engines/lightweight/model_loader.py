"""
Lightweight model loader for small/medium models

Handles efficient loading and management of models ≤30B parameters
with optimizations for single-GPU execution.
"""

import logging
import time
from typing import Dict, Any, Optional
import torch
from pathlib import Path

# Import existing vLLM components
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.model_loader import get_model
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available, using fallback implementation")

from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig


logger = logging.getLogger(__name__)


class LightweightModelLoader:
    """Optimized model loader for lightweight engine"""
    
    def __init__(self):
        self._cache_dir = Path("./model_cache")
        self._cache_dir.mkdir(exist_ok=True)
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the model loader"""
        try:
            # Check GPU availability (allow CPU-only for testing)
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU-only mode for testing")
            
            # Check vLLM availability
            if not VLLM_AVAILABLE:
                logger.warning("vLLM not available, using basic fallback")
            
            self._initialized = True
            logger.info("Lightweight model loader initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model loader: {e}")
            return False
    
    def load_model(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Load model with optimizations for lightweight engine
        
        Args:
            model_config: Enhanced model configuration
            
        Returns:
            Dict containing model, tokenizer, and metadata
        """
        if not self._initialized:
            raise RuntimeError("Model loader not initialized")
        
        logger.info(f"Loading model {model_config.model_name}")
        load_start = time.time()
        
        try:
            if VLLM_AVAILABLE:
                return self._load_with_vllm(model_config)
            else:
                return self._load_with_fallback(model_config)
                
        except Exception as e:
            logger.error(f"Failed to load model {model_config.model_name}: {e}")
            raise
    
    def _load_with_vllm(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Load model using vLLM backend"""
        # Get vLLM configuration
        vllm_config = model_config.get_vllm_config()
        
        # Apply lightweight optimizations
        if hasattr(model_config, 'lightweight_optimizations'):
            opts = model_config.lightweight_optimizations
            if opts.fast_loading:
                vllm_config["load_format"] = "auto"
            if opts.memory_mapping:
                vllm_config["enable_lora"] = False  # Disable LoRA for simpler loading
        
        logger.info(f"Loading {model_config.model_name} with vLLM config: {vllm_config}")
        
        # Create LLM instance
        llm = LLM(**vllm_config)
        
        # Create sampling parameters (can be adjusted per request)
        sampling_params = SamplingParams(
            temperature=model_config.agent_temperature,
            max_tokens=512,  # Default, can be overridden
            top_p=0.9
        )
        
        return {
            "model": llm,
            "tokenizer": llm.get_tokenizer(),
            "sampling_params": sampling_params,
            "backend": "vllm",
            "config": vllm_config,
            "model_name": model_config.model_name,
            "size_gb": model_config.size_gb
        }
    
    def _load_with_fallback(self, model_config: EnhancedModelConfig) -> Dict[str, Any]:
        """Fallback implementation without vLLM"""
        logger.warning(f"Loading {model_config.model_name} with fallback implementation")
        
        # This would implement a basic transformers loading approach
        # For now, return a mock implementation for testing
        return {
            "model": MockModel(model_config.model_name),
            "tokenizer": MockTokenizer(),
            "backend": "fallback",
            "config": {},
            "model_name": model_config.model_name,
            "size_gb": model_config.size_gb
        }
    
    def validate_model_compatibility(self, model_config: EnhancedModelConfig) -> bool:
        """Check if model is compatible with lightweight loader"""
        # Size check
        if model_config.size_gb > 60.0:
            logger.warning(f"Model {model_config.model_name} too large for lightweight loader")
            return False
        
        # Multi-GPU check
        if model_config.tensor_parallel_size > 1 or model_config.pipeline_parallel_size > 1:
            logger.warning(f"Model {model_config.model_name} requires multi-GPU, not suitable for lightweight loader")
            return False
        
        return True
    
    def get_memory_requirements(self, model_config: EnhancedModelConfig) -> Dict[str, float]:
        """Estimate memory requirements for model loading"""
        base_memory = model_config.size_gb
        
        # Add overhead for activation memory and evaluation
        total_memory = base_memory * 2.0  # Model + activations
        total_memory += 2.0  # Evaluation overhead
        
        # Apply optimization adjustments
        if hasattr(model_config, 'lightweight_optimizations'):
            opts = model_config.lightweight_optimizations
            if opts.memory_mapping:
                total_memory *= 0.9
            if opts.use_flash_attention:
                total_memory *= 0.85
        
        return {
            "model_memory_gb": base_memory,
            "total_gpu_memory_gb": total_memory,
            "system_memory_gb": max(8.0, base_memory * 0.5)
        }


class MockModel:
    """Mock model for testing without actual model loading"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompts, sampling_params=None):
        """Mock generation for testing"""
        return [MockOutput(f"Generated response for: {prompt[:50]}...") for prompt in prompts]


class MockTokenizer:
    """Mock tokenizer for testing"""
    
    def encode(self, text: str):
        return list(range(len(text.split())))
    
    def decode(self, tokens):
        return " ".join([f"token_{i}" for i in tokens])


class MockOutput:
    """Mock output for testing"""
    
    def __init__(self, text: str):
        self.outputs = [MockCompletionOutput(text)]


class MockCompletionOutput:
    """Mock completion output for testing"""
    
    def __init__(self, text: str):
        self.text = text