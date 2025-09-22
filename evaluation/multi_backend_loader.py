#!/usr/bin/env python3
"""
Multi-Backend Model Loader
==========================

Supports multiple inference backends for different model types:
- vLLM: For supported causal language models (fast inference)
- Transformers: For BERT, encoder models, unsupported architectures
- Direct PyTorch: For special cases

This allows us to evaluate ALL models regardless of vLLM support.
"""

import logging
import torch
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BackendType(Enum):
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    PYTORCH = "pytorch"
    SENTENCE_TRANSFORMERS = "sentence_transformers"

@dataclass
class ModelBackendConfig:
    """Configuration for model backend selection"""
    backend: BackendType
    model_class: str
    tokenizer_class: str = "AutoTokenizer"
    requires_trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"

# Backend selection rules based on model architecture/type
BACKEND_RULES = {
    # vLLM supported architectures (fast inference)
    "causal_lm": {
        "architectures": [
            "LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM", 
            "Phi3ForCausalLM", "GemmaForCausalLM", "CodeLlamaForCausalLM",
            "DeepseekForCausalLM", "ChatGLMForConditionalGeneration"
        ],
        "backend": BackendType.VLLM,
        "model_class": "vLLM"
    },
    
    # BERT-style models (use transformers)
    "bert_models": {
        "architectures": [
            "BertModel", "BertForMaskedLM", "BertForSequenceClassification",
            "RobertaModel", "RobertaForMaskedLM", "DistilBertModel",
            "AlbertModel", "ElectraModel"
        ],
        "backend": BackendType.TRANSFORMERS,
        "model_class": "AutoModel"
    },
    
    # Encoder models for embeddings
    "encoder_models": {
        "architectures": [
            "BertModel", "RobertaModel", "DistilBertModel",
            "SentenceTransformer"
        ],
        "backend": BackendType.TRANSFORMERS,
        "model_class": "AutoModel"
    },
    
    # Vision-Language models
    "vision_language": {
        "architectures": [
            "Qwen2VLForConditionalGeneration", "LlavaForConditionalGeneration",
            "InstructBlipForConditionalGeneration", "BlipForConditionalGeneration"
        ],
        "backend": BackendType.TRANSFORMERS,
        "model_class": "AutoModelForVision2Seq"
    },
    
    # Special models
    "gpt_models": {
        "architectures": [
            "BioGptForCausalLM", "GPT2LMHeadModel", "GPTNeoForCausalLM"
        ],
        "backend": BackendType.TRANSFORMERS,
        "model_class": "AutoModelForCausalLM"
    }
}

# Model-specific overrides (when we know exactly what works)
MODEL_OVERRIDES = {
    # Known vLLM compatible models
    "qwen25_0_5b": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    "qwen25_3b": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    "qwen3_8b": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    "qwen3_14b": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    "qwen25_math_7b": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    "qwen25_7b": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    "phi35_mini": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    "qwen3_coder_30b": ModelBackendConfig(BackendType.VLLM, "vLLM"),
    
    # Known transformers-only models
    "biogpt": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM"),
    "biogpt_large": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM"),
    "pubmedbert_large": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModel"),
    "scibert_base": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModel"),
    "layoutlmv3_base": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModel"),
    
    # Multimodal models
    "qwen2_vl_7b": ModelBackendConfig(BackendType.TRANSFORMERS, "Qwen2VLForConditionalGeneration"),
    "qwen25_vl_7b": ModelBackendConfig(BackendType.TRANSFORMERS, "Qwen2VLForConditionalGeneration"),
    
    # Models requiring authentication (try transformers)
    "llama31_8b": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM"),
    "llama33_70b": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM"),
    "mistral_7b": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM"),
    "gemma2_9b": ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM"),
}

def detect_model_backend(model_id: str, model_config: Any) -> ModelBackendConfig:
    """
    Detect the appropriate backend for a model based on its configuration
    """
    # First check for explicit overrides
    if model_id in MODEL_OVERRIDES:
        logger.info(f"Using override backend for {model_id}: {MODEL_OVERRIDES[model_id].backend.value}")
        return MODEL_OVERRIDES[model_id]
    
    # Try to get architecture from model config
    huggingface_id = getattr(model_config, 'huggingface_id', model_id)
    
    try:
        # For gated models, this might fail
        from transformers import AutoConfig
        import os
        
        # Get authentication token for gated models
        auth_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        
        hf_config = AutoConfig.from_pretrained(
            huggingface_id, 
            trust_remote_code=True,
            token=auth_token
        )
        architectures = getattr(hf_config, 'architectures', [])
        
        if architectures:
            arch = architectures[0]
            logger.info(f"Detected architecture for {model_id}: {arch}")
            
            # Check backend rules
            for rule_name, rule in BACKEND_RULES.items():
                if arch in rule["architectures"]:
                    backend = rule["backend"]
                    model_class = rule["model_class"]
                    logger.info(f"Matched rule {rule_name} for {model_id} -> {backend.value}")
                    return ModelBackendConfig(backend, model_class)
        
    except Exception as e:
        logger.warning(f"Could not detect architecture for {model_id}: {e}")
    
    # Default fallback logic
    model_path = huggingface_id.lower()
    
    # Special patterns
    if any(x in model_path for x in ['qwen', 'phi']):
        return ModelBackendConfig(BackendType.VLLM, "vLLM")
    elif any(x in model_path for x in ['bert', 'roberta', 'distil']):
        return ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModel")
    elif any(x in model_path for x in ['gpt', 'llama', 'mistral', 'gemma']):
        return ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM")
    elif any(x in model_path for x in ['vl', 'vision', 'multimodal']):
        return ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForVision2Seq")
    else:
        # Default to transformers for unknown models
        return ModelBackendConfig(BackendType.TRANSFORMERS, "AutoModelForCausalLM")

class MultiBackendModelLoader:
    """
    Model loader that can use different backends based on model requirements
    """
    
    def __init__(self):
        self.loaded_models = {}
        self.backend_configs = {}
    
    def load_model(self, model_id: str, model_config: Any, preset: str = "balanced") -> tuple[bool, Any]:
        """
        Load a model using the appropriate backend
        
        Returns:
            (success: bool, model: Any)
        """
        try:
            # Detect backend
            backend_config = detect_model_backend(model_id, model_config)
            self.backend_configs[model_id] = backend_config
            
            logger.info(f"Loading {model_id} using {backend_config.backend.value} backend")
            
            if backend_config.backend == BackendType.VLLM:
                return self._load_vllm_model(model_id, model_config, preset)
            elif backend_config.backend == BackendType.TRANSFORMERS:
                return self._load_transformers_model(model_id, model_config, backend_config, preset)
            else:
                logger.error(f"Backend {backend_config.backend} not implemented yet")
                return False, None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False, None
    
    def _load_vllm_model(self, model_id: str, model_config: Any, preset: str) -> tuple[bool, Any]:
        """Load model using vLLM backend"""
        try:
            # Import vLLM directly and create simple wrapper
            from vllm import LLM, SamplingParams
            import torch
            
            huggingface_id = getattr(model_config, 'huggingface_id', model_id)
            
            # Get authentication token for gated models
            import os
            auth_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
            
            # Basic vLLM configuration
            vllm_args = {
                "model": huggingface_id,
                "trust_remote_code": True,
                "max_model_len": getattr(model_config, 'context_window', 32768),
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 64,
                "tensor_parallel_size": getattr(model_config, 'tensor_parallel_size', 1),
                "enable_prefix_caching": True,
                "block_size": 16
            }
            
            # Add token for gated models
            if auth_token:
                vllm_args["download_dir"] = None  # Use default
                # Set environment for vLLM to use the token
                os.environ['HF_TOKEN'] = auth_token
            
            logger.info(f"Loading {model_id} using vLLM backend")
            llm = LLM(**vllm_args)
            
            # Create a simple wrapper
            class VLLMModelWrapper:
                def __init__(self, llm_instance, model_name):
                    self.llm = llm_instance
                    self.model_name = model_name
                    
                def generate(self, prompts, **kwargs):
                    sampling_params = SamplingParams(
                        temperature=kwargs.get('temperature', 0.1),
                        top_p=kwargs.get('top_p', 0.9),
                        max_tokens=kwargs.get('max_tokens', 1024)
                    )
                    return self.llm.generate(prompts, sampling_params)
                
                def generate_response(self, prompt: str, **kwargs) -> str:
                    """Generate a single response for a single prompt (compatibility method)"""
                    outputs = self.generate([prompt], **kwargs)
                    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            
            wrapper = VLLMModelWrapper(llm, model_id)
            self.loaded_models[model_id] = wrapper
            return True, wrapper
                
        except Exception as e:
            logger.error(f"vLLM loading failed for {model_id}: {e}")
            return False, None
    
    def _load_transformers_model(self, model_id: str, model_config: Any, backend_config: ModelBackendConfig, preset: str) -> tuple[bool, Any]:
        """Load model using Transformers backend"""
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
            import torch
            
            huggingface_id = getattr(model_config, 'huggingface_id', model_id)
            
            # Get authentication token for gated models
            import os
            auth_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                huggingface_id,
                trust_remote_code=True,
                padding_side="left",
                token=auth_token
            )
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on model class
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "token": auth_token
            }
            
            if backend_config.model_class == "AutoModel":
                model = AutoModel.from_pretrained(huggingface_id, **model_kwargs)
            elif backend_config.model_class == "AutoModelForCausalLM":
                model = AutoModelForCausalLM.from_pretrained(huggingface_id, **model_kwargs)
            elif backend_config.model_class.startswith("Qwen2VL"):
                # Special handling for Qwen2-VL
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(huggingface_id, **model_kwargs)
            else:
                # Try AutoModelForCausalLM as default
                model = AutoModelForCausalLM.from_pretrained(huggingface_id, **model_kwargs)
            
            # Create a wrapper that mimics vLLM interface
            wrapper = TransformersModelWrapper(model, tokenizer, model_id)
            self.loaded_models[model_id] = wrapper
            
            logger.info(f"✅ Successfully loaded {model_id} using Transformers backend")
            return True, wrapper
            
        except Exception as e:
            logger.error(f"Transformers loading failed for {model_id}: {e}")
            return False, None

class TransformersModelWrapper:
    """
    Wrapper to make Transformers models compatible with the existing evaluation interface
    """
    
    def __init__(self, model, tokenizer, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_name = model_id  # Add missing model_name attribute
        self._is_loaded = True
    
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using Transformers backend"""
        try:
            # Handle single prompt
            if isinstance(prompts, str):
                prompts = [prompts]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            )
            
            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', 256),
                    temperature=kwargs.get('temperature', 0.1),
                    do_sample=kwargs.get('temperature', 0.1) > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode responses
            responses = []
            for i, output in enumerate(outputs):
                # Remove input tokens from output
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response.strip())
            
            return responses
            
        except Exception as e:
            logger.error(f"Generation failed for {self.model_id}: {e}")
            return [""] * len(prompts)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a single response for a single prompt (compatibility method)"""
        responses = self.generate([prompt], **kwargs)
        return responses[0] if responses else ""
    
    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        torch.cuda.empty_cache()

# Export the main loader
def create_multi_backend_loader() -> MultiBackendModelLoader:
    """Create a new multi-backend model loader instance"""
    return MultiBackendModelLoader()