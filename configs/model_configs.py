"""
Model configurations for individual LLM evaluation on H100 GPU
Optimized for 80GB VRAM and agentic system development
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    model_name: str
    huggingface_id: str
    license: str
    size_gb: float
    context_window: int
    quantization_method: str = "awq"
    max_model_len: int = 32768  # Conservative for agents
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    priority: str = "HIGH"  # HIGH, MEDIUM, LOW
    agent_optimized: bool = True
    
    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM engine arguments"""
        return {
            "model": self.huggingface_id,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.torch_dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization_method if self.quantization_method != "none" else None,
        }

# Primary Model Configurations (Under 16B Parameters)
MODEL_CONFIGS = {
    "qwen3_8b": ModelConfig(
        model_name="Qwen-3 8B Instruct",
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        license="Apache 2.0",
        size_gb=7.5,
        context_window=128000,
        quantization_method="awq",
        max_model_len=32768,  # Agent workloads
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True
    ),
    
    "qwen3_14b": ModelConfig(
        model_name="Qwen-3 14B Instruct", 
        huggingface_id="Qwen/Qwen2.5-14B-Instruct",
        license="Apache 2.0",
        size_gb=14.0,
        context_window=128000,
        quantization_method="awq",
        max_model_len=24576,  # Slightly reduced for 14B
        gpu_memory_utilization=0.80,
        priority="HIGH",
        agent_optimized=True
    ),
    
    "deepseek_coder_16b": ModelConfig(
        model_name="DeepSeek-Coder-V2-Lite 16B",
        huggingface_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", 
        license="Custom (Commercial OK)",
        size_gb=16.0,
        context_window=128000,
        quantization_method="awq",
        max_model_len=16384,  # MoE considerations
        gpu_memory_utilization=0.75,  # Conservative for MoE
        priority="HIGH",
        agent_optimized=True
    ),
    
    "llama31_8b": ModelConfig(
        model_name="Llama 3.1 8B Instruct",
        huggingface_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        license="Meta Community License",
        size_gb=8.0,
        context_window=128000,
        quantization_method="awq", 
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True
    ),
    
    "mistral_7b": ModelConfig(
        model_name="Mistral 7B Instruct",
        huggingface_id="mistralai/Mistral-7B-Instruct-v0.3",
        license="Apache 2.0",
        size_gb=7.0,
        context_window=32768,
        quantization_method="awq",
        max_model_len=24576,
        gpu_memory_utilization=0.85,
        priority="MEDIUM", 
        agent_optimized=True
    ),
    
    "olmo2_13b": ModelConfig(
        model_name="OLMo-2 13B Instruct",
        huggingface_id="allenai/OLMo-2-1124-13B-Instruct",
        license="Apache 2.0", 
        size_gb=13.0,
        context_window=4096,  # Limited context
        quantization_method="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=False  # Research focus
    ),
    
    "yi_9b": ModelConfig(
        model_name="Yi-1.5 9B Chat",
        huggingface_id="01-ai/Yi-1.5-9B-Chat",
        license="Apache 2.0",
        size_gb=9.0, 
        context_window=32768,
        quantization_method="awq",
        max_model_len=24576,
        gpu_memory_utilization=0.85,
        priority="LOW",
        agent_optimized=False
    ),
    
    "phi35_mini": ModelConfig(
        model_name="Phi-3.5 Mini",
        huggingface_id="microsoft/Phi-3.5-mini-instruct", 
        license="MIT",
        size_gb=3.8,
        context_window=128000,
        quantization_method="none",  # Small enough for FP16
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        priority="LOW",
        agent_optimized=False  # Too small for complex agents
    )
}

# Evaluation comparison models (restricted licenses)
COMPARISON_MODELS = {
    "gemma2_9b": ModelConfig(
        model_name="Gemma 2 9B Instruct",
        huggingface_id="google/gemma-2-9b-it",
        license="Gemma License (Restricted)",
        size_gb=9.0,
        context_window=8192,
        quantization_method="awq", 
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        priority="COMPARISON",
        agent_optimized=True
    )
}

def get_high_priority_models():
    """Get models marked as HIGH priority"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.priority == "HIGH"}

def get_apache_licensed_models():
    """Get models with Apache 2.0 license (safe for commercial use)"""
    return {k: v for k, v in MODEL_CONFIGS.items() if "Apache" in v.license}

def get_agent_optimized_models():
    """Get models suitable for agentic systems"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.agent_optimized}

def estimate_memory_usage(config: ModelConfig, context_length: int = None) -> Dict[str, float]:
    """Estimate VRAM usage for a given configuration"""
    if context_length is None:
        context_length = config.max_model_len
    
    # Base model size (quantized)
    if config.quantization_method == "awq":
        base_size = config.size_gb * 0.25  # 4-bit quantization
    elif config.quantization_method == "gptq":
        base_size = config.size_gb * 0.25  # 4-bit quantization  
    else:
        base_size = config.size_gb * 0.5   # FP16
    
    # KV cache estimation (rough)
    # For transformer models: context_len * hidden_size * 2 * layers * 2 (K+V) / 1e9
    # Simplified estimation: ~2-4 MB per 1K context tokens for 7-8B models
    kv_cache_gb = (context_length / 1000) * (config.size_gb / 4) * 0.002
    
    # Framework overhead (vLLM, etc.)
    overhead_gb = 2.0
    
    total_estimated = base_size + kv_cache_gb + overhead_gb
    
    return {
        "base_model_gb": base_size,
        "kv_cache_gb": kv_cache_gb, 
        "overhead_gb": overhead_gb,
        "total_estimated_gb": total_estimated,
        "h100_utilization": total_estimated / 80.0  # H100 has 80GB
    }

# Test the configurations
if __name__ == "__main__":
    print("=== High Priority Models ===")
    for name, config in get_high_priority_models().items():
        memory_est = estimate_memory_usage(config)
        print(f"{name}: {config.model_name}")
        print(f"  Size: {config.size_gb}GB, License: {config.license}")
        print(f"  Estimated VRAM: {memory_est['total_estimated_gb']:.1f}GB ({memory_est['h100_utilization']:.1%})")
        print()