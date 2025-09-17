"""
Model configurations for individual LLM evaluation on H100 GPU
Optimized for 80GB VRAM and agentic system development
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import warnings

@dataclass
class ModelConfig:
    # Core model information
    model_name: str
    huggingface_id: str
    license: str
    size_gb: float
    context_window: int
    
    # Configuration preset (optimal balance approach)
    preset: str = "balanced"  # "balanced", "performance", "memory_optimized"
    
    # Basic model settings
    quantization_method: str = "none"  # AWQ not available for this model in vLLM 0.10.2
    max_model_len: int = 32768  # Conservative for agents
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    priority: str = "HIGH"  # HIGH, MEDIUM, LOW
    agent_optimized: bool = True
    
    # Advanced vLLM settings (optimal defaults)
    max_num_seqs: int = 64  # Reasonable batch size for agents
    enable_prefix_caching: bool = True  # Speeds up repeated prompts
    use_v2_block_manager: bool = True  # Better memory management
    enforce_eager: bool = False  # Allow CUDA graphs when beneficial
    
    # Agent-specific optimizations
    function_calling_format: str = "json"  # "json", "xml", "natural"
    max_function_calls_per_turn: int = 5
    agent_temperature: float = 0.1  # Low for consistent agent behavior
    
    # Evaluation settings
    evaluation_batch_size: int = 8
    benchmark_iterations: int = 3  # Balanced - not too slow, not too fast
    
    # Advanced settings (optional, populated by preset)
    _vllm_overrides: Dict[str, Any] = field(default_factory=dict)
    
    
    def apply_preset(self):
        """Apply preset-specific optimizations"""
        if self.preset == "performance":
            # Maximize speed, higher memory usage - H100 optimized
            self.gpu_memory_utilization = 0.95  # Very aggressive for H100
            self.max_num_seqs = 512  # Maximum throughput batching
            self.enforce_eager = False  # Allow CUDA graphs
            self.evaluation_batch_size = 64  # Large evaluation batches
            self._vllm_overrides.update({
                "max_num_batched_tokens": 32768,  # Large token batches for H100
                "disable_log_stats": True,
                "block_size": 32,  # Larger blocks for H100
                "enable_chunked_prefill": True,  # H100 optimization
                "kv_cache_dtype": "fp8",  # Use H100's native FP8 support
                "use_v2_block_manager": True,
                "swap_space": 8,  # 8GB swap for overflow handling
            })
            
        elif self.preset == "memory_optimized":
            # Minimize VRAM usage
            self.gpu_memory_utilization = 0.70
            self.max_num_seqs = 32
            self.max_model_len = min(self.max_model_len, 16384)  # Reduce context
            self.enable_prefix_caching = False  # Saves memory
            self.evaluation_batch_size = 4
            self._vllm_overrides.update({
                "max_num_batched_tokens": 2048,
                "block_size": 8,  # Smaller blocks
            })
            
        else:  # balanced (default)
            # Optimal balance - no changes needed, using defaults
            pass
    
    def validate_h100_compatibility(self) -> List[str]:
        """Validate configuration for H100 GPU (80GB VRAM)"""
        warnings_list = []
        
        # Estimate memory usage
        memory_est = estimate_memory_usage(self)
        estimated_usage = memory_est["total_estimated_gb"]
        
        if estimated_usage > 75:  # Leave 5GB buffer
            warnings_list.append(f"High VRAM usage: {estimated_usage:.1f}GB (may not fit on H100)")
        
        if self.max_model_len > 100000 and self.size_gb > 10:
            warnings_list.append("Large context + large model may cause OOM")
        
        if self.gpu_memory_utilization > 0.95:
            warnings_list.append("GPU memory utilization too high, reducing to 0.90")
            self.gpu_memory_utilization = 0.90
        
        # Check quantization compatibility
        if self.quantization_method not in ["awq", "awq_marlin", "gptq", "none"]:
            warnings_list.append(f"Unsupported quantization: {self.quantization_method}, using 'awq'")
            self.quantization_method = "awq"
        
        return warnings_list
    
    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM engine arguments with preset optimizations"""
        # Apply preset before generating args
        self.apply_preset()
        
        # Run validation
        warnings_list = self.validate_h100_compatibility()
        for warning in warnings_list:
            warnings.warn(f"ModelConfig: {warning}")
        
        # Base vLLM arguments
        vllm_args = {
            "model": self.huggingface_id,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.torch_dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization_method if self.quantization_method != "none" else None,
            
            # Advanced optimizations
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            # "use_v2_block_manager": self.use_v2_block_manager,  # Not available in vLLM 0.10.2
            "enforce_eager": self.enforce_eager,
            
            # Single GPU optimization
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        }
        
        # Apply preset-specific overrides
        vllm_args.update(self._vllm_overrides)
        
        return vllm_args
    
    def get_agent_sampling_params(self) -> Dict[str, Any]:
        """Get optimized sampling parameters for agent tasks"""
        return {
            "temperature": self.agent_temperature,
            "top_p": 0.9,
            "max_tokens": 2048,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": self._get_model_specific_stops(),
        }
    
    def _get_model_specific_stops(self) -> List[str]:
        """Get model-specific stop tokens"""
        # Basic stops - would be expanded per model family
        stops = ["<|endoftext|>", "\n\nUser:", "\n\nHuman:"]
        
        if "qwen" in self.huggingface_id.lower():
            stops.extend(["<|im_end|>", "<|endoftext|>"])
        elif "llama" in self.huggingface_id.lower():
            stops.extend(["<|eot_id|>"])
        elif "mistral" in self.huggingface_id.lower():
            stops.extend(["</s>"])
        
        return stops
    
    def create_preset_variant(self, preset: str) -> 'ModelConfig':
        """Create a new config with different preset"""
        import copy
        new_config = copy.deepcopy(self)
        new_config.preset = preset
        new_config._vllm_overrides = {}  # Reset overrides
        return new_config

# Primary Model Configurations (Under 16B Parameters)
MODEL_CONFIGS = {
    "qwen3_8b": ModelConfig(
        model_name="Qwen-3 8B Instruct",
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        license="Apache 2.0",
        size_gb=7.5,
        context_window=128000,
        preset="balanced",  # Using new preset system
        quantization_method="none",  # No quantization for now
        max_model_len=32768,  # Agent workloads
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.1,  # New: optimized for agents
        max_function_calls_per_turn=5,
        evaluation_batch_size=8
    ),
    
    "qwen3_14b": ModelConfig(
        model_name="Qwen-3 14B Instruct", 
        huggingface_id="Qwen/Qwen2.5-14B-Instruct-AWQ",  # Use official AWQ model
        license="Apache 2.0",
        size_gb=14.0,
        context_window=128000,
        preset="performance",  # Changed to performance for maximum utilization
        quantization_method="awq_marlin",  # Use AWQ-Marlin kernel for 5x speedup!
        max_model_len=49152,  # Increased context length for better H100 utilization (48K)
        gpu_memory_utilization=0.90,  # Increased to 90% for maximum memory usage
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.1,
        max_function_calls_per_turn=5,
        evaluation_batch_size=32,  # Significantly increased batch size for throughput
        max_num_seqs=256,  # Much higher sequence batching for H100
        benchmark_iterations=5  # More iterations for better benchmarking
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

# Preset configuration factory functions
def create_qwen3_8b_configs() -> Dict[str, ModelConfig]:
    """Create Qwen-3 8B configurations for all presets"""
    base_config = MODEL_CONFIGS["qwen3_8b"]
    
    return {
        "qwen3_8b_balanced": base_config,
        "qwen3_8b_performance": base_config.create_preset_variant("performance"),
        "qwen3_8b_memory_optimized": base_config.create_preset_variant("memory_optimized"),
    }

def create_qwen3_14b_configs() -> Dict[str, ModelConfig]:
    """Create Qwen-3 14B configurations for all presets"""
    base_config = MODEL_CONFIGS["qwen3_14b"]
    
    return {
        "qwen3_14b_balanced": base_config,
        "qwen3_14b_performance": base_config.create_preset_variant("performance"),
        "qwen3_14b_memory_optimized": base_config.create_preset_variant("memory_optimized"),
    }

def get_all_qwen_variants() -> Dict[str, ModelConfig]:
    """Get all Qwen model variants across all presets"""
    all_configs = {}
    all_configs.update(create_qwen3_8b_configs())
    all_configs.update(create_qwen3_14b_configs())
    return all_configs

def recommend_config_for_task(task_type: str, available_memory_gb: int = 80) -> Optional[ModelConfig]:
    """Recommend optimal configuration based on task and available memory"""
    if task_type == "agent_development":
        # Prioritize balanced configs with good agent capabilities
        candidates = [MODEL_CONFIGS["qwen3_8b"], MODEL_CONFIGS["qwen3_14b"]]
    elif task_type == "performance_testing":
        # Use performance presets
        candidates = [
            MODEL_CONFIGS["qwen3_8b"].create_preset_variant("performance"),
            MODEL_CONFIGS["qwen3_14b"].create_preset_variant("performance")
        ]
    elif task_type == "memory_constrained":
        # Use memory-optimized presets
        candidates = [
            MODEL_CONFIGS["qwen3_8b"].create_preset_variant("memory_optimized"),
            MODEL_CONFIGS["qwen3_14b"].create_preset_variant("memory_optimized")
        ]
    else:
        # Default to balanced
        candidates = [MODEL_CONFIGS["qwen3_8b"], MODEL_CONFIGS["qwen3_14b"]]
    
    # Filter by memory constraints
    for config in candidates:
        memory_est = estimate_memory_usage(config)
        if memory_est["total_estimated_gb"] <= available_memory_gb * 0.9:  # 10% buffer
            return config
    
    return None

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
    print("=== Enhanced Model Configuration System ===")
    print()
    
    # Test preset variants for Qwen-3 8B
    print("=== Qwen-3 8B Preset Variants ===")
    qwen_variants = create_qwen3_8b_configs()
    
    for name, config in qwen_variants.items():
        memory_est = estimate_memory_usage(config)
        vllm_args = config.to_vllm_args()
        
        print(f"{name}:")
        print(f"  Preset: {config.preset}")
        print(f"  GPU Memory Util: {config.gpu_memory_utilization}")
        print(f"  Max Sequences: {config.max_num_seqs}")
        print(f"  Estimated VRAM: {memory_est['total_estimated_gb']:.1f}GB ({memory_est['h100_utilization']:.1%})")
        print(f"  vLLM Args: {len(vllm_args)} parameters configured")
        print()
    
    # Test configuration recommendations
    print("=== Configuration Recommendations ===")
    
    task_scenarios = [
        ("agent_development", 80),
        ("performance_testing", 80),
        ("memory_constrained", 40),
    ]
    
    for task, memory in task_scenarios:
        recommended = recommend_config_for_task(task, memory)
        if recommended:
            print(f"Task: {task} ({memory}GB available)")
            print(f"  Recommended: {recommended.model_name}")
            print(f"  Preset: {recommended.preset}")
            print(f"  Agent Optimized: {recommended.agent_optimized}")
            print()
        else:
            print(f"Task: {task} ({memory}GB available) - No suitable config found")
            print()
    
    # Test validation warnings
    print("=== Configuration Validation ===")
    
    # Create a problematic config for testing
    test_config = MODEL_CONFIGS["qwen3_8b"].create_preset_variant("performance")
    test_config.gpu_memory_utilization = 0.98  # Too high
    test_config.max_model_len = 150000  # Very large context
    
    warnings = test_config.validate_h100_compatibility()
    print("Test config warnings:")
    for warning in warnings:
        print(f"  ⚠️  {warning}")
    print()
    
    print("=== High Priority Models (Updated) ===")
    for name, config in get_high_priority_models().items():
        memory_est = estimate_memory_usage(config)
        print(f"{name}: {config.model_name}")
        print(f"  Size: {config.size_gb}GB, License: {config.license}")
        print(f"  Preset: {config.preset}, Agent Temp: {config.agent_temperature}")
        print(f"  Estimated VRAM: {memory_est['total_estimated_gb']:.1f}GB ({memory_est['h100_utilization']:.1%})")
        print()