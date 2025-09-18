"""
Core model registry with ModelConfig dataclass and main model configurations
Centralized registry for all model configurations used in LLM evaluation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import copy

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
    
    # NEW: Specialization metadata
    specialization_category: str = "general"  # "text_generation", "code_generation", "data_science", "mathematics", "bioinformatics", "multimodal", "efficiency", "research"
    specialization_subcategory: str = "general_purpose"  # More specific specialization
    primary_use_cases: List[str] = field(default_factory=lambda: ["general"])
    
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
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: Optional[str] = None
    kv_cache_dtype: Optional[str] = None
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    swap_space: int = 4
    cpu_offload_gb: int = 0
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 64
    max_paddings: int = 64
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    
    # Internal preset handling
    _vllm_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Apply preset configurations after initialization"""
        self._apply_preset()
    
    def _apply_preset(self):
        """Apply preset-specific optimizations"""
        if self.preset == "performance":
            # Maximize throughput and performance
            self.gpu_memory_utilization = 0.95
            self.max_num_seqs = 256
            self.evaluation_batch_size = min(32, self.evaluation_batch_size * 4)
            self.benchmark_iterations = 5
            self.enable_prefix_caching = True
            self.use_v2_block_manager = True
            
        elif self.preset == "memory_optimized":
            # Optimize for memory efficiency
            self.gpu_memory_utilization = 0.75
            self.max_num_seqs = 32
            self.evaluation_batch_size = max(2, self.evaluation_batch_size // 2)
            self.cpu_offload_gb = 4
            self.swap_space = 8
            
        elif self.preset == "balanced":
            # Default balanced configuration (no changes needed)
            pass
            
    def get_vllm_config(self) -> Dict[str, Any]:
        """Generate vLLM configuration from this model config"""
        config = {
            "model": self.huggingface_id,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.torch_dtype,  # Changed from torch_dtype to dtype for vLLM compatibility
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            # "use_v2_block_manager": self.use_v2_block_manager,  # Not available in current vLLM version
            "enforce_eager": self.enforce_eager,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "block_size": self.block_size,
            "swap_space": self.swap_space,
            "cpu_offload_gb": self.cpu_offload_gb,
            "disable_log_stats": self.disable_log_stats,
        }
        
        # Add quantization if specified
        if self.quantization_method and self.quantization_method != "none":
            config["quantization"] = self.quantization_method
            
        # Add optional parameters if set
        if self.dtype:
            config["dtype"] = self.dtype
        if self.kv_cache_dtype:
            config["kv_cache_dtype"] = self.kv_cache_dtype
        if self.max_parallel_loading_workers:
            config["max_parallel_loading_workers"] = self.max_parallel_loading_workers
        if self.max_num_batched_tokens:
            config["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.revision:
            config["revision"] = self.revision
        if self.tokenizer_revision:
            config["tokenizer_revision"] = self.tokenizer_revision
            
        # Apply any manual overrides
        config.update(self._vllm_overrides)
        
        return config
    
    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM engine arguments (backward compatibility method)
        
        This method maintains compatibility with existing evaluation code
        that expects to_vllm_args() method on ModelConfig objects.
        """
        # Use the existing get_vllm_config method
        return self.get_vllm_config()
    
    def get_agent_sampling_params(self) -> Dict[str, Any]:
        """Get optimized sampling parameters for agent tasks"""
        return {
            "temperature": self.agent_temperature,
            "top_p": 0.9,
            "max_tokens": 2048,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["<|endoftext|>", "\n\nUser:", "\n\nHuman:"],
        }
    
    def apply_preset(self, preset: str) -> None:
        """Apply preset configuration (public method for evaluation engine)"""
        self.preset = preset
        self._apply_preset()
    
    def create_preset_variant(self, preset: str) -> 'ModelConfig':
        """Create a new config with different preset"""
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
        max_model_len=32768,  # Fixed: Match model's max_position_embeddings (32K)
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
        quantization_method="none",  # No AWQ version available
        max_model_len=32768,  # Increased to match other working models
        gpu_memory_utilization=0.85,  # Match working model pattern
        priority="HIGH",
        agent_optimized=True
    ),
    
    "codestral_22b": ModelConfig(
        model_name="Codestral 22B",
        huggingface_id="mistralai/Codestral-22B-v0.1",
        license="Mistral AI Non-Production License",
        size_gb=22.0,
        context_window=32768,
        quantization_method="none",  # No AWQ version available
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        specialization_category="code_generation",
        specialization_subcategory="coding_specialist",
        primary_use_cases=["code_generation", "code_completion", "code_analysis"]
    ),
    
    "qwen3_coder_30b": ModelConfig(
        model_name="Qwen3 Coder 30B",
        huggingface_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # Using the 32B instruct version
        license="Apache 2.0",
        size_gb=32.0,
        context_window=32768,
        quantization_method="none",  # No AWQ version available for this size
        max_model_len=32768,
        gpu_memory_utilization=0.90,  # Higher utilization for larger model
        priority="HIGH",
        agent_optimized=True,
        specialization_category="code_generation",
        specialization_subcategory="coding_specialist",
        primary_use_cases=["code_generation", "code_completion", "code_analysis"],
        max_num_seqs=32  # Lower batch size for larger model
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
    ),
    
    # Comparison Models (restricted licenses)
    "gemma2_9b": ModelConfig(
        model_name="Gemma 2 9B Instruct",
        huggingface_id="google/gemma-2-9b-it",
        license="Gemma License (Restricted)",
        size_gb=9.0,
        context_window=8192,
        quantization_method="awq", 
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        priority="LOW",
        agent_optimized=False
    ),
    
    "claude_sonnet": ModelConfig(
        model_name="Claude 3.5 Sonnet",
        huggingface_id="anthropic/claude-3-5-sonnet",  # Placeholder - would need API integration
        license="Commercial (API Only)",
        size_gb=0,  # API model
        context_window=200000,
        priority="LOW",
        agent_optimized=True
    )
}

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get model configuration by name"""
    return MODEL_CONFIGS.get(model_name)

def get_all_model_names() -> List[str]:
    """Get all available model names"""
    return list(MODEL_CONFIGS.keys())

def get_models_by_license(license_filter: str) -> Dict[str, ModelConfig]:
    """Get models by license type"""
    return {k: v for k, v in MODEL_CONFIGS.items() if license_filter.lower() in v.license.lower()}

def get_models_by_priority(priority: str) -> Dict[str, ModelConfig]:
    """Get models by priority level"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.priority == priority.upper()}