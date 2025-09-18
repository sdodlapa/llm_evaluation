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
    _vllm_overrides: Dict[str, Any] = field(default_factory=dict)
    
    
    def apply_preset(self):
        """Apply preset-specific optimizations"""
        if self.preset == "performance":
            # Maximize speed, higher memory usage - H100 optimized
            # Ensure max_model_len doesn't exceed model capabilities (32K for Qwen models)
            self.max_model_len = min(self.max_model_len, 32768)  # Safe limit for all models
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
                # "use_v2_block_manager": True,  # Not available in vLLM 0.10.2
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
                "block_size": 16,  # Smaller blocks (must be multiple of 16 for Flash Attention)
            })
            
        else:  # balanced (default)
            # Optimal balance - ensure safe max_model_len for all models
            self.max_model_len = min(self.max_model_len, 32768)  # Safe limit for all models
            # Other defaults remain unchanged
    
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
    ),
    
    # === NEW RECOMMENDED QWEN MODELS ===
    
    # Qwen2.5-7B - Popular baseline model
    "qwen25_7b": ModelConfig(
        model_name="Qwen2.5 7B Instruct",
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        license="Apache 2.0",
        size_gb=7.0,
        context_window=128000,
        preset="balanced",
        quantization_method="none",  # Excellent performance without quantization
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.1,
        max_function_calls_per_turn=5,
        evaluation_batch_size=8
    ),
    
    # Qwen3-Coder - Specialized coding model
    "qwen3_coder_30b": ModelConfig(
        model_name="Qwen3-Coder 30B-A3B Instruct",
        huggingface_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        license="Apache 2.0",
        size_gb=31.0,  # MoE with 3B active parameters
        context_window=128000,
        preset="performance",  # Maximize coding performance
        quantization_method="awq_marlin",  # Use AWQ for efficiency
        max_model_len=32768,
        gpu_memory_utilization=0.88,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.05,  # Very low for precise coding
        max_function_calls_per_turn=8,  # More calls for complex coding tasks
        evaluation_batch_size=4,  # Smaller batch for large model
        benchmark_iterations=5
    ),
    
    # Qwen2-VL - Multimodal vision-language model
    "qwen2_vl_7b": ModelConfig(
        model_name="Qwen2-VL 7B Instruct",
        huggingface_id="Qwen/Qwen2-VL-7B-Instruct",
        license="Apache 2.0",
        size_gb=8.5,  # Slightly larger due to vision components
        context_window=128000,
        preset="balanced",
        quantization_method="none",  # Vision models work best unquantized
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.1,
        max_function_calls_per_turn=5,
        evaluation_batch_size=6  # Slightly smaller due to vision processing
    ),
    
    # Qwen2.5-Math - Mathematics specialist
    "qwen25_math_7b": ModelConfig(
        model_name="Qwen2.5-Math 7B Instruct",
        huggingface_id="Qwen/Qwen2.5-Math-7B-Instruct",
        license="Apache 2.0",
        size_gb=7.0,
        context_window=128000,
        preset="balanced",
        quantization_method="none",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.05,  # Very precise for mathematical reasoning
        max_function_calls_per_turn=6,  # Extra calls for multi-step math
        evaluation_batch_size=8
    ),
    
    # Qwen2.5-0.5B - Efficient small model
    "qwen25_0_5b": ModelConfig(
        model_name="Qwen2.5 0.5B Instruct", 
        huggingface_id="Qwen/Qwen2.5-0.5B-Instruct",
        license="Apache 2.0",
        size_gb=0.5,
        context_window=128000,
        preset="performance",  # Can afford to be aggressive with small model
        quantization_method="none",  # No need to quantize small model
        max_model_len=32768,
        gpu_memory_utilization=0.95,  # Can use high memory since model is tiny
        priority="MEDIUM",
        agent_optimized=False,  # Too small for complex agents
        agent_temperature=0.2,
        max_function_calls_per_turn=3,
        evaluation_batch_size=32  # Large batches for throughput
    ),
    
    # Qwen2.5-3B - Medium efficiency model
    "qwen25_3b": ModelConfig(
        model_name="Qwen2.5 3B Instruct",
        huggingface_id="Qwen/Qwen2.5-3B-Instruct", 
        license="Apache 2.0",
        size_gb=3.0,
        context_window=128000,
        preset="balanced",
        quantization_method="none",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        priority="HIGH",
        agent_optimized=True,  # Good balance of size and capability
        agent_temperature=0.1,
        max_function_calls_per_turn=4,
        evaluation_batch_size=16
    ),
    
    # === SPECIALIZED MODELS FOR GENOMIC DATA ===
    
    # Qwen2.5-1.5B - Efficient for genomic sequence analysis
    "qwen25_1_5b_genomic": ModelConfig(
        model_name="Qwen2.5 1.5B Instruct (Genomic Optimized)",
        huggingface_id="Qwen/Qwen2.5-1.5B-Instruct",
        license="Apache 2.0",
        size_gb=1.5,
        context_window=128000,  # Large context for long genomic sequences
        preset="performance",
        quantization_method="none",
        max_model_len=65536,  # Extended context for genomic data
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        agent_temperature=0.01,  # Very precise for genomic analysis
        max_function_calls_per_turn=10,  # Many calls for genomic pipelines
        evaluation_batch_size=8
    ),
    
    # Qwen2.5-72B-AWQ - Large model for complex genomic reasoning
    "qwen25_72b_genomic": ModelConfig(
        model_name="Qwen2.5 72B Instruct (Genomic Analysis)",
        huggingface_id="Qwen/Qwen2.5-72B-Instruct-AWQ",
        license="Apache 2.0",
        size_gb=72.0,  # Large model
        context_window=128000,
        preset="memory_optimized",  # Need to be careful with 72B model
        quantization_method="awq_marlin",
        max_model_len=32768,  # Conservative for large model
        gpu_memory_utilization=0.78,  # Conservative for 72B
        priority="LOW",  # Resource intensive
        agent_optimized=True,
        agent_temperature=0.01,  # Extremely precise
        max_function_calls_per_turn=15,  # Complex genomic workflows
        evaluation_batch_size=1  # Single sample processing for large model
    ),
    
    # === STRATEGIC NON-QWEN ADDITIONS (FILL EVALUATION GAPS) ===
    
    # Mistral-NeMo 12B - Long context specialist (128K) for comparison
    "mistral_nemo_12b": ModelConfig(
        model_name="Mistral-NeMo 12B Instruct",
        huggingface_id="nvidia/Mistral-NeMo-12B-Instruct",
        license="Apache 2.0",
        size_gb=12.0,
        context_window=128000,  # Key feature: 128K context
        preset="balanced",
        quantization_method="awq",
        max_model_len=32768,  # Conservative but can handle long contexts
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.1,
        max_function_calls_per_turn=5,
        evaluation_batch_size=6
    ),
    
    # IBM Granite 3.1 8B - Production-focused enterprise model
    "granite_3_1_8b": ModelConfig(
        model_name="IBM Granite 3.1 8B Instruct",
        huggingface_id="ibm-granite/granite-3.1-8b-instruct",
        license="Apache 2.0",
        size_gb=8.0,
        context_window=128000,  # Long context support
        preset="balanced",
        quantization_method="awq",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.05,  # Production precision
        max_function_calls_per_turn=6,  # Enterprise workflow support
        evaluation_batch_size=8
    ),
    
    # OLMo-2 13B - Research transparency standard
    "olmo2_13b_research": ModelConfig(
        model_name="OLMo-2 13B Instruct (Research Standard)",
        huggingface_id="allenai/OLMo-2-1124-13B-Instruct",
        license="Apache 2.0",
        size_gb=13.0,
        context_window=4096,  # Limited but sufficient for research
        preset="balanced",
        quantization_method="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=False,  # Research-focused, not production agents
        agent_temperature=0.2,  # Research exploration
        max_function_calls_per_turn=3,
        evaluation_batch_size=6
    ),
    
    # Yi-1.5 34B - Multilingual powerhouse
    "yi_1_5_34b": ModelConfig(
        model_name="Yi-1.5 34B Chat",
        huggingface_id="01-ai/Yi-1.5-34B-Chat",
        license="Apache 2.0",
        size_gb=34.0,
        context_window=32768,
        preset="memory_optimized",  # Large model, need careful memory management
        quantization_method="awq_marlin",
        max_model_len=24576,
        gpu_memory_utilization=0.80,  # Conservative for 34B
        priority="MEDIUM",
        agent_optimized=True,
        agent_temperature=0.1,
        max_function_calls_per_turn=5,
        evaluation_batch_size=3  # Smaller batches for large model
    ),
    
    # === ADVANCED SPECIALIZED MODELS (FILL REMAINING GAPS) ===
    
    # CodeLlama-34B-Instruct - Advanced code generation specialist
    "codellama_34b_instruct": ModelConfig(
        model_name="CodeLlama 34B Instruct",
        huggingface_id="codellama/CodeLlama-34b-Instruct-hf",
        license="Custom CodeLlama License",
        size_gb=34.0,
        context_window=16384,  # CodeLlama context limit
        preset="memory_optimized",  # Large model needs careful memory management
        quantization_method="awq_marlin",
        max_model_len=16384,
        gpu_memory_utilization=0.80,  # Conservative for 34B
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.02,  # Very precise for code generation
        max_function_calls_per_turn=12,  # Complex coding workflows
        evaluation_batch_size=2  # Small batches for large model
    ),
    
    # CodeLlama-34B-Python - Data science and analytics specialist
    "codellama_34b_python": ModelConfig(
        model_name="CodeLlama 34B Python",
        huggingface_id="codellama/CodeLlama-34b-Python-hf",
        license="Custom CodeLlama License",
        size_gb=34.0,
        context_window=16384,
        preset="memory_optimized",
        quantization_method="awq_marlin",
        max_model_len=16384,
        gpu_memory_utilization=0.80,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.01,  # Extremely precise for data analysis
        max_function_calls_per_turn=15,  # Complex data science pipelines
        evaluation_batch_size=2
    ),
    
    # StarCoder2-15B - Advanced code generation (alternative to CodeLlama)
    "starcoder2_15b": ModelConfig(
        model_name="StarCoder2 15B",
        huggingface_id="bigcode/starcoder2-15b",
        license="OpenRAIL-M",
        size_gb=15.0,
        context_window=16384,
        preset="balanced",
        quantization_method="awq",
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        agent_temperature=0.05,  # Precise code generation
        max_function_calls_per_turn=8,
        evaluation_batch_size=4
    ),
    
    # DeepSeek-Coder-V2-Lite-16B - Already have but optimize for advanced coding
    "deepseek_coder_v2_advanced": ModelConfig(
        model_name="DeepSeek-Coder-V2-Lite 16B (Advanced Optimization)",
        huggingface_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        license="Custom (Commercial OK)",
        size_gb=16.0,
        context_window=128000,
        preset="performance",  # Optimize for advanced coding
        quantization_method="awq_marlin",
        max_model_len=32768,  # Extended context for complex code
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.01,  # Very precise
        max_function_calls_per_turn=10,
        evaluation_batch_size=4
    ),
    
    # WizardMath-70B - Advanced mathematical reasoning (if we have VRAM)
    "wizardmath_70b": ModelConfig(
        model_name="WizardMath 70B V1.1",
        huggingface_id="WizardLM/WizardMath-70B-V1.1",
        license="Custom License",
        size_gb=70.0,
        context_window=4096,  # Limited context but powerful math
        preset="memory_optimized",  # Essential for 70B
        quantization_method="awq_marlin",
        max_model_len=4096,
        gpu_memory_utilization=0.75,  # Very conservative
        priority="LOW",  # Resource intensive
        agent_optimized=False,  # Math specialist, not general agent
        agent_temperature=0.01,
        max_function_calls_per_turn=20,  # Complex mathematical workflows
        evaluation_batch_size=1
    ),
    
    # === SCIENTIFIC & BIOMEDICAL MODELS ===
    
    # Biomedical Literature Models
    "biomistral_7b": ModelConfig(
        model_name="BioMistral 7B Instruct",
        huggingface_id="BioMistral/BioMistral-7B-Instruct",
        license="Apache 2.0",
        size_gb=7.0,
        context_window=32768,
        specialization_category="biomedical",
        specialization_subcategory="literature_qa",
        primary_use_cases=["biomedical_qa", "literature_summarization", "medical_reasoning"],
        preset="balanced",
        quantization_method="awq",
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.1,
        max_function_calls_per_turn=6,
        evaluation_batch_size=8
    ),
    
    "biogpt_large": ModelConfig(
        model_name="BioGPT Large",
        huggingface_id="microsoft/BioGPT-Large",
        license="MIT",
        size_gb=1.6,
        context_window=1024,
        specialization_category="biomedical", 
        specialization_subcategory="pubmed_generation",
        primary_use_cases=["pubmed_qa", "relation_extraction", "biomedical_ner"],
        preset="balanced",
        quantization_method="none",  # Small enough
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        priority="MEDIUM",
        agent_optimized=False,  # Specialized generation tool
        agent_temperature=0.2,
        evaluation_batch_size=16
    ),
    
    "clinical_t5_large": ModelConfig(
        model_name="Clinical T5 Large",
        huggingface_id="microsoft/clinical-t5-large",
        license="MIT",
        size_gb=3.0,
        context_window=512,
        specialization_category="biomedical",
        specialization_subcategory="clinical_text",
        primary_use_cases=["clinical_notes", "medical_summarization"],
        preset="balanced",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        agent_temperature=0.05,  # High precision for clinical
        evaluation_batch_size=12
    ),
    
    # Scientific Embedding Models
    "specter2_base": ModelConfig(
        model_name="SPECTER2 Scientific Paper Embeddings",
        huggingface_id="allenai/specter2_base",
        license="Apache 2.0",
        size_gb=0.4,
        context_window=512,
        specialization_category="scientific_embeddings",
        specialization_subcategory="paper_retrieval",
        primary_use_cases=["scientific_rag", "paper_similarity", "literature_search"],
        preset="performance",  # Optimized for throughput
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.95,
        priority="HIGH",
        agent_optimized=True,
        evaluation_batch_size=64  # High throughput for embeddings
    ),
    
    "scibert_base": ModelConfig(
        model_name="SciBERT Scientific Text Encoder",
        huggingface_id="allenai/scibert_scivocab_uncased",
        license="Apache 2.0",
        size_gb=0.4,
        context_window=512,
        specialization_category="scientific_embeddings",
        specialization_subcategory="scientific_text",
        primary_use_cases=["scientific_classification", "rag_retrieval", "concept_extraction"],
        preset="performance",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.95,
        priority="MEDIUM",
        agent_optimized=True,
        evaluation_batch_size=64
    ),
    
    # Document Understanding Models
    "donut_base": ModelConfig(
        model_name="Donut Document Understanding",
        huggingface_id="naver-clova-ix/donut-base",
        license="MIT",
        size_gb=0.8,
        context_window=1024,
        specialization_category="document_understanding",
        specialization_subcategory="ocr_free_vqa",
        primary_use_cases=["document_parsing", "form_understanding", "table_extraction"],
        preset="balanced",
        quantization_method="none",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        agent_temperature=0.1,
        evaluation_batch_size=16
    ),
    
    "layoutlmv3_base": ModelConfig(
        model_name="LayoutLMv3 Base",
        huggingface_id="microsoft/layoutlmv3-base",
        license="MIT",
        size_gb=0.5,
        context_window=512,
        specialization_category="document_understanding",
        specialization_subcategory="layout_analysis",
        primary_use_cases=["document_classification", "layout_analysis", "entity_extraction"],
        preset="balanced",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.90,
        priority="MEDIUM",
        agent_optimized=True,
        evaluation_batch_size=24
    ),
    
    # Strategic Gap Models
    "longformer_large": ModelConfig(
        model_name="Longformer Large",
        huggingface_id="allenai/longformer-large-4096",
        license="Apache 2.0",
        size_gb=1.3,
        context_window=4096,
        specialization_category="long_context",
        specialization_subcategory="document_analysis",
        primary_use_cases=["long_document_qa", "research_paper_analysis"],
        preset="balanced",
        quantization_method="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        evaluation_batch_size=8
    ),
    
    "safety_bert": ModelConfig(
        model_name="Safety BERT Classifier",
        huggingface_id="unitary/toxic-bert",
        license="Apache 2.0",
        size_gb=0.4,
        context_window=512,
        specialization_category="safety_alignment",
        specialization_subcategory="toxicity_detection",
        primary_use_cases=["safety_classification", "bias_detection"],
        preset="performance",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.95,
        priority="LOW",
        agent_optimized=False,  # Classification tool
        evaluation_batch_size=32
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

def create_qwen25_7b_configs() -> Dict[str, ModelConfig]:
    """Create Qwen2.5 7B configurations for all presets"""
    base_config = MODEL_CONFIGS["qwen25_7b"]
    
    return {
        "qwen25_7b_balanced": base_config,
        "qwen25_7b_performance": base_config.create_preset_variant("performance"),
        "qwen25_7b_memory_optimized": base_config.create_preset_variant("memory_optimized"),
    }

def create_qwen3_coder_configs() -> Dict[str, ModelConfig]:
    """Create Qwen3-Coder configurations for all presets"""
    base_config = MODEL_CONFIGS["qwen3_coder_30b"]
    
    return {
        "qwen3_coder_30b_balanced": base_config,
        "qwen3_coder_30b_performance": base_config.create_preset_variant("performance"),
        "qwen3_coder_30b_memory_optimized": base_config.create_preset_variant("memory_optimized"),
    }

def get_all_qwen_variants() -> Dict[str, ModelConfig]:
    """Get all Qwen model variants across all presets"""
    all_configs = {}
    all_configs.update(create_qwen3_8b_configs())
    all_configs.update(create_qwen3_14b_configs())
    all_configs.update(create_qwen25_7b_configs())
    all_configs.update(create_qwen3_coder_configs())
    return all_configs

def get_coding_optimized_models() -> Dict[str, ModelConfig]:
    """Get models optimized for coding tasks"""
    coding_models = {
        "qwen3_coder_30b": MODEL_CONFIGS["qwen3_coder_30b"],
        "qwen25_7b": MODEL_CONFIGS["qwen25_7b"],  # Excellent general coding
        "qwen3_8b": MODEL_CONFIGS["qwen3_8b"],    # Good baseline coding
        "deepseek_coder_16b": MODEL_CONFIGS["deepseek_coder_16b"],  # Original DeepSeek
    }
    return coding_models

def get_advanced_code_generation_models() -> Dict[str, ModelConfig]:
    """Get models specialized for advanced code generation"""
    return {
        "codellama_34b_instruct": MODEL_CONFIGS["codellama_34b_instruct"],
        "starcoder2_15b": MODEL_CONFIGS["starcoder2_15b"],
        "deepseek_coder_v2_advanced": MODEL_CONFIGS["deepseek_coder_v2_advanced"],
        "qwen3_coder_30b": MODEL_CONFIGS["qwen3_coder_30b"],  # Include for comparison
    }

def get_data_science_models() -> Dict[str, ModelConfig]:
    """Get models optimized for data science and analytics"""
    return {
        "codellama_34b_python": MODEL_CONFIGS["codellama_34b_python"],
        "qwen25_math_7b": MODEL_CONFIGS["qwen25_math_7b"],  # Mathematical foundation
        "granite_3_1_8b": MODEL_CONFIGS["granite_3_1_8b"],  # Enterprise analytics
    }

def get_mathematical_reasoning_models() -> Dict[str, ModelConfig]:
    """Get models specialized for mathematical reasoning"""
    return {
        "qwen25_math_7b": MODEL_CONFIGS["qwen25_math_7b"],
        "wizardmath_70b": MODEL_CONFIGS["wizardmath_70b"],
        "qwen25_7b": MODEL_CONFIGS["qwen25_7b"],  # General math capability
    }

def get_text_generation_models() -> Dict[str, ModelConfig]:
    """Get models optimized for general text generation and reasoning"""
    return {
        "qwen25_7b": MODEL_CONFIGS["qwen25_7b"],
        "qwen3_8b": MODEL_CONFIGS["qwen3_8b"],
        "qwen3_14b": MODEL_CONFIGS["qwen3_14b"],
        "mistral_nemo_12b": MODEL_CONFIGS["mistral_nemo_12b"],
        "granite_3_1_8b": MODEL_CONFIGS["granite_3_1_8b"],
        "olmo2_13b_research": MODEL_CONFIGS["olmo2_13b_research"],
        "yi_1_5_34b": MODEL_CONFIGS["yi_1_5_34b"],
    }

def get_all_specialization_categories() -> Dict[str, Dict[str, ModelConfig]]:
    """Get all models organized by specialization category"""
    return {
        "text_generation": get_text_generation_models(),
        "code_generation": get_advanced_code_generation_models(),
        "data_science": get_data_science_models(), 
        "mathematics": get_mathematical_reasoning_models(),
        "bioinformatics": get_genomic_optimized_models(),
        "multimodal": get_multimodal_models(),
        "efficiency": get_efficiency_models(),
        "research": {"olmo2_13b_research": MODEL_CONFIGS["olmo2_13b_research"]}
    }

def get_models_by_specialization(category: str) -> Dict[str, ModelConfig]:
    """Get models by specialization category"""
    all_categories = get_all_specialization_categories()
    return all_categories.get(category, {})

def get_genomic_optimized_models() -> Dict[str, ModelConfig]:
    """Get models optimized for genomic data analysis"""
    genomic_models = {
        "qwen25_1_5b_genomic": MODEL_CONFIGS["qwen25_1_5b_genomic"],
        "qwen25_72b_genomic": MODEL_CONFIGS["qwen25_72b_genomic"],
        "qwen25_math_7b": MODEL_CONFIGS["qwen25_math_7b"],  # Mathematical reasoning for genomics
        "qwen2_vl_7b": MODEL_CONFIGS["qwen2_vl_7b"],        # For genomic visualization
    }
    return genomic_models

def get_multimodal_models() -> Dict[str, ModelConfig]:
    """Get models with multimodal capabilities"""
    return {
        "qwen2_vl_7b": MODEL_CONFIGS["qwen2_vl_7b"],
    }

def get_efficiency_models() -> Dict[str, ModelConfig]:
    """Get small efficient models for resource-constrained scenarios"""
    return {
        "qwen25_0_5b": MODEL_CONFIGS["qwen25_0_5b"],
        "qwen25_1_5b_genomic": MODEL_CONFIGS["qwen25_1_5b_genomic"],
        "qwen25_3b": MODEL_CONFIGS["qwen25_3b"],
    }

def _get_specialization(model_name: str) -> str:
    """Get specialization description for a model"""
    if "coder" in model_name:
        return "Coding & Software Development"
    elif "math" in model_name:
        return "Mathematics & Reasoning"
    elif "vl" in model_name:
        return "Vision-Language (Multimodal)"
    elif "genomic" in model_name:
        return "Genomic Data Analysis"
    elif "0_5b" in model_name:
        return "Efficiency & Edge Deployment"
    elif "3b" in model_name:
        return "Balanced Efficiency & Performance"
    elif "72b" in model_name:
        return "Maximum Performance & Complex Reasoning"
    else:
        return "General Purpose"

def _get_specialization_from_config(config: ModelConfig) -> str:
    """Get specialization from config object"""
    model_id = config.huggingface_id.lower()
    if "coder" in model_id:
        return "Coding Specialist"
    elif "math" in model_id:
        return "Mathematics Specialist"
    elif "vl" in model_id:
        return "Vision-Language"
    elif config.size_gb >= 50:
        return "Large Scale Reasoning"
    elif config.size_gb <= 2:
        return "Efficiency Optimized"
    else:
        return "General Purpose"

def recommend_config_for_task(task_type: str, available_memory_gb: int = 80) -> Optional[ModelConfig]:
    """Recommend optimal configuration based on task and available memory"""
    if task_type == "agent_development":
        # Prioritize balanced configs with good agent capabilities
        candidates = [MODEL_CONFIGS["qwen25_7b"], MODEL_CONFIGS["qwen3_8b"], MODEL_CONFIGS["qwen3_14b"]]
    elif task_type == "coding":
        # Use coding-specialized models
        candidates = [MODEL_CONFIGS["qwen3_coder_30b"], MODEL_CONFIGS["qwen25_7b"], MODEL_CONFIGS["qwen3_8b"]]
    elif task_type == "genomic_analysis":
        # Use genomic-optimized models
        candidates = [MODEL_CONFIGS["qwen25_1_5b_genomic"], MODEL_CONFIGS["qwen25_math_7b"], MODEL_CONFIGS["qwen25_72b_genomic"]]
    elif task_type == "multimodal":
        # Use vision-language models
        candidates = [MODEL_CONFIGS["qwen2_vl_7b"]]
    elif task_type == "mathematics":
        # Use math-specialized models
        candidates = [MODEL_CONFIGS["qwen25_math_7b"], MODEL_CONFIGS["qwen25_7b"]]
    elif task_type == "efficiency":
        # Use small efficient models
        candidates = [MODEL_CONFIGS["qwen25_3b"], MODEL_CONFIGS["qwen25_1_5b_genomic"], MODEL_CONFIGS["qwen25_0_5b"]]
    elif task_type == "performance_testing":
        # Use performance presets
        candidates = [
            MODEL_CONFIGS["qwen3_coder_30b"].create_preset_variant("performance"),
            MODEL_CONFIGS["qwen25_7b"].create_preset_variant("performance"),
            MODEL_CONFIGS["qwen3_14b"].create_preset_variant("performance")
        ]
    elif task_type == "memory_constrained":
        # Use memory-optimized presets
        candidates = [
            MODEL_CONFIGS["qwen25_3b"].create_preset_variant("memory_optimized"),
            MODEL_CONFIGS["qwen25_7b"].create_preset_variant("memory_optimized"),
            MODEL_CONFIGS["qwen3_8b"].create_preset_variant("memory_optimized")
        ]
    else:
        # Default to balanced approach with new recommended models
        candidates = [MODEL_CONFIGS["qwen25_7b"], MODEL_CONFIGS["qwen3_8b"], MODEL_CONFIGS["qwen25_3b"]]
    
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

def get_biomedical_models() -> Dict[str, ModelConfig]:
    """Get biomedical specialist models"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.specialization_category == "biomedical"}

def get_scientific_embedding_models() -> Dict[str, ModelConfig]:
    """Get scientific embedding models for RAG and retrieval"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.specialization_category == "scientific_embeddings"}

def get_document_understanding_models() -> Dict[str, ModelConfig]:
    """Get document understanding and analysis models"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.specialization_category == "document_understanding"}

def get_long_context_models() -> Dict[str, ModelConfig]:
    """Get long context specialist models"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.specialization_category == "long_context"}

def get_safety_alignment_models() -> Dict[str, ModelConfig]:
    """Get safety and alignment specialist models"""
    return {k: v for k, v in MODEL_CONFIGS.items() if v.specialization_category == "safety_alignment"}

def get_all_scientific_models() -> Dict[str, ModelConfig]:
    """Get all scientific and biomedical specialist models"""
    scientific_categories = ["biomedical", "scientific_embeddings", "document_understanding", "long_context", "safety_alignment"]
    return {k: v for k, v in MODEL_CONFIGS.items() if v.specialization_category in scientific_categories}

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
    print("=== Enhanced Model Configuration System with New Qwen Models ===")
    print()
    
    # Show all new Qwen models added
    print("=== NEW QWEN MODELS ADDED ===")
    new_models = [
        "qwen25_7b", "qwen3_coder_30b", "qwen2_vl_7b", "qwen25_math_7b", 
        "qwen25_0_5b", "qwen25_3b", "qwen25_1_5b_genomic", "qwen25_72b_genomic"
    ]
    
    for model_name in new_models:
        if model_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_name]
            memory_est = estimate_memory_usage(config)
            print(f"{model_name}: {config.model_name}")
            print(f"  Size: {config.size_gb}GB, License: {config.license}")
            print(f"  Specialized for: {_get_specialization(model_name)}")
            print(f"  Priority: {config.priority}, Agent Temp: {config.agent_temperature}")
            print(f"  Estimated VRAM: {memory_est['total_estimated_gb']:.1f}GB ({memory_est['h100_utilization']:.1%})")
            print()
    
    # Test specialized model groups
    print("=== CODING-OPTIMIZED MODELS ===")
    coding_models = get_coding_optimized_models()
    for name, config in coding_models.items():
        memory_est = estimate_memory_usage(config)
        print(f"{name}: {config.model_name}")
        print(f"  Agent Temp: {config.agent_temperature} (lower = more precise)")
        print(f"  Function Calls: {config.max_function_calls_per_turn}")
        print(f"  VRAM: {memory_est['total_estimated_gb']:.1f}GB")
        print()
    
    print("=== GENOMIC-OPTIMIZED MODELS ===")
    genomic_models = get_genomic_optimized_models()
    for name, config in genomic_models.items():
        memory_est = estimate_memory_usage(config)
        print(f"{name}: {config.model_name}")
        print(f"  Context Window: {config.context_window:,} tokens")
        print(f"  Max Model Len: {config.max_model_len:,} tokens")
        print(f"  Agent Temp: {config.agent_temperature} (precision for genomics)")
        print(f"  VRAM: {memory_est['total_estimated_gb']:.1f}GB")
        print()
    
    print("=== EFFICIENCY MODELS ===")
    efficiency_models = get_efficiency_models()
    for name, config in efficiency_models.items():
        memory_est = estimate_memory_usage(config)
        print(f"{name}: {config.model_name}")
        print(f"  Size: {config.size_gb}GB, Batch Size: {config.evaluation_batch_size}")
        print(f"  VRAM: {memory_est['total_estimated_gb']:.1f}GB ({memory_est['h100_utilization']:.1%})")
        print()
    
    # Test task-specific recommendations
    print("=== TASK-SPECIFIC RECOMMENDATIONS ===")
    
    task_scenarios = [
        ("coding", 80),
        ("genomic_analysis", 80), 
        ("multimodal", 80),
        ("mathematics", 80),
        ("efficiency", 40),
        ("agent_development", 80),
    ]
    
    for task, memory in task_scenarios:
        recommended = recommend_config_for_task(task, memory)
        if recommended:
            memory_est = estimate_memory_usage(recommended)
            print(f"Task: {task} ({memory}GB available)")
            print(f"  Recommended: {recommended.model_name}")
            print(f"  Preset: {recommended.preset}")
            print(f"  Specialization: {_get_specialization_from_config(recommended)}")
            print(f"  VRAM Usage: {memory_est['total_estimated_gb']:.1f}GB")
            print()
        else:
            print(f"Task: {task} ({memory}GB available) - No suitable config found")
            print()
    
    # Show comprehensive model count
    print("=== COMPREHENSIVE MODEL INVENTORY ===")
    all_variants = get_all_qwen_variants()
    print(f"Total Qwen model variants (all presets): {len(all_variants)}")
    print(f"Base Qwen models: {len([k for k in MODEL_CONFIGS.keys() if 'qwen' in k])}")
    print(f"Coding models: {len(get_coding_optimized_models())}")
    print(f"Genomic models: {len(get_genomic_optimized_models())}")
    print(f"Multimodal models: {len(get_multimodal_models())}")
    print(f"Efficiency models: {len(get_efficiency_models())}")
    print(f"High priority models: {len(get_high_priority_models())}")
    print()


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get model configuration by name"""
    return MODEL_CONFIGS.get(model_name)