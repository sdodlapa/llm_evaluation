"""
Model preset configurations and variants
Handles different configuration presets for performance, memory optimization, and balanced usage
"""

from typing import Dict
try:
    from .model_registry import ModelConfig, MODEL_CONFIGS
except ImportError:
    from configs.model_registry import ModelConfig, MODEL_CONFIGS

# Extended Model Configurations with Qwen Variants
QWEN_MODEL_CONFIGS = {
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
    
    # Qwen3-Coder - Specialized coding model (using 7B AWQ since 30B AWQ doesn't exist)
    "qwen3_coder_30b": ModelConfig(
        model_name="Qwen2.5-Coder 7B Instruct",
        huggingface_id="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",  # Use available AWQ version
        license="Apache 2.0",
        size_gb=7.0,  # Updated size for 7B model
        context_window=128000,
        preset="performance",  # Maximize coding performance
        quantization_method="awq_marlin",  # Keep AWQ with correct model
        max_model_len=32768,
        gpu_memory_utilization=0.85,  # Reduced for smaller model
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.05,  # Very low for precise coding
        max_function_calls_per_turn=8,  # More calls for complex coding tasks
        evaluation_batch_size=8,  # Increased batch for smaller model
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
    
    # Specialized Genomic Models
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
    
    # Additional Strategic Models
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
        agent_temperature=0.1,
        max_function_calls_per_turn=5,
        evaluation_batch_size=8
    ),
    
    # Research Models
    "olmo2_13b_research": ModelConfig(
        model_name="OLMo-2 13B Instruct (Research)",
        huggingface_id="allenai/OLMo-2-1124-13B-Instruct",
        license="Apache 2.0",
        size_gb=13.0,
        context_window=4096,
        preset="balanced",
        quantization_method="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=False,  # Research focus
        agent_temperature=0.2,
        evaluation_batch_size=8
    ),
    
    "yi_1_5_34b": ModelConfig(
        model_name="Yi-1.5 34B Chat",
        huggingface_id="01-ai/Yi-1.5-34B-Chat-AWQ",
        license="Apache 2.0", 
        size_gb=34.0,
        context_window=4096,
        preset="memory_optimized",  # Large model needs careful memory management
        quantization_method="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        priority="LOW",
        agent_optimized=False,
        agent_temperature=0.1,
        evaluation_batch_size=2
    ),
    
    # Large Mathematical Model
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
    )
}

def create_qwen3_8b_configs() -> Dict[str, ModelConfig]:
    """Create all preset variants for Qwen3 8B"""
    base_config = QWEN_MODEL_CONFIGS["qwen25_7b"]  # Using 2.5 as base
    return {
        "qwen3_8b_balanced": base_config,
        "qwen3_8b_performance": base_config.create_preset_variant("performance"),
        "qwen3_8b_memory_optimized": base_config.create_preset_variant("memory_optimized")
    }

def create_qwen3_14b_configs() -> Dict[str, ModelConfig]:
    """Create all preset variants for Qwen3 14B"""
    base_config = MODEL_CONFIGS["qwen3_14b"]
    return {
        "qwen3_14b_balanced": base_config.create_preset_variant("balanced"),
        "qwen3_14b_performance": base_config,
        "qwen3_14b_memory_optimized": base_config.create_preset_variant("memory_optimized")
    }

def create_qwen25_7b_configs() -> Dict[str, ModelConfig]:
    """Create all preset variants for Qwen2.5 7B"""
    base_config = QWEN_MODEL_CONFIGS["qwen25_7b"]
    return {
        "qwen25_7b_balanced": base_config,
        "qwen25_7b_performance": base_config.create_preset_variant("performance"),
        "qwen25_7b_memory_optimized": base_config.create_preset_variant("memory_optimized")
    }

def create_qwen3_coder_configs() -> Dict[str, ModelConfig]:
    """Create all preset variants for Qwen3 Coder"""
    base_config = QWEN_MODEL_CONFIGS["qwen3_coder_30b"]
    return {
        "qwen3_coder_balanced": base_config.create_preset_variant("balanced"),
        "qwen3_coder_performance": base_config,
        "qwen3_coder_memory_optimized": base_config.create_preset_variant("memory_optimized")
    }

def get_all_qwen_variants() -> Dict[str, ModelConfig]:
    """Get all Qwen model variants with all presets"""
    variants = {}
    variants.update(create_qwen3_8b_configs())
    variants.update(create_qwen3_14b_configs())
    variants.update(create_qwen25_7b_configs())
    variants.update(create_qwen3_coder_configs())
    
    # Add individual models from QWEN_MODEL_CONFIGS
    variants.update(QWEN_MODEL_CONFIGS)
    
    return variants

def get_preset_variants(model_name: str) -> Dict[str, ModelConfig]:
    """Get all preset variants for a specific model"""
    if model_name in ["qwen3_8b", "qwen25_7b"]:
        return create_qwen25_7b_configs()
    elif model_name == "qwen3_14b":
        return create_qwen3_14b_configs()
    elif model_name == "qwen3_coder_30b":
        return create_qwen3_coder_configs()
    else:
        # Return base model with all presets
        base_config = MODEL_CONFIGS.get(model_name) or QWEN_MODEL_CONFIGS.get(model_name)
        if base_config:
            return {
                f"{model_name}_balanced": base_config.create_preset_variant("balanced"),
                f"{model_name}_performance": base_config.create_preset_variant("performance"),
                f"{model_name}_memory_optimized": base_config.create_preset_variant("memory_optimized")
            }
        return {}

# Merge all configurations
ALL_MODEL_CONFIGS = {**MODEL_CONFIGS, **QWEN_MODEL_CONFIGS}