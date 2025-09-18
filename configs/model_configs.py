"""
Model configurations for individual LLM evaluation on H100 GPU
Optimized for 80GB VRAM and agentic system development

This module provides backward compatibility while the configuration system
has been refactored into separate modules for better organization.
"""

# Import core components from new modular structure
try:
    # Try relative imports first (when imported as module)
    from .model_registry import (
        ModelConfig, 
        MODEL_CONFIGS as BASE_MODEL_CONFIGS, 
        get_model_config,
        get_all_model_names,
        get_models_by_license,
        get_models_by_priority
    )

    from .preset_configs import (
        QWEN_MODEL_CONFIGS,
        ALL_MODEL_CONFIGS,
        create_qwen3_8b_configs,
        create_qwen3_14b_configs,
        create_qwen25_7b_configs,
        create_qwen3_coder_configs,
        get_all_qwen_variants,
        get_preset_variants
    )

    from .scientific_configs import (
        SCIENTIFIC_MODEL_CONFIGS,
        get_biomedical_models,
        get_scientific_embedding_models,
        get_document_understanding_models,
        get_long_context_models,
        get_safety_alignment_models,
        get_all_scientific_models,
        get_models_by_use_case,
        get_embedding_models,
        get_clinical_models,
        get_literature_models
    )

    from .config_validation import (
        get_coding_optimized_models,
        get_advanced_code_generation_models,
        get_data_science_models,
        get_mathematical_reasoning_models,
        get_text_generation_models,
        get_all_specialization_categories,
        get_models_by_specialization,
        get_genomic_optimized_models,
        get_multimodal_models,
        get_efficiency_models,
        _get_specialization,
        _get_specialization_from_config,
        recommend_config_for_task,
        get_high_priority_models,
        get_apache_licensed_models,
        get_agent_optimized_models,
        estimate_memory_usage,
        validate_model_config,
        get_all_model_configs,
        get_model_by_huggingface_id,
        get_models_by_size_range
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from configs.model_registry import (
        ModelConfig, 
        MODEL_CONFIGS as BASE_MODEL_CONFIGS, 
        get_model_config,
        get_all_model_names,
        get_models_by_license,
        get_models_by_priority
    )

    from configs.preset_configs import (
        QWEN_MODEL_CONFIGS,
        ALL_MODEL_CONFIGS,
        create_qwen3_8b_configs,
        create_qwen3_14b_configs,
        create_qwen25_7b_configs,
        create_qwen3_coder_configs,
        get_all_qwen_variants,
        get_preset_variants
    )

    from configs.scientific_configs import (
        SCIENTIFIC_MODEL_CONFIGS,
        get_biomedical_models,
        get_scientific_embedding_models,
        get_document_understanding_models,
        get_long_context_models,
        get_safety_alignment_models,
        get_all_scientific_models,
        get_models_by_use_case,
        get_embedding_models,
        get_clinical_models,
        get_literature_models
    )

    from configs.config_validation import (
        get_coding_optimized_models,
        get_advanced_code_generation_models,
        get_data_science_models,
        get_mathematical_reasoning_models,
        get_text_generation_models,
        get_all_specialization_categories,
        get_models_by_specialization,
        get_genomic_optimized_models,
        get_multimodal_models,
        get_efficiency_models,
        _get_specialization,
        _get_specialization_from_config,
        recommend_config_for_task,
        get_high_priority_models,
        get_apache_licensed_models,
        get_agent_optimized_models,
        estimate_memory_usage,
        validate_model_config,
        get_all_model_configs,
        get_model_by_huggingface_id,
        get_models_by_size_range
    )

# Maintain backward compatibility by making all configs available at module level
# This ensures existing code continues to work without modification
def _merge_all_configs():
    """Merge all configuration dictionaries for backward compatibility"""
    all_configs = {}
    all_configs.update(BASE_MODEL_CONFIGS)
    all_configs.update(QWEN_MODEL_CONFIGS)  
    all_configs.update(SCIENTIFIC_MODEL_CONFIGS)
    return all_configs

# Create unified MODEL_CONFIGS for backward compatibility
MODEL_CONFIGS = _merge_all_configs()

# Legacy comparison models section (if needed)
COMPARISON_MODELS = {
    "gemma2_9b": MODEL_CONFIGS.get("gemma2_9b"),
    "claude_sonnet": MODEL_CONFIGS.get("claude_sonnet")
}

# Remove None values from comparison models
COMPARISON_MODELS = {k: v for k, v in COMPARISON_MODELS.items() if v is not None}

# Test functionality if run directly
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
    print(f"Scientific models: {len(get_all_scientific_models())}")
    print(f"Total unique model configurations: {len(MODEL_CONFIGS)}")
    print()
    
    # Test validation
    print("=== CONFIGURATION VALIDATION TEST ===")
    validation_errors = 0
    for name, config in list(MODEL_CONFIGS.items())[:5]:  # Test first 5
        issues = validate_model_config(config)
        if issues:
            print(f"{name}: {len(issues)} issues found")
            validation_errors += len(issues)
        else:
            print(f"{name}: ✓ Valid")
    
    print(f"Total validation errors in sample: {validation_errors}")
    print()