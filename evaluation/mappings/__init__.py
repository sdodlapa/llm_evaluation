"""
Category-based Model-Dataset Mappings
====================================

This module provides category-based mapping between models and datasets,
enabling systematic evaluation of model categories against appropriate datasets.

Main Components:
- ModelCategory: Category definition with models, datasets, and configs
- CategoryMappingManager: Orchestrates category-based evaluations
- CODING_SPECIALISTS: First category implementation
"""

# Core category system
from .model_categories import (
    ModelCategory,
    CODING_SPECIALISTS,
    CATEGORY_REGISTRY,
    get_category_for_model,
    get_models_in_category,
    get_datasets_for_category,
    is_valid_model_dataset_pair,
    get_category_evaluation_config
)

# Mapping management
from .category_mappings import (
    CategoryMappingManager,
    EvaluationTask,
    get_coding_specialists_manager,
    quick_coding_evaluation_plan,
    validate_coding_readiness
)

__all__ = [
    # Category definitions
    "ModelCategory",
    "CODING_SPECIALISTS", 
    "CATEGORY_REGISTRY",
    
    # Category utilities
    "get_category_for_model",
    "get_models_in_category", 
    "get_datasets_for_category",
    "is_valid_model_dataset_pair",
    "get_category_evaluation_config",
    
    # Mapping management
    "CategoryMappingManager",
    "EvaluationTask",
    
    # Convenience functions
    "get_coding_specialists_manager",
    "quick_coding_evaluation_plan",
    "validate_coding_readiness"
]