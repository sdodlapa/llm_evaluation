"""
Enhanced model configuration system for hybrid architecture

Provides backward-compatible ModelConfig enhancements with multi-GPU
support and engine selection capabilities.
"""

from .enhanced_model_config import EnhancedModelConfig, ModelSizeCategory
from .model_registry_enhanced import EnhancedModelRegistry

__all__ = [
    'EnhancedModelConfig',
    'ModelSizeCategory', 
    'EnhancedModelRegistry'
]