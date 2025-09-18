"""
Simple Model Registry

Provides a centralized registry for model creation functions, replacing hard-coded
model instantiation logic with a clean, extensible dictionary-based approach.

This is the simplified approach chosen over complex factory patterns to maintain
clear architecture while supporting multiple model families.
"""

import logging
from typing import Dict, Callable, Optional
from models.base_model import BaseModelImplementation
from models.qwen_implementation import create_qwen3_8b, create_qwen3_14b

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple registry for model creation functions"""
    
    def __init__(self):
        """Initialize the model registry with available models"""
        self._registry: Dict[str, Callable] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register all available model creation functions"""
        # Qwen family models
        self.register_model("qwen3_8b", create_qwen3_8b)
        self.register_model("qwen3_14b", create_qwen3_14b)
        
        # Aliases for common name variations
        self.register_model("qwen_8b", create_qwen3_8b)
        self.register_model("qwen_14b", create_qwen3_14b)
        self.register_model("qwen2.5_8b", create_qwen3_8b)
        self.register_model("qwen2.5_14b", create_qwen3_14b)
    
    def register_model(self, model_name: str, creation_function: Callable):
        """Register a new model creation function
        
        Args:
            model_name: Identifier for the model (e.g., 'qwen3_8b')
            creation_function: Function that creates the model instance
        """
        self._registry[model_name.lower()] = creation_function
        logger.debug(f"Registered model: {model_name}")
    
    def create_model(self, model_name: str, preset: str = "balanced", 
                    cache_dir: Optional[str] = None) -> Optional[BaseModelImplementation]:
        """Create a model instance using the registry
        
        Args:
            model_name: Name of the model to create
            preset: Model preset configuration
            cache_dir: Optional cache directory
            
        Returns:
            Model instance or None if not found
        """
        normalized_name = model_name.lower()
        
        if normalized_name in self._registry:
            try:
                creation_func = self._registry[normalized_name]
                logger.info(f"Creating {model_name} instance with preset: {preset}")
                
                # Call creation function with appropriate parameters
                if cache_dir:
                    return creation_func(preset=preset, cache_dir=cache_dir)
                else:
                    return creation_func(preset=preset)
                    
            except Exception as e:
                logger.error(f"Failed to create model {model_name}: {e}")
                return None
        else:
            logger.warning(f"Model {model_name} not found in registry. Available models: {self.list_models()}")
            return None
    
    def list_models(self) -> list:
        """Get list of all registered model names"""
        return sorted(list(self._registry.keys()))
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in the registry"""
        return model_name.lower() in self._registry
    
    def get_model_families(self) -> Dict[str, list]:
        """Get models organized by family"""
        families = {}
        
        for model_name in self._registry.keys():
            # Extract family name (everything before size indicator)
            if "qwen" in model_name:
                family = "qwen"
            else:
                family = "other"
            
            if family not in families:
                families[family] = []
            families[family].append(model_name)
        
        return families


# Global registry instance
model_registry = ModelRegistry()


def create_model(model_name: str, preset: str = "balanced", 
                cache_dir: Optional[str] = None) -> Optional[BaseModelImplementation]:
    """Convenience function to create models using the global registry"""
    return model_registry.create_model(model_name, preset, cache_dir)


def register_model(model_name: str, creation_function: Callable):
    """Convenience function to register new models"""
    model_registry.register_model(model_name, creation_function)


def list_available_models() -> list:
    """Convenience function to list all available models"""
    return model_registry.list_models()


def get_model_families() -> Dict[str, list]:
    """Convenience function to get model families"""
    return model_registry.get_model_families()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    return model_registry