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
from models.qwen_implementation import create_qwen3_8b, create_qwen3_14b, Qwen3Implementation

logger = logging.getLogger(__name__)


def create_generic_model(model_name: str, preset: str = "balanced", cache_dir: Optional[str] = None):
    """Generic model creation function that works with any model in the configuration system"""
    try:
        # Import here to avoid circular imports
        from configs.model_configs import MODEL_CONFIGS
        
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Model {model_name} not found in configuration system")
            return None
        
        model_config = MODEL_CONFIGS[model_name]
        
        # For now, use QwenImplementation as the base implementation for all models
        # This is a simplification - in a full system, we'd have different implementations
        # for different model families (Qwen, LLaMA, Mistral, BioGPT, etc.)
        logger.info(f"Creating {model_name} using generic implementation with preset: {preset}")
        
        model_instance = Qwen3Implementation(
            config=model_config.create_preset_variant(preset)
        )
        
        logger.info(f"Model instance created. Attempting to load model...")
        
        # Load the model after creation
        load_success = model_instance.load_model()
        logger.info(f"Model load_model() returned: {load_success}")
        logger.info(f"Model is_loaded status: {model_instance.is_loaded}")
        
        if not load_success:
            logger.error(f"Failed to load model {model_name} - load_model() returned False")
            return None
            
        # Double-check that model is actually loaded
        if not model_instance.is_loaded:
            logger.error(f"Model {model_name} load_model() succeeded but is_loaded is False")
            return None
            
        logger.info(f"✅ Model {model_name} successfully loaded and ready")
        return model_instance
        
    except Exception as e:
        logger.error(f"Failed to create generic model {model_name}: {e}")
        return None


class ModelRegistry:
    """Simple registry for model creation functions"""
    
    def __init__(self):
        """Initialize the model registry with available models"""
        self._registry: Dict[str, Callable] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register all available model creation functions"""
        # Import here to avoid circular imports
        try:
            from configs.model_configs import MODEL_CONFIGS
            
            # Register specific implementations for Qwen models
            self.register_model("qwen3_8b", create_qwen3_8b)
            self.register_model("qwen3_14b", create_qwen3_14b)
            
            # Register generic implementation for all other models
            for model_name in MODEL_CONFIGS.keys():
                if model_name not in ["qwen3_8b", "qwen3_14b"]:
                    # Create a closure to capture the model_name
                    def make_creation_func(name):
                        return lambda preset="balanced", cache_dir=None: create_generic_model(name, preset, cache_dir)
                    
                    self.register_model(model_name, make_creation_func(model_name))
            
            # Aliases for common name variations
            self.register_model("qwen_8b", create_qwen3_8b)
            self.register_model("qwen_14b", create_qwen3_14b)
            self.register_model("qwen2.5_8b", create_qwen3_8b)
            self.register_model("qwen2.5_14b", create_qwen3_14b)
            
            logger.info(f"Registered {len(self._registry)} models in registry")
            
        except ImportError as e:
            logger.error(f"Failed to import model configurations: {e}")
            # Fall back to basic Qwen models only
            self.register_model("qwen3_8b", create_qwen3_8b)
            self.register_model("qwen3_14b", create_qwen3_14b)
    
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
            elif "bio" in model_name or "clinical" in model_name:
                family = "biomedical"
            elif "coder" in model_name or "code" in model_name:
                family = "coding"
            elif "math" in model_name:
                family = "mathematics"
            elif "mistral" in model_name:
                family = "mistral"
            elif "llama" in model_name:
                family = "llama"
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