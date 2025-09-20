"""
Enhanced Model Registry for hybrid architecture

Provides enhanced model management with backward compatibility,
intelligent engine selection, and multi-GPU support.
"""

from typing import Dict, List, Optional, Any, Callable
import json
import logging
from pathlib import Path

from .enhanced_model_config import EnhancedModelConfig, ModelSizeCategory, EnginePreference
from ..interfaces.evaluation_interfaces import EngineType, ResourceRequirements


logger = logging.getLogger(__name__)


class EnhancedModelRegistry:
    """Enhanced model registry with intelligent engine selection"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._models: Dict[str, EnhancedModelConfig] = {}
        self._model_groups: Dict[str, List[str]] = {}
        self._engine_compatibility_cache: Dict[str, Dict[str, bool]] = {}
        
        # Load existing configurations if path provided
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def register_model(self, config: EnhancedModelConfig) -> bool:
        """Register a model configuration
        
        Args:
            config: Enhanced model configuration
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate configuration
            validation_errors = self._validate_model_config(config)
            if validation_errors:
                logger.error(f"Model validation failed for {config.model_name}: {validation_errors}")
                return False
            
            # Register the model
            self._models[config.model_name] = config
            
            # Update model groups
            category = config.specialization_category
            if category not in self._model_groups:
                self._model_groups[category] = []
            if config.model_name not in self._model_groups[category]:
                self._model_groups[category].append(config.model_name)
            
            # Cache engine compatibility
            self._cache_engine_compatibility(config)
            
            logger.info(f"Registered model: {config.model_name} (Category: {category})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {config.model_name}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[EnhancedModelConfig]:
        """Get model configuration by name"""
        return self._models.get(model_name)
    
    def get_models_by_category(self, category: str) -> List[EnhancedModelConfig]:
        """Get all models in a specific category"""
        model_names = self._model_groups.get(category, [])
        return [self._models[name] for name in model_names if name in self._models]
    
    def get_models_by_size_category(self, size_category: ModelSizeCategory) -> List[EnhancedModelConfig]:
        """Get models by size category"""
        return [config for config in self._models.values() 
                if config.model_size_category == size_category]
    
    def get_compatible_models(self, engine_type: EngineType, 
                            resource_constraints: Optional[ResourceRequirements] = None) -> List[EnhancedModelConfig]:
        """Get models compatible with specific engine type and resource constraints"""
        compatible = []
        
        for config in self._models.values():
            # Check engine compatibility
            if not self._is_model_compatible_with_engine(config, engine_type):
                continue
            
            # Check resource constraints
            if resource_constraints and not self._meets_resource_constraints(config, resource_constraints):
                continue
            
            compatible.append(config)
        
        return compatible
    
    def recommend_engine(self, model_name: str, 
                        resource_constraints: Optional[ResourceRequirements] = None) -> Optional[EngineType]:
        """Recommend optimal engine for a model"""
        config = self.get_model(model_name)
        if not config:
            return None
        
        # Get preferred engine from model config
        preferred = config.get_optimal_engine()
        
        if preferred == EnginePreference.LIGHTWEIGHT:
            return EngineType.LIGHTWEIGHT
        elif preferred == EnginePreference.DISTRIBUTED:
            return EngineType.DISTRIBUTED
        else:
            # AUTO selection logic
            return self._auto_select_engine(config, resource_constraints)
    
    def get_resource_requirements(self, model_name: str, 
                                engine_type: Optional[EngineType] = None) -> Optional[ResourceRequirements]:
        """Get resource requirements for a model with specific engine"""
        config = self.get_model(model_name)
        if not config:
            return None
        
        # Get recommended engine if not specified
        if engine_type is None:
            engine_type = self.recommend_engine(model_name)
        
        # Calculate base requirements
        model_reqs = config.get_resource_requirements()
        
        # Convert to ResourceRequirements object
        requirements = ResourceRequirements(
            gpu_memory_gb=model_reqs["gpu_memory_gb"],
            cpu_cores=model_reqs["cpu_cores"],
            system_memory_gb=model_reqs["total_memory_gb"] * 0.5,  # Half for system
            disk_space_gb=model_reqs["disk_space_gb"],
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            min_gpu_count=model_reqs["gpu_count"]
        )
        
        # Adjust based on engine type
        if engine_type == EngineType.LIGHTWEIGHT:
            # Lightweight engine may use less memory
            requirements.gpu_memory_gb *= 0.9
        elif engine_type == EngineType.DISTRIBUTED:
            # Distributed engine may need more overhead
            requirements.gpu_memory_gb *= 1.1
            requirements.system_memory_gb *= 1.2
        
        return requirements
    
    def migrate_from_legacy_registry(self, legacy_registry_path: str) -> int:
        """Migrate models from legacy registry format
        
        Args:
            legacy_registry_path: Path to legacy model registry
            
        Returns:
            int: Number of models migrated
        """
        try:
            # This would integrate with existing model_registry.py
            # For now, return 0 as placeholder
            logger.info(f"Migration from legacy registry not yet implemented: {legacy_registry_path}")
            return 0
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return 0
    
    def export_configurations(self, output_path: str) -> bool:
        """Export all configurations to file"""
        try:
            export_data = {
                "models": {},
                "model_groups": self._model_groups,
                "metadata": {
                    "total_models": len(self._models),
                    "categories": list(self._model_groups.keys()),
                    "export_format_version": "1.0"
                }
            }
            
            # Convert model configs to serializable format
            for name, config in self._models.items():
                # Use to_legacy_config() for backward compatibility
                export_data["models"][name] = config.to_legacy_config()
                export_data["models"][name]["enhanced_features"] = {
                    "model_size_category": config.model_size_category.value,
                    "preferred_engine": config.preferred_engine.value,
                    "estimated_parameters_b": config.estimated_parameters_b,
                    "is_large_model": config.is_large_model()
                }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(self._models)} model configurations to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> int:
        """Load configurations from file
        
        Returns:
            int: Number of models loaded
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            loaded_count = 0
            models_data = data.get("models", {})
            
            for model_name, model_data in models_data.items():
                try:
                    # Create EnhancedModelConfig from loaded data
                    config = EnhancedModelConfig(**model_data)
                    if self.register_model(config):
                        loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
            
            logger.info(f"Loaded {loaded_count} models from {file_path}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load from {file_path}: {e}")
            return 0
    
    def _validate_model_config(self, config: EnhancedModelConfig) -> List[str]:
        """Validate model configuration"""
        errors = []
        
        # Basic validation
        if not config.model_name:
            errors.append("Model name is required")
        
        if not config.huggingface_id:
            errors.append("Hugging Face ID is required")
        
        if config.size_gb <= 0:
            errors.append("Model size must be positive")
        
        # Multi-GPU validation
        if config.tensor_parallel_size < 1:
            errors.append("tensor_parallel_size must be >= 1")
        
        if config.pipeline_parallel_size < 1:
            errors.append("pipeline_parallel_size must be >= 1")
        
        # Memory validation
        if config.gpu_memory_utilization <= 0 or config.gpu_memory_utilization > 1:
            errors.append("gpu_memory_utilization must be between 0 and 1")
        
        return errors
    
    def _cache_engine_compatibility(self, config: EnhancedModelConfig):
        """Cache engine compatibility for model"""
        compatibility = config.get_engine_compatibility()
        self._engine_compatibility_cache[config.model_name] = compatibility
    
    def _is_model_compatible_with_engine(self, config: EnhancedModelConfig, engine_type: EngineType) -> bool:
        """Check if model is compatible with engine type"""
        compatibility = self._engine_compatibility_cache.get(config.model_name)
        if not compatibility:
            compatibility = config.get_engine_compatibility()
            self._engine_compatibility_cache[config.model_name] = compatibility
        
        if engine_type == EngineType.LIGHTWEIGHT:
            return compatibility.get("lightweight", False)
        elif engine_type == EngineType.DISTRIBUTED:
            return compatibility.get("distributed", True)
        
        return True
    
    def _meets_resource_constraints(self, config: EnhancedModelConfig, 
                                  constraints: ResourceRequirements) -> bool:
        """Check if model meets resource constraints"""
        model_reqs = config.get_resource_requirements()
        
        # Check GPU memory
        if model_reqs["gpu_memory_gb"] > constraints.gpu_memory_gb:
            return False
        
        # Check GPU count
        if model_reqs["gpu_count"] > constraints.min_gpu_count:
            return False
        
        return True
    
    def _auto_select_engine(self, config: EnhancedModelConfig, 
                          constraints: Optional[ResourceRequirements] = None) -> EngineType:
        """Automatically select optimal engine"""
        # Simple selection logic - can be enhanced
        if config.is_large_model():
            return EngineType.DISTRIBUTED
        else:
            return EngineType.LIGHTWEIGHT
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        size_distribution = {}
        engine_distribution = {}
        category_distribution = {}
        
        for config in self._models.values():
            # Size distribution
            size_cat = config.model_size_category.value
            size_distribution[size_cat] = size_distribution.get(size_cat, 0) + 1
            
            # Engine distribution
            recommended_engine = self.recommend_engine(config.model_name)
            if recommended_engine:
                engine_name = recommended_engine.value
                engine_distribution[engine_name] = engine_distribution.get(engine_name, 0) + 1
            
            # Category distribution
            cat = config.specialization_category
            category_distribution[cat] = category_distribution.get(cat, 0) + 1
        
        return {
            "total_models": len(self._models),
            "size_distribution": size_distribution,
            "engine_distribution": engine_distribution,
            "category_distribution": category_distribution,
            "model_groups": {k: len(v) for k, v in self._model_groups.items()}
        }