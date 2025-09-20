"""
Integration adapter for hybrid architecture testing

Provides bridge between new hybrid architecture and existing evaluation pipeline
for validation and testing purposes.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add core shared to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core_shared.interfaces.evaluation_interfaces import (
    EvaluationRequest, EvaluationResult, EngineType, ResourceRequirements
)
from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
from core_shared.model_registry.model_registry_enhanced import EnhancedModelRegistry
from core_shared.orchestration.evaluation_orchestrator import EvaluationOrchestrator, OrchestratorConfig
from engines.lightweight.lightweight_engine import LightweightEvaluationEngine

# Import existing pipeline components
try:
    from configs.model_registry import ModelConfig, get_model_registry
    from evaluation.evaluation_engine import EvaluationEngine as LegacyEvaluationEngine
except ImportError:
    # Fallback for testing
    ModelConfig = None
    LegacyEvaluationEngine = None


logger = logging.getLogger(__name__)


class HybridIntegrationAdapter:
    """Adapter for integrating hybrid architecture with existing pipeline"""
    
    def __init__(self):
        self.enhanced_registry = EnhancedModelRegistry()
        self.orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        self.lightweight_engine = LightweightEvaluationEngine()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the hybrid integration adapter"""
        try:
            logger.info("Initializing hybrid integration adapter")
            
            # Initialize orchestrator
            await self.orchestrator.start()
            
            # Initialize and register lightweight engine
            if self.lightweight_engine.initialize():
                self.orchestrator.register_engine(self.lightweight_engine)
                logger.info("Lightweight engine registered with orchestrator")
            else:
                logger.error("Failed to initialize lightweight engine")
                return False
            
            # Migrate existing models if available
            migrated_count = await self._migrate_existing_models()
            logger.info(f"Migrated {migrated_count} models from existing registry")
            
            self._initialized = True
            logger.info("Hybrid integration adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize integration adapter: {e}")
            return False
    
    async def evaluate_model(self, model_name: str, datasets: List[str], 
                           evaluation_params: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Evaluate a model using the hybrid architecture
        
        Args:
            model_name: Name of the model to evaluate
            datasets: List of dataset names
            evaluation_params: Optional evaluation parameters
            
        Returns:
            EvaluationResult from the hybrid system
        """
        if not self._initialized:
            raise RuntimeError("Integration adapter not initialized")
        
        # Get model configuration
        model_config = self.enhanced_registry.get_model(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Create evaluation request
        request = EvaluationRequest(
            request_id=f"test_{model_name}_{int(asyncio.get_event_loop().time())}",
            model_config=model_config,
            datasets=datasets,
            evaluation_params=evaluation_params or {},
            resource_constraints=ResourceRequirements()
        )
        
        # Submit to orchestrator
        request_id = await self.orchestrator.submit_request(request)
        
        # Wait for completion (simplified for testing)
        max_wait_time = 300  # 5 minutes
        wait_interval = 1.0
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            context = await self.orchestrator.get_request_status(request_id)
            if context and context.result:
                return context.result
            
            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval
        
        raise TimeoutError(f"Evaluation timed out after {max_wait_time} seconds")
    
    async def test_engine_selection(self, model_configs: List[EnhancedModelConfig]) -> Dict[str, Any]:
        """Test engine selection logic with different model configurations"""
        results = {}
        
        for config in model_configs:
            # Register model
            self.enhanced_registry.register_model(config)
            
            # Test engine recommendation
            recommended_engine = self.enhanced_registry.recommend_engine(config.model_name)
            
            # Test resource requirements
            requirements = self.enhanced_registry.get_resource_requirements(config.model_name, recommended_engine)
            
            results[config.model_name] = {
                "recommended_engine": recommended_engine.value if recommended_engine else None,
                "resource_requirements": {
                    "gpu_memory_gb": requirements.gpu_memory_gb if requirements else 0,
                    "gpu_count": requirements.min_gpu_count if requirements else 1,
                    "tensor_parallel_size": requirements.tensor_parallel_size if requirements else 1
                },
                "model_size_gb": config.size_gb,
                "is_large_model": config.is_large_model() if hasattr(config, 'is_large_model') else False
            }
        
        return results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of the hybrid system"""
        return {
            "orchestrator_metrics": self.orchestrator.get_metrics(),
            "lightweight_engine_status": {
                "initialized": self.lightweight_engine.is_initialized(),
                "capabilities": {
                    "max_model_size_gb": self.lightweight_engine.capabilities.max_model_size_gb,
                    "max_gpu_count": self.lightweight_engine.capabilities.max_gpu_count,
                    "engine_type": self.lightweight_engine.capabilities.engine_type.value
                }
            },
            "registry_stats": self.enhanced_registry.get_statistics()
        }
    
    async def cleanup(self):
        """Clean up adapter resources"""
        logger.info("Cleaning up hybrid integration adapter")
        
        if self.orchestrator:
            await self.orchestrator.stop()
        
        if self.lightweight_engine:
            self.lightweight_engine.cleanup()
        
        self._initialized = False
    
    async def _migrate_existing_models(self) -> int:
        """Migrate models from existing registry to enhanced registry"""
        migrated_count = 0
        
        try:
            if ModelConfig is None:
                # Create test models for validation
                test_models = self._create_test_models()
                for model in test_models:
                    if self.enhanced_registry.register_model(model):
                        migrated_count += 1
                return migrated_count
            
            # Get existing registry (placeholder for actual integration)
            # existing_registry = get_model_registry()
            # for model_name, legacy_config in existing_registry.items():
            #     enhanced_config = self._convert_legacy_to_enhanced(legacy_config)
            #     if self.enhanced_registry.register_model(enhanced_config):
            #         migrated_count += 1
            
        except Exception as e:
            logger.error(f"Failed to migrate existing models: {e}")
        
        return migrated_count
    
    def _create_test_models(self) -> List[EnhancedModelConfig]:
        """Create test model configurations for validation"""
        return [
            EnhancedModelConfig(
                model_name="test_small_model",
                huggingface_id="microsoft/DialoGPT-small",
                license="MIT",
                size_gb=2.5,
                context_window=1024,
                specialization_category="general",
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            ),
            EnhancedModelConfig(
                model_name="test_medium_model",
                huggingface_id="microsoft/DialoGPT-medium",
                license="MIT", 
                size_gb=15.0,
                context_window=2048,
                specialization_category="general",
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            ),
            EnhancedModelConfig(
                model_name="test_large_model",
                huggingface_id="test/large-model",
                license="Apache-2.0",
                size_gb=45.0,
                context_window=4096,
                specialization_category="general",
                tensor_parallel_size=2,
                pipeline_parallel_size=1
            )
        ]
    
    def _convert_legacy_to_enhanced(self, legacy_config: Any) -> EnhancedModelConfig:
        """Convert legacy ModelConfig to EnhancedModelConfig"""
        # Placeholder implementation
        return EnhancedModelConfig(
            model_name=getattr(legacy_config, 'model_name', 'unknown'),
            huggingface_id=getattr(legacy_config, 'huggingface_id', 'unknown'),
            license=getattr(legacy_config, 'license', 'unknown'),
            size_gb=getattr(legacy_config, 'size_gb', 1.0),
            context_window=getattr(legacy_config, 'context_window', 2048),
            tensor_parallel_size=getattr(legacy_config, 'tensor_parallel_size', 1),
            pipeline_parallel_size=getattr(legacy_config, 'pipeline_parallel_size', 1)
        )