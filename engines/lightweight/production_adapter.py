"""
Production Integration Adapter for Lightweight Evaluation Engine

This module provides seamless integration between the new lightweight evaluation engine
and the existing evaluation pipeline, ensuring backward compatibility and smooth deployment.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Import existing framework components
try:
    from evaluation import run_evaluation
    from core_shared.model_registry.enhanced_model_config import EnhancedModelConfig
    from core_shared.interfaces.evaluation_interfaces import EngineType, EvaluationRequest, EvaluationResult
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProductionIntegrationAdapter:
    """
    Adapter that enables the lightweight engine to work seamlessly with existing production pipeline
    """
    
    def __init__(self, lightweight_engine=None):
        """
        Initialize production integration adapter
        
        Args:
            lightweight_engine: Instance of LightweightEvaluationEngine
        """
        self.lightweight_engine = lightweight_engine
        self.compatibility_mode = True
        self.legacy_function_mapping = {}
        self.setup_function_mapping()
        
        logger.info("Production integration adapter initialized")
    
    def setup_function_mapping(self):
        """Setup mapping between legacy functions and new engine methods"""
        self.legacy_function_mapping = {
            'run_evaluation': self.run_evaluation_adapter,
            'evaluate_model': self.evaluate_model_adapter,
            'batch_evaluate': self.batch_evaluate_adapter,
            'get_evaluation_results': self.get_evaluation_results_adapter,
            'cleanup_evaluation': self.cleanup_evaluation_adapter
        }
    
    def run_evaluation_adapter(self, 
                              model_names: Union[str, List[str]],
                              dataset_names: Union[str, List[str]],
                              **kwargs) -> Union[EvaluationResult, List[EvaluationResult]]:
        """
        Adapter for the legacy run_evaluation function
        
        Args:
            model_names: Model name(s) to evaluate
            dataset_names: Dataset name(s) to use
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation result(s) compatible with existing pipeline
        """
        
        if not self.lightweight_engine:
            logger.warning("Lightweight engine not available, falling back to legacy evaluation")
            if FRAMEWORK_AVAILABLE:
                return run_evaluation(model_names, dataset_names, **kwargs)
            else:
                raise RuntimeError("Neither lightweight engine nor legacy framework available")
        
        # Convert inputs to lists if needed
        if isinstance(model_names, str):
            model_names = [model_names]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        results = []
        
        for model_name in model_names:
            try:
                # Create evaluation request
                request = EvaluationRequest(
                    model_name=model_name,
                    datasets=dataset_names,
                    parameters=kwargs.get('eval_params', {}),
                    priority=kwargs.get('priority', 'normal'),
                    timeout_seconds=kwargs.get('timeout', 3600)
                )
                
                # Run evaluation using lightweight engine
                result = self.lightweight_engine.evaluate(request)
                results.append(result)
                
                logger.info(f"Completed evaluation for {model_name} using lightweight engine")
                
            except Exception as e:
                logger.error(f"Lightweight engine evaluation failed for {model_name}: {e}")
                
                # Fallback to legacy evaluation if available
                if FRAMEWORK_AVAILABLE and kwargs.get('allow_fallback', True):
                    logger.info(f"Falling back to legacy evaluation for {model_name}")
                    try:
                        legacy_result = run_evaluation(model_name, dataset_names, **kwargs)
                        results.append(legacy_result)
                    except Exception as legacy_e:
                        logger.error(f"Legacy evaluation also failed for {model_name}: {legacy_e}")
                        # Create error result
                        error_result = EvaluationResult(
                            request_id=f"error_{model_name}_{int(time.time())}",
                            model_name=model_name,
                            dataset_name=",".join(dataset_names),
                            metrics={},
                            engine_used=EngineType.LIGHTWEIGHT,
                            success=False,
                            error_message=str(e),
                            started_at=time.time(),
                            completed_at=time.time()
                        )
                        results.append(error_result)
                else:
                    # Create error result
                    error_result = EvaluationResult(
                        request_id=f"error_{model_name}_{int(time.time())}",
                        model_name=model_name,
                        dataset_name=",".join(dataset_names),
                        metrics={},
                        engine_used=EngineType.LIGHTWEIGHT,
                        success=False,
                        error_message=str(e),
                        started_at=time.time(),
                        completed_at=time.time()
                    )
                    results.append(error_result)
        
        # Return single result if single model, list if multiple
        if len(results) == 1 and isinstance(model_names, list) and len(model_names) == 1:
            return results[0]
        return results
    
    def evaluate_model_adapter(self, model_config: Union[str, EnhancedModelConfig],
                              datasets: List[str],
                              **kwargs) -> EvaluationResult:
        """
        Adapter for model evaluation with enhanced configuration
        
        Args:
            model_config: Model configuration or model name
            datasets: List of datasets to evaluate on
            **kwargs: Additional parameters
            
        Returns:
            Evaluation result
        """
        
        if isinstance(model_config, str):
            model_name = model_config
        elif hasattr(model_config, 'model_name'):
            model_name = model_config.model_name
        else:
            raise ValueError(f"Invalid model_config type: {type(model_config)}")
        
        return self.run_evaluation_adapter(model_name, datasets, **kwargs)
    
    def batch_evaluate_adapter(self, 
                              evaluation_configs: List[Dict[str, Any]],
                              **kwargs) -> List[EvaluationResult]:
        """
        Adapter for batch evaluation
        
        Args:
            evaluation_configs: List of evaluation configurations
            **kwargs: Additional parameters
            
        Returns:
            List of evaluation results
        """
        
        results = []
        
        for config in evaluation_configs:
            model_name = config.get('model_name')
            datasets = config.get('datasets', [])
            eval_params = config.get('parameters', {})
            
            if not model_name or not datasets:
                logger.warning(f"Invalid evaluation config: {config}")
                continue
            
            try:
                result = self.run_evaluation_adapter(
                    model_name, 
                    datasets, 
                    eval_params=eval_params,
                    **kwargs
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch evaluation failed for {model_name}: {e}")
                error_result = EvaluationResult(
                    request_id=f"batch_error_{model_name}_{int(time.time())}",
                    model_name=model_name,
                    dataset_name=",".join(datasets),
                    metrics={},
                    engine_used=EngineType.LIGHTWEIGHT,
                    success=False,
                    error_message=str(e),
                    started_at=time.time(),
                    completed_at=time.time()
                )
                results.append(error_result)
        
        return results
    
    def get_evaluation_results_adapter(self, 
                                     request_ids: Optional[List[str]] = None,
                                     model_names: Optional[List[str]] = None,
                                     **kwargs) -> List[EvaluationResult]:
        """
        Adapter for retrieving evaluation results
        
        Args:
            request_ids: Optional list of request IDs to filter by
            model_names: Optional list of model names to filter by
            **kwargs: Additional filter parameters
            
        Returns:
            List of matching evaluation results
        """
        
        if not self.lightweight_engine:
            logger.warning("Lightweight engine not available for result retrieval")
            return []
        
        # Use engine's result storage if available
        if hasattr(self.lightweight_engine, 'get_evaluation_history'):
            return self.lightweight_engine.get_evaluation_history(
                request_ids=request_ids,
                model_names=model_names,
                **kwargs
            )
        
        logger.warning("Result retrieval not implemented in lightweight engine")
        return []
    
    def cleanup_evaluation_adapter(self, **kwargs) -> None:
        """
        Adapter for evaluation cleanup
        
        Args:
            **kwargs: Cleanup parameters
        """
        
        if self.lightweight_engine:
            try:
                self.lightweight_engine.cleanup()
                logger.info("Lightweight engine cleanup completed")
            except Exception as e:
                logger.error(f"Lightweight engine cleanup failed: {e}")
        
        # Additional cleanup if needed
        if kwargs.get('deep_cleanup', False):
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def enable_compatibility_mode(self, enable: bool = True) -> None:
        """
        Enable or disable compatibility mode
        
        Args:
            enable: Whether to enable compatibility mode
        """
        self.compatibility_mode = enable
        logger.info(f"Compatibility mode {'enabled' if enable else 'disabled'}")
    
    def check_engine_compatibility(self, model_name: str) -> bool:
        """
        Check if a model is compatible with the lightweight engine
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if compatible, False otherwise
        """
        
        if not self.lightweight_engine:
            return False
        
        try:
            # Check if model can be handled by lightweight engine
            if hasattr(self.lightweight_engine, 'can_handle_model'):
                return self.lightweight_engine.can_handle_model(model_name)
            
            # Default compatibility check
            return True
            
        except Exception as e:
            logger.error(f"Compatibility check failed for {model_name}: {e}")
            return False
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get status of the lightweight engine and adapter
        
        Returns:
            Dictionary containing status information
        """
        
        status = {
            'adapter_version': '1.0.0',
            'compatibility_mode': self.compatibility_mode,
            'lightweight_engine_available': self.lightweight_engine is not None,
            'legacy_framework_available': FRAMEWORK_AVAILABLE,
            'supported_functions': list(self.legacy_function_mapping.keys())
        }
        
        if self.lightweight_engine:
            if hasattr(self.lightweight_engine, 'get_status'):
                status['engine_status'] = self.lightweight_engine.get_status()
            else:
                status['engine_status'] = 'Available'
        
        return status
    
    def create_fallback_pipeline(self) -> Dict[str, callable]:
        """
        Create fallback pipeline for when lightweight engine is not available
        
        Returns:
            Dictionary of fallback functions
        """
        
        fallback_pipeline = {}
        
        if FRAMEWORK_AVAILABLE:
            fallback_pipeline['run_evaluation'] = run_evaluation
            logger.info("Legacy framework available for fallback")
        else:
            # Create mock functions for testing
            def mock_evaluation(*args, **kwargs):
                logger.warning("Mock evaluation function called - no backend available")
                return EvaluationResult(
                    request_id="mock_" + str(int(time.time())),
                    model_name=args[0] if args else "unknown",
                    dataset_name="mock_dataset",
                    metrics={'mock_score': 0.5},
                    engine_used=EngineType.LIGHTWEIGHT,
                    success=True,
                    started_at=time.time(),
                    completed_at=time.time()
                )
            
            fallback_pipeline['run_evaluation'] = mock_evaluation
            logger.warning("No evaluation backend available - using mock functions")
        
        return fallback_pipeline
    
    def migrate_from_legacy(self, legacy_config_path: str) -> Dict[str, Any]:
        """
        Migrate configuration from legacy evaluation setup
        
        Args:
            legacy_config_path: Path to legacy configuration file
            
        Returns:
            Migrated configuration for lightweight engine
        """
        
        migration_report = {
            'status': 'not_implemented',
            'migrated_settings': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Read legacy configuration
            legacy_config_file = Path(legacy_config_path)
            if not legacy_config_file.exists():
                migration_report['errors'].append(f"Legacy config file not found: {legacy_config_path}")
                return migration_report
            
            # TODO: Implement actual migration logic based on legacy config format
            migration_report['status'] = 'placeholder'
            migration_report['warnings'].append("Migration logic not yet implemented")
            
            logger.info(f"Migration attempted for {legacy_config_path}")
            
        except Exception as e:
            migration_report['errors'].append(f"Migration failed: {e}")
            logger.error(f"Migration failed: {e}")
        
        return migration_report

# Factory function for easy integration
def create_production_adapter(lightweight_engine=None) -> ProductionIntegrationAdapter:
    """
    Factory function to create production integration adapter
    
    Args:
        lightweight_engine: Optional lightweight engine instance
        
    Returns:
        Configured production integration adapter
    """
    
    adapter = ProductionIntegrationAdapter(lightweight_engine)
    
    # Auto-detect and configure adapter
    if adapter.lightweight_engine:
        logger.info("Production adapter configured with lightweight engine")
    else:
        logger.warning("Production adapter configured without lightweight engine - fallback mode only")
    
    return adapter

# Convenience functions for backward compatibility
def run_evaluation_compatible(*args, **kwargs):
    """Backward compatible run_evaluation function"""
    adapter = create_production_adapter()
    return adapter.run_evaluation_adapter(*args, **kwargs)

def evaluate_model_compatible(*args, **kwargs):
    """Backward compatible evaluate_model function"""
    adapter = create_production_adapter()
    return adapter.evaluate_model_adapter(*args, **kwargs)