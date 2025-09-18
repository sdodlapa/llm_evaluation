"""
Enhanced Dataset Manager - Modular implementation with backward compatibility
Main interface for dataset management with improved architecture
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import from new modular components
from .dataset_registry import DatasetInfo, dataset_registry
from .dataset_loader import DatasetLoader
from .dataset_processor import DatasetProcessor
from .dataset_utils import DatasetUtils

logger = logging.getLogger(__name__)

class EnhancedDatasetManager:
    """
    Enhanced dataset manager with modular architecture
    Provides backward compatibility while using new modular components
    """
    
    def __init__(self, base_data_path: str = "evaluation_data"):
        """Initialize the enhanced dataset manager
        
        Args:
            base_data_path: Base path for dataset files
        """
        self.base_data_path = base_data_path
        
        # Initialize modular components
        self.registry = dataset_registry
        self.loader = DatasetLoader(base_data_path)
        self.processor = DatasetProcessor()
        self.utils = DatasetUtils(base_data_path)
        
        logger.info(f"Enhanced dataset manager initialized with {len(self.registry.get_all_datasets())} datasets")
    
    # Core dataset operations (delegate to appropriate modules)
    
    def get_available_datasets(self) -> List[str]:
        """Get list of all available dataset names"""
        return list(self.registry.get_all_datasets().keys())
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a specific dataset"""
        return self.registry.get_dataset_info(dataset_name)
    
    def load_dataset(self, dataset_name: str, sample_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load dataset samples"""
        return self.loader.load_dataset(dataset_name, sample_count)
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """Validate that a dataset exists and is properly formatted"""
        return self.loader.validate_dataset(dataset_name)
    
    def prepare_evaluation_sample(self, dataset_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a sample for evaluation"""
        return self.processor.prepare_evaluation_sample(dataset_name, sample)
    
    def get_evaluation_metrics(self, dataset_name: str) -> List[str]:
        """Get appropriate evaluation metrics for a dataset"""
        return self.processor.get_evaluation_metrics(dataset_name)
    
    # Utility and analysis functions
    
    def get_recommended_datasets(self, task_types: Optional[List[str]] = None) -> List[str]:
        """Get recommended datasets for evaluation"""
        return self.utils.get_recommended_datasets(task_types)
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all datasets"""
        return self.utils.get_dataset_summary()
    
    def get_evaluation_plan(self, model_capabilities: List[str], time_budget: str = "medium") -> Dict[str, Any]:
        """Generate an evaluation plan"""
        return self.utils.get_evaluation_plan(model_capabilities, time_budget)
    
    # Backward compatibility methods for existing code
    
    def get_all_datasets(self) -> Dict[str, DatasetInfo]:
        """Get all dataset configurations (backward compatibility)"""
        return self.registry.get_all_datasets()
    
    def get_implemented_datasets(self) -> Dict[str, DatasetInfo]:
        """Get only implemented datasets"""
        implemented_names = self.registry.get_implemented_datasets()
        return {name: self.registry.get_dataset_info(name) for name in implemented_names}
    
    def get_unimplemented_datasets(self) -> Dict[str, DatasetInfo]:
        """Get datasets not yet implemented"""
        unimplemented_names = self.registry.get_unimplemented_datasets()
        return {name: self.registry.get_dataset_info(name) for name in unimplemented_names}
    
    def get_datasets_by_task_type(self, task_type: str) -> Dict[str, DatasetInfo]:
        """Get datasets filtered by task type"""
        return self.registry.get_datasets_by_task_type(task_type)
    
    def create_missing_datasets(self, dataset_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Create missing dataset files"""
        return self.loader.create_missing_datasets(dataset_names)
    
    def analyze_dataset_gaps(self) -> Dict[str, Any]:
        """Analyze gaps in dataset coverage"""
        return self.utils.analyze_dataset_gaps()
    
    # Enhanced functionality
    
    def bulk_load_datasets(self, dataset_names: List[str], 
                          sample_counts: Optional[Dict[str, int]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Load multiple datasets at once
        
        Args:
            dataset_names: List of dataset names to load
            sample_counts: Optional dict mapping dataset names to sample counts
            
        Returns:
            Dict mapping dataset names to their loaded samples
        """
        results = {}
        
        for dataset_name in dataset_names:
            try:
                sample_count = sample_counts.get(dataset_name) if sample_counts else None
                results[dataset_name] = self.load_dataset(dataset_name, sample_count)
                logger.info(f"Successfully loaded {len(results[dataset_name])} samples from {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                results[dataset_name] = []
        
        return results
    
    def validate_all_datasets(self) -> Dict[str, bool]:
        """Validate all implemented datasets"""
        implemented_datasets = self.get_implemented_datasets()
        validation_results = {}
        
        for dataset_name in implemented_datasets.keys():
            try:
                validation_results[dataset_name] = self.validate_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Validation failed for {dataset_name}: {e}")
                validation_results[dataset_name] = False
        
        return validation_results
    
    def prepare_batch_evaluation(self, dataset_names: List[str], 
                               sample_counts: Optional[Dict[str, int]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Prepare multiple datasets for batch evaluation
        
        Args:
            dataset_names: List of dataset names to prepare
            sample_counts: Optional dict mapping dataset names to sample counts
            
        Returns:
            Dict mapping dataset names to prepared evaluation samples
        """
        raw_datasets = self.bulk_load_datasets(dataset_names, sample_counts)
        prepared_datasets = {}
        
        for dataset_name, samples in raw_datasets.items():
            if samples:  # Only process if we have samples
                prepared_samples = []
                for sample in samples:
                    try:
                        prepared_sample = self.prepare_evaluation_sample(dataset_name, sample)
                        prepared_samples.append(prepared_sample)
                    except Exception as e:
                        logger.warning(f"Failed to prepare sample from {dataset_name}: {e}")
                
                prepared_datasets[dataset_name] = prepared_samples
                logger.info(f"Prepared {len(prepared_samples)} samples for {dataset_name}")
            else:
                prepared_datasets[dataset_name] = []
        
        return prepared_datasets
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics"""
        summary = self.get_dataset_summary()
        validation_results = self.validate_all_datasets()
        
        stats = {
            'total_datasets': summary['total_datasets'],
            'implemented_datasets': summary['implemented_datasets'],
            'validated_datasets': sum(1 for valid in validation_results.values() if valid),
            'task_type_distribution': summary['task_type_distribution'],
            'evaluation_readiness': {},
            'recommendations': []
        }
        
        # Calculate readiness by task type
        for task_type, count in summary['task_type_distribution'].items():
            task_datasets = self.get_datasets_by_task_type(task_type)
            implemented_count = len([d for d in task_datasets.values() if d.implemented])
            validated_count = sum(1 for name in task_datasets.keys() 
                                if validation_results.get(name, False))
            
            stats['evaluation_readiness'][task_type] = {
                'total': count,
                'implemented': implemented_count,
                'validated': validated_count,
                'readiness_score': validated_count / count if count > 0 else 0
            }
        
        # Add recommendations
        low_readiness = [task for task, data in stats['evaluation_readiness'].items() 
                        if data['readiness_score'] < 0.5]
        
        if low_readiness:
            stats['recommendations'].append(
                f"Improve implementation and validation for: {', '.join(low_readiness)}"
            )
        
        if stats['validated_datasets'] < stats['implemented_datasets']:
            stats['recommendations'].append(
                "Some implemented datasets failed validation - check data files"
            )
        
        return stats
    
    def export_dataset_config(self, output_path: str) -> bool:
        """Export current dataset configuration to JSON file"""
        try:
            config_data = {}
            
            for name, info in self.get_all_datasets().items():
                config_data[name] = {
                    'task_type': info.task_type,
                    'file_path': info.file_path,
                    'sample_count': info.sample_count,
                    'evaluation_type': info.evaluation_type,
                    'implemented': info.implemented,
                    'description': info.description
                }
            
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Dataset configuration exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export dataset configuration: {e}")
            return False


# Convenience function for backward compatibility
def create_dataset_manager(base_data_path: str = "evaluation_data") -> EnhancedDatasetManager:
    """Create and return a dataset manager instance"""
    return EnhancedDatasetManager(base_data_path)


# Global instance for backward compatibility
dataset_manager = create_dataset_manager()