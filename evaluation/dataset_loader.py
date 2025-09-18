"""
Dataset Loader - Data loading, validation, and file operations
Handles loading datasets, metadata, and validating data integrity
"""

import os
import json
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

try:
    from .dataset_registry import DatasetInfo, dataset_registry
except ImportError:
    from evaluation.dataset_registry import DatasetInfo, dataset_registry

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Handles loading and validation of dataset files"""
    
    def __init__(self, base_data_path: str = "evaluation_data"):
        self.base_path = Path(base_data_path)
        self.registry = dataset_registry
    
    def load_dataset(self, dataset_name: str, num_samples: int = None) -> List[Dict[str, Any]]:
        """Load dataset with optional sampling
        
        Args:
            dataset_name: Name of the dataset to load
            num_samples: Number of samples to return (random sample if less than total)
            
        Returns:
            List of dataset samples
        """
        dataset_info = self.registry.get_dataset_info(dataset_name)
        data_path = self.base_path / dataset_info.data_path
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different data formats
            if isinstance(data, dict):
                samples = data.get('data', data.get('examples', [data]))
                if not isinstance(samples, list):
                    samples = [data]  # Single sample format
            elif isinstance(data, list):
                samples = data
            else:
                raise ValueError(f"Unexpected data format in {dataset_name}")
            
            # Apply sampling if requested
            if num_samples and len(samples) > num_samples:
                samples = random.sample(samples, num_samples)
                
            logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            return samples
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {dataset_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def load_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset metadata
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing metadata
        """
        dataset_info = self.registry.get_dataset_info(dataset_name)
        
        if not dataset_info.metadata_path:
            return {
                'dataset_name': dataset_name,
                'task_type': dataset_info.task_type,
                'sample_count': dataset_info.sample_count,
                'description': dataset_info.description
            }
        
        metadata_path = self.base_path / dataset_info.metadata_path
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {
                'dataset_name': dataset_name,
                'task_type': dataset_info.task_type,
                'sample_count': dataset_info.sample_count,
                'description': dataset_info.description,
                'note': 'Metadata file missing'
            }
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata for {dataset_name}: {e}")
            return {
                'dataset_name': dataset_name,
                'error': str(e)
            }
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset structure and content
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            dataset_info = self.registry.get_dataset_info(dataset_name)
            validation_result = {
                'dataset_name': dataset_name,
                'valid': True,
                'issues': [],
                'sample_count': 0,
                'required_fields': [],
                'missing_fields': [],
                'field_coverage': {}
            }
            
            # Check if data file exists
            data_path = self.base_path / dataset_info.data_path
            if not data_path.exists():
                validation_result['valid'] = False
                validation_result['issues'].append(f"Data file not found: {data_path}")
                return validation_result
            
            # Load and validate samples
            try:
                samples = self.load_dataset(dataset_name, num_samples=10)  # Sample for validation
                validation_result['sample_count'] = len(samples)
                
                if not samples:
                    validation_result['valid'] = False
                    validation_result['issues'].append("Dataset contains no samples")
                    return validation_result
                
                # Get required fields for this task type
                required_fields = self._get_required_fields(dataset_info.task_type)
                validation_result['required_fields'] = required_fields
                
                # Check field coverage
                field_counts = {}
                for sample in samples:
                    for field in sample.keys():
                        field_counts[field] = field_counts.get(field, 0) + 1
                
                validation_result['field_coverage'] = {
                    field: count / len(samples) for field, count in field_counts.items()
                }
                
                # Check for missing required fields
                missing_fields = []
                for field in required_fields:
                    if field not in field_counts or field_counts[field] < len(samples) * 0.8:
                        missing_fields.append(field)
                
                validation_result['missing_fields'] = missing_fields
                
                if missing_fields:
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"Missing required fields: {missing_fields}")
                
            except Exception as e:
                validation_result['valid'] = False
                validation_result['issues'].append(f"Failed to load samples: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            return {
                'dataset_name': dataset_name,
                'valid': False,
                'issues': [f"Validation failed: {str(e)}"],
                'sample_count': 0
            }
    
    def _get_required_fields(self, task_type: str) -> List[str]:
        """Get required fields for a given task type"""
        field_requirements = {
            'coding': ['problem', 'solution'],
            'reasoning': ['question', 'answer'],
            'qa': ['question', 'answer'],
            'instruction_following': ['instruction', 'expected_output'],
            'function_calling': ['function_description', 'expected_call'],
            'biomedical_qa': ['question', 'answer'],
            'relation_extraction': ['text', 'relations'],
            'biomedical_ner': ['text', 'entities'],
            'sequence_classification': ['sequence', 'label'],
            'summarization': ['text', 'summary'],
            'scientific_ner': ['text', 'entities'],
            'document_vqa': ['image', 'question', 'answer'],
            'safety_classification': ['text', 'label']
        }
        return field_requirements.get(task_type, ['text', 'label'])
    
    def create_missing_datasets(self):
        """Create placeholder files for missing datasets"""
        logger.info("Creating placeholder files for missing datasets...")
        
        for name, info in self.registry.get_all_datasets().items():
            if not info.implemented:
                continue
                
            # Create data file if missing
            data_path = self.base_path / info.data_path
            if not data_path.exists():
                data_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create minimal placeholder
                placeholder_data = {
                    'dataset_name': name,
                    'task_type': info.task_type,
                    'description': info.description,
                    'data': [],
                    'note': 'Placeholder file - actual data needs to be added'
                }
                
                with open(data_path, 'w') as f:
                    json.dump(placeholder_data, f, indent=2)
                
                logger.info(f"Created placeholder data file: {data_path}")
            
            # Create metadata file if missing
            if info.metadata_path:
                metadata_path = self.base_path / info.metadata_path
                if not metadata_path.exists():
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    placeholder_metadata = {
                        'dataset_name': name,
                        'task_type': info.task_type,
                        'total_samples': info.sample_count,
                        'evaluation_type': info.evaluation_type,
                        'description': info.description,
                        'created': 'placeholder'
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(placeholder_metadata, f, indent=2)
                    
                    logger.info(f"Created placeholder metadata file: {metadata_path}")
    
    def check_dataset_files(self) -> Dict[str, Dict[str, bool]]:
        """Check which dataset files exist
        
        Returns:
            Dictionary mapping dataset names to file existence status
        """
        file_status = {}
        
        for name, info in self.registry.get_all_datasets().items():
            data_path = self.base_path / info.data_path
            metadata_path = self.base_path / info.metadata_path if info.metadata_path else None
            
            file_status[name] = {
                'data_exists': data_path.exists(),
                'metadata_exists': metadata_path.exists() if metadata_path else True,
                'fully_available': data_path.exists() and (not metadata_path or metadata_path.exists())
            }
        
        return file_status
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            samples = self.load_dataset(dataset_name, num_samples=100)  # Sample for stats
            metadata = self.load_metadata(dataset_name)
            
            stats = {
                'dataset_name': dataset_name,
                'total_samples': len(samples),
                'metadata': metadata,
                'field_distribution': {},
                'sample_lengths': []
            }
            
            # Analyze field distribution
            all_fields = set()
            for sample in samples:
                all_fields.update(sample.keys())
            
            for field in all_fields:
                field_count = sum(1 for sample in samples if field in sample and sample[field])
                stats['field_distribution'][field] = {
                    'count': field_count,
                    'coverage': field_count / len(samples)
                }
            
            # Analyze text lengths (for text fields)
            text_fields = ['text', 'question', 'problem', 'instruction', 'prompt']
            for field in text_fields:
                if field in all_fields:
                    lengths = [len(str(sample.get(field, ''))) for sample in samples 
                             if sample.get(field)]
                    if lengths:
                        stats['sample_lengths'].append({
                            'field': field,
                            'min_length': min(lengths),
                            'max_length': max(lengths),
                            'avg_length': sum(lengths) / len(lengths)
                        })
            
            return stats
            
        except Exception as e:
            return {
                'dataset_name': dataset_name,
                'error': str(e)
            }