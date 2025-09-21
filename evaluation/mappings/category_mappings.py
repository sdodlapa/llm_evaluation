"""
Category Mapping Manager
========================

Manages model-category-dataset mappings and provides evaluation orchestration
for category-based evaluations.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

from .model_categories import (
    ModelCategory, 
    CATEGORY_REGISTRY,
    get_category_for_model,
    get_datasets_for_category,
    is_valid_model_dataset_pair,
    get_category_evaluation_config
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationTask:
    """Represents a single evaluation task (model + dataset + config)"""
    model_name: str
    dataset_name: str 
    category: str
    sample_limit: int
    evaluation_config: Dict[str, Any]
    priority: str = "NORMAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "category": self.category,
            "sample_limit": self.sample_limit,
            "priority": self.priority,
            "config": self.evaluation_config
        }


class CategoryMappingManager:
    """
    Manages category-based model-dataset mappings and evaluation orchestration.
    Central component for organizing and executing category-based evaluations.
    """
    
    def __init__(self, evaluation_data_dir: str = "evaluation_data"):
        self.evaluation_data_dir = Path(evaluation_data_dir)
        self.available_datasets: Set[str] = set()
        self.model_registry: Dict[str, Any] = {}
        
        # Initialize
        self._discover_datasets()
        self._load_model_registry()
    
    def _discover_datasets(self) -> None:
        """Discover all available datasets in the evaluation_data directory"""
        logger.info(f"Discovering datasets in {self.evaluation_data_dir}")
        
        if not self.evaluation_data_dir.exists():
            logger.warning(f"Evaluation data directory {self.evaluation_data_dir} does not exist")
            return
        
        # Scan all subdirectories for JSON files (both direct and nested)
        for category_dir in self.evaluation_data_dir.iterdir():
            if category_dir.is_dir() and category_dir.name not in ["meta", "download_logs"]:
                
                # Check if this is a top-level dataset directory (like ai2d, scienceqa)
                json_files = list(category_dir.glob("*.json"))
                if json_files:
                    has_standard_names = any(f.name in ["train.json", "test.json", "val.json", "validation.json"] for f in json_files)
                    
                    if has_standard_names:
                        # This is a top-level dataset directory
                        dataset_name = category_dir.name
                        self.available_datasets.add(dataset_name)
                        logger.debug(f"Found top-level dataset: {dataset_name}")
                    else:
                        # Regular JSON files in category directory
                        for dataset_file in json_files:
                            if not any(skip in dataset_file.name.lower() for skip in ["summary", "metadata", "config", "download"]):
                                dataset_name = dataset_file.stem
                                self.available_datasets.add(dataset_name)
                                logger.debug(f"Found dataset: {dataset_name} in {category_dir.name}")
                
                # JSON files in subdirectories (for nested dataset structure)
                for subdirectory in category_dir.iterdir():
                    if subdirectory.is_dir():
                        json_files = list(subdirectory.glob("*.json"))
                        if json_files:
                            # Check if any files suggest this is a dataset directory
                            has_standard_names = any(f.name in ["train.json", "test.json", "val.json", "validation.json"] for f in json_files)
                            
                            # Check if files start with the directory name (like medqa/medqa_train.json)
                            has_prefix_pattern = any(f.stem.startswith(subdirectory.name) for f in json_files)
                            
                            if has_standard_names or has_prefix_pattern:
                                # Use subdirectory name as dataset name for structured datasets
                                dataset_name = subdirectory.name
                                self.available_datasets.add(dataset_name)
                                logger.debug(f"Found nested dataset: {dataset_name} in {category_dir.name}/{subdirectory.name}")
                            else:
                                # For non-standard files, use individual file names
                                for dataset_file in json_files:
                                    if not any(skip in dataset_file.name.lower() for skip in ["summary", "metadata", "config", "download"]):
                                        dataset_name = dataset_file.stem
                                        self.available_datasets.add(dataset_name)
                                        logger.debug(f"Found nested dataset: {dataset_name} in {category_dir.name}/{subdirectory.name}")
        
        logger.info(f"Discovered {len(self.available_datasets)} datasets: {sorted(self.available_datasets)}")
    
    def _load_model_registry(self) -> None:
        """Load model registry - stub for now, will integrate with existing registry"""
        # TODO: Integrate with existing model registry from configs/model_configs.py
        logger.info("Model registry loading - will integrate with existing system")
    
    def get_available_datasets(self) -> List[str]:
        """Get all discovered datasets"""
        return sorted(self.available_datasets)
    
    def get_missing_datasets_for_category(self, category_name: str) -> Tuple[List[str], List[str]]:
        """
        Check which datasets are missing for a category
        Returns: (missing_primary, missing_optional)
        """
        category = CATEGORY_REGISTRY.get(category_name.lower())
        if not category:
            return [], []
        
        missing_primary = [d for d in category['primary_datasets'] if d not in self.available_datasets]
        missing_optional = [d for d in category['optional_datasets'] if d not in self.available_datasets]
        
        return missing_primary, missing_optional
    
    def validate_category_readiness(self, category_name: str) -> Dict[str, Any]:
        """Validate if a category is ready for evaluation"""
        category = CATEGORY_REGISTRY.get(category_name.lower())
        if not category:
            return {"ready": False, "error": f"Category '{category_name}' not found"}
        
        missing_primary, missing_optional = self.get_missing_datasets_for_category(category_name)
        
        # Category is ready if ALL primary datasets are available
        ready = len(missing_primary) == 0
        
        available_primary = [d for d in category['primary_datasets'] if d in self.available_datasets]
        available_optional = [d for d in category['optional_datasets'] if d in self.available_datasets]
        
        return {
            "ready": ready,
            "category": category_name,
            "models": category['models'],
            "primary_datasets": {
                "total": len(category['primary_datasets']),
                "available": len(available_primary),
                "missing": missing_primary,
                "available_list": available_primary
            },
            "optional_datasets": {
                "total": len(category['optional_datasets']),
                "available": len(available_optional), 
                "missing": missing_optional,
                "available_list": available_optional
            },
            "evaluation_config": category['category_config']
        }
    
    def generate_evaluation_tasks(
        self, 
        category_name: str, 
        sample_limit: int = 5,
        include_optional: bool = False,
        specific_models: Optional[List[str]] = None,
        specific_datasets: Optional[List[str]] = None
    ) -> List[EvaluationTask]:
        """
        Generate evaluation tasks for a category
        
        Args:
            category_name: Name of the category
            sample_limit: Number of samples per dataset
            include_optional: Whether to include optional datasets
            specific_models: Specific models to evaluate (subset of category)
            specific_datasets: Specific datasets to use (subset of category)
        """
        category = CATEGORY_REGISTRY.get(category_name.lower())
        if not category:
            logger.error(f"Category '{category_name}' not found")
            return []
        
        # Determine models to evaluate
        models_to_eval = specific_models if specific_models else category['models']
        
        # Determine datasets to evaluate  
        if specific_datasets:
            datasets_to_eval = specific_datasets
        else:
            datasets_to_eval = category['primary_datasets'].copy()
            if include_optional:
                datasets_to_eval.extend(category['optional_datasets'])
        
        # Filter to only available datasets
        available_datasets = [d for d in datasets_to_eval if d in self.available_datasets]
        
        if not available_datasets:
            logger.warning(f"No available datasets found for category '{category_name}'")
            return []
        
        # Generate tasks
        tasks = []
        evaluation_config = category['category_config']
        
        for model in models_to_eval:
            for dataset in available_datasets:
                task = EvaluationTask(
                    model_name=model,
                    dataset_name=dataset,
                    category=category_name,
                    sample_limit=sample_limit,
                    evaluation_config=evaluation_config,
                    priority=category['priority']
                )
                tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} evaluation tasks for category '{category_name}'")
        return tasks
    
    def get_category_summary(self, category_name: str) -> Dict[str, Any]:
        """Get comprehensive summary of a category"""
        validation = self.validate_category_readiness(category_name)
        
        if not validation["ready"]:
            return validation
        
        # Add task generation preview
        tasks = self.generate_evaluation_tasks(category_name, sample_limit=5)
        
        validation.update({
            "potential_tasks": len(tasks),
            "task_breakdown": {
                "models": len(validation["models"]),
                "available_datasets": len(validation["primary_datasets"]["available_list"]),
                "total_combinations": len(validation["models"]) * len(validation["primary_datasets"]["available_list"])
            }
        })
        
        return validation
    
    def get_all_categories_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all categories"""
        status = {}
        
        for category_name in CATEGORY_REGISTRY.keys():
            status[category_name] = self.get_category_summary(category_name)
        
        return status
    
    def suggest_evaluation_plan(
        self, 
        category_name: str, 
        max_tasks: int = 25
    ) -> Dict[str, Any]:
        """
        Suggest an optimal evaluation plan for a category
        Considers dataset availability, model count, and resource constraints
        """
        validation = self.validate_category_readiness(category_name)
        
        if not validation["ready"]:
            return {
                "feasible": False,
                "reason": "Category not ready for evaluation",
                "details": validation
            }
        
        category = CATEGORY_REGISTRY[category_name.lower()]
        
        # Calculate task combinations
        available_datasets = validation["primary_datasets"]["available_list"]
        model_count = len(category['models'])
        total_combinations = model_count * len(available_datasets)
        
        # Suggest sample limits based on task count
        if total_combinations <= max_tasks:
            suggested_sample_limit = min(100, max_tasks // total_combinations * 5)
        else:
            suggested_sample_limit = 5  # Conservative for many combinations
        
        return {
            "feasible": True,
            "category": category_name,
            "recommendation": {
                "sample_limit": suggested_sample_limit,
                "include_optional": total_combinations <= max_tasks // 2,
                "estimated_tasks": total_combinations,
                "estimated_total_samples": total_combinations * suggested_sample_limit
            },
            "breakdown": {
                "models": model_count,
                "available_datasets": len(available_datasets),
                "dataset_list": available_datasets
            },
            "execution_estimate": {
                "total_evaluations": total_combinations,
                "estimated_time_minutes": total_combinations * 2,  # Rough estimate
                "resource_requirements": "moderate" if total_combinations <= 20 else "high"
            }
        }


# ================================
# CONVENIENCE FUNCTIONS
# ================================

def get_coding_specialists_manager() -> CategoryMappingManager:
    """Get a configured manager for coding specialists evaluation"""
    manager = CategoryMappingManager()
    return manager


def quick_coding_evaluation_plan(sample_limit: int = 5) -> Dict[str, Any]:
    """Get a quick evaluation plan for coding specialists"""
    manager = get_coding_specialists_manager()
    return manager.suggest_evaluation_plan("coding_specialists", max_tasks=30)


def validate_coding_readiness() -> Dict[str, Any]:
    """Quick validation of coding specialists category readiness"""
    manager = get_coding_specialists_manager()
    return manager.validate_category_readiness("coding_specialists")