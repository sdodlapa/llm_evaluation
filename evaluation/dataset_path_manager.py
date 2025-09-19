"""
Dataset Path Manager - Centralized dataset discovery and validation
Resolves path inconsistencies and provides authoritative dataset locations
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetPathManager:
    """Manages dataset path resolution and validation"""
    
    def __init__(self, base_data_dir: str = "evaluation_data"):
        self.base_data_dir = Path(base_data_dir)
        self.category_mappings = self._initialize_category_mappings()
        self.path_cache = {}
        
    def _initialize_category_mappings(self) -> Dict[str, List[str]]:
        """Initialize category to folder mappings"""
        return {
            "coding": ["coding"],
            "mathematical": ["mathematical", "math"],
            "biomedical": ["biomedical", "scientific", "bio"],  # Multiple possible locations
            "multimodal": ["multimodal", "vision", "document"],
            "qa": ["qa", "question_answering"],
            "safety": ["safety", "alignment"],
            "function_calling": ["function_calling", "tools"],
            "scientific": ["scientific", "biomedical"],  # Alias for biomedical
        }
    
    def resolve_dataset_path(self, dataset_name: str, registry_path: str) -> Tuple[Optional[str], bool]:
        """
        Resolve actual dataset path, checking multiple possible locations
        Returns: (actual_path, path_exists)
        """
        # Check cache first
        cache_key = f"{dataset_name}:{registry_path}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Try registry path first
        registry_full_path = self.base_data_dir / registry_path
        if registry_full_path.exists():
            result = (str(registry_full_path), True)
            self.path_cache[cache_key] = result
            return result
        
        # Extract category from registry path
        category = Path(registry_path).parts[0] if Path(registry_path).parts else None
        filename = Path(registry_path).name
        
        # Try alternative locations based on category mappings
        if category in self.category_mappings:
            for alt_category in self.category_mappings[category]:
                alt_path = self.base_data_dir / alt_category / filename
                if alt_path.exists():
                    logger.warning(f"Dataset {dataset_name} found at {alt_path} instead of expected {registry_full_path}")
                    result = (str(alt_path), True)
                    self.path_cache[cache_key] = result
                    return result
        
        # Try scanning all subdirectories as last resort
        for subdir in self.base_data_dir.iterdir():
            if subdir.is_dir():
                candidate_path = subdir / filename
                if candidate_path.exists():
                    logger.warning(f"Dataset {dataset_name} found at {candidate_path} instead of expected {registry_full_path}")
                    result = (str(candidate_path), True)
                    self.path_cache[cache_key] = result
                    return result
        
        # Dataset not found
        logger.error(f"Dataset {dataset_name} not found. Expected: {registry_full_path}")
        result = (None, False)
        self.path_cache[cache_key] = result
        return result
    
    def validate_all_datasets(self, dataset_registry) -> Dict[str, Dict[str, any]]:
        """Validate all datasets in registry and report issues"""
        validation_results = {}
        
        for dataset_name, dataset_info in dataset_registry.datasets.items():
            actual_path, exists = self.resolve_dataset_path(dataset_name, dataset_info.data_path)
            
            expected_full_path = str(self.base_data_dir / dataset_info.data_path)
            
            validation_results[dataset_name] = {
                "expected_path": dataset_info.data_path,
                "expected_full_path": expected_full_path,
                "actual_path": actual_path,
                "exists": exists,
                "needs_path_correction": actual_path != expected_full_path if exists else False
            }
        
        return validation_results
    
    def generate_path_correction_script(self, validation_results: Dict) -> List[Dict]:
        """Generate script to fix dataset path issues"""
        corrections = []
        
        for dataset_name, result in validation_results.items():
            if result["needs_path_correction"] and result["exists"]:
                corrections.append({
                    "dataset": dataset_name,
                    "current_path": result["actual_path"],
                    "expected_path": result["expected_path"],
                    "action": "update_registry" if Path(result["actual_path"]).exists() else "move_file"
                })
        
        return corrections
    
    def create_missing_directories(self) -> List[str]:
        """Create missing category directories"""
        created_dirs = []
        
        for category, alt_names in self.category_mappings.items():
            primary_dir = self.base_data_dir / category
            if not primary_dir.exists():
                primary_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(primary_dir))
                logger.info(f"Created directory: {primary_dir}")
        
        return created_dirs
    
    def get_dataset_info_summary(self, dataset_registry) -> Dict[str, any]:
        """Get summary of dataset path validation"""
        validation_results = self.validate_all_datasets(dataset_registry)
        
        summary = {
            "total_datasets": len(validation_results),
            "existing_datasets": sum(1 for r in validation_results.values() if r["exists"]),
            "missing_datasets": sum(1 for r in validation_results.values() if not r["exists"]),
            "path_mismatches": sum(1 for r in validation_results.values() if r["needs_path_correction"]),
            "categories_found": set(),
            "missing_datasets_list": [],
            "path_mismatch_list": []
        }
        
        for dataset_name, result in validation_results.items():
            if result["exists"] and result["actual_path"]:
                category = Path(result["actual_path"]).parent.name
                summary["categories_found"].add(category)
            
            if not result["exists"]:
                summary["missing_datasets_list"].append(dataset_name)
            
            if result["needs_path_correction"]:
                summary["path_mismatch_list"].append({
                    "dataset": dataset_name,
                    "expected": result["expected_path"],
                    "actual": result["actual_path"]
                })
        
        summary["categories_found"] = list(summary["categories_found"])
        
        return summary

# Global instance
dataset_path_manager = DatasetPathManager()