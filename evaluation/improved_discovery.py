"""
Improved Dataset Discovery Engine - Version 2.0
Simple, predictable, and maintainable dataset discovery for hybrid structure
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass

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

class ImprovedDatasetDiscovery:
    """
    Simplified dataset discovery for hybrid structure:
    evaluation_data/datasets/category_name/dataset_name.json
    """
    
    def __init__(self, base_path: str = "evaluation_data"):
        self.base_path = Path(base_path)
        self.datasets_path = self.base_path / "datasets"
        self.registry_path = self.base_path / "registry"
        self.category_registry_file = self.registry_path / "category_registry.json"
        
        # Cache for performance
        self._discovered_datasets: Optional[Dict[str, Set[str]]] = None
        self._category_registry: Optional[Dict[str, Any]] = None
    
    def discover_datasets(self, force_refresh: bool = False) -> Dict[str, Set[str]]:
        """
        Discover datasets organized by category.
        Simple and predictable: scan each category directory for JSON files.
        
        Returns:
            Dict[category_name, Set[dataset_names]]
        """
        if self._discovered_datasets is not None and not force_refresh:
            return self._discovered_datasets
        
        logger.info(f"Discovering datasets in {self.datasets_path}")
        
        if not self.datasets_path.exists():
            logger.warning(f"Datasets directory {self.datasets_path} does not exist")
            self._discovered_datasets = {}
            return self._discovered_datasets
        
        categories = {}
        total_datasets = 0
        
        # Scan each category directory
        for category_dir in self.datasets_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                datasets = set()
                
                # All JSON files in category are datasets (except metadata files)
                for dataset_file in category_dir.glob("*.json"):
                    # Skip obvious non-dataset files
                    if not any(skip in dataset_file.name.lower() 
                             for skip in ["metadata", "summary", "config", "download"]):
                        datasets.add(dataset_file.stem)
                        logger.debug(f"Found dataset: {dataset_file.stem} in {category_name}")
                
                categories[category_name] = datasets
                total_datasets += len(datasets)
                logger.info(f"Category {category_name}: {len(datasets)} datasets")
        
        logger.info(f"Discovered {total_datasets} datasets across {len(categories)} categories")
        self._discovered_datasets = categories
        return categories
    
    def get_available_datasets(self, category: Optional[str] = None) -> List[str]:
        """Get list of available datasets, optionally filtered by category"""
        all_datasets = self.discover_datasets()
        
        if category:
            return sorted(all_datasets.get(category, set()))
        else:
            # Return all datasets across all categories
            all_dataset_names = set()
            for datasets in all_datasets.values():
                all_dataset_names.update(datasets)
            return sorted(all_dataset_names)
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        categories = self.discover_datasets()
        return sorted(categories.keys())
    
    def validate_category_readiness(self, category_name: str, required_datasets: List[str]) -> Dict[str, Any]:
        """
        Validate if a category is ready for evaluation.
        Simple check: do all required datasets exist?
        """
        available_datasets = self.get_available_datasets(category_name)
        available_set = set(available_datasets)
        required_set = set(required_datasets)
        
        missing_datasets = required_set - available_set
        extra_datasets = available_set - required_set
        
        is_ready = len(missing_datasets) == 0
        
        return {
            "ready": is_ready,
            "category": category_name,
            "required_datasets": required_datasets,
            "available_datasets": available_datasets,
            "missing_datasets": list(missing_datasets),
            "extra_datasets": list(extra_datasets),
            "dataset_count": len(available_datasets)
        }
    
    def get_dataset_path(self, dataset_name: str, category: Optional[str] = None) -> Optional[Path]:
        """Get the full path to a dataset file"""
        if category:
            # Direct lookup if category is known
            dataset_path = self.datasets_path / category / f"{dataset_name}.json"
            if dataset_path.exists():
                return dataset_path
        else:
            # Search across all categories
            all_datasets = self.discover_datasets()
            for cat_name, datasets in all_datasets.items():
                if dataset_name in datasets:
                    dataset_path = self.datasets_path / cat_name / f"{dataset_name}.json"
                    if dataset_path.exists():
                        return dataset_path
        
        return None
    
    def load_category_registry(self) -> Dict[str, Any]:
        """Load category registry if available"""
        if self._category_registry is not None:
            return self._category_registry
        
        if self.category_registry_file.exists():
            try:
                with open(self.category_registry_file, 'r') as f:
                    self._category_registry = json.load(f)
                    return self._category_registry
            except Exception as e:
                logger.warning(f"Could not load category registry: {e}")
        
        # Fallback: create registry from discovery
        return self.create_category_registry()
    
    def create_category_registry(self) -> Dict[str, Any]:
        """Create category registry from current structure"""
        discovered = self.discover_datasets()
        
        registry = {
            "version": "2.0",
            "structure_type": "hybrid",
            "discovery_timestamp": None,
            "categories": {}
        }
        
        for category, datasets in discovered.items():
            registry["categories"][category] = {
                "directory": category,
                "datasets": sorted(datasets),
                "dataset_count": len(datasets)
            }
        
        # Save registry
        if not self.registry_path.exists():
            self.registry_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.category_registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Created category registry: {self.category_registry_file}")
        except Exception as e:
            logger.warning(f"Could not save category registry: {e}")
        
        self._category_registry = registry
        return registry
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset structure"""
        discovered = self.discover_datasets()
        
        stats = {
            "total_categories": len(discovered),
            "total_datasets": sum(len(datasets) for datasets in discovered.values()),
            "categories": {},
            "largest_category": None,
            "smallest_category": None
        }
        
        category_sizes = []
        for category, datasets in discovered.items():
            size = len(datasets)
            stats["categories"][category] = {
                "dataset_count": size,
                "datasets": sorted(datasets)
            }
            category_sizes.append((category, size))
        
        if category_sizes:
            category_sizes.sort(key=lambda x: x[1], reverse=True)
            stats["largest_category"] = {"name": category_sizes[0][0], "count": category_sizes[0][1]}
            stats["smallest_category"] = {"name": category_sizes[-1][0], "count": category_sizes[-1][1]}
        
        return stats


# Compatibility layer for existing code
class CategoryMappingManager:
    """
    Compatibility wrapper to maintain existing API while using improved discovery
    """
    
    def __init__(self, evaluation_data_dir: str = "evaluation_data"):
        self.evaluation_data_dir = Path(evaluation_data_dir)
        self.discovery_engine = ImprovedDatasetDiscovery(evaluation_data_dir)
        
        # Category definitions (updated to match model_categories.py)
        self._category_definitions = {
            "CODING_SPECIALISTS": {
                "models": ["qwen3_8b", "qwen3_14b", "codestral_22b", "qwen3_coder_30b", "deepseek_coder_16b", "starcoder2_15b"],
                "primary_datasets": ["humaneval", "mbpp", "bigcodebench"],
                "category_dir": "coding"
            },
            "MATHEMATICAL_REASONING": {
                "models": ["qwen25_math_7b", "deepseek_math_7b", "wizardmath_70b", "metamath_70b", "qwen25_7b"],
                "primary_datasets": ["gsm8k", "enhanced_math_fixed"],
                "category_dir": "math"
            },
            "BIOMEDICAL_SPECIALISTS": {
                "models": ["biomistral_7b", "biomistral_7b_unquantized", "biomedlm_7b", "medalpaca_7b", "biogpt", 
                          "medalpaca_13b", "clinical_camel_70b", "biogpt_large"],
                "primary_datasets": ["bioasq", "pubmedqa", "mediqa", "medqa"],
                "category_dir": "biomedical",
                "classification_models": ["bio_clinicalbert", "pubmedbert_large"]
            },
            "MULTIMODAL_PROCESSING": {
                "models": ["qwen2_vl_7b", "donut_base", "layoutlmv3_base", "qwen25_vl_7b", "minicpm_v_26", 
                          "llava_next_vicuna_7b", "internvl2_8b", "llama32_vision_90b"],
                "primary_datasets": ["docvqa", "multimodal_sample", "ai2d", "scienceqa"],
                "category_dir": "multimodal"
            },
            "SCIENTIFIC_RESEARCH": {
                "models": ["longformer_large"],
                "primary_datasets": ["scientific_papers", "scierc"],
                "category_dir": "scientific",
                "classification_models": ["scibert_base", "specter2_base"]
            },
            "EFFICIENCY_OPTIMIZED": {
                "models": ["qwen25_0_5b", "qwen25_3b", "phi35_mini"],
                "primary_datasets": ["humaneval", "gsm8k", "arc_challenge"],
                "category_dir": "efficiency"
            },
            "GENERAL_PURPOSE": {
                "models": ["llama31_8b", "mistral_7b", "mistral_nemo_12b", "olmo2_13b", "yi_9b", "yi_1_5_34b", 
                          "gemma2_9b", "llama31_70b", "gemma2_27b", "internlm2_20b"],
                "primary_datasets": ["arc_challenge", "hellaswag", "mt_bench", "mmlu"],
                "category_dir": "general"
            },
            "SAFETY_ALIGNMENT": {
                "models": ["safety_bert", "biomistral_7b", "qwen25_7b"],
                "primary_datasets": ["truthfulness_fixed"],  # Fixed: using actual available dataset name
                "category_dir": "safety",
                "optional_datasets": ["toxicity_detection"]  # This is available
            },
            "MIXTURE_OF_EXPERTS": {
                "models": ["mixtral_8x7b"],
                "primary_datasets": ["mmlu", "hellaswag", "arc_challenge", "humaneval"],
                "category_dir": "general"  # Most datasets (3/4) are in general, humaneval will be found via cross-search
            },
            "REASONING_SPECIALIZED": {
                "models": ["deepseek_r1_distill_llama_70b"],
                "primary_datasets": ["gsm8k", "enhanced_math_fixed", "arc_challenge", "mmlu"],
                "category_dir": "math"  # Math datasets are in math directory, general datasets will be found via cross-search
            },
            "TEXT_GEOSPATIAL": {
                "models": ["qwen25_7b", "qwen3_8b", "qwen3_14b", "mistral_nemo_12b"],
                "primary_datasets": ["spatial_reasoning", "coordinate_processing", "address_parsing", "location_ner", "ner_locations"],
                "category_dir": "geospatial"
            }
        }
        
        # Initialize discovery
        logger.info("Improved discovery engine initialized")
    
    def get_available_datasets(self) -> List[str]:
        """Get all available datasets across all categories"""
        return self.discovery_engine.get_available_datasets()
    
    def validate_category_readiness(self, category_name: str) -> Dict[str, Any]:
        """Validate if a category is ready for evaluation"""
        category_key = category_name.upper()
        if category_key not in self._category_definitions:
            return {"ready": False, "error": f"Unknown category: {category_name}"}
        
        category_info = self._category_definitions[category_key]
        required_datasets = category_info["primary_datasets"]
        category_dir = category_info["category_dir"]
        
        # For these categories, always use cross-directory search since their datasets span multiple directories
        if category_key in ["MIXTURE_OF_EXPERTS", "REASONING_SPECIALIZED"]:
            return self._validate_cross_directory_category(category_key, required_datasets)
        
        # Special handling for other cross-directory categories
        if category_dir == "cross_directory":
            return self._validate_cross_directory_category(category_key, required_datasets)
        
        # Standard single-directory validation
        return self.discovery_engine.validate_category_readiness(category_dir, required_datasets)
    
    def _validate_cross_directory_category(self, category_key: str, required_datasets: List[str]) -> Dict[str, Any]:
        """Handle categories where datasets span multiple directories"""
        all_discovered = self.discovery_engine.discover_datasets()
        available_datasets = []
        missing_datasets = []
        
        # Search for each required dataset across all directories
        for dataset_name in required_datasets:
            found = False
            for category_dir, datasets in all_discovered.items():
                if dataset_name in datasets:
                    available_datasets.append(dataset_name)
                    found = True
                    break
            
            if not found:
                missing_datasets.append(dataset_name)
        
        is_ready = len(missing_datasets) == 0
        
        return {
            "ready": is_ready,
            "category": category_key.lower(),
            "required_datasets": required_datasets,
            "available_datasets": available_datasets,
            "missing_datasets": missing_datasets,
            "extra_datasets": [],  # Not applicable for cross-directory
            "dataset_count": len(available_datasets)
        }
    
    def get_category_datasets(self, category_name: str) -> List[str]:
        """Get datasets for a specific category"""
        category_key = category_name.upper()
        if category_key not in self._category_definitions:
            return []
        
        category_dir = self._category_definitions[category_key]["category_dir"]
        return self.discovery_engine.get_available_datasets(category_dir)
    
    def get_category_status(self, category_name: str) -> Dict[str, Any]:
        """Get category status in the format expected by legacy code"""
        category_key = category_name.upper()
        if category_key not in self._category_definitions:
            return {
                "ready": False,
                "error": f"Unknown category: {category_name}",
                "primary_datasets": {"required": [], "available": [], "missing": []},
                "models": []
            }
        
        category_info = self._category_definitions[category_key]
        required_datasets = category_info["primary_datasets"]
        category_dir = category_info["category_dir"]
        
        # Get validation results using the manager's method (which handles cross-directory search)
        validation = self.validate_category_readiness(category_name)
        
        # Convert to legacy format
        return {
            "ready": validation["ready"],
            "category": category_name,
            "models": category_info["models"],
            "primary_datasets": {
                "required": validation["required_datasets"],
                "available": validation["available_datasets"],
                "missing": validation["missing_datasets"]
            },
            "all_datasets": validation["available_datasets"]
        }

    def get_all_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get all category definitions with readiness status"""
        categories = {}
        
        for category_name, category_info in self._category_definitions.items():
            readiness = self.validate_category_readiness(category_name)
            categories[category_name] = {
                "models": category_info["models"],
                "primary_datasets": category_info["primary_datasets"],
                "category_dir": category_info["category_dir"],
                "readiness": readiness
            }
        
        return categories

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
        category_key = category_name.upper()
        if category_key not in self._category_definitions:
            logger.error(f"Category '{category_name}' not found")
            return []

        category_info = self._category_definitions[category_key]
        
        # Determine models to evaluate
        models_to_eval = specific_models if specific_models else category_info['models']
        
        # Determine datasets to evaluate  
        if specific_datasets:
            datasets_to_eval = specific_datasets
        else:
            datasets_to_eval = category_info['primary_datasets'].copy()
            # Note: optional_datasets not implemented in improved discovery yet
            # if include_optional:
            #     datasets_to_eval.extend(category_info['optional_datasets'])
        
        # Get available datasets for this category
        category_dir = category_info["category_dir"]
        available_datasets_in_category = self.discovery_engine.get_available_datasets(category_dir)
        
        # Filter to only available datasets
        available_datasets = [d for d in datasets_to_eval if d in available_datasets_in_category]
        
        if not available_datasets:
            logger.warning(f"No available datasets found for category '{category_name}'")
            return []
        
        # Generate tasks
        tasks = []
        evaluation_config = category_info.get('category_config', {})
        
        for model in models_to_eval:
            for dataset in available_datasets:
                task = EvaluationTask(
                    model_name=model,
                    dataset_name=dataset,
                    category=category_name.lower(),
                    sample_limit=sample_limit,
                    evaluation_config=evaluation_config,
                    priority="HIGH"
                )
                tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} evaluation tasks for category '{category_name}'")
        return tasks