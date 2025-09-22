#!/usr/bin/env python3
"""
Dataset Structure Migration Script
Migrates from current mixed structure to improved hybrid approach
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetMigrator:
    def __init__(self, base_path: str = "evaluation_data"):
        self.base_path = Path(base_path)
        self.old_structure = self.base_path
        self.new_structure = self.base_path / "datasets"
        self.backup_path = self.base_path / "backup_original_structure"
        self.registry_path = self.base_path / "registry"
        self.metadata_path = self.base_path / "metadata"
        
        # Category mapping for migration
        self.category_mappings = {
            "coding": {
                "directory": "coding",
                "datasets": ["humaneval", "mbpp", "bigcodebench", "codecontests", "apps"]
            },
            "mathematical": {
                "directory": "math", 
                "datasets": ["gsm8k", "enhanced_math_fixed", "advanced_math_sample"]
            },
            "biomedical": {
                "directory": "biomedical",
                "datasets": ["bioasq", "pubmedqa", "mediqa", "medqa", "pubmedqa_full", "pubmedqa_sample", 
                           "biomedical_extended", "biomedical_sample", "bc5cdr", "ddi"]
            },
            "multimodal": {
                "directory": "multimodal",
                "datasets": ["ai2d", "docvqa", "multimodal_sample", "scienceqa", "textvqa", "chartqa"]
            },
            "scientific": {
                "directory": "scientific",
                "datasets": ["scientific_papers", "scierc"]
            },
            "efficiency": {
                "directory": "efficiency",
                "datasets": ["humaneval", "gsm8k", "arc_challenge"]  # Will be symlinks
            },
            "general": {
                "directory": "general", 
                "datasets": ["arc_challenge", "hellaswag", "mt_bench", "mmlu"]
            },
            "safety": {
                "directory": "safety",
                "datasets": ["toxicity_detection", "truthfulness_fixed"]
            },
            "geospatial": {
                "directory": "geospatial",
                "datasets": ["spatial_reasoning", "coordinate_processing", "address_parsing", 
                           "location_ner", "ner_locations", "geographic_demand", "geographic_features"]
            }
        }

    def create_backup(self):
        """Create backup of original structure"""
        logger.info("Creating backup of original structure...")
        
        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)
        
        # Copy current structure to backup
        shutil.copytree(self.old_structure, self.backup_path, 
                       ignore=shutil.ignore_patterns("backup_*", "datasets", "registry", "metadata"))
        
        logger.info(f"Backup created at: {self.backup_path}")

    def create_new_structure(self):
        """Create new directory structure"""
        logger.info("Creating new directory structure...")
        
        # Create main directories
        self.new_structure.mkdir(exist_ok=True)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)
        
        # Create category directories
        for category_info in self.category_mappings.values():
            category_dir = self.new_structure / category_info["directory"]
            category_dir.mkdir(exist_ok=True)
            logger.info(f"Created category directory: {category_dir}")

    def discover_current_datasets(self) -> Dict[str, Path]:
        """Discover all current dataset files"""
        logger.info("Discovering current datasets...")
        
        datasets = {}
        
        # Scan for JSON files (excluding summaries, metadata, etc.)
        for json_file in self.old_structure.rglob("*.json"):
            # Skip files in backup, new structure, and excluded patterns
            if any(skip in str(json_file) for skip in ["backup_", "datasets/", "registry/", "metadata/"]):
                continue
            
            if any(skip in json_file.name.lower() for skip in ["summary", "metadata", "config", "download"]):
                continue
            
            # Handle nested datasets like medqa
            if json_file.parent.name in ["medqa"] and json_file.stem.startswith(json_file.parent.name):
                dataset_name = json_file.parent.name
            else:
                dataset_name = json_file.stem
            
            datasets[dataset_name] = json_file
            logger.debug(f"Found dataset: {dataset_name} at {json_file}")
        
        logger.info(f"Discovered {len(datasets)} datasets")
        return datasets

    def migrate_dataset(self, dataset_name: str, source_path: Path, target_category: str):
        """Migrate a single dataset to new structure"""
        target_dir = self.new_structure / target_category
        target_file = target_dir / f"{dataset_name}.json"
        
        try:
            # Handle special cases
            if source_path.is_dir():
                # For directories like medqa/, find the main file
                main_files = list(source_path.glob(f"{dataset_name}*.json"))
                if main_files:
                    source_path = main_files[0]  # Use first matching file
                else:
                    # Merge multiple files if needed
                    self.merge_dataset_files(source_path, target_file, dataset_name)
                    return
            
            # Copy file to new location
            shutil.copy2(source_path, target_file)
            logger.info(f"Migrated {dataset_name}: {source_path} → {target_file}")
            
        except Exception as e:
            logger.error(f"Failed to migrate {dataset_name}: {e}")

    def merge_dataset_files(self, source_dir: Path, target_file: Path, dataset_name: str):
        """Merge multiple files for datasets like medqa"""
        logger.info(f"Merging files for dataset: {dataset_name}")
        
        merged_data = {
            "dataset_name": dataset_name,
            "description": f"Merged dataset from {source_dir}",
            "data": []
        }
        
        for json_file in source_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        merged_data["data"].extend(file_data)
                    elif isinstance(file_data, dict) and "data" in file_data:
                        merged_data["data"].extend(file_data["data"])
                    else:
                        merged_data["data"].append(file_data)
            except Exception as e:
                logger.warning(f"Could not merge {json_file}: {e}")
        
        # Save merged data
        with open(target_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        logger.info(f"Merged {len(merged_data['data'])} items into {target_file}")

    def find_dataset_category(self, dataset_name: str) -> str:
        """Find which category a dataset belongs to"""
        for category, info in self.category_mappings.items():
            if dataset_name in info["datasets"]:
                return info["directory"]
        
        # Default category for unmatched datasets
        return "general"

    def create_symlinks_for_shared_datasets(self):
        """Create symlinks for datasets used by multiple categories"""
        logger.info("Creating symlinks for shared datasets...")
        
        shared_datasets = {
            "humaneval": ["coding", "efficiency"],
            "gsm8k": ["mathematical", "efficiency"], 
            "arc_challenge": ["general", "efficiency"]
        }
        
        for dataset_name, categories in shared_datasets.items():
            primary_category = categories[0]
            primary_file = self.new_structure / self.category_mappings[primary_category]["directory"] / f"{dataset_name}.json"
            
            if primary_file.exists():
                for secondary_category in categories[1:]:
                    secondary_dir = self.new_structure / self.category_mappings[secondary_category]["directory"]
                    symlink_path = secondary_dir / f"{dataset_name}.json"
                    
                    if not symlink_path.exists():
                        relative_path = os.path.relpath(primary_file, secondary_dir)
                        symlink_path.symlink_to(relative_path)
                        logger.info(f"Created symlink: {symlink_path} → {primary_file}")

    def create_category_registry(self):
        """Create category registry file"""
        logger.info("Creating category registry...")
        
        registry = {
            "version": "2.0",
            "structure_type": "hybrid",
            "categories": {}
        }
        
        for category, info in self.category_mappings.items():
            category_dir = self.new_structure / info["directory"]
            datasets = [f.stem for f in category_dir.glob("*.json")]
            
            registry["categories"][category] = {
                "directory": info["directory"],
                "datasets": sorted(datasets),
                "dataset_count": len(datasets)
            }
        
        registry_file = self.registry_path / "category_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Created category registry: {registry_file}")
        return registry

    def validate_migration(self, original_datasets: Dict[str, Path]) -> bool:
        """Validate that migration was successful"""
        logger.info("Validating migration...")
        
        success = True
        migrated_count = 0
        
        for category_info in self.category_mappings.values():
            category_dir = self.new_structure / category_info["directory"]
            migrated_files = list(category_dir.glob("*.json"))
            migrated_count += len(migrated_files)
            
            logger.info(f"Category {category_info['directory']}: {len(migrated_files)} datasets")
        
        logger.info(f"Migration validation: {migrated_count} datasets migrated from {len(original_datasets)} discovered")
        
        if migrated_count < len(original_datasets) * 0.8:  # Allow for some datasets being filtered out
            logger.warning("Migration may be incomplete - fewer datasets than expected")
            success = False
        
        return success

    def run_migration(self):
        """Execute complete migration"""
        logger.info("Starting dataset structure migration...")
        
        # Step 1: Backup original structure
        self.create_backup()
        
        # Step 2: Create new structure
        self.create_new_structure()
        
        # Step 3: Discover current datasets
        original_datasets = self.discover_current_datasets()
        
        # Step 4: Migrate datasets
        for dataset_name, source_path in original_datasets.items():
            target_category = self.find_dataset_category(dataset_name)
            self.migrate_dataset(dataset_name, source_path, target_category)
        
        # Step 5: Create symlinks for shared datasets
        self.create_symlinks_for_shared_datasets()
        
        # Step 6: Create registry
        registry = self.create_category_registry()
        
        # Step 7: Validate migration
        success = self.validate_migration(original_datasets)
        
        if success:
            logger.info("✅ Dataset migration completed successfully!")
            logger.info(f"📊 Migrated datasets organized into {len(registry['categories'])} categories")
            logger.info(f"📁 New structure available at: {self.new_structure}")
            logger.info(f"🗃️  Registry created at: {self.registry_path}")
            logger.info(f"💾 Backup available at: {self.backup_path}")
        else:
            logger.error("❌ Migration completed with warnings - please review")
        
        return success

if __name__ == "__main__":
    migrator = DatasetMigrator()
    success = migrator.run_migration()
    
    if success:
        print("🎉 Migration successful! Ready to update discovery code.")
    else:
        print("⚠️  Migration completed with issues. Please review logs.")