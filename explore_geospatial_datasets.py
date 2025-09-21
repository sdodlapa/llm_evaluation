#!/usr/bin/env python3
"""
Geospatial Dataset Explorer
==========================

Analyzes the downloaded text-based geospatial datasets and prepares them for integration.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialDatasetExplorer:
    """Explores and analyzes text-based geospatial datasets"""
    
    def __init__(self, base_dir: str = "evaluation_data/text_geospatial"):
        self.base_dir = Path(base_dir)
        self.dataset_summary = {}
    
    def analyze_all_datasets(self):
        """Analyze all downloaded datasets"""
        logger.info("Analyzing geospatial datasets...")
        
        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir():
                self.analyze_dataset(dataset_dir.name)
        
        self.generate_integration_summary()
        return self.dataset_summary
    
    def analyze_dataset(self, dataset_name: str):
        """Analyze a specific dataset"""
        dataset_dir = self.base_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory {dataset_name} not found")
            return
        
        logger.info(f"Analyzing dataset: {dataset_name}")
        
        # Read metadata if available
        metadata_file = dataset_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        
        # Analyze data files
        data_analysis = {}
        for data_file in dataset_dir.glob("*.json"):
            if data_file.name != "metadata.json":
                analysis = self.analyze_data_file(data_file)
                data_analysis[data_file.stem] = analysis
        
        # Combine analysis
        self.dataset_summary[dataset_name] = {
            "metadata": metadata,
            "data_analysis": data_analysis,
            "integration_status": self.assess_integration_readiness(dataset_name, metadata, data_analysis)
        }
    
    def analyze_data_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single data file"""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if not data:
                return {"error": "Empty dataset"}
            
            sample_count = len(data)
            sample_item = data[0] if data else {}
            
            # Analyze structure
            structure_analysis = {
                "sample_count": sample_count,
                "fields": list(sample_item.keys()) if isinstance(sample_item, dict) else [],
                "sample_preview": sample_item,
                "task_types": self.extract_task_types(data),
                "quality_assessment": self.assess_data_quality(data)
            }
            
            return structure_analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_task_types(self, data: List[Dict]) -> List[str]:
        """Extract unique task types from dataset"""
        task_types = set()
        for item in data[:10]:  # Check first 10 samples
            if isinstance(item, dict) and 'task_type' in item:
                task_types.add(item['task_type'])
        return list(task_types)
    
    def assess_data_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """Assess data quality and readiness"""
        if not data:
            return {"status": "empty"}
        
        sample = data[0]
        required_fields = ['input', 'task_type', 'instruction']
        
        quality = {
            "has_required_fields": all(field in sample for field in required_fields),
            "missing_fields": [field for field in required_fields if field not in sample],
            "has_output": 'output' in sample or 'expected_entities' in sample,
            "consistent_structure": self.check_structure_consistency(data[:5])
        }
        
        if quality["has_required_fields"] and quality["has_output"]:
            quality["status"] = "ready"
        elif quality["has_required_fields"]:
            quality["status"] = "needs_output_format"
        else:
            quality["status"] = "needs_restructuring"
        
        return quality
    
    def check_structure_consistency(self, samples: List[Dict]) -> bool:
        """Check if samples have consistent structure"""
        if not samples:
            return False
        
        reference_keys = set(samples[0].keys())
        return all(set(sample.keys()) == reference_keys for sample in samples)
    
    def assess_integration_readiness(self, dataset_name: str, metadata: Dict, data_analysis: Dict) -> Dict[str, Any]:
        """Assess how ready a dataset is for integration"""
        
        integration_status = {
            "ready_for_integration": False,
            "required_modifications": [],
            "evaluation_suitability": "unknown",
            "recommended_task_types": []
        }
        
        # Check data quality across all splits
        all_ready = True
        for split_name, analysis in data_analysis.items():
            if "error" in analysis:
                all_ready = False
                integration_status["required_modifications"].append(f"Fix {split_name}: {analysis['error']}")
            elif analysis.get("quality_assessment", {}).get("status") != "ready":
                all_ready = False
                status = analysis.get("quality_assessment", {}).get("status", "unknown")
                integration_status["required_modifications"].append(f"Fix {split_name}: {status}")
        
        integration_status["ready_for_integration"] = all_ready
        
        # Assess evaluation suitability
        if dataset_name in ["spatial_reasoning", "coordinate_processing", "address_parsing"]:
            integration_status["evaluation_suitability"] = "high"
            integration_status["recommended_task_types"] = ["qa_accuracy", "exact_match"]
        elif "ner" in dataset_name.lower():
            integration_status["evaluation_suitability"] = "high"
            integration_status["recommended_task_types"] = ["ner_f1", "entity_extraction"]
        elif "geographic" in dataset_name.lower():
            integration_status["evaluation_suitability"] = "medium"
            integration_status["recommended_task_types"] = ["qa_accuracy", "knowledge_assessment"]
        else:
            integration_status["evaluation_suitability"] = "low"
        
        return integration_status
    
    def generate_integration_summary(self):
        """Generate summary for integration planning"""
        ready_datasets = []
        needs_work = []
        
        for dataset_name, analysis in self.dataset_summary.items():
            if analysis["integration_status"]["ready_for_integration"]:
                ready_datasets.append({
                    "name": dataset_name,
                    "task_types": analysis["integration_status"]["recommended_task_types"],
                    "suitability": analysis["integration_status"]["evaluation_suitability"],
                    "sample_count": sum(da.get("sample_count", 0) for da in analysis["data_analysis"].values())
                })
            else:
                needs_work.append({
                    "name": dataset_name,
                    "issues": analysis["integration_status"]["required_modifications"]
                })
        
        integration_summary = {
            "ready_for_integration": ready_datasets,
            "needs_modification": needs_work,
            "integration_priority": sorted(ready_datasets, key=lambda x: x["sample_count"], reverse=True),
            "recommended_category_config": self.generate_category_config(ready_datasets)
        }
        
        # Save integration summary
        with open(self.base_dir / "integration_analysis.json", 'w') as f:
            json.dump({
                "dataset_analysis": self.dataset_summary,
                "integration_summary": integration_summary
            }, f, indent=2)
        
        logger.info(f"Integration analysis saved to {self.base_dir}/integration_analysis.json")
        
        return integration_summary
    
    def generate_category_config(self, ready_datasets: List[Dict]) -> Dict[str, Any]:
        """Generate configuration for text_geospatial category"""
        
        # Map datasets to appropriate evaluation types
        dataset_mapping = {
            "spatial_reasoning": "spatial_reasoning_qa",
            "coordinate_processing": "coordinate_math",
            "address_parsing": "address_standardization",
            "location_ner": "location_ner",
            "ner_locations": "location_ner",
            "geographic_features": "geographic_knowledge",
            "geographic_demand": "geographic_analysis"
        }
        
        primary_datasets = []
        optional_datasets = []
        
        for dataset in ready_datasets:
            if dataset["suitability"] == "high":
                primary_datasets.append(dataset_mapping.get(dataset["name"], dataset["name"]))
            else:
                optional_datasets.append(dataset_mapping.get(dataset["name"], dataset["name"]))
        
        category_config = {
            "category_name": "text_geospatial",
            "description": "Text-based geographic understanding and spatial reasoning",
            "models": [
                "qwen25_7b",  # Good geographic knowledge
                "qwen3_8b", 
                "qwen3_14b",
                "mistral_nemo_12b"  # Long context for complex queries
            ],
            "primary_datasets": primary_datasets,
            "optional_datasets": optional_datasets,
            "evaluation_metrics": [
                "spatial_reasoning_accuracy",
                "geographic_f1", 
                "coordinate_accuracy",
                "address_match_score",
                "qa_accuracy"
            ],
            "category_config": {
                "default_sample_limit": 20,
                "timeout_per_sample": 30,
                "temperature": 0.1,
                "top_p": 0.9,
                "enable_coordinate_validation": True,
                "require_geographic_context": True
            }
        }
        
        return category_config
    
    def print_analysis_summary(self):
        """Print a human-readable analysis summary"""
        print("\n" + "="*60)
        print("GEOSPATIAL DATASETS ANALYSIS SUMMARY")
        print("="*60)
        
        for dataset_name, analysis in self.dataset_summary.items():
            print(f"\n📊 {dataset_name.upper()}")
            print("-" * 40)
            
            # Basic info
            metadata = analysis.get("metadata", {})
            print(f"Description: {metadata.get('description', 'N/A')}")
            print(f"Task Type: {metadata.get('task_type', 'N/A')}")
            
            # Data analysis
            total_samples = sum(da.get("sample_count", 0) for da in analysis["data_analysis"].values())
            print(f"Total Samples: {total_samples}")
            
            # Integration status
            integration = analysis["integration_status"]
            status = "✅ Ready" if integration["ready_for_integration"] else "⚠️ Needs Work"
            print(f"Integration Status: {status}")
            
            if integration["required_modifications"]:
                print("Required Modifications:")
                for mod in integration["required_modifications"]:
                    print(f"  - {mod}")
            
            print(f"Evaluation Suitability: {integration['evaluation_suitability']}")

def main():
    """Main execution function"""
    explorer = GeospatialDatasetExplorer()
    explorer.analyze_all_datasets()
    integration_summary = explorer.generate_integration_summary()
    explorer.print_analysis_summary()
    
    print("\n" + "="*60)
    print("INTEGRATION RECOMMENDATIONS")
    print("="*60)
    
    ready_count = len(integration_summary["ready_for_integration"])
    needs_work_count = len(integration_summary["needs_modification"])
    
    print(f"✅ Ready for integration: {ready_count} datasets")
    print(f"⚠️ Need modifications: {needs_work_count} datasets")
    
    if ready_count > 0:
        print("\nPriority datasets for integration:")
        for i, dataset in enumerate(integration_summary["integration_priority"][:3], 1):
            print(f"{i}. {dataset['name']} ({dataset['sample_count']} samples, {dataset['suitability']} suitability)")
    
    print(f"\nNext step: Create text_geospatial category with {ready_count} datasets")

if __name__ == "__main__":
    main()