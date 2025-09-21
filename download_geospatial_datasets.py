#!/usr/bin/env python3
"""
Geospatial Dataset Downloader
=============================

Downloads text-based geospatial datasets for integration into our LLM evaluation framework.
Focuses on text-only datasets that work with our existing text-based pipeline.
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging
from datasets import load_dataset
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialDatasetDownloader:
    """Downloads and processes text-based geospatial datasets"""
    
    def __init__(self, base_dir: str = "evaluation_data/text_geospatial"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Define target datasets
        self.datasets = {
            "location_ner": {
                "huggingface_id": "dshut002/LocationData",
                "description": "Location entity recognition dataset",
                "task_type": "named_entity_recognition",
                "sample_limit": 200
            },
            "ner_locations": {
                "huggingface_id": "NochnoyRitzar/ner_locations_dataset_pretokenized_xlm_roberta_large_conll",
                "description": "Pretokenized location NER dataset",
                "task_type": "named_entity_recognition", 
                "sample_limit": 150
            },
            "geographic_features": {
                "huggingface_id": "LeroyDyer/geographic_features",
                "description": "Geographic feature descriptions",
                "task_type": "geographic_knowledge",
                "sample_limit": 100
            },
            "wikidata_geographic": {
                "huggingface_id": "saikeerthana00/Wikidata_Geographic_Datasets",
                "description": "Wikidata geographic knowledge",
                "task_type": "geographic_qa",
                "sample_limit": 100
            },
            "geographic_demand": {
                "huggingface_id": "neuralsorcerer/geographic-product-demand",
                "description": "Geographic product demand data",
                "task_type": "geographic_analysis",
                "sample_limit": 50
            }
        }
        
        # Custom datasets to create
        self.custom_datasets = {
            "spatial_reasoning": {
                "description": "Custom spatial reasoning questions",
                "task_type": "spatial_reasoning"
            },
            "coordinate_processing": {
                "description": "Coordinate conversion and distance calculation",
                "task_type": "coordinate_math"
            },
            "address_parsing": {
                "description": "Address standardization tasks",
                "task_type": "address_parsing"
            }
        }
    
    def download_huggingface_dataset(self, dataset_name: str, config: Dict[str, Any]) -> bool:
        """Download a dataset from HuggingFace"""
        try:
            logger.info(f"Downloading {dataset_name} from {config['huggingface_id']}")
            
            # Create dataset directory
            dataset_dir = self.base_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Load dataset
            dataset = load_dataset(config['huggingface_id'], trust_remote_code=True)
            
            # Process and save each split
            for split_name, split_data in dataset.items():
                logger.info(f"Processing split: {split_name} ({len(split_data)} samples)")
                
                # Limit samples if specified
                sample_limit = config.get('sample_limit', len(split_data))
                if len(split_data) > sample_limit:
                    split_data = split_data.select(range(sample_limit))
                
                # Convert to our format
                processed_data = self._process_dataset_split(split_data, config['task_type'])
                
                # Save as JSON
                output_file = dataset_dir / f"{split_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved {len(processed_data)} samples to {output_file}")
            
            # Save dataset metadata
            metadata = {
                "name": dataset_name,
                "huggingface_id": config['huggingface_id'],
                "description": config['description'],
                "task_type": config['task_type'],
                "splits": list(dataset.keys()),
                "total_samples": sum(len(split) for split in dataset.values()),
                "processed_samples": sum(len(split.select(range(min(len(split), config.get('sample_limit', len(split)))))) 
                                       for split in dataset.values())
            }
            
            with open(dataset_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _process_dataset_split(self, split_data, task_type: str) -> List[Dict[str, Any]]:
        """Process dataset split into our standard format"""
        processed = []
        
        for i, sample in enumerate(split_data):
            try:
                if task_type == "named_entity_recognition":
                    processed_sample = self._process_ner_sample(sample, i)
                elif task_type == "geographic_knowledge":
                    processed_sample = self._process_knowledge_sample(sample, i)
                elif task_type == "geographic_qa":
                    processed_sample = self._process_qa_sample(sample, i)
                elif task_type == "geographic_analysis":
                    processed_sample = self._process_analysis_sample(sample, i)
                else:
                    processed_sample = self._process_generic_sample(sample, i)
                
                if processed_sample:
                    processed.append(processed_sample)
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {str(e)}")
                continue
        
        return processed
    
    def _process_ner_sample(self, sample: Dict, idx: int) -> Dict[str, Any]:
        """Process NER sample for location extraction"""
        # Handle different NER formats
        if 'text' in sample:
            text = sample['text']
        elif 'tokens' in sample:
            text = ' '.join(sample['tokens'])
        else:
            text = str(sample)
        
        # Extract location entities if available
        entities = []
        if 'entities' in sample:
            entities = [ent for ent in sample['entities'] if 'location' in ent.get('type', '').lower()]
        elif 'tags' in sample or 'labels' in sample:
            # Convert BIO tags to entities
            entities = self._bio_to_entities(sample)
        
        return {
            "id": f"ner_{idx}",
            "task_type": "location_ner",
            "input": text,
            "expected_entities": entities,
            "instruction": "Extract all location entities from the following text:",
            "metadata": {
                "source": "huggingface_ner",
                "entity_count": len(entities)
            }
        }
    
    def _process_knowledge_sample(self, sample: Dict, idx: int) -> Dict[str, Any]:
        """Process geographic knowledge sample"""
        if 'text' in sample:
            text = sample['text']
        elif 'description' in sample:
            text = sample['description']
        else:
            text = str(sample)
        
        return {
            "id": f"knowledge_{idx}",
            "task_type": "geographic_knowledge",
            "input": text,
            "instruction": "Answer questions about this geographic information:",
            "metadata": {
                "source": "geographic_knowledge",
                "type": sample.get('type', 'unknown')
            }
        }
    
    def _process_qa_sample(self, sample: Dict, idx: int) -> Dict[str, Any]:
        """Process geographic Q&A sample"""
        question = sample.get('question', sample.get('text', str(sample)))
        answer = sample.get('answer', sample.get('label', ''))
        
        return {
            "id": f"geo_qa_{idx}",
            "task_type": "geographic_qa",
            "input": question,
            "output": answer,
            "instruction": "Answer this geographic question:",
            "metadata": {
                "source": "geographic_qa",
                "difficulty": sample.get('difficulty', 'unknown')
            }
        }
    
    def _process_analysis_sample(self, sample: Dict, idx: int) -> Dict[str, Any]:
        """Process geographic analysis sample"""
        return {
            "id": f"analysis_{idx}",
            "task_type": "geographic_analysis", 
            "input": str(sample),
            "instruction": "Analyze this geographic data:",
            "metadata": {
                "source": "geographic_analysis"
            }
        }
    
    def _process_generic_sample(self, sample: Dict, idx: int) -> Dict[str, Any]:
        """Process generic sample"""
        return {
            "id": f"generic_{idx}",
            "task_type": "geographic_text",
            "input": str(sample),
            "instruction": "Process this geographic text:",
            "metadata": {
                "source": "generic"
            }
        }
    
    def _bio_to_entities(self, sample: Dict) -> List[Dict[str, Any]]:
        """Convert BIO tags to entity list"""
        entities = []
        # Simplified entity extraction - would need proper BIO tag processing
        if 'tags' in sample:
            for i, tag in enumerate(sample['tags']):
                if 'LOC' in str(tag) or 'LOCATION' in str(tag):
                    entities.append({
                        "text": sample.get('tokens', [''])[i] if 'tokens' in sample else '',
                        "type": "LOCATION",
                        "position": i
                    })
        return entities
    
    def create_custom_datasets(self):
        """Create custom geospatial datasets"""
        
        # Spatial Reasoning Dataset
        spatial_reasoning_data = [
            {
                "id": "spatial_001",
                "task_type": "spatial_reasoning",
                "input": "What direction is France from Spain?",
                "output": "North",
                "instruction": "Determine the directional relationship between these locations:",
                "metadata": {"relation_type": "directional", "difficulty": "easy"}
            },
            {
                "id": "spatial_002", 
                "task_type": "spatial_reasoning",
                "input": "Which city is between New York and Boston?",
                "output": "There are several cities between New York and Boston, including New Haven, Hartford, and Springfield.",
                "instruction": "Identify locations between the given places:",
                "metadata": {"relation_type": "between", "difficulty": "medium"}
            },
            {
                "id": "spatial_003",
                "task_type": "spatial_reasoning", 
                "input": "What country contains the Alps mountain range?",
                "output": "The Alps span across multiple countries including France, Switzerland, Italy, Austria, and others.",
                "instruction": "Identify which countries contain this geographic feature:",
                "metadata": {"relation_type": "containment", "difficulty": "medium"}
            },
            {
                "id": "spatial_004",
                "task_type": "spatial_reasoning",
                "input": "What is the capital of the country that borders both France and Germany?",
                "output": "The countries that border both France and Germany are Switzerland (capital: Bern) and Luxembourg (capital: Luxembourg City).",
                "instruction": "Use spatial relationships to answer this geographic question:",
                "metadata": {"relation_type": "borders", "difficulty": "hard"}
            },
            {
                "id": "spatial_005",
                "task_type": "spatial_reasoning",
                "input": "Which ocean is west of California?",
                "output": "Pacific Ocean",
                "instruction": "Determine the directional relationship between these locations:",
                "metadata": {"relation_type": "directional", "difficulty": "easy"}
            }
        ]
        
        # Coordinate Processing Dataset
        coordinate_data = [
            {
                "id": "coord_001",
                "task_type": "coordinate_math",
                "input": "Calculate the approximate distance between New York City (40.7128° N, 74.0060° W) and Los Angeles (34.0522° N, 118.2437° W).",
                "output": "Approximately 2,445 miles or 3,935 kilometers",
                "instruction": "Calculate the distance between these coordinates:",
                "metadata": {"calculation_type": "distance", "difficulty": "medium"}
            },
            {
                "id": "coord_002",
                "task_type": "coordinate_math", 
                "input": "Convert the coordinates 40.7128° N, 74.0060° W to decimal degrees format.",
                "output": "40.7128, -74.0060",
                "instruction": "Convert these coordinates to the specified format:",
                "metadata": {"calculation_type": "conversion", "difficulty": "easy"}
            },
            {
                "id": "coord_003",
                "task_type": "coordinate_math",
                "input": "What is the timezone for coordinates 51.5074° N, 0.1278° W?",
                "output": "GMT (Greenwich Mean Time) / UTC+0",
                "instruction": "Determine the timezone for these coordinates:",
                "metadata": {"calculation_type": "timezone", "difficulty": "easy"}
            }
        ]
        
        # Address Parsing Dataset
        address_data = [
            {
                "id": "addr_001",
                "task_type": "address_parsing",
                "input": "123 Main St, New York, NY 10001",
                "output": {
                    "street_number": "123",
                    "street_name": "Main St",
                    "city": "New York", 
                    "state": "NY",
                    "postal_code": "10001",
                    "country": "USA"
                },
                "instruction": "Parse this address into structured components:",
                "metadata": {"address_type": "US_standard", "difficulty": "easy"}
            },
            {
                "id": "addr_002",
                "task_type": "address_parsing",
                "input": "221B Baker Street, London NW1 6XE, UK",
                "output": {
                    "street_number": "221B",
                    "street_name": "Baker Street",
                    "city": "London",
                    "postal_code": "NW1 6XE",
                    "country": "UK"
                },
                "instruction": "Parse this address into structured components:",
                "metadata": {"address_type": "UK_standard", "difficulty": "medium"}
            }
        ]
        
        # Save custom datasets
        custom_data = {
            "spatial_reasoning": spatial_reasoning_data,
            "coordinate_processing": coordinate_data,
            "address_parsing": address_data
        }
        
        for dataset_name, data in custom_data.items():
            dataset_dir = self.base_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            with open(dataset_dir / "train.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            # Create metadata
            metadata = {
                "name": dataset_name,
                "description": self.custom_datasets[dataset_name]["description"],
                "task_type": self.custom_datasets[dataset_name]["task_type"],
                "splits": ["train"],
                "total_samples": len(data),
                "custom_dataset": True
            }
            
            with open(dataset_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created custom dataset {dataset_name} with {len(data)} samples")
    
    def download_all(self):
        """Download all geospatial datasets"""
        logger.info("Starting geospatial dataset download...")
        
        # Download HuggingFace datasets
        results = {}
        for dataset_name, config in self.datasets.items():
            results[dataset_name] = self.download_huggingface_dataset(dataset_name, config)
        
        # Create custom datasets
        self.create_custom_datasets()
        
        # Generate summary
        successful = sum(1 for success in results.values() if success)
        total_hf = len(self.datasets)
        total_custom = len(self.custom_datasets)
        
        logger.info(f"Dataset download complete!")
        logger.info(f"HuggingFace datasets: {successful}/{total_hf} successful")
        logger.info(f"Custom datasets: {total_custom} created")
        logger.info(f"Total datasets: {successful + total_custom}")
        
        # Create overall summary
        summary = {
            "download_date": pd.Timestamp.now().isoformat(),
            "huggingface_datasets": {name: success for name, success in results.items()},
            "custom_datasets": list(self.custom_datasets.keys()),
            "total_successful": successful + total_custom,
            "base_directory": str(self.base_dir)
        }
        
        with open(self.base_dir / "download_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main execution function"""
    downloader = GeospatialDatasetDownloader()
    summary = downloader.download_all()
    
    print("\n" + "="*50)
    print("GEOSPATIAL DATASET DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Base directory: {downloader.base_dir}")
    print(f"Total datasets: {summary['total_successful']}")
    print(f"HuggingFace datasets: {len([s for s in summary['huggingface_datasets'].values() if s])}")
    print(f"Custom datasets: {len(summary['custom_datasets'])}")
    print("\nDatasets ready for exploration and integration!")

if __name__ == "__main__":
    main()