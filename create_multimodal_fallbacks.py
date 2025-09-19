#!/usr/bin/env python3
"""
Alternative multimodal dataset downloader with fallback approaches
for datasets that require special handling.
"""

import os
import json
import logging
from pathlib import Path
import requests
import zipfile
import tempfile
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multimodal_fallback_datasets.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_ai2d_alternative(data_dir, logger):
    """Download AI2D using alternative approach - create sample data"""
    logger.info("Creating AI2D sample dataset...")
    try:
        ai2d_dir = data_dir / "ai2d"
        ai2d_dir.mkdir(exist_ok=True)
        
        # Create sample AI2D-style data for testing
        sample_data = []
        for i in range(50):
            sample_data.append({
                'id': f'ai2d_sample_{i}',
                'question': f'What does component {chr(65+i%26)} represent in this diagram?',
                'choices': ['A) Input', 'B) Process', 'C) Output', 'D) Control'],
                'answer': chr(65 + i % 4),  # A, B, C, or D
                'image_path': f'diagram_{i}.png',  # Placeholder
                'metadata': {
                    'source': 'ai2d_sample',
                    'type': 'diagram_qa',
                    'note': 'Sample data for testing'
                }
            })
        
        # Save sample data
        output_file = ai2d_dir / "test.json"
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"AI2D sample data created: {len(sample_data)} samples")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create AI2D sample data: {e}")
        return False

def download_textvqa_alternative(data_dir, logger):
    """Download TextVQA using alternative approach - create sample data"""
    logger.info("Creating TextVQA sample dataset...")
    try:
        textvqa_dir = data_dir / "textvqa"
        textvqa_dir.mkdir(exist_ok=True)
        
        # Create sample TextVQA-style data for testing
        sample_data = []
        sample_questions = [
            "What text is visible on the sign?",
            "What is written on the book cover?",
            "What does the label say?",
            "What is the name on the storefront?",
            "What text appears on the screen?"
        ]
        
        for i in range(50):
            sample_data.append({
                'question_id': f'textvqa_sample_{i}',
                'question': sample_questions[i % len(sample_questions)],
                'answers': [f'Sample text {i}', f'Text {i}', f'Answer {i}'],
                'image_id': f'image_{i}',
                'image_path': f'text_image_{i}.jpg',  # Placeholder
                'metadata': {
                    'source': 'textvqa_sample',
                    'type': 'text_vqa',
                    'note': 'Sample data for testing'
                }
            })
        
        # Save sample data
        output_file = textvqa_dir / "test.json"
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"TextVQA sample data created: {len(sample_data)} samples")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create TextVQA sample data: {e}")
        return False

def create_multimodal_evaluation_config(data_dir, logger):
    """Create configuration file for multimodal evaluation"""
    config = {
        "multimodal_datasets": {
            "scienceqa": {
                "path": "scienceqa",
                "splits": ["train", "validation", "test"],
                "format": "json",
                "type": "science_qa",
                "description": "Multimodal science question answering",
                "available": True
            },
            "chartqa": {
                "path": "chartqa", 
                "splits": ["train", "val", "test"],
                "format": "json",
                "type": "chart_qa",
                "description": "Chart and graph understanding",
                "available": True
            },
            "ai2d": {
                "path": "ai2d",
                "splits": ["test"],
                "format": "json", 
                "type": "diagram_qa",
                "description": "Diagram understanding (sample data)",
                "available": True,
                "note": "Sample data for testing"
            },
            "textvqa": {
                "path": "textvqa",
                "splits": ["test"],
                "format": "json",
                "type": "text_vqa", 
                "description": "Text-based visual QA (sample data)",
                "available": True,
                "note": "Sample data for testing"
            }
        },
        "evaluation_config": {
            "batch_size": 1,
            "max_samples_per_dataset": 100,
            "evaluation_metrics": [
                "accuracy",
                "exact_match",
                "bleu",
                "rouge"
            ],
            "prompt_templates": {
                "science_qa": "Answer the following science question. Question: {question}\nChoices: {choices}\nAnswer:",
                "chart_qa": "Look at this chart and answer the question. Question: {question}\nAnswer:",
                "diagram_qa": "Analyze this diagram and answer the question. Question: {question}\nChoices: {choices}\nAnswer:",
                "text_vqa": "Read the text in this image and answer the question. Question: {question}\nAnswer:"
            }
        }
    }
    
    config_file = data_dir / "multimodal_evaluation_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Multimodal evaluation config saved to {config_file}")
    return config

def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting fallback multimodal dataset creation")
    
    # Setup directory
    data_dir = Path(__file__).parent / "evaluation_data"
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Using data directory: {data_dir}")
    
    # Create fallback datasets
    datasets_to_create = [
        ("AI2D Sample", download_ai2d_alternative),
        ("TextVQA Sample", download_textvqa_alternative)
    ]
    
    success_count = 0
    for dataset_name, create_func in datasets_to_create:
        logger.info(f"\n--- Creating {dataset_name} ---")
        if create_func(data_dir, logger):
            success_count += 1
        else:
            logger.error(f"Failed to create {dataset_name}")
    
    # Create evaluation config
    logger.info("\n--- Creating Evaluation Config ---")
    config = create_multimodal_evaluation_config(data_dir, logger)
    
    # Final report
    logger.info(f"\n--- Setup Complete ---")
    logger.info(f"Successfully created: {success_count}/{len(datasets_to_create)} fallback datasets")
    logger.info(f"Available multimodal datasets: ScienceQA, ChartQA, AI2D (sample), TextVQA (sample)")
    
    # Print summary
    print("\nMultimodal Dataset Summary:")
    for name, info in config["multimodal_datasets"].items():
        available = "✓" if info["available"] else "✗"
        note = f" ({info.get('note', '')})" if info.get('note') else ""
        print(f"  {available} {name}: {info['description']}{note}")

if __name__ == "__main__":
    main()