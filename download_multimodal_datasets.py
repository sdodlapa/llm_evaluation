#!/usr/bin/env python3
"""
Download and prepare multimodal datasets for evaluation.
Focuses on vision-language datasets: AI2D, ScienceQA, ChartQA, TextVQA
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multimodal_dataset_download.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_dataset_directory():
    """Ensure the evaluation_data directory exists"""
    data_dir = Path(__file__).parent / "evaluation_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def download_ai2d(data_dir, logger):
    """Download AI2D dataset for diagram understanding"""
    logger.info("Downloading AI2D dataset...")
    try:
        # AI2D is available on HuggingFace
        dataset = load_dataset("allenai/ai2d", trust_remote_code=True)
        
        # Create AI2D directory
        ai2d_dir = data_dir / "ai2d"
        ai2d_dir.mkdir(exist_ok=True)
        
        # Save dataset splits
        for split_name, split_data in dataset.items():
            output_file = ai2d_dir / f"{split_name}.json"
            logger.info(f"Saving AI2D {split_name} split to {output_file}")
            
            # Convert to list of dictionaries for easier processing
            samples = []
            for item in split_data:
                samples.append({
                    'id': item.get('id', ''),
                    'question': item.get('question', ''),
                    'choices': item.get('choices', []),
                    'answer': item.get('answer', ''),
                    'image': item.get('image', None),  # PIL Image object
                    'metadata': {
                        'source': 'ai2d',
                        'type': 'diagram_qa'
                    }
                })
            
            with open(output_file, 'w') as f:
                json.dump(samples[:100], f, indent=2, default=str)  # Limit to first 100 for testing
            
            logger.info(f"AI2D {split_name}: {len(samples)} samples (saved first 100)")
        
        logger.info("AI2D dataset downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download AI2D: {e}")
        return False

def download_scienceqa(data_dir, logger):
    """Download ScienceQA dataset for multimodal science reasoning"""
    logger.info("Downloading ScienceQA dataset...")
    try:
        # ScienceQA is available on HuggingFace
        dataset = load_dataset("derek-thomas/ScienceQA", trust_remote_code=True)
        
        # Create ScienceQA directory
        scienceqa_dir = data_dir / "scienceqa"
        scienceqa_dir.mkdir(exist_ok=True)
        
        # Save dataset splits
        for split_name, split_data in dataset.items():
            output_file = scienceqa_dir / f"{split_name}.json"
            logger.info(f"Saving ScienceQA {split_name} split to {output_file}")
            
            # Convert to list of dictionaries
            samples = []
            for item in split_data:
                samples.append({
                    'id': item.get('id', ''),
                    'question': item.get('question', ''),
                    'choices': item.get('choices', []),
                    'answer': item.get('answer', ''),
                    'image': item.get('image', None),
                    'subject': item.get('subject', ''),
                    'category': item.get('category', ''),
                    'metadata': {
                        'source': 'scienceqa',
                        'type': 'science_qa'
                    }
                })
            
            with open(output_file, 'w') as f:
                json.dump(samples[:100], f, indent=2, default=str)  # Limit to first 100 for testing
            
            logger.info(f"ScienceQA {split_name}: {len(samples)} samples (saved first 100)")
        
        logger.info("ScienceQA dataset downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download ScienceQA: {e}")
        return False

def download_chartqa(data_dir, logger):
    """Download ChartQA dataset for chart understanding"""
    logger.info("Downloading ChartQA dataset...")
    try:
        # ChartQA is available on HuggingFace
        dataset = load_dataset("HuggingFaceM4/ChartQA", trust_remote_code=True)
        
        # Create ChartQA directory
        chartqa_dir = data_dir / "chartqa"
        chartqa_dir.mkdir(exist_ok=True)
        
        # Save dataset splits
        for split_name, split_data in dataset.items():
            output_file = chartqa_dir / f"{split_name}.json"
            logger.info(f"Saving ChartQA {split_name} split to {output_file}")
            
            # Convert to list of dictionaries
            samples = []
            for item in split_data:
                samples.append({
                    'id': item.get('id', ''),
                    'question': item.get('query', ''),
                    'answer': item.get('label', ''),
                    'image': item.get('image', None),
                    'chart_type': item.get('chart_type', ''),
                    'metadata': {
                        'source': 'chartqa',
                        'type': 'chart_qa'
                    }
                })
            
            with open(output_file, 'w') as f:
                json.dump(samples[:100], f, indent=2, default=str)  # Limit to first 100 for testing
            
            logger.info(f"ChartQA {split_name}: {len(samples)} samples (saved first 100)")
        
        logger.info("ChartQA dataset downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download ChartQA: {e}")
        return False

def download_textvqa(data_dir, logger):
    """Download TextVQA dataset for text-based visual question answering"""
    logger.info("Downloading TextVQA dataset...")
    try:
        # TextVQA is available on HuggingFace
        dataset = load_dataset("textvqa", trust_remote_code=True)
        
        # Create TextVQA directory
        textvqa_dir = data_dir / "textvqa"
        textvqa_dir.mkdir(exist_ok=True)
        
        # Save dataset splits
        for split_name, split_data in dataset.items():
            output_file = textvqa_dir / f"{split_name}.json"
            logger.info(f"Saving TextVQA {split_name} split to {output_file}")
            
            # Convert to list of dictionaries
            samples = []
            for item in split_data:
                samples.append({
                    'id': item.get('question_id', ''),
                    'question': item.get('question', ''),
                    'answers': item.get('answers', []),
                    'image': item.get('image', None),
                    'image_id': item.get('image_id', ''),
                    'metadata': {
                        'source': 'textvqa',
                        'type': 'text_vqa'
                    }
                })
            
            with open(output_file, 'w') as f:
                json.dump(samples[:100], f, indent=2, default=str)  # Limit to first 100 for testing
            
            logger.info(f"TextVQA {split_name}: {len(samples)} samples (saved first 100)")
        
        logger.info("TextVQA dataset downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download TextVQA: {e}")
        return False

def create_dataset_summary(data_dir, logger):
    """Create a summary of downloaded datasets"""
    summary = {
        "multimodal_datasets": {
            "ai2d": {
                "description": "Diagram understanding and reasoning",
                "type": "Visual Question Answering",
                "splits": [],
                "sample_count": 0
            },
            "scienceqa": {
                "description": "Multimodal science question answering",
                "type": "Science QA",
                "splits": [],
                "sample_count": 0
            },
            "chartqa": {
                "description": "Chart and graph understanding",
                "type": "Chart QA",
                "splits": [],
                "sample_count": 0
            },
            "textvqa": {
                "description": "Text-based visual question answering",
                "type": "Text VQA",
                "splits": [],
                "sample_count": 0
            }
        },
        "download_timestamp": str(Path().cwd()),
        "note": "Sample datasets limited to 100 items each for testing"
    }
    
    # Count actual files and samples
    for dataset_name in summary["multimodal_datasets"].keys():
        dataset_dir = data_dir / dataset_name
        if dataset_dir.exists():
            splits = []
            total_samples = 0
            for json_file in dataset_dir.glob("*.json"):
                split_name = json_file.stem
                splits.append(split_name)
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        total_samples += len(data)
                except:
                    pass
            
            summary["multimodal_datasets"][dataset_name]["splits"] = splits
            summary["multimodal_datasets"][dataset_name]["sample_count"] = total_samples
    
    # Save summary
    summary_file = data_dir / "multimodal_datasets_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Dataset summary saved to {summary_file}")
    return summary

def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting multimodal dataset download process")
    
    # Setup directory
    data_dir = ensure_dataset_directory()
    logger.info(f"Using data directory: {data_dir}")
    
    # Download datasets
    datasets_to_download = [
        ("AI2D", download_ai2d),
        ("ScienceQA", download_scienceqa),
        ("ChartQA", download_chartqa),
        ("TextVQA", download_textvqa)
    ]
    
    success_count = 0
    for dataset_name, download_func in datasets_to_download:
        logger.info(f"\n--- Downloading {dataset_name} ---")
        if download_func(data_dir, logger):
            success_count += 1
        else:
            logger.error(f"Failed to download {dataset_name}")
    
    # Create summary
    logger.info("\n--- Creating Dataset Summary ---")
    summary = create_dataset_summary(data_dir, logger)
    
    # Final report
    logger.info(f"\n--- Download Complete ---")
    logger.info(f"Successfully downloaded: {success_count}/{len(datasets_to_download)} datasets")
    logger.info(f"Data directory: {data_dir}")
    
    # Print summary
    print("\nDataset Summary:")
    for name, info in summary["multimodal_datasets"].items():
        print(f"  {name}: {info['sample_count']} samples, splits: {info['splits']}")

if __name__ == "__main__":
    main()