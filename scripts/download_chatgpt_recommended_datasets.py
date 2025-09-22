#!/usr/bin/env python3
"""
ChatGPT-Recommended Dataset Downloader
=====================================

Downloads plug-and-play evaluation datasets recommended by ChatGPT for H100 cluster evaluation.
All datasets are available via datasets.load_dataset() and work out-of-the-box with no custom preprocessing.

Categories covered:
- General-purpose / Knowledge & Commonsense
- Reasoning / Math  
- Coding
- Function-calling / Tools
- Vision-Language

Based on ChatGPT recommendations for comprehensive LLM evaluation.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict, Dataset
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_download_chatgpt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for a dataset download"""
    hf_id: str
    category: str
    description: str
    splits: List[str]
    size_estimate_mb: int
    priority: str  # "HIGH", "MEDIUM", "LOW"
    evaluation_ready: bool = True
    requires_preprocessing: bool = False
    notes: str = ""

class ChatGPTDatasetDownloader:
    """Downloads ChatGPT-recommended evaluation datasets"""
    
    def __init__(self, base_data_dir: str = "/home/sdodl001_odu_edu/llm_evaluation/evaluation_data"):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track download stats
        self.download_stats = {
            "total_datasets": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "skipped_existing": 0,
            "total_size_mb": 0
        }
        
        # Define ChatGPT-recommended datasets
        self.dataset_configs = self._initialize_dataset_configs()
        
    def _initialize_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """Initialize all ChatGPT-recommended dataset configurations"""
        return {
            # ================================
            # GENERAL-PURPOSE / KNOWLEDGE & COMMONSENSE
            # ================================
            "mmlu": DatasetConfig(
                hf_id="cais/mmlu",
                category="general_knowledge",
                description="Massive Multitask Language Understanding - 57 subjects, MCQ",
                splits=["test", "validation", "dev"],
                size_estimate_mb=150,
                priority="HIGH",
                notes="Core benchmark for general knowledge evaluation"
            ),
            
            "arc_challenge": DatasetConfig(
                hf_id="allenai/ai2_arc",
                category="general_knowledge", 
                description="AI2 Reasoning Challenge - Challenge split",
                splits=["test", "validation"],
                size_estimate_mb=25,
                priority="HIGH",
                notes="Use ARC-Challenge split for difficulty"
            ),
            
            "hellaswag": DatasetConfig(
                hf_id="Rowan/hellaswag",
                category="general_knowledge",
                description="Common sense reasoning benchmark",
                splits=["test", "validation"],
                size_estimate_mb=50,
                priority="HIGH",
                notes="Ready MC format from Rowan"
            ),
            
            "truthfulqa_mc": DatasetConfig(
                hf_id="EleutherAI/truthful_qa_mc",
                category="general_knowledge",
                description="TruthfulQA Multiple Choice format",
                splits=["test", "validation"],
                size_estimate_mb=15,
                priority="HIGH",
                notes="Tests truthfulness and factual accuracy"
            ),
            
            "winogrande": DatasetConfig(
                hf_id="allenai/winogrande",
                category="general_knowledge",
                description="Winograd Schema Challenge",
                splits=["test", "validation"],
                size_estimate_mb=20,
                priority="MEDIUM",
                notes="Commonsense reasoning"
            ),
            
            "piqa": DatasetConfig(
                hf_id="ybisk/piqa",
                category="general_knowledge",
                description="Physical Interaction QA",
                splits=["test", "validation"],
                size_estimate_mb=30,
                priority="MEDIUM",
                notes="Physical commonsense reasoning"
            ),
            
            "boolq": DatasetConfig(
                hf_id="google/boolq",
                category="general_knowledge",
                description="Boolean Questions for reading comprehension",
                splits=["test", "validation"],
                size_estimate_mb=40,
                priority="MEDIUM",
                notes="Yes/No reading comprehension"
            ),
            
            # ================================
            # REASONING / MATH
            # ================================
            "gsm8k": DatasetConfig(
                hf_id="openai/gsm8k",
                category="mathematical_reasoning",
                description="Grade School Math 8K problems",
                splits=["test", "train"],
                size_estimate_mb=25,
                priority="HIGH",
                notes="Core math reasoning benchmark"
            ),
            
            "math_competition": DatasetConfig(
                hf_id="hendrycks/competition_math",
                category="mathematical_reasoning",
                description="MATH - competition mathematics problems",
                splits=["test", "train"],
                size_estimate_mb=100,
                priority="HIGH",
                notes="Advanced mathematical reasoning"
            ),
            
            "bigbench_hard": DatasetConfig(
                hf_id="lukaemon/bbh",
                category="reasoning_specialized",
                description="BIG-bench Hard - challenging reasoning tasks",
                splits=["test"],
                size_estimate_mb=75,
                priority="HIGH",
                notes="Task collection for advanced reasoning"
            ),
            
            "gpqa": DatasetConfig(
                hf_id="Idavidrein/gpqa",
                category="reasoning_specialized",
                description="Graduate-level Google-Proof Q&A in STEM",
                splits=["test"],
                size_estimate_mb=35,
                priority="HIGH",
                notes="Expert-written STEM questions, MCQ format"
            ),
            
            # ================================
            # CODING
            # ================================
            "humaneval": DatasetConfig(
                hf_id="openai/openai_humaneval",
                category="coding_specialists",
                description="HumanEval - Python code generation with unit tests",
                splits=["test"],
                size_estimate_mb=10,
                priority="HIGH",
                notes="Core coding benchmark with automatic evaluation"
            ),
            
            "mbpp": DatasetConfig(
                hf_id="Muennighoff/mbpp",
                category="coding_specialists",
                description="Mostly Basic Python Problems",
                splits=["test", "train", "validation"],
                size_estimate_mb=15,
                priority="HIGH",
                notes="Python coding problems with test cases"
            ),
            
            "apps": DatasetConfig(
                hf_id="codeparrot/apps",
                category="coding_specialists", 
                description="APPS - Automated Programming Progress Standard",
                splits=["test", "train"],
                size_estimate_mb=200,
                priority="MEDIUM",
                notes="Programming problems with test cases"
            ),
            
            "code_contests": DatasetConfig(
                hf_id="deepmind/code_contests",
                category="coding_specialists",
                description="CodeContests - programming competitions",
                splits=["test", "train", "valid"],
                size_estimate_mb=300,
                priority="MEDIUM",
                notes="Competitive programming problems"
            ),
            
            "bigcodebench": DatasetConfig(
                hf_id="bigcode/bigcodebench",
                category="coding_specialists",
                description="BigCodeBench - comprehensive code evaluation",
                splits=["test"],
                size_estimate_mb=50,
                priority="HIGH",
                notes="Ready dataset with pip harness for evaluation"
            ),
            
            # ================================
            # FUNCTION-CALLING / TOOLS
            # ================================
            "bfcl": DatasetConfig(
                hf_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                category="function_calling",
                description="Berkeley Function-Calling Leaderboard",
                splits=["test"],
                size_estimate_mb=25,
                priority="HIGH",
                notes="CSV tasks with function calling evaluation"
            ),
            
            # ================================
            # VISION-LANGUAGE (MULTIMODAL)
            # ================================
            "mmmu": DatasetConfig(
                hf_id="MMMU/MMMU",
                category="multimodal_processing",
                description="Massive Multi-discipline Multimodal Understanding",
                splits=["test", "validation", "dev"],
                size_estimate_mb=500,
                priority="HIGH",
                notes="Images + Q/A packaged for direct load with HF + PIL"
            ),
            
            "scienceqa": DatasetConfig(
                hf_id="lmms-lab/ScienceQA",
                category="multimodal_processing",
                description="Science Question Answering with diagrams",
                splits=["test", "train", "val"],
                size_estimate_mb=300,
                priority="HIGH",
                notes="Image + diagram Q/A for scientific reasoning"
            ),
            
            "docvqa": DatasetConfig(
                hf_id="lmms-lab/DocVQA",
                category="multimodal_processing",
                description="Document Visual Question Answering",
                splits=["test", "validation"],
                size_estimate_mb=200,
                priority="MEDIUM",
                notes="Formatted for lmms-eval one-click evaluation"
            ),
            
            "chartqa": DatasetConfig(
                hf_id="lmms-lab/ChartQA",
                category="multimodal_processing",
                description="Chart Question Answering",
                splits=["test", "train", "val"],
                size_estimate_mb=150,
                priority="MEDIUM",
                notes="Chart understanding and analysis"
            ),
            
            "textcaps": DatasetConfig(
                hf_id="lmms-lab/TextCaps",
                category="multimodal_processing",
                description="TextCaps - OCR-centric VQA variant",
                splits=["test", "train", "val"],
                size_estimate_mb=250,
                priority="LOW",
                notes="OCR and text understanding in images"
            )
        }
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """Download a specific dataset with robust error handling"""
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.base_data_dir / config.category / dataset_name
        
        # Check if already exists
        if dataset_dir.exists() and not force_redownload:
            logger.info(f"✓ Dataset {dataset_name} already exists, skipping")
            self.download_stats["skipped_existing"] += 1
            return True
            
        logger.info(f"📥 Downloading {dataset_name} ({config.hf_id})")
        logger.info(f"   Category: {config.category}")
        logger.info(f"   Description: {config.description}")
        logger.info(f"   Estimated size: {config.size_estimate_mb}MB")
        
        try:
            # Create category directory
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            start_time = time.time()
            dataset = None
            
            # Handle special cases for different dataset structures
            if dataset_name == "arc_challenge":
                # For ARC, we want the Challenge subset
                dataset = load_dataset(config.hf_id, "ARC-Challenge")
            elif dataset_name == "mmlu":
                # MMLU needs 'all' config or specific subject
                try:
                    dataset = load_dataset(config.hf_id, "all")
                except:
                    # Fallback to default config
                    dataset = load_dataset(config.hf_id)
            elif dataset_name == "winogrande":
                # WinoGrande needs specific config
                try:
                    dataset = load_dataset(config.hf_id, "winogrande_xl")
                except:
                    dataset = load_dataset(config.hf_id, "winogrande_l")
            elif dataset_name == "truthfulqa_mc":
                # TruthfulQA MC needs specific config
                try:
                    dataset = load_dataset(config.hf_id, "multiple_choice")
                except:
                    dataset = load_dataset(config.hf_id)
            elif dataset_name == "piqa":
                # PIQA sometimes needs specific handling
                dataset = load_dataset(config.hf_id)
            elif dataset_name == "gsm8k":
                # GSM8K needs main config
                try:
                    dataset = load_dataset(config.hf_id, "main")
                except:
                    dataset = load_dataset(config.hf_id)
            elif dataset_name == "math_competition":
                # MATH competition dataset
                dataset = load_dataset(config.hf_id)
            elif dataset_name == "bigbench_hard":
                # BBH is a task collection
                dataset = load_dataset(config.hf_id)
            elif dataset_name == "gpqa":
                # GPQA might need specific config
                try:
                    dataset = load_dataset(config.hf_id, "gpqa_main")
                except:
                    try:
                        dataset = load_dataset(config.hf_id, "main")
                    except:
                        dataset = load_dataset(config.hf_id)
            elif dataset_name == "mbpp":
                # MBPP might need specific config
                try:
                    dataset = load_dataset(config.hf_id, "full")
                except:
                    dataset = load_dataset(config.hf_id)
            elif dataset_name == "apps":
                # APPS might be gated or need config
                try:
                    dataset = load_dataset(config.hf_id, "all")
                except:
                    try:
                        dataset = load_dataset(config.hf_id, "train")
                    except:
                        logger.warning(f"APPS dataset may be gated or require authentication")
                        dataset = load_dataset(config.hf_id)
            elif dataset_name == "bfcl":
                # BFCL might need special handling for CSV format
                try:
                    dataset = load_dataset(config.hf_id)
                except:
                    logger.warning(f"BFCL dataset might need authentication or different access method")
                    return False
            elif dataset_name == "mmmu":
                # MMMU might need config
                try:
                    dataset = load_dataset(config.hf_id, "all")
                except:
                    dataset = load_dataset(config.hf_id)
            elif dataset_name == "scienceqa":
                # ScienceQA might need specific config
                try:
                    dataset = load_dataset(config.hf_id, "no_image")  # Try text-only first
                except:
                    dataset = load_dataset(config.hf_id)
            elif dataset_name == "docvqa":
                # DocVQA might need specific config
                dataset = load_dataset(config.hf_id)
            else:
                # Standard download with fallback
                try:
                    dataset = load_dataset(config.hf_id)
                except Exception as e:
                    if "gated" in str(e).lower() or "authentication" in str(e).lower():
                        logger.warning(f"Dataset {dataset_name} appears to be gated. You may need to request access at https://huggingface.co/{config.hf_id}")
                        return False
                    else:
                        raise e
            
            if dataset is None:
                logger.error(f"Failed to load dataset {dataset_name}")
                return False
            
            # Save dataset locally
            dataset.save_to_disk(str(dataset_dir))
            
            download_time = time.time() - start_time
            
            # Save metadata
            metadata = {
                "dataset_name": dataset_name,
                "hf_id": config.hf_id,
                "category": config.category,
                "description": config.description,
                "download_time_seconds": download_time,
                "splits": list(dataset.keys()) if hasattr(dataset, 'keys') else ['unknown'],
                "priority": config.priority,
                "evaluation_ready": config.evaluation_ready,
                "notes": config.notes,
                "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(dataset_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Successfully downloaded {dataset_name} in {download_time:.1f}s")
            if hasattr(dataset, 'keys'):
                logger.info(f"   Splits: {list(dataset.keys())}")
            logger.info(f"   Saved to: {dataset_dir}")
            
            self.download_stats["successful_downloads"] += 1
            self.download_stats["total_size_mb"] += config.size_estimate_mb
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "authentication" in error_msg.lower():
                logger.warning(f"⚠️  Dataset {dataset_name} is gated and requires access approval")
                logger.warning(f"   Request access at: https://huggingface.co/{config.hf_id}")
            elif "config" in error_msg.lower():
                logger.warning(f"⚠️  Dataset {dataset_name} has configuration issues: {error_msg}")
                logger.warning(f"   May need specific config name or manual handling")
            else:
                logger.error(f"❌ Failed to download {dataset_name}: {error_msg}")
            
            self.download_stats["failed_downloads"] += 1
            return False
    
    def download_by_category(self, category: str, priority_filter: Optional[str] = None) -> Dict[str, bool]:
        """Download all datasets in a specific category"""
        logger.info(f"📦 Downloading datasets for category: {category}")
        
        results = {}
        for name, config in self.dataset_configs.items():
            if config.category == category:
                if priority_filter and config.priority != priority_filter:
                    continue
                results[name] = self.download_dataset(name)
                
        return results
    
    def download_by_priority(self, priority: str) -> Dict[str, bool]:
        """Download all datasets with specified priority"""
        logger.info(f"🚀 Downloading {priority} priority datasets")
        
        results = {}
        for name, config in self.dataset_configs.items():
            if config.priority == priority:
                results[name] = self.download_dataset(name)
                
        return results
    
    def download_core_benchmarks(self) -> Dict[str, bool]:
        """Download essential core benchmarks for immediate evaluation"""
        core_datasets = [
            "mmlu", "hellaswag", "arc_challenge", "gsm8k", 
            "humaneval", "truthfulqa_mc", "bigbench_hard"
        ]
        
        logger.info("🎯 Downloading core benchmarks for immediate evaluation")
        
        results = {}
        for dataset_name in core_datasets:
            results[dataset_name] = self.download_dataset(dataset_name)
            
        return results
    
    def download_multimodal_datasets(self) -> Dict[str, bool]:
        """Download multimodal datasets for vision-language evaluation"""
        multimodal_datasets = ["mmmu", "scienceqa", "docvqa", "chartqa"]
        
        logger.info("🖼️ Downloading multimodal datasets")
        
        results = {}
        for dataset_name in multimodal_datasets:
            results[dataset_name] = self.download_dataset(dataset_name)
            
        return results
    
    def download_all(self, skip_low_priority: bool = True) -> Dict[str, bool]:
        """Download all ChatGPT-recommended datasets"""
        logger.info("🌟 Downloading all ChatGPT-recommended datasets")
        
        results = {}
        for name, config in self.dataset_configs.items():
            if skip_low_priority and config.priority == "LOW":
                logger.info(f"⏭️ Skipping low priority dataset: {name}")
                continue
            results[name] = self.download_dataset(name)
            
        return results
    
    def retry_failed_downloads(self) -> Dict[str, bool]:
        """Retry downloading failed datasets with improved error handling"""
        logger.info("🔄 Retrying failed dataset downloads with improved handling")
        
        # List of previously failed datasets based on our log
        failed_datasets = [
            "mmlu", "truthfulqa_mc", "winogrande", "piqa", 
            "gsm8k", "math_competition", "bigbench_hard", "gpqa",
            "mbpp", "apps", "bfcl", "mmmu", "scienceqa", "docvqa"
        ]
        
        results = {}
        for dataset_name in failed_datasets:
            if dataset_name in self.dataset_configs:
                logger.info(f"\n📥 Retrying {dataset_name}...")
                results[dataset_name] = self.download_dataset(dataset_name)
            else:
                logger.warning(f"Dataset {dataset_name} not found in configurations")
                results[dataset_name] = False
        
        return results
    
    def print_dataset_summary(self):
        """Print summary of available datasets"""
        print("\n" + "="*80)
        print("CHATGPT-RECOMMENDED DATASETS SUMMARY")
        print("="*80)
        
        by_category = {}
        for name, config in self.dataset_configs.items():
            if config.category not in by_category:
                by_category[config.category] = []
            by_category[config.category].append((name, config))
        
        for category, datasets in by_category.items():
            print(f"\n📁 {category.upper().replace('_', ' ')}:")
            for name, config in datasets:
                priority_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}[config.priority]
                print(f"  {priority_icon} {name}")
                print(f"     HF ID: {config.hf_id}")
                print(f"     Size: ~{config.size_estimate_mb}MB | Priority: {config.priority}")
                if config.notes:
                    print(f"     Notes: {config.notes}")
        
        total_size = sum(config.size_estimate_mb for config in self.dataset_configs.values())
        high_priority_size = sum(config.size_estimate_mb for config in self.dataset_configs.values() 
                                if config.priority == "HIGH")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total datasets: {len(self.dataset_configs)}")
        print(f"   HIGH priority: {len([c for c in self.dataset_configs.values() if c.priority == 'HIGH'])}")
        print(f"   Total estimated size: ~{total_size}MB ({total_size/1024:.1f}GB)")
        print(f"   HIGH priority size: ~{high_priority_size}MB ({high_priority_size/1024:.1f}GB)")
        print("="*80)
    
    def print_download_stats(self):
        """Print download statistics"""
        print("\n" + "="*60)
        print("DOWNLOAD STATISTICS")
        print("="*60)
        print(f"Total datasets processed: {self.download_stats['total_datasets']}")
        print(f"Successful downloads: {self.download_stats['successful_downloads']}")
        print(f"Failed downloads: {self.download_stats['failed_downloads']}")
        print(f"Skipped (already exist): {self.download_stats['skipped_existing']}")
        print(f"Total size downloaded: ~{self.download_stats['total_size_mb']}MB")
        print("="*60)

def main():
    """Main execution function"""
    downloader = ChatGPTDatasetDownloader()
    
    # Print dataset summary
    downloader.print_dataset_summary()
    
    # Get user choice
    print("\nChoose download option:")
    print("1. Core benchmarks only (essential for immediate testing)")
    print("2. HIGH priority datasets only")
    print("3. All datasets except LOW priority")
    print("4. Multimodal datasets only")
    print("5. All datasets including LOW priority")
    print("6. Specific category")
    print("7. Retry failed downloads with improved error handling")
    print("8. Show summary only (no download)")
    
    try:
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == "1":
            results = downloader.download_core_benchmarks()
        elif choice == "2":
            results = downloader.download_by_priority("HIGH")
        elif choice == "3":
            results = downloader.download_all(skip_low_priority=True)
        elif choice == "4":
            results = downloader.download_multimodal_datasets()
        elif choice == "5":
            results = downloader.download_all(skip_low_priority=False)
        elif choice == "6":
            categories = ["general_knowledge", "mathematical_reasoning", "reasoning_specialized", 
                         "coding_specialists", "function_calling", "multimodal_processing"]
            print(f"Available categories: {', '.join(categories)}")
            category = input("Enter category: ").strip()
            results = downloader.download_by_category(category)
        elif choice == "7":
            results = downloader.retry_failed_downloads()
        elif choice == "8":
            print("\n✓ Summary displayed. No downloads performed.")
            return
        else:
            print("Invalid choice. Exiting.")
            return
            
        # Update total datasets processed
        downloader.download_stats["total_datasets"] = len(results)
        
        # Print results
        downloader.print_download_stats()
        
        print("\n✅ Dataset download process completed!")
        print("📁 Downloaded datasets are available in:")
        print(f"   {downloader.base_data_dir}")
        print("\n📋 Ready for evaluation with:")
        print("   - lm-evaluation-harness")
        print("   - BigCodeBench pip + scripts")
        print("   - lmms-eval / VLMEvalKit")
        print("   - Your custom evaluation pipeline")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Download interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()