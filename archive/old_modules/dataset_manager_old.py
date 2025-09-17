"""
Dataset management for LLM evaluation
Handles downloading, caching, and analysis of evaluation datasets
"""

import os
import json
import gzip
import requests
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset, Dataset
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about an evaluation dataset"""
    name: str
    description: str
    task_type: str  # "function_calling", "coding", "reasoning", "qa", "instruction_following"
    size_mb: float
    num_samples: int
    has_labels: bool
    source_type: str  # "huggingface", "direct_download", "github"
    source_url: str
    license: str
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]

class EvaluationDatasetManager:
    """Manages download and caching of evaluation datasets"""
    
    def __init__(self, cache_dir: str = "evaluation_data", max_total_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_total_size_gb = max_total_size_gb
        self.datasets_info = self._get_dataset_catalog()
        
        # Create subdirectories for different types
        (self.cache_dir / "function_calling").mkdir(exist_ok=True)
        (self.cache_dir / "coding").mkdir(exist_ok=True)
        (self.cache_dir / "reasoning").mkdir(exist_ok=True)
        (self.cache_dir / "qa").mkdir(exist_ok=True)
        (self.cache_dir / "instruction_following").mkdir(exist_ok=True)
        (self.cache_dir / "meta").mkdir(exist_ok=True)
    
    def _get_dataset_catalog(self) -> Dict[str, DatasetInfo]:
        """Define catalog of evaluation datasets"""
        return {
            # Function Calling & Agent Tasks
            "bfcl": DatasetInfo(
                name="Berkeley Function Calling Leaderboard",
                description="Comprehensive function calling benchmark with real APIs",
                task_type="function_calling",
                size_mb=50.0,  # Using subset for now
                num_samples=2000,
                has_labels=True,
                source_type="huggingface",
                source_url="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                license="Apache-2.0"
            ),
            "toolllama": DatasetInfo(
                name="ToolLLaMA",
                description="Tool learning dataset for LLMs",
                task_type="function_calling", 
                size_mb=100.0,  # Using subset
                num_samples=1500,
                has_labels=True,
                source_type="huggingface",
                source_url="ToolBench/ToolBench",
                license="Apache-2.0"
            ),
            
            # Code Generation & Understanding
            "humaneval": DatasetInfo(
                name="HumanEval",
                description="Hand-written programming problems for code generation",
                task_type="coding",
                size_mb=5.0,
                num_samples=164,
                has_labels=True,
                source_type="huggingface",
                source_url="openai_humaneval",
                license="MIT"
            ),
            "mbpp": DatasetInfo(
                name="MBPP",
                description="Mostly Basic Python Problems",
                task_type="coding",
                size_mb=10.0,
                num_samples=974,
                has_labels=True,
                source_type="huggingface",
                source_url="mbpp",
                license="CC-BY-4.0"
            ),
            "codet5": DatasetInfo(
                name="CodeT5",
                description="Code understanding and generation tasks",
                task_type="coding",
                size_mb=150.0,
                num_samples=5000,
                has_labels=True,
                source_type="huggingface", 
                source_url="code_x_glue_ct_code_to_text",
                license="Apache-2.0"
            ),
            
            # Reasoning & Problem Solving
            "gsm8k": DatasetInfo(
                name="GSM8K",
                description="Grade school math word problems",
                task_type="reasoning",
                size_mb=3.0,
                num_samples=1319,
                has_labels=True,
                source_type="huggingface",
                source_url="gsm8k",
                license="MIT"
            ),
            "arc_challenge": DatasetInfo(
                name="ARC-Challenge",
                description="AI2 Reasoning Challenge - challenging set",
                task_type="reasoning",
                size_mb=2.0,
                num_samples=1172,
                has_labels=True,
                source_type="huggingface",
                source_url="ai2_arc",
                license="CC-BY-SA-4.0"
            ),
            "hellaswag": DatasetInfo(
                name="HellaSwag",
                description="Commonsense reasoning about physical situations",
                task_type="reasoning",
                size_mb=20.0,
                num_samples=10042,
                has_labels=True,
                source_type="huggingface",
                source_url="hellaswag",
                license="MIT"
            ),
            
            # Instruction Following
            "alpaca_eval": DatasetInfo(
                name="AlpacaEval",
                description="Instruction following evaluation dataset",
                task_type="instruction_following",
                size_mb=15.0,
                num_samples=805,
                has_labels=True,
                source_type="huggingface",
                source_url="tatsu-lab/alpaca_eval",
                license="Apache-2.0"
            ),
            "mt_bench": DatasetInfo(
                name="MT-Bench",
                description="Multi-turn conversation benchmark",
                task_type="instruction_following",
                size_mb=5.0,
                num_samples=160,
                has_labels=True,
                source_type="huggingface",
                source_url="lmsys/mt_bench_human_judgments",
                license="Apache-2.0"
            ),
            
            # Knowledge & QA
            "mmlu": DatasetInfo(
                name="MMLU",
                description="Massive Multitask Language Understanding",
                task_type="qa",
                size_mb=200.0,
                num_samples=14042,
                has_labels=True,
                source_type="huggingface",
                source_url="cais/mmlu",
                license="MIT"
            ),
            "truthfulqa": DatasetInfo(
                name="TruthfulQA",
                description="Questions that humans would answer falsely due to misconceptions",
                task_type="qa",
                size_mb=5.0,
                num_samples=817,
                has_labels=True,
                source_type="huggingface",
                source_url="truthful_qa",
                license="Apache-2.0"
            )
        }
    
    def get_recommended_datasets(self, task_types: Optional[List[str]] = None, 
                               max_size_gb: Optional[float] = None) -> List[str]:
        """Get recommended datasets for evaluation"""
        if task_types is None:
            # Default: focus on agent/coding capabilities
            task_types = ["function_calling", "coding", "reasoning", "instruction_following"]
        
        if max_size_gb is None:
            max_size_gb = self.max_total_size_gb
        
        # Filter and sort by priority
        recommended = []
        total_size = 0.0
        
        # Priority order for each task type
        priorities = {
            "function_calling": ["bfcl", "toolllama"],
            "coding": ["humaneval", "mbpp", "codet5"],
            "reasoning": ["gsm8k", "arc_challenge", "hellaswag"],
            "instruction_following": ["alpaca_eval", "mt_bench"],
            "qa": ["mmlu", "truthfulqa"]
        }
        
        for task_type in task_types:
            if task_type in priorities:
                for dataset_name in priorities[task_type]:
                    if dataset_name in self.datasets_info:
                        dataset_info = self.datasets_info[dataset_name]
                        if total_size + dataset_info.size_mb/1024 <= max_size_gb:
                            recommended.append(dataset_name)
                            total_size += dataset_info.size_mb/1024
        
        logger.info(f"Recommended {len(recommended)} datasets, total size: {total_size:.2f}GB")
        return recommended
    
    def download_dataset(self, dataset_name: str, subset: Optional[str] = None, 
                        split: Optional[str] = None) -> Dict[str, Any]:
        """Download and cache a specific dataset"""
        if dataset_name not in self.datasets_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.datasets_info[dataset_name]
        cache_path = self.cache_dir / dataset_info.task_type / f"{dataset_name}.json"
        
        # Check if already cached
        if cache_path.exists():
            logger.info(f"Dataset {dataset_name} already cached")
            with open(cache_path) as f:
                return json.load(f)
        
        logger.info(f"Downloading {dataset_name} from {dataset_info.source_url}")
        
        try:
            if dataset_info.source_type == "huggingface":
                dataset = self._download_huggingface_dataset(dataset_info, subset, split)
            elif dataset_info.source_type == "direct_download":
                dataset = self._download_direct_dataset(dataset_info)
            else:
                raise ValueError(f"Unsupported source type: {dataset_info.source_type}")
            
            # Convert to standard format and cache
            processed_data = self._process_dataset(dataset, dataset_info)
            
            # Save to cache
            with open(cache_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            # Save metadata
            self._save_dataset_metadata(dataset_name, processed_data)
            
            logger.info(f"Successfully cached {dataset_name}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            raise
    
    def _download_huggingface_dataset(self, dataset_info: DatasetInfo, 
                                    subset: Optional[str], split: Optional[str]) -> Dataset:
        """Download dataset from Hugging Face"""
        try:
            # Handle special cases
            if dataset_info.name == "ARC-Challenge":
                dataset = load_dataset(dataset_info.source_url, "ARC-Challenge", split="test")
            elif dataset_info.name == "MMLU":
                dataset = load_dataset(dataset_info.source_url, "all", split="test")
            elif dataset_info.name == "TruthfulQA":
                dataset = load_dataset(dataset_info.source_url, "generation", split="validation")
            elif dataset_info.name == "GSM8K":
                dataset = load_dataset(dataset_info.source_url, "main", split="test")
            elif subset:
                dataset = load_dataset(dataset_info.source_url, subset, split=split or "test")
            else:
                # Try common split names
                for split_name in ["test", "validation", "eval", "dev"]:
                    try:
                        dataset = load_dataset(dataset_info.source_url, split=split_name)
                        break
                    except:
                        continue
                else:
                    # Default to first available split
                    dataset = load_dataset(dataset_info.source_url)
                    if isinstance(dataset, dict):
                        dataset = dataset[list(dataset.keys())[0]]
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {dataset_info.source_url}: {e}")
            raise
    
    def _download_direct_dataset(self, dataset_info: DatasetInfo) -> Dict:
        """Download dataset from direct URL"""
        # This would handle direct downloads (GitHub, etc.)
        # For now, placeholder implementation
        raise NotImplementedError("Direct download not yet implemented")
    
    def _process_dataset(self, dataset: Dataset, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Process dataset into standard format"""
        processed = {
            "name": dataset_info.name,
            "task_type": dataset_info.task_type,
            "downloaded_at": datetime.now().isoformat(),
            "samples": [],
            "metadata": {
                "total_samples": len(dataset),
                "has_labels": dataset_info.has_labels,
                "source": dataset_info.source_url,
                "license": dataset_info.license
            }
        }
        
        # Convert samples to standard format
        for i, sample in enumerate(dataset):
            if i >= 1000 and dataset_info.size_mb > 50:  # Limit large datasets for now
                break
                
            processed_sample = self._standardize_sample(sample, dataset_info.task_type)
            processed["samples"].append(processed_sample)
        
        processed["metadata"]["processed_samples"] = len(processed["samples"])
        return processed
    
    def _standardize_sample(self, sample: Dict, task_type: str) -> Dict[str, Any]:
        """Standardize sample format across different datasets"""
        if task_type == "coding":
            return {
                "id": sample.get("task_id", sample.get("id", "")),
                "prompt": sample.get("prompt", sample.get("text", sample.get("question", ""))),
                "expected_output": sample.get("canonical_solution", sample.get("code", sample.get("answer", ""))),
                "test_cases": sample.get("test", sample.get("test_cases", [])),
                "difficulty": sample.get("difficulty", "unknown")
            }
        elif task_type == "function_calling":
            return {
                "id": sample.get("id", ""),
                "prompt": sample.get("question", sample.get("prompt", "")),
                "functions": sample.get("function", sample.get("tools", [])),
                "expected_calls": sample.get("answers", sample.get("expected_output", [])),
                "category": sample.get("category", "general")
            }
        elif task_type == "reasoning":
            return {
                "id": sample.get("id", ""),
                "question": sample.get("question", sample.get("prompt", "")),
                "choices": sample.get("choices", sample.get("options", [])),
                "answer": sample.get("answer", sample.get("answerKey", "")),
                "explanation": sample.get("explanation", sample.get("solution", ""))
            }
        elif task_type == "instruction_following":
            return {
                "id": sample.get("id", ""),
                "instruction": sample.get("instruction", sample.get("prompt", "")),
                "input": sample.get("input", ""),
                "expected_output": sample.get("output", sample.get("response", "")),
                "category": sample.get("category", "general")
            }
        elif task_type == "qa":
            return {
                "id": sample.get("id", ""),
                "question": sample.get("question", sample.get("prompt", "")),
                "choices": sample.get("choices", sample.get("options", [])),
                "answer": sample.get("answer", sample.get("correct_answer", "")),
                "subject": sample.get("subject", sample.get("category", "general"))
            }
        else:
            # Generic format
            return {
                "id": sample.get("id", ""),
                "input": sample.get("input", sample.get("prompt", sample.get("question", ""))),
                "output": sample.get("output", sample.get("answer", sample.get("response", ""))),
                "metadata": {k: v for k, v in sample.items() if k not in ["input", "output", "id"]}
            }
    
    def _save_dataset_metadata(self, dataset_name: str, processed_data: Dict):
        """Save dataset metadata for analysis"""
        metadata_file = self.cache_dir / "meta" / f"{dataset_name}_metadata.json"
        
        metadata = {
            "dataset_name": dataset_name,
            "download_info": self.datasets_info[dataset_name].__dict__,
            "processing_info": processed_data["metadata"],
            "sample_analysis": self._analyze_samples(processed_data["samples"]),
            "cached_at": datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _analyze_samples(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze dataset samples for statistics"""
        if not samples:
            return {}
        
        analysis = {
            "total_samples": len(samples),
            "sample_keys": list(samples[0].keys()) if samples else [],
            "avg_input_length": 0,
            "avg_output_length": 0,
            "empty_outputs": 0,
            "unique_ids": len(set(s.get("id", i) for i, s in enumerate(samples)))
        }
        
        # Analyze text lengths
        input_lengths = []
        output_lengths = []
        
        for sample in samples:
            # Get input text
            input_text = (sample.get("prompt", "") + " " + 
                         sample.get("question", "") + " " + 
                         sample.get("instruction", "") + " " +
                         sample.get("input", "")).strip()
            
            # Get output text  
            output_text = (sample.get("expected_output", "") + " " +
                          sample.get("answer", "") + " " +
                          sample.get("expected_calls", "") + " " +
                          sample.get("output", "")).strip()
            
            input_lengths.append(len(input_text))
            output_lengths.append(len(output_text))
            
            if not output_text:
                analysis["empty_outputs"] += 1
        
        if input_lengths:
            analysis["avg_input_length"] = sum(input_lengths) / len(input_lengths)
            analysis["max_input_length"] = max(input_lengths)
            analysis["min_input_length"] = min(input_lengths)
        
        if output_lengths:
            analysis["avg_output_length"] = sum(output_lengths) / len(output_lengths)
            analysis["max_output_length"] = max(output_lengths)
            analysis["min_output_length"] = min(output_lengths)
        
        return analysis
    
    def download_recommended_datasets(self, task_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Download all recommended datasets"""
        recommended = self.get_recommended_datasets(task_types)
        results = {}
        
        total_size = 0
        for dataset_name in recommended:
            try:
                logger.info(f"Downloading {dataset_name}...")
                data = self.download_dataset(dataset_name)
                results[dataset_name] = data
                
                dataset_info = self.datasets_info[dataset_name]
                total_size += dataset_info.size_mb
                
                logger.info(f"✅ {dataset_name} downloaded successfully ({dataset_info.size_mb}MB)")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {dataset_name}: {e}")
                results[dataset_name] = {"error": str(e)}
        
        logger.info(f"Downloaded {len([r for r in results.values() if 'error' not in r])} datasets, total size: {total_size/1024:.2f}GB")
        return results
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of all available and cached datasets"""
        summary = {
            "available_datasets": len(self.datasets_info),
            "cached_datasets": [],
            "total_available_size_mb": sum(info.size_mb for info in self.datasets_info.values()),
            "total_cached_size_mb": 0,
            "by_task_type": {},
            "datasets": {}
        }
        
        # Check what's cached
        for task_type in ["function_calling", "coding", "reasoning", "instruction_following", "qa"]:
            task_dir = self.cache_dir / task_type
            if task_dir.exists():
                cached_files = list(task_dir.glob("*.json"))
                for cached_file in cached_files:
                    dataset_name = cached_file.stem
                    if dataset_name in self.datasets_info:
                        summary["cached_datasets"].append(dataset_name)
                        summary["total_cached_size_mb"] += self.datasets_info[dataset_name].size_mb
        
        # Organize by task type
        for name, info in self.datasets_info.items():
            task_type = info.task_type
            if task_type not in summary["by_task_type"]:
                summary["by_task_type"][task_type] = {
                    "count": 0,
                    "total_size_mb": 0,
                    "datasets": []
                }
            
            summary["by_task_type"][task_type]["count"] += 1
            summary["by_task_type"][task_type]["total_size_mb"] += info.size_mb
            summary["by_task_type"][task_type]["datasets"].append(name)
            
            summary["datasets"][name] = {
                "info": info.__dict__,
                "cached": name in summary["cached_datasets"]
            }
        
        return summary
    
    def load_cached_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load a cached dataset"""
        if dataset_name not in self.datasets_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.datasets_info[dataset_name]
        cache_path = self.cache_dir / dataset_info.task_type / f"{dataset_name}.json"
        
        if not cache_path.exists():
            logger.warning(f"Dataset {dataset_name} not cached. Call download_dataset() first.")
            return None
        
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cached dataset {dataset_name}: {e}")
            return None

def main():
    """Command line interface for dataset management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage evaluation datasets")
    parser.add_argument("--cache-dir", default="evaluation_data", help="Cache directory")
    parser.add_argument("--download", nargs="+", help="Download specific datasets")
    parser.add_argument("--download-recommended", action="store_true", help="Download recommended datasets")
    parser.add_argument("--summary", action="store_true", help="Show dataset summary")
    parser.add_argument("--task-types", nargs="+", 
                       choices=["function_calling", "coding", "reasoning", "instruction_following", "qa"],
                       help="Limit to specific task types")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = EvaluationDatasetManager(args.cache_dir)
    
    if args.summary:
        summary = manager.get_dataset_summary()
        print(f"\n📊 Dataset Summary:")
        print(f"Available datasets: {summary['available_datasets']}")
        print(f"Cached datasets: {len(summary['cached_datasets'])}")
        print(f"Total available size: {summary['total_available_size_mb']/1024:.2f}GB")
        print(f"Total cached size: {summary['total_cached_size_mb']/1024:.2f}GB")
        
        print(f"\n📋 By Task Type:")
        for task_type, info in summary['by_task_type'].items():
            print(f"  {task_type}: {info['count']} datasets ({info['total_size_mb']/1024:.2f}GB)")
    
    if args.download_recommended:
        print(f"\n📥 Downloading recommended datasets...")
        results = manager.download_recommended_datasets(args.task_types)
        successful = len([r for r in results.values() if 'error' not in r])
        print(f"✅ Successfully downloaded {successful}/{len(results)} datasets")
    
    if args.download:
        for dataset_name in args.download:
            try:
                print(f"\n📥 Downloading {dataset_name}...")
                manager.download_dataset(dataset_name)
                print(f"✅ {dataset_name} downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download {dataset_name}: {e}")

if __name__ == "__main__":
    main()