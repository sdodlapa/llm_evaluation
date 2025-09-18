#!/usr/bin/env python3
"""
Comprehensive Dataset Downloader for LLM Evaluation Framework
Downloads and prepares datasets for specialized model evaluation
"""

import os
import json
import time
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import shutil
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetSource:
    """Configuration for a dataset source"""
    name: str
    category: str  # coding, mathematics, genomics, multimodal, efficiency, etc.
    url: str
    format: str  # json, csv, jsonl, zip, tar.gz
    description: str
    expected_samples: int
    license: str
    evaluation_type: str
    preprocessing_required: bool = True
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DatasetDownloader:
    """Automated dataset downloader with validation and preprocessing"""
    
    def __init__(self, base_path: str = "evaluation_data"):
        self.base_path = Path(base_path)
        self.download_log_path = self.base_path / "download_logs"
        self.temp_path = self.base_path / "temp"
        
        # Create necessary directories
        for path in [self.base_path, self.download_log_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset catalog
        self.dataset_sources = self._initialize_dataset_catalog()
        
    def _initialize_dataset_catalog(self) -> Dict[str, DatasetSource]:
        """Initialize comprehensive dataset catalog"""
        return {
            # ===== CODING DATASETS =====
            "bigcodebench": DatasetSource(
                name="bigcodebench",
                category="coding",
                url="https://huggingface.co/datasets/bigcode/bigcodebench/resolve/main/data/train-00000-of-00001.parquet",
                format="parquet",
                description="Comprehensive code generation benchmark with complex tasks",
                expected_samples=1140,
                license="Apache 2.0",
                evaluation_type="code_execution",
                metadata={"difficulty": "hard", "languages": ["python"], "domains": ["algorithms", "data_structures"]}
            ),
            
            "codecontests": DatasetSource(
                name="codecontests",
                category="coding", 
                url="https://huggingface.co/datasets/deepmind/code_contests/resolve/main/train.jsonl",
                format="jsonl",
                description="Programming contest problems from competitive programming",
                expected_samples=13500,
                license="Apache 2.0", 
                evaluation_type="code_execution",
                metadata={"difficulty": "medium-hard", "languages": ["python", "cpp"], "source": "competitive_programming"}
            ),
            
            "apps": DatasetSource(
                name="apps",
                category="coding",
                url="https://huggingface.co/datasets/codeparrot/apps/resolve/main/train/train-00000-of-00001.parquet",
                format="parquet", 
                description="Measuring coding challenge competence with 10,000 problems",
                expected_samples=5000,
                license="Apache 2.0",
                evaluation_type="code_execution",
                metadata={"difficulty": "introductory-interview", "languages": ["python"], "source": "coding_interviews"}
            ),
            
            # ===== MATHEMATICS DATASETS =====
            "math_competition": DatasetSource(
                name="math_competition", 
                category="mathematics",
                url="https://huggingface.co/datasets/hendrycks/competition_math/resolve/main/train.jsonl",
                format="jsonl",
                description="MATH dataset - Competition mathematics problems",
                expected_samples=7500,
                license="MIT",
                evaluation_type="numerical_accuracy",
                metadata={"difficulty": "high_school_competition", "domains": ["algebra", "geometry", "number_theory"]}
            ),
            
            "mathqa": DatasetSource(
                name="mathqa",
                category="mathematics",
                url="https://huggingface.co/datasets/math_qa/resolve/main/train.json",
                format="json", 
                description="Math word problems with operation programs",
                expected_samples=29837,
                license="Apache 2.0",
                evaluation_type="numerical_accuracy",
                metadata={"difficulty": "grade_school", "format": "word_problems"}
            ),
            
            "aime": DatasetSource(
                name="aime",
                category="mathematics",
                url="https://huggingface.co/datasets/AI-MO/aimo-validation-aime/resolve/main/aime_validation.jsonl",
                format="jsonl",
                description="American Invitational Mathematics Examination problems",
                expected_samples=240,
                license="MIT",
                evaluation_type="numerical_accuracy", 
                metadata={"difficulty": "competition", "source": "AIME", "target": "high_school"}
            ),
            
            # ===== MULTIMODAL DATASETS =====
            "scienceqa": DatasetSource(
                name="scienceqa", 
                category="multimodal",
                url="https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/main/train/train-00000-of-00001.parquet",
                format="parquet",
                description="Science question answering with images and text",
                expected_samples=19206,
                license="Creative Commons",
                evaluation_type="multiple_choice_accuracy",
                metadata={"modalities": ["text", "images"], "domains": ["science"], "grade_levels": ["3-12"]}
            ),
            
            "vqa_v2": DatasetSource(
                name="vqa_v2",
                category="multimodal", 
                url="https://huggingface.co/datasets/HuggingFaceM4/VQAv2/resolve/main/train.jsonl",
                format="jsonl",
                description="Visual Question Answering dataset v2.0",
                expected_samples=443757,
                license="Creative Commons",
                evaluation_type="free_form_accuracy",
                metadata={"modalities": ["text", "images"], "domains": ["natural_images"], "task": "vqa"}
            ),
            
            "chartqa": DatasetSource(
                name="chartqa",
                category="multimodal",
                url="https://huggingface.co/datasets/HuggingFaceM4/ChartQA/resolve/main/train.jsonl",
                format="jsonl", 
                description="Question answering on charts and graphs",
                expected_samples=18271,
                license="MIT",
                evaluation_type="free_form_accuracy",
                metadata={"modalities": ["text", "charts"], "domains": ["data_visualization"], "task": "chart_qa"}
            ),
            
            # ===== GENOMICS/BIOINFORMATICS DATASETS =====
            "genomics_benchmark": DatasetSource(
                name="genomics_benchmark",
                category="genomics",
                url="https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/resolve/main/human_dataset.csv",
                format="csv",
                description="Long-range genomics benchmark for sequence analysis",
                expected_samples=6000,
                license="Apache 2.0",
                evaluation_type="sequence_classification",
                metadata={"species": "human", "task": "promoter_detection", "sequence_length": "long_range"}
            ),
            
            "protein_sequences": DatasetSource(
                name="protein_sequences", 
                category="genomics",
                url="https://huggingface.co/datasets/Rostlab/ProstT5/resolve/main/train.csv",
                format="csv",
                description="Protein sequence and structure prediction dataset",
                expected_samples=25000,
                license="MIT",
                evaluation_type="sequence_classification", 
                metadata={"task": "protein_structure", "domains": ["structural_biology"]}
            ),
            
            "bioasq": DatasetSource(
                name="bioasq",
                category="genomics",
                url="https://huggingface.co/datasets/microsoft/msr_genomics_kbcomp/resolve/main/train.jsonl",
                format="jsonl", 
                description="Biomedical semantic indexing and question answering",
                expected_samples=3000,
                license="Creative Commons",
                evaluation_type="qa_accuracy",
                metadata={"domains": ["biomedical", "genomics"], "task": "biomedical_qa"}
            ),
            
            # ===== EFFICIENCY DATASETS =====
            "efficiency_bench": DatasetSource(
                name="efficiency_bench",
                category="efficiency",
                url="https://huggingface.co/datasets/efficiency/efficiency_benchmark/resolve/main/latency_test.json",
                format="json",
                description="Efficiency benchmarking with latency constraints",
                expected_samples=1000,
                license="Apache 2.0",
                evaluation_type="speed_accuracy_tradeoff",
                metadata={"metrics": ["latency", "memory", "throughput"], "constraints": ["mobile", "edge"]}
            ),
            
            "mobile_benchmark": DatasetSource(
                name="mobile_benchmark", 
                category="efficiency",
                url="https://huggingface.co/datasets/mobile/mobile_efficiency/resolve/main/mobile_tasks.jsonl",
                format="jsonl",
                description="Mobile device efficiency evaluation tasks",
                expected_samples=2000,
                license="MIT", 
                evaluation_type="resource_efficiency",
                metadata={"target": "mobile_devices", "constraints": ["memory_limited", "battery_limited"]}
            )
        }
        
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """Download a specific dataset"""
        if dataset_name not in self.dataset_sources:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        dataset = self.dataset_sources[dataset_name]
        output_dir = self.base_path / dataset.category / dataset_name
        
        # Check if already downloaded
        if output_dir.exists() and not force_redownload:
            logger.info(f"Dataset {dataset_name} already exists. Use force_redownload=True to redownload.")
            return True
            
        logger.info(f"Downloading dataset: {dataset_name}")
        
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            temp_file = self.temp_path / f"{dataset_name}_temp"
            success = self._download_file(dataset.url, temp_file)
            
            if not success:
                logger.error(f"Failed to download {dataset_name}")
                return False
                
            # Validate download
            if not self._validate_download(temp_file, dataset):
                logger.error(f"Validation failed for {dataset_name}")
                return False
                
            # Process and move to final location
            final_file = output_dir / f"{dataset_name}.{dataset.format}"
            if dataset.format in ['zip', 'tar.gz']:
                success = self._extract_archive(temp_file, output_dir)
            else:
                success = self._move_file(temp_file, final_file)
                
            if success:
                # Create metadata file
                self._create_metadata_file(output_dir, dataset)
                # Log successful download
                self._log_download(dataset_name, True, dataset.expected_samples)
                logger.info(f"Successfully downloaded {dataset_name}")
                return True
            else:
                logger.error(f"Failed to process {dataset_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {str(e)}")
            return False
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
                
    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return False
            
    def _validate_download(self, file_path: Path, dataset: DatasetSource) -> bool:
        """Validate downloaded file"""
        if not file_path.exists():
            return False
            
        # Check file size
        file_size = file_path.stat().st_size
        if file_size < 1024:  # Less than 1KB is likely an error
            logger.error(f"Downloaded file too small: {file_size} bytes")
            return False
            
        # Check format-specific validation
        if dataset.format == 'json':
            return self._validate_json(file_path)
        elif dataset.format == 'jsonl':
            return self._validate_jsonl(file_path)
        elif dataset.format in ['csv', 'parquet']:
            return True  # Basic size check sufficient for now
            
        return True
        
    def _validate_json(self, file_path: Path) -> bool:
        """Validate JSON file"""
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            logger.error("Invalid JSON format")
            return False
            
    def _validate_jsonl(self, file_path: Path) -> bool:
        """Validate JSONL file"""
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    json.loads(line.strip())
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSONL format: {str(e)}")
            return False
            
    def _extract_archive(self, archive_path: Path, output_dir: Path) -> bool:
        """Extract zip or tar.gz archives"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
            elif archive_path.suffix == '.gz':
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(output_dir)
            return True
        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            return False
            
    def _move_file(self, source: Path, destination: Path) -> bool:
        """Move file to final location"""
        try:
            shutil.move(str(source), str(destination))
            return True
        except Exception as e:
            logger.error(f"Move error: {str(e)}")
            return False
            
    def _create_metadata_file(self, output_dir: Path, dataset: DatasetSource):
        """Create metadata file for the dataset"""
        metadata = {
            "name": dataset.name,
            "category": dataset.category,
            "description": dataset.description,
            "expected_samples": dataset.expected_samples,
            "license": dataset.license,
            "evaluation_type": dataset.evaluation_type,
            "download_date": datetime.now().isoformat(),
            "format": dataset.format,
            "source_url": dataset.url,
            "metadata": dataset.metadata
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _log_download(self, dataset_name: str, success: bool, sample_count: int):
        """Log download attempt"""
        log_entry = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "sample_count": sample_count
        }
        
        log_file = self.download_log_path / "download_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def download_by_category(self, category: str) -> Dict[str, bool]:
        """Download all datasets for a specific category"""
        results = {}
        category_datasets = [name for name, ds in self.dataset_sources.items() 
                           if ds.category == category]
        
        logger.info(f"Downloading {len(category_datasets)} datasets for category: {category}")
        
        for dataset_name in category_datasets:
            results[dataset_name] = self.download_dataset(dataset_name)
            time.sleep(1)  # Be respectful to servers
            
        return results
        
    def download_all(self) -> Dict[str, bool]:
        """Download all datasets"""
        results = {}
        for dataset_name in self.dataset_sources.keys():
            results[dataset_name] = self.download_dataset(dataset_name)
            time.sleep(1)  # Be respectful to servers
            
        return results
        
    def get_download_status(self) -> Dict[str, Any]:
        """Get status of all datasets"""
        status = {
            "categories": {},
            "total_datasets": len(self.dataset_sources),
            "downloaded": 0,
            "missing": []
        }
        
        for name, dataset in self.dataset_sources.items():
            category = dataset.category
            if category not in status["categories"]:
                status["categories"][category] = {"total": 0, "downloaded": 0, "datasets": []}
            
            status["categories"][category]["total"] += 1
            status["categories"][category]["datasets"].append(name)
            
            output_dir = self.base_path / dataset.category / name
            if output_dir.exists():
                status["downloaded"] += 1
                status["categories"][category]["downloaded"] += 1
            else:
                status["missing"].append(name)
                
        return status
        
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """List all available datasets by category"""
        by_category = {}
        for name, dataset in self.dataset_sources.items():
            if dataset.category not in by_category:
                by_category[dataset.category] = []
            by_category[dataset.category].append(name)
            
        return by_category

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for LLM evaluation")
    parser.add_argument("--dataset", help="Specific dataset to download")
    parser.add_argument("--category", help="Download all datasets for a category")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.list:
        datasets = downloader.list_available_datasets()
        print("\n📊 Available Datasets by Category:")
        for category, dataset_list in datasets.items():
            print(f"\n🔹 {category.upper()}: {len(dataset_list)} datasets")
            for dataset in dataset_list:
                print(f"   • {dataset}")
                
    elif args.status:
        status = downloader.get_download_status()
        print(f"\n📈 Download Status: {status['downloaded']}/{status['total_datasets']} datasets")
        for category, info in status["categories"].items():
            print(f"🔹 {category}: {info['downloaded']}/{info['total']} downloaded")
        if status["missing"]:
            print(f"\n❌ Missing datasets: {', '.join(status['missing'])}")
            
    elif args.dataset:
        success = downloader.download_dataset(args.dataset, force_redownload=args.force)
        print(f"✅ Dataset '{args.dataset}' download: {'Success' if success else 'Failed'}")
        
    elif args.category:
        results = downloader.download_by_category(args.category)
        successful = sum(1 for success in results.values() if success)
        print(f"✅ Category '{args.category}': {successful}/{len(results)} datasets downloaded successfully")
        
    elif args.all:
        results = downloader.download_all()
        successful = sum(1 for success in results.values() if success)
        print(f"✅ All datasets: {successful}/{len(results)} downloaded successfully")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()