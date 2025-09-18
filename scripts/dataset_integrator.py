#!/usr/bin/env python3
"""
Dataset Integration Script for LLM Evaluation Framework
Integrates newly downloaded datasets into EnhancedDatasetManager
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetIntegration:
    """Configuration for integrating a dataset into the framework"""
    name: str
    category: str
    source_path: str
    target_format: str  # How to format for evaluation
    preprocessing_function: str  # Name of preprocessing function
    evaluation_adapter: str  # How to adapt for evaluation
    sample_limit: Optional[int] = None

class DatasetIntegrator:
    """Integrates newly downloaded datasets into the evaluation framework"""
    
    def __init__(self, data_path: str = "evaluation_data"):
        self.data_path = Path(data_path)
        self.integration_configs = self._initialize_integration_configs()
        
    def _initialize_integration_configs(self) -> Dict[str, DatasetIntegration]:
        """Initialize integration configurations for each dataset"""
        return {
            # ===== CODING DATASETS =====
            "bigcodebench": DatasetIntegration(
                name="bigcodebench",
                category="coding",
                source_path="coding/bigcodebench/bigcodebench.parquet",
                target_format="humaneval_style",
                preprocessing_function="preprocess_bigcodebench",
                evaluation_adapter="code_execution_adapter",
                sample_limit=500  # Limit for efficiency
            ),
            
            "codecontests": DatasetIntegration(
                name="codecontests", 
                category="coding",
                source_path="coding/codecontests/codecontests.jsonl",
                target_format="humaneval_style",
                preprocessing_function="preprocess_codecontests",
                evaluation_adapter="code_execution_adapter",
                sample_limit=300
            ),
            
            "apps": DatasetIntegration(
                name="apps",
                category="coding", 
                source_path="coding/apps/apps.parquet",
                target_format="humaneval_style",
                preprocessing_function="preprocess_apps",
                evaluation_adapter="code_execution_adapter",
                sample_limit=400
            ),
            
            # ===== MATHEMATICS DATASETS =====
            "math_competition": DatasetIntegration(
                name="math_competition",
                category="mathematics",
                source_path="mathematics/math_competition/math_competition.jsonl", 
                target_format="gsm8k_style",
                preprocessing_function="preprocess_math_competition",
                evaluation_adapter="numerical_accuracy_adapter",
                sample_limit=1000
            ),
            
            "mathqa": DatasetIntegration(
                name="mathqa",
                category="mathematics",
                source_path="mathematics/mathqa/mathqa.json",
                target_format="gsm8k_style", 
                preprocessing_function="preprocess_mathqa",
                evaluation_adapter="numerical_accuracy_adapter",
                sample_limit=2000
            ),
            
            "aime": DatasetIntegration(
                name="aime",
                category="mathematics",
                source_path="mathematics/aime/aime.jsonl",
                target_format="gsm8k_style",
                preprocessing_function="preprocess_aime", 
                evaluation_adapter="numerical_accuracy_adapter"
            ),
            
            # ===== MULTIMODAL DATASETS =====
            "scienceqa": DatasetIntegration(
                name="scienceqa",
                category="multimodal",
                source_path="multimodal/scienceqa/scienceqa.parquet",
                target_format="vqa_style",
                preprocessing_function="preprocess_scienceqa",
                evaluation_adapter="multimodal_accuracy_adapter",
                sample_limit=1000
            ),
            
            "vqa_v2": DatasetIntegration(
                name="vqa_v2", 
                category="multimodal",
                source_path="multimodal/vqa_v2/vqa_v2.jsonl",
                target_format="vqa_style",
                preprocessing_function="preprocess_vqa_v2",
                evaluation_adapter="multimodal_accuracy_adapter",
                sample_limit=2000
            ),
            
            "chartqa": DatasetIntegration(
                name="chartqa",
                category="multimodal",
                source_path="multimodal/chartqa/chartqa.jsonl", 
                target_format="vqa_style",
                preprocessing_function="preprocess_chartqa",
                evaluation_adapter="multimodal_accuracy_adapter"
            ),
            
            # ===== GENOMICS DATASETS =====
            "genomics_benchmark": DatasetIntegration(
                name="genomics_benchmark",
                category="genomics",
                source_path="genomics/genomics_benchmark/genomics_benchmark.csv",
                target_format="classification_style",
                preprocessing_function="preprocess_genomics_benchmark", 
                evaluation_adapter="sequence_classification_adapter"
            ),
            
            "protein_sequences": DatasetIntegration(
                name="protein_sequences",
                category="genomics",
                source_path="genomics/protein_sequences/protein_sequences.csv",
                target_format="classification_style",
                preprocessing_function="preprocess_protein_sequences",
                evaluation_adapter="sequence_classification_adapter"
            ),
            
            "bioasq": DatasetIntegration(
                name="bioasq",
                category="genomics", 
                source_path="genomics/bioasq/bioasq.jsonl",
                target_format="qa_style",
                preprocessing_function="preprocess_bioasq",
                evaluation_adapter="qa_accuracy_adapter"
            ),
            
            # ===== EFFICIENCY DATASETS =====
            "efficiency_bench": DatasetIntegration(
                name="efficiency_bench",
                category="efficiency",
                source_path="efficiency/efficiency_bench/efficiency_bench.json",
                target_format="speed_test_style",
                preprocessing_function="preprocess_efficiency_bench",
                evaluation_adapter="efficiency_adapter"
            ),
            
            "mobile_benchmark": DatasetIntegration(
                name="mobile_benchmark",
                category="efficiency",
                source_path="efficiency/mobile_benchmark/mobile_benchmark.jsonl",
                target_format="speed_test_style", 
                preprocessing_function="preprocess_mobile_benchmark",
                evaluation_adapter="efficiency_adapter"
            )
        }
        
    def integrate_dataset(self, dataset_name: str) -> bool:
        """Integrate a specific dataset into the framework"""
        if dataset_name not in self.integration_configs:
            logger.error(f"No integration config for dataset: {dataset_name}")
            return False
            
        config = self.integration_configs[dataset_name]
        source_file = self.data_path / config.source_path
        
        if not source_file.exists():
            logger.error(f"Source file not found: {source_file}")
            return False
            
        logger.info(f"Integrating dataset: {dataset_name}")
        
        try:
            # Load and preprocess the dataset
            preprocessor = getattr(self, config.preprocessing_function)
            processed_data = preprocessor(source_file, config)
            
            if not processed_data:
                logger.error(f"Preprocessing failed for {dataset_name}")
                return False
                
            # Save in standard format
            output_file = self.data_path / config.category / f"{dataset_name}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
                
            # Create metadata
            self._create_integration_metadata(dataset_name, config, len(processed_data))
            
            logger.info(f"Successfully integrated {dataset_name}: {len(processed_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error integrating {dataset_name}: {str(e)}")
            return False
            
    def _create_integration_metadata(self, dataset_name: str, config: DatasetIntegration, sample_count: int):
        """Create metadata for integrated dataset"""
        metadata = {
            "name": dataset_name,
            "category": config.category,
            "sample_count": sample_count,
            "target_format": config.target_format,
            "evaluation_adapter": config.evaluation_adapter,
            "integration_date": pd.Timestamp.now().isoformat(),
            "status": "integrated"
        }
        
        metadata_file = self.data_path / "meta" / f"{dataset_name}_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    # ===== PREPROCESSING FUNCTIONS =====
    
    def preprocess_bigcodebench(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess BigCodeBench dataset"""
        try:
            df = pd.read_parquet(source_file)
            processed = []
            
            for idx, row in df.iterrows():
                if config.sample_limit and len(processed) >= config.sample_limit:
                    break
                    
                item = {
                    "task_id": f"bigcodebench_{idx}",
                    "prompt": row.get("prompt", ""),
                    "canonical_solution": row.get("canonical_solution", ""),
                    "test": row.get("test", ""),
                    "entry_point": row.get("entry_point", "main"),
                    "difficulty": row.get("difficulty", "medium")
                }
                processed.append(item)
                
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing BigCodeBench: {str(e)}")
            return []
            
    def preprocess_codecontests(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess CodeContests dataset"""
        try:
            processed = []
            
            with open(source_file, 'r') as f:
                for idx, line in enumerate(f):
                    if config.sample_limit and len(processed) >= config.sample_limit:
                        break
                        
                    data = json.loads(line.strip())
                    item = {
                        "task_id": f"codecontests_{idx}",
                        "prompt": data.get("description", ""),
                        "canonical_solution": data.get("solutions", {}).get("python", [""])[0],
                        "test": self._create_tests_from_examples(data.get("public_tests", {})),
                        "entry_point": "solve",
                        "difficulty": data.get("difficulty", "medium")
                    }
                    processed.append(item)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing CodeContests: {str(e)}")
            return []
            
    def preprocess_apps(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess APPS dataset"""
        try:
            df = pd.read_parquet(source_file)
            processed = []
            
            for idx, row in df.iterrows():
                if config.sample_limit and len(processed) >= config.sample_limit:
                    break
                    
                item = {
                    "task_id": f"apps_{idx}",
                    "prompt": row.get("question", ""),
                    "canonical_solution": row.get("solutions", ""),
                    "test": row.get("input_output", ""),
                    "entry_point": "main",
                    "difficulty": row.get("difficulty", "interview")
                }
                processed.append(item)
                
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing APPS: {str(e)}")
            return []
            
    def preprocess_math_competition(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess MATH competition dataset"""
        try:
            processed = []
            
            with open(source_file, 'r') as f:
                for idx, line in enumerate(f):
                    if config.sample_limit and len(processed) >= config.sample_limit:
                        break
                        
                    data = json.loads(line.strip())
                    item = {
                        "question": data.get("problem", ""),
                        "answer": data.get("solution", ""),
                        "level": data.get("level", "unknown"),
                        "type": data.get("type", "unknown"),
                        "source": "MATH_competition"
                    }
                    processed.append(item)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing MATH competition: {str(e)}")
            return []
            
    def preprocess_mathqa(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess MathQA dataset"""
        try:
            with open(source_file, 'r') as f:
                data = json.load(f)
                
            processed = []
            for idx, item in enumerate(data):
                if config.sample_limit and len(processed) >= config.sample_limit:
                    break
                    
                processed_item = {
                    "question": item.get("Problem", ""),
                    "answer": item.get("correct", ""),
                    "rationale": item.get("Rationale", ""),
                    "options": item.get("options", ""),
                    "source": "MathQA"
                }
                processed.append(processed_item)
                
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing MathQA: {str(e)}")
            return []
            
    def preprocess_aime(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess AIME dataset"""
        try:
            processed = []
            
            with open(source_file, 'r') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line.strip())
                    item = {
                        "question": data.get("problem", ""),
                        "answer": data.get("answer", ""),
                        "year": data.get("year", "unknown"),
                        "problem_number": data.get("problem_number", idx + 1),
                        "source": "AIME"
                    }
                    processed.append(item)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing AIME: {str(e)}")
            return []
            
    def preprocess_scienceqa(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess ScienceQA dataset"""
        try:
            df = pd.read_parquet(source_file)
            processed = []
            
            for idx, row in df.iterrows():
                if config.sample_limit and len(processed) >= config.sample_limit:
                    break
                    
                item = {
                    "question": row.get("question", ""),
                    "choices": row.get("choices", []),
                    "answer": row.get("answer", ""),
                    "image": row.get("image", None),
                    "subject": row.get("subject", "science"),
                    "grade": row.get("grade", "unknown")
                }
                processed.append(item)
                
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing ScienceQA: {str(e)}")
            return []
            
    def preprocess_vqa_v2(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess VQA v2 dataset"""
        try:
            processed = []
            
            with open(source_file, 'r') as f:
                for idx, line in enumerate(f):
                    if config.sample_limit and len(processed) >= config.sample_limit:
                        break
                        
                    data = json.loads(line.strip())
                    item = {
                        "question": data.get("question", ""),
                        "image_id": data.get("image_id", ""),
                        "answers": data.get("answers", []),
                        "question_type": data.get("question_type", "unknown"),
                        "answer_type": data.get("answer_type", "unknown")
                    }
                    processed.append(item)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing VQA v2: {str(e)}")
            return []
            
    def preprocess_chartqa(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess ChartQA dataset"""
        try:
            processed = []
            
            with open(source_file, 'r') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line.strip())
                    item = {
                        "question": data.get("query", ""),
                        "chart_image": data.get("imgname", ""),
                        "answer": data.get("label", ""),
                        "chart_type": data.get("chart_type", "unknown")
                    }
                    processed.append(item)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing ChartQA: {str(e)}")
            return []
            
    def preprocess_genomics_benchmark(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess genomics benchmark dataset"""
        try:
            df = pd.read_csv(source_file)
            processed = []
            
            for idx, row in df.iterrows():
                item = {
                    "sequence": row.get("sequence", ""),
                    "label": row.get("label", ""),
                    "sequence_length": len(row.get("sequence", "")),
                    "task": "promoter_detection"
                }
                processed.append(item)
                
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing genomics benchmark: {str(e)}")
            return []
            
    def preprocess_protein_sequences(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess protein sequences dataset"""
        try:
            df = pd.read_csv(source_file)
            processed = []
            
            for idx, row in df.iterrows():
                item = {
                    "sequence": row.get("sequence", ""),
                    "structure": row.get("structure", ""),
                    "protein_id": row.get("protein_id", f"protein_{idx}"),
                    "task": "structure_prediction"
                }
                processed.append(item)
                
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing protein sequences: {str(e)}")
            return []
            
    def preprocess_bioasq(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess BioASQ dataset"""
        try:
            processed = []
            
            with open(source_file, 'r') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line.strip())
                    item = {
                        "question": data.get("question", ""),
                        "answer": data.get("answer", ""),
                        "context": data.get("context", ""),
                        "domain": "biomedical"
                    }
                    processed.append(item)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing BioASQ: {str(e)}")
            return []
            
    def preprocess_efficiency_bench(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess efficiency benchmark dataset"""
        try:
            with open(source_file, 'r') as f:
                data = json.load(f)
                
            processed = []
            for item in data:
                processed_item = {
                    "task": item.get("task", ""),
                    "input": item.get("input", ""),
                    "expected_output": item.get("expected_output", ""),
                    "max_latency_ms": item.get("max_latency_ms", 1000),
                    "max_memory_mb": item.get("max_memory_mb", 512)
                }
                processed.append(processed_item)
                
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing efficiency benchmark: {str(e)}")
            return []
            
    def preprocess_mobile_benchmark(self, source_file: Path, config: DatasetIntegration) -> List[Dict]:
        """Preprocess mobile benchmark dataset"""
        try:
            processed = []
            
            with open(source_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    item = {
                        "task": data.get("task", ""),
                        "input": data.get("input", ""),
                        "expected_output": data.get("expected_output", ""),
                        "device_constraints": data.get("device_constraints", {}),
                        "target_platform": "mobile"
                    }
                    processed.append(item)
                    
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing mobile benchmark: {str(e)}")
            return []
            
    def _create_tests_from_examples(self, examples: Dict) -> str:
        """Create test cases from contest examples"""
        if not examples:
            return ""
            
        test_cases = []
        inputs = examples.get("input", [])
        outputs = examples.get("output", [])
        
        for inp, out in zip(inputs, outputs):
            test_cases.append(f"assert solve({repr(inp)}) == {repr(out)}")
            
        return "\n".join(test_cases)
        
    def integrate_all(self) -> Dict[str, bool]:
        """Integrate all available datasets"""
        results = {}
        for dataset_name in self.integration_configs.keys():
            results[dataset_name] = self.integrate_dataset(dataset_name)
            
        return results
        
    def integrate_by_category(self, category: str) -> Dict[str, bool]:
        """Integrate all datasets for a specific category"""
        results = {}
        category_datasets = [name for name, config in self.integration_configs.items() 
                           if config.category == category]
        
        for dataset_name in category_datasets:
            results[dataset_name] = self.integrate_dataset(dataset_name)
            
        return results

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate datasets into LLM evaluation framework")
    parser.add_argument("--dataset", help="Specific dataset to integrate")
    parser.add_argument("--category", help="Integrate all datasets for a category")
    parser.add_argument("--all", action="store_true", help="Integrate all datasets")
    
    args = parser.parse_args()
    
    integrator = DatasetIntegrator()
    
    if args.dataset:
        success = integrator.integrate_dataset(args.dataset)
        print(f"✅ Dataset '{args.dataset}' integration: {'Success' if success else 'Failed'}")
        
    elif args.category:
        results = integrator.integrate_by_category(args.category)
        successful = sum(1 for success in results.values() if success)
        print(f"✅ Category '{args.category}': {successful}/{len(results)} datasets integrated successfully")
        
    elif args.all:
        results = integrator.integrate_all()
        successful = sum(1 for success in results.values() if success)
        print(f"✅ All datasets: {successful}/{len(results)} integrated successfully")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()