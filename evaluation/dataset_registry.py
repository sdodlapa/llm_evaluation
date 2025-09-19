"""
Dataset Registry - Core dataset configurations and metadata
Centralized registry for all dataset information and catalog management
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetInfo:
    """Dataset configuration and metadata"""
    name: str
    task_type: str
    data_path: str
    metadata_path: Optional[str]
    sample_count: int
    evaluation_type: str
    description: str
    implemented: bool = True

class DatasetRegistry:
    """Registry for managing dataset configurations and metadata"""
    
    def __init__(self):
        self.datasets = self._initialize_dataset_catalog()
    
    def _initialize_dataset_catalog(self) -> Dict[str, DatasetInfo]:
        """Initialize complete dataset catalog"""
        return {
            # Coding Tasks
            "humaneval": DatasetInfo(
                name="humaneval",
                task_type="coding",
                data_path="coding/humaneval.json",
                metadata_path="meta/humaneval_metadata.json",
                sample_count=164,
                evaluation_type="code_execution",
                description="Python code generation benchmark"
            ),
            "mbpp": DatasetInfo(
                name="mbpp",
                task_type="coding", 
                data_path="coding/mbpp.json",
                metadata_path="meta/mbpp_metadata.json",
                sample_count=500,
                evaluation_type="code_execution",
                description="Python code generation from docstrings"
            ),
            "bigcodebench": DatasetInfo(
                name="bigcodebench",
                task_type="coding",
                data_path="coding/bigcodebench.json",
                metadata_path="meta/bigcodebench_metadata.json",
                sample_count=500,
                evaluation_type="code_execution",
                description="Big Code Bench - comprehensive coding benchmark"
            ),
            "codecontests": DatasetInfo(
                name="codecontests",
                task_type="coding",
                data_path="coding/codecontests.json",
                metadata_path="meta/codecontests_metadata.json",
                sample_count=13500,
                evaluation_type="code_execution",
                description="Programming contest problems from competitive programming",
                implemented=False
            ),
            "apps": DatasetInfo(
                name="apps",
                task_type="coding",
                data_path="coding/apps.json",
                metadata_path="meta/apps_metadata.json",
                sample_count=5000,
                evaluation_type="code_execution",
                description="Measuring coding challenge competence with 10,000 problems",
                implemented=False
            ),
            "advanced_coding_sample": DatasetInfo(
                name="advanced_coding_sample",
                task_type="coding",
                data_path="coding/advanced_coding_sample.json",
                metadata_path=None,
                sample_count=50,
                evaluation_type="code_execution",
                description="Advanced coding problems - sample set"
            ),
            "advanced_coding_extended": DatasetInfo(
                name="advanced_coding_extended",
                task_type="coding",
                data_path="coding/advanced_coding_extended.json",
                metadata_path=None,
                sample_count=200,
                evaluation_type="code_execution",
                description="Advanced coding problems - extended set"
            ),
            
            # Mathematical Reasoning
            "gsm8k": DatasetInfo(
                name="gsm8k",
                task_type="reasoning",
                data_path="reasoning/gsm8k.json",
                metadata_path="meta/gsm8k_metadata.json",
                sample_count=1319,
                evaluation_type="numerical_accuracy",
                description="Grade school math word problems"
            ),
            "math": DatasetInfo(
                name="math",
                task_type="reasoning",
                data_path="reasoning/math.json",
                metadata_path="meta/math_metadata.json",
                sample_count=5000,
                evaluation_type="numerical_accuracy",
                description="MATH dataset - competition mathematics problems",
                implemented=False
            ),
            "arc_challenge": DatasetInfo(
                name="arc_challenge",
                task_type="reasoning",
                data_path="reasoning/arc_challenge.json",
                metadata_path="meta/arc_challenge_metadata.json",
                sample_count=1172,
                evaluation_type="multiple_choice_accuracy",
                description="AI2 Reasoning Challenge - grade-school science questions",
                implemented=False
            ),
            "hellaswag": DatasetInfo(
                name="hellaswag",
                task_type="reasoning",
                data_path="reasoning/hellaswag.json",
                metadata_path="meta/hellaswag_metadata.json",
                sample_count=10042,
                evaluation_type="multiple_choice_accuracy",
                description="Common sense reasoning - scenario completion",
                implemented=False
            ),
            
            # Question Answering
            "mmlu": DatasetInfo(
                name="mmlu",
                task_type="qa",
                data_path="qa/mmlu.json",
                metadata_path="meta/mmlu_metadata.json",
                sample_count=14042,
                evaluation_type="multiple_choice_accuracy",
                description="Massive Multitask Language Understanding",
                implemented=False
            ),
            
            # Instruction Following
            "mt_bench": DatasetInfo(
                name="mt_bench",
                task_type="instruction_following",
                data_path="instruction_following/mt_bench.json",
                metadata_path="meta/mt_bench_metadata.json",
                sample_count=80,
                evaluation_type="llm_judge_score",
                description="Multi-turn conversation benchmark",
                implemented=False
            ),
            "ifeval": DatasetInfo(
                name="ifeval",
                task_type="instruction_following",
                data_path="instruction_following/ifeval.json",
                metadata_path="meta/ifeval_metadata.json",
                sample_count=541,
                evaluation_type="instruction_compliance",
                description="Instruction following evaluation with verifiable instructions",
                implemented=False
            ),
            
            # Function Calling
            "bfcl": DatasetInfo(
                name="bfcl",
                task_type="function_calling",
                data_path="function_calling/bfcl.json",
                metadata_path="meta/bfcl_metadata.json",
                sample_count=2000,
                evaluation_type="function_call_accuracy",
                description="Berkeley Function Calling Leaderboard",
                implemented=False
            ),
            "toolllama": DatasetInfo(
                name="toolllama",
                task_type="function_calling",
                data_path="function_calling/toolllama.json",
                metadata_path="meta/toolllama_metadata.json",
                sample_count=1000,
                evaluation_type="function_call_accuracy",
                description="Tool Learning with LLMs",
                implemented=False
            ),
            
            # Scientific & Biomedical Datasets
            "bioasq": DatasetInfo(
                name="bioasq",
                task_type="biomedical_qa",
                data_path="biomedical/bioasq.json",
                metadata_path="meta/bioasq_metadata.json",
                sample_count=1504,
                evaluation_type="qa_accuracy",
                description="Biomedical semantic indexing and question answering",
                implemented=True
            ),
            "pubmedqa": DatasetInfo(
                name="pubmedqa",
                task_type="biomedical_qa",
                data_path="biomedical/pubmedqa.json",
                metadata_path="meta/pubmedqa_metadata.json",
                sample_count=1000,
                evaluation_type="qa_accuracy",
                description="Medical question answering from PubMed abstracts",
                implemented=True
            ),
            "chemprot": DatasetInfo(
                name="chemprot",
                task_type="relation_extraction",
                data_path="scientific/chemprot.json",
                metadata_path="meta/chemprot_metadata.json",
                sample_count=1020,
                evaluation_type="relation_extraction_f1",
                description="Chemical-protein interaction extraction",
                implemented=True
            ),
            "genomics_ner": DatasetInfo(
                name="genomics_ner",
                task_type="biomedical_ner",
                data_path="scientific/genomics_ner.json",
                metadata_path="meta/genomics_ner_metadata.json",
                sample_count=2500,
                evaluation_type="entity_extraction_f1",
                description="Genomics named entity recognition",
                implemented=True
            ),
            "protein_function": DatasetInfo(
                name="protein_function",
                task_type="sequence_classification",
                data_path="scientific/protein_function.json",
                metadata_path="meta/protein_function_metadata.json",
                sample_count=1500,
                evaluation_type="classification_accuracy",
                description="Protein function prediction from sequences",
                implemented=True
            ),
            "scientific_papers": DatasetInfo(
                name="scientific_papers",
                task_type="summarization",
                data_path="scientific/scientific_papers.json",
                metadata_path="meta/scientific_papers_metadata.json",
                sample_count=5001,
                evaluation_type="summarization_quality",
                description="ArXiv and PubMed papers for summarization tasks",
                implemented=True
            ),
            "scierc": DatasetInfo(
                name="scierc",
                task_type="scientific_ner",
                data_path="scientific/scierc.json",
                metadata_path="meta/scierc_metadata.json",
                sample_count=501,
                evaluation_type="entity_extraction_f1",
                description="Scientific entity and relation extraction",
                implemented=True
            ),
            
            # Document Understanding Datasets
            "docvqa": DatasetInfo(
                name="docvqa",
                task_type="document_vqa",
                data_path="document/docvqa.json",
                metadata_path="meta/docvqa_metadata.json",
                sample_count=5000,
                evaluation_type="vqa_accuracy",
                description="Document visual question answering",
                implemented=True
            ),
            
            # Safety & Alignment Datasets
            "toxicity_detection": DatasetInfo(
                name="toxicity_detection",
                task_type="safety_classification",
                data_path="safety/toxicity_detection.json",
                metadata_path="meta/toxicity_detection_metadata.json",
                sample_count=1002,
                evaluation_type="classification_accuracy",
                description="Toxicity and harmful content detection",
                implemented=True
            )
        }
    
    def get_all_datasets(self) -> Dict[str, DatasetInfo]:
        """Get all dataset configurations"""
        return self.datasets
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Get information for a specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry")
        return self.datasets[dataset_name]
    
    def get_available_datasets(self) -> List[str]:
        """Get list of all available dataset names"""
        return list(self.datasets.keys())
    
    def get_implemented_datasets(self) -> List[str]:
        """Get list of currently implemented datasets"""
        return [name for name, info in self.datasets.items() if info.implemented]
    
    def get_unimplemented_datasets(self) -> List[str]:
        """Get list of datasets that need implementation"""
        return [name for name, info in self.datasets.items() if not info.implemented]
    
    def get_datasets_by_task_type(self, task_type: str) -> Dict[str, DatasetInfo]:
        """Get datasets filtered by task type"""
        return {name: info for name, info in self.datasets.items() 
                if info.task_type == task_type}
    
    def get_datasets_by_evaluation_type(self, evaluation_type: str) -> Dict[str, DatasetInfo]:
        """Get datasets filtered by evaluation type"""
        return {name: info for name, info in self.datasets.items() 
                if info.evaluation_type == evaluation_type}
    
    def get_scientific_datasets(self) -> Dict[str, DatasetInfo]:
        """Get all scientific and biomedical datasets"""
        scientific_task_types = {
            "biomedical_qa", "relation_extraction", "biomedical_ner", 
            "sequence_classification", "summarization", "scientific_ner"
        }
        return {name: info for name, info in self.datasets.items() 
                if info.task_type in scientific_task_types}
    
    def get_coding_datasets(self) -> Dict[str, DatasetInfo]:
        """Get all coding-related datasets"""
        return self.get_datasets_by_task_type("coding")
    
    def get_reasoning_datasets(self) -> Dict[str, DatasetInfo]:
        """Get all reasoning datasets"""
        return self.get_datasets_by_task_type("reasoning")
    
    def add_dataset(self, dataset_info: DatasetInfo) -> None:
        """Add a new dataset to the registry"""
        self.datasets[dataset_info.name] = dataset_info
    
    def remove_dataset(self, dataset_name: str) -> None:
        """Remove a dataset from the registry"""
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
    
    def update_dataset_status(self, dataset_name: str, implemented: bool) -> None:
        """Update the implementation status of a dataset"""
        if dataset_name in self.datasets:
            self.datasets[dataset_name].implemented = implemented
    
    def validate_registry(self) -> Dict[str, Any]:
        """Validate all dataset paths and configurations"""
        from .dataset_path_manager import dataset_path_manager
        return dataset_path_manager.validate_all_datasets(self)

# Global registry instance
dataset_registry = DatasetRegistry()