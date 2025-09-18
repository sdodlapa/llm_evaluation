"""
Enhanced Dataset Manager
Handles all 12 documented datasets with proper sampling and evaluation
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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

class EnhancedDatasetManager:
    """Enhanced dataset manager supporting all 12 datasets"""
    
    def __init__(self, base_data_path: str = "evaluation_data"):
        self.base_path = Path(base_data_path)
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
                description="Mathematical competition problems",
                implemented=False
            ),
            "math_competition": DatasetInfo(
                name="math_competition",
                task_type="mathematics",
                data_path="mathematics/math_competition.json",
                metadata_path="meta/math_competition_metadata.json",
                sample_count=7500,
                evaluation_type="numerical_accuracy",
                description="MATH dataset - Competition mathematics problems",
                implemented=False
            ),
            "mathqa": DatasetInfo(
                name="mathqa",
                task_type="mathematics",
                data_path="mathematics/mathqa.json",
                metadata_path="meta/mathqa_metadata.json",
                sample_count=29837,
                evaluation_type="numerical_accuracy",
                description="Math word problems with operation programs",
                implemented=False
            ),
            "aime": DatasetInfo(
                name="aime",
                task_type="mathematics",
                data_path="mathematics/aime.json",
                metadata_path="meta/aime_metadata.json",
                sample_count=240,
                evaluation_type="numerical_accuracy",
                description="American Invitational Mathematics Examination problems",
                implemented=False
            ),
            
            # Function Calling & Agent Tasks
            "bfcl": DatasetInfo(
                name="bfcl",
                task_type="function_calling",
                data_path="function_calling/bfcl.json",
                metadata_path="meta/bfcl_metadata.json",
                sample_count=2000,
                evaluation_type="function_accuracy",
                description="Berkeley Function-Calling Leaderboard",
                implemented=False
            ),
            "toolllama": DatasetInfo(
                name="toolllama",
                task_type="function_calling", 
                data_path="function_calling/toolllama.json",
                metadata_path="meta/toolllama_metadata.json",
                sample_count=3000,
                evaluation_type="tool_usage_accuracy",
                description="Tool usage and API calling benchmark",
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
            "arc_challenge": DatasetInfo(
                name="arc_challenge",
                task_type="qa",
                data_path="reasoning/arc_challenge.json", 
                metadata_path="meta/arc_challenge_metadata.json",
                sample_count=1172,
                evaluation_type="multiple_choice_accuracy",
                description="AI2 Reasoning Challenge",
                implemented=True
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
                implemented=True
            ),
            "ifeval": DatasetInfo(
                name="ifeval",
                task_type="instruction_following",
                data_path="instruction_following/ifeval.json",
                metadata_path="meta/ifeval_metadata.json",
                sample_count=500,
                evaluation_type="instruction_compliance",
                description="Instruction following evaluation",
                implemented=False
            ),
            
            # General Knowledge & Common Sense
            "hellaswag": DatasetInfo(
                name="hellaswag",
                task_type="reasoning",
                data_path="reasoning/hellaswag.json",
                metadata_path="meta/hellaswag_metadata.json",
                sample_count=10042,
                evaluation_type="multiple_choice_accuracy",
                description="Commonsense reasoning benchmark",
                implemented=True
            ),
            "winogrande": DatasetInfo(
                name="winogrande",
                task_type="reasoning",
                data_path="reasoning/winogrande.json",
                metadata_path="meta/winogrande_metadata.json",
                sample_count=1767,
                evaluation_type="multiple_choice_accuracy",
                description="Commonsense reasoning with pronoun resolution",
                implemented=False
            ),
            
            # Multimodal Datasets  
            "scienceqa": DatasetInfo(
                name="scienceqa",
                task_type="multimodal",
                data_path="multimodal/scienceqa.json",
                metadata_path="meta/scienceqa_metadata.json",
                sample_count=19206,
                evaluation_type="multiple_choice_accuracy",
                description="Science question answering with images and text",
                implemented=False
            ),
            "vqa_v2": DatasetInfo(
                name="vqa_v2",
                task_type="multimodal",
                data_path="multimodal/vqa_v2.json",
                metadata_path="meta/vqa_v2_metadata.json",
                sample_count=443757,
                evaluation_type="free_form_accuracy",
                description="Visual Question Answering dataset v2.0",
                implemented=False
            ),
            "chartqa": DatasetInfo(
                name="chartqa",
                task_type="multimodal",
                data_path="multimodal/chartqa.json",
                metadata_path="meta/chartqa_metadata.json",
                sample_count=18271,
                evaluation_type="free_form_accuracy",
                description="Question answering on charts and graphs",
                implemented=False
            ),
            
            # Genomics/Bioinformatics Datasets
            "genomics_benchmark": DatasetInfo(
                name="genomics_benchmark",
                task_type="genomics",
                data_path="genomics/genomics_benchmark.json",
                metadata_path="meta/genomics_benchmark_metadata.json",
                sample_count=6000,
                evaluation_type="sequence_classification",
                description="Long-range genomics benchmark for sequence analysis",
                implemented=False
            ),
            "protein_sequences": DatasetInfo(
                name="protein_sequences",
                task_type="genomics",
                data_path="genomics/protein_sequences.json",
                metadata_path="meta/protein_sequences_metadata.json",
                sample_count=25000,
                evaluation_type="sequence_classification",
                description="Protein sequence and structure prediction dataset",
                implemented=False
            ),
            "bioasq": DatasetInfo(
                name="bioasq",
                task_type="genomics",
                data_path="genomics/bioasq.json",
                metadata_path="meta/bioasq_metadata.json",
                sample_count=3000,
                evaluation_type="qa_accuracy",
                description="Biomedical semantic indexing and question answering",
                implemented=False
            ),
            
            # Efficiency Datasets
            "efficiency_bench": DatasetInfo(
                name="efficiency_bench",
                task_type="efficiency",
                data_path="efficiency/efficiency_bench.json",
                metadata_path="meta/efficiency_bench_metadata.json",
                sample_count=1000,
                evaluation_type="speed_accuracy_tradeoff",
                description="Efficiency benchmarking with latency constraints",
                implemented=False
            ),
            "mobile_benchmark": DatasetInfo(
                name="mobile_benchmark",
                task_type="efficiency",
                data_path="efficiency/mobile_benchmark.json",
                metadata_path="meta/mobile_benchmark_metadata.json",
                sample_count=2000,
                evaluation_type="resource_efficiency",
                description="Mobile device efficiency evaluation tasks",
                implemented=False
            ),
            
            # === SCIENTIFIC & BIOMEDICAL DATASETS ===
            
            # Biomedical Datasets
            "pubmedqa": DatasetInfo(
                name="pubmedqa",
                task_type="biomedical_qa",
                data_path="biomedical/pubmedqa.json",
                metadata_path="meta/pubmedqa_metadata.json",
                sample_count=1000,
                evaluation_type="qa_accuracy",
                description="Biomedical question answering from PubMed abstracts",
                implemented=True
            ),
            "bioasq": DatasetInfo(
                name="bioasq",
                task_type="biomedical_qa",
                data_path="biomedical/bioasq.json",
                metadata_path="meta/bioasq_metadata.json",
                sample_count=3000,
                evaluation_type="qa_accuracy",
                description="Biomedical semantic indexing and question answering",
                implemented=True
            ),
            "mediqa": DatasetInfo(
                name="mediqa",
                task_type="clinical_qa",
                data_path="biomedical/mediqa.json",
                metadata_path="meta/mediqa_metadata.json",
                sample_count=1000,
                evaluation_type="clinical_accuracy",
                description="Medical question answering and summarization",
                implemented=True
            ),
            
            # Scientific Datasets
            "scientific_papers": DatasetInfo(
                name="scientific_papers",
                task_type="scientific_summarization",
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
    
    def get_available_datasets(self) -> List[str]:
        """Get list of all available dataset names"""
        return list(self.datasets.keys())
    
    def get_implemented_datasets(self) -> List[str]:
        """Get list of currently implemented datasets"""
        return [name for name, info in self.datasets.items() if info.implemented]
    
    def get_unimplemented_datasets(self) -> List[str]:
        """Get list of datasets that need implementation"""
        return [name for name, info in self.datasets.items() if not info.implemented]
    
    def get_recommended_datasets(self, task_types: Optional[List[str]] = None, 
                               include_experimental: bool = True) -> List[str]:
        """Get recommended datasets for evaluation based on task types and availability
        
        Args:
            task_types: List of task types to focus on. If None, includes all major categories
            include_experimental: Whether to include datasets marked as not fully implemented
        
        Returns:
            List of dataset names recommended for evaluation
        """
        if task_types is None:
            # Default: comprehensive evaluation across all capabilities
            task_types = ["coding", "reasoning", "qa", "instruction_following", "function_calling"]
        
        recommended = []
        
        # Priority datasets by task type
        task_priorities = {
            "coding": ["humaneval", "mbpp", "bigcodebench"],  # All implemented
            "reasoning": ["gsm8k", "arc_challenge", "hellaswag", "math"],  # math needs verification
            "qa": ["mmlu"],  # May need implementation check
            "instruction_following": ["mt_bench", "ifeval"],  # ifeval needs verification  
            "function_calling": ["bfcl", "toolllama"]  # Both need verification
        }
        
        # Add datasets based on requested task types
        for task_type in task_types:
            if task_type in task_priorities:
                for dataset in task_priorities[task_type]:
                    if dataset in self.datasets:
                        dataset_info = self.datasets[dataset]
                        # Include if implemented OR if experimental datasets allowed
                        if dataset_info.implemented or include_experimental:
                            if dataset not in recommended:
                                recommended.append(dataset)
        
        # Always include core evaluation datasets regardless of task type
        core_datasets = ["humaneval", "gsm8k", "mt_bench"]
        for dataset in core_datasets:
            if dataset in self.datasets and dataset not in recommended:
                recommended.append(dataset)
        
        return recommended
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Get detailed information about a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.datasets[dataset_name]
    
    def load_dataset(self, dataset_name: str, num_samples: int = None) -> List[Dict[str, Any]]:
        """Load dataset with optional sampling"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        dataset_info = self.datasets[dataset_name]
        
        if not dataset_info.implemented:
            raise NotImplementedError(f"Dataset {dataset_name} is not yet implemented")
        
        # Check if data file exists
        data_path = self.base_path / dataset_info.data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Handle different data structures
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            if 'data' in data:
                samples = data['data']
            elif 'samples' in data:
                samples = data['samples']
            elif 'problems' in data:
                samples = data['problems']
            else:
                # Assume the dict values are the samples
                samples = list(data.values())
        else:
            raise ValueError(f"Unexpected data format in {dataset_name}")
        
        # Apply sampling if requested
        if num_samples and num_samples < len(samples):
            random.seed(42)  # Reproducible sampling
            samples = random.sample(samples, num_samples)
            logger.info(f"Sampled {num_samples} from {len(samples)} available samples in {dataset_name}")
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    
    def load_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset metadata if available"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.datasets[dataset_name]
        
        if not dataset_info.metadata_path:
            return {}
        
        metadata_path = self.base_path / dataset_info.metadata_path
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset integrity and return diagnostics"""
        diagnostics = {
            'dataset_name': dataset_name,
            'exists': False,
            'sample_count': 0,
            'metadata_exists': False,
            'structure_valid': False,
            'errors': []
        }
        
        try:
            dataset_info = self.get_dataset_info(dataset_name)
            
            # Check data file
            data_path = self.base_path / dataset_info.data_path
            if data_path.exists():
                diagnostics['exists'] = True
                
                try:
                    samples = self.load_dataset(dataset_name)
                    diagnostics['sample_count'] = len(samples)
                    diagnostics['structure_valid'] = True
                    
                    # Basic structure validation
                    if samples and isinstance(samples[0], dict):
                        sample_keys = set(samples[0].keys())
                        diagnostics['sample_keys'] = list(sample_keys)
                        
                        # Check for required fields based on task type
                        required_fields = self._get_required_fields(dataset_info.task_type)
                        missing_fields = set(required_fields) - sample_keys
                        if missing_fields:
                            diagnostics['errors'].append(f"Missing required fields: {missing_fields}")
                        
                except Exception as e:
                    diagnostics['errors'].append(f"Failed to load data: {str(e)}")
            else:
                diagnostics['errors'].append(f"Data file not found: {data_path}")
            
            # Check metadata
            if dataset_info.metadata_path:
                metadata_path = self.base_path / dataset_info.metadata_path
                if metadata_path.exists():
                    diagnostics['metadata_exists'] = True
                else:
                    diagnostics['errors'].append(f"Metadata file not found: {metadata_path}")
                    
        except Exception as e:
            diagnostics['errors'].append(f"Validation error: {str(e)}")
        
        return diagnostics
    
    def _get_required_fields(self, task_type: str) -> List[str]:
        """Get required fields for each task type"""
        field_mapping = {
            'coding': ['prompt', 'canonical_solution'],
            'reasoning': ['question', 'answer'],
            'qa': ['question', 'answer'],
            'function_calling': ['query', 'tools', 'expected_result'],
            'instruction_following': ['instruction', 'expected_behavior']
        }
        return field_mapping.get(task_type, ['input', 'output'])
    
    def get_evaluation_strategy(self, dataset_name: str) -> str:
        """Get appropriate evaluation strategy for dataset"""
        dataset_info = self.get_dataset_info(dataset_name)
        return dataset_info.evaluation_type
    
    def prepare_evaluation_sample(self, dataset_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a sample for evaluation based on dataset type"""
        dataset_info = self.get_dataset_info(dataset_name)
        
        # Standardize sample format for evaluation
        prepared_sample = {
            'dataset': dataset_name,
            'task_type': dataset_info.task_type,
            'evaluation_type': dataset_info.evaluation_type,
            'original_sample': sample
        }
        
        # Extract standardized fields based on task type
        if dataset_info.task_type == 'coding':
            prepared_sample.update({
                'prompt': sample.get('prompt', sample.get('problem', '')),
                'expected_output': sample.get('canonical_solution', sample.get('solution', '')),
                'test_cases': sample.get('test', sample.get('test_cases', []))
            })
        
        elif dataset_info.task_type == 'reasoning':
            prepared_sample.update({
                'question': sample.get('question', sample.get('problem', '')),
                'expected_answer': sample.get('answer', sample.get('solution', '')),
                'options': sample.get('choices', sample.get('options', []))
            })
        
        elif dataset_info.task_type == 'qa':
            prepared_sample.update({
                'question': sample.get('question', sample.get('query', '')),
                'expected_answer': sample.get('answer', sample.get('correct_answer', '')),
                'options': sample.get('choices', sample.get('options', []))
            })
        
        elif dataset_info.task_type == 'function_calling':
            prepared_sample.update({
                'query': sample.get('query', sample.get('instruction', '')),
                'available_tools': sample.get('tools', sample.get('functions', [])),
                'expected_result': sample.get('expected_result', sample.get('answer', ''))
            })
        
        elif dataset_info.task_type == 'instruction_following':
            prepared_sample.update({
                'instruction': sample.get('instruction', sample.get('prompt', '')),
                'expected_behavior': sample.get('expected_behavior', sample.get('target', '')),
                'constraints': sample.get('constraints', [])
            })
        
        return prepared_sample
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all datasets"""
        summary = {
            'total_datasets': len(self.datasets),
            'implemented_datasets': len(self.get_implemented_datasets()),
            'unimplemented_datasets': len(self.get_unimplemented_datasets()),
            'task_type_distribution': {},
            'datasets': {}
        }
        
        # Analyze task type distribution
        for dataset_info in self.datasets.values():
            task_type = dataset_info.task_type
            if task_type not in summary['task_type_distribution']:
                summary['task_type_distribution'][task_type] = 0
            summary['task_type_distribution'][task_type] += 1
        
        # Add detailed info for each dataset
        for name, info in self.datasets.items():
            summary['datasets'][name] = {
                'task_type': info.task_type,
                'sample_count': info.sample_count,
                'evaluation_type': info.evaluation_type,
                'implemented': info.implemented,
                'description': info.description
            }
        
        return summary
    
    def create_missing_datasets(self):
        """Create placeholder files for missing datasets"""
        logger.info("Creating placeholder files for missing datasets...")
        
        for name, info in self.datasets.items():
            if not info.implemented:
                continue
                
            # Create data file if missing
            data_path = self.base_path / info.data_path
            if not data_path.exists():
                data_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create minimal placeholder
                placeholder_data = {
                    'dataset_name': name,
                    'task_type': info.task_type,
                    'description': info.description,
                    'data': [],
                    'note': 'Placeholder file - actual data needs to be added'
                }
                
                with open(data_path, 'w') as f:
                    json.dump(placeholder_data, f, indent=2)
                
                logger.info(f"Created placeholder data file: {data_path}")
            
            # Create metadata file if missing
            if info.metadata_path:
                metadata_path = self.base_path / info.metadata_path
                if not metadata_path.exists():
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    placeholder_metadata = {
                        'dataset_name': name,
                        'task_type': info.task_type,
                        'total_samples': info.sample_count,
                        'evaluation_type': info.evaluation_type,
                        'description': info.description,
                        'created': 'placeholder'
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(placeholder_metadata, f, indent=2)
                    
                    logger.info(f"Created placeholder metadata file: {metadata_path}")
    
    def get_recommended_sample_counts(self) -> Dict[str, int]:
        """Get recommended sample counts for comprehensive evaluation"""
        recommendations = {}
        
        for name, info in self.datasets.items():
            if info.sample_count <= 100:
                # Use all samples for small datasets
                recommendations[name] = info.sample_count
            elif info.sample_count <= 500:
                # Use 80% for medium datasets
                recommendations[name] = min(200, int(info.sample_count * 0.8))
            else:
                # Use fixed 200 samples for large datasets
                recommendations[name] = 200
        
        return recommendations