"""
Dataset Utilities - Recommendations, summaries, and helper functions
Utility functions for dataset management, recommendations, and analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    from .dataset_registry import DatasetInfo, dataset_registry
    from .dataset_loader import DatasetLoader
except ImportError:
    from evaluation.dataset_registry import DatasetInfo, dataset_registry
    from evaluation.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)

class DatasetUtils:
    """Utility functions for dataset analysis and recommendations"""
    
    def __init__(self, base_data_path: str = "evaluation_data"):
        self.registry = dataset_registry
        self.loader = DatasetLoader(base_data_path)
    
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
            "function_calling": ["bfcl", "toolllama"],  # Both need verification
            "biomedical": ["bioasq", "pubmedqa", "chemprot"],
            "scientific": ["scientific_papers", "scierc", "genomics_ner"]
        }
        
        # Add datasets based on requested task types
        for task_type in task_types:
            if task_type in task_priorities:
                for dataset in task_priorities[task_type]:
                    if dataset in self.registry.get_all_datasets():
                        dataset_info = self.registry.get_dataset_info(dataset)
                        # Include if implemented OR if experimental datasets allowed
                        if dataset_info.implemented or include_experimental:
                            if dataset not in recommended:
                                recommended.append(dataset)
        
        # Always include core evaluation datasets regardless of task type
        core_datasets = ["humaneval", "gsm8k"]
        for dataset in core_datasets:
            if dataset in self.registry.get_all_datasets() and dataset not in recommended:
                recommended.append(dataset)
        
        return recommended
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all datasets"""
        all_datasets = self.registry.get_all_datasets()
        
        summary = {
            'total_datasets': len(all_datasets),
            'implemented_datasets': len(self.registry.get_implemented_datasets()),
            'unimplemented_datasets': len(self.registry.get_unimplemented_datasets()),
            'task_type_distribution': {},
            'evaluation_type_distribution': {},
            'datasets': {}
        }
        
        # Analyze distributions
        for dataset_info in all_datasets.values():
            # Task type distribution
            task_type = dataset_info.task_type
            if task_type not in summary['task_type_distribution']:
                summary['task_type_distribution'][task_type] = 0
            summary['task_type_distribution'][task_type] += 1
            
            # Evaluation type distribution
            eval_type = dataset_info.evaluation_type
            if eval_type not in summary['evaluation_type_distribution']:
                summary['evaluation_type_distribution'][eval_type] = 0
            summary['evaluation_type_distribution'][eval_type] += 1
        
        # Add detailed info for each dataset
        for name, info in all_datasets.items():
            summary['datasets'][name] = {
                'task_type': info.task_type,
                'sample_count': info.sample_count,
                'evaluation_type': info.evaluation_type,
                'implemented': info.implemented,
                'description': info.description
            }
        
        return summary
    
    def get_recommended_sample_counts(self) -> Dict[str, int]:
        """Get recommended sample counts for comprehensive evaluation"""
        recommendations = {}
        
        for name, info in self.registry.get_all_datasets().items():
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
    
    def get_evaluation_plan(self, model_capabilities: List[str], 
                          time_budget: str = "medium") -> Dict[str, Any]:
        """Generate an evaluation plan based on model capabilities and time budget
        
        Args:
            model_capabilities: List of model capabilities (e.g., ['coding', 'reasoning', 'biomedical'])
            time_budget: 'quick', 'medium', or 'comprehensive'
            
        Returns:
            Dictionary with evaluation plan
        """
        # Map time budgets to sample multipliers
        time_multipliers = {
            'quick': 0.3,
            'medium': 0.6,
            'comprehensive': 1.0
        }
        
        multiplier = time_multipliers.get(time_budget, 0.6)
        
        # Get recommended datasets for capabilities
        recommended_datasets = self.get_recommended_datasets(model_capabilities)
        sample_counts = self.get_recommended_sample_counts()
        
        plan = {
            'model_capabilities': model_capabilities,
            'time_budget': time_budget,
            'total_datasets': len(recommended_datasets),
            'estimated_samples': 0,
            'datasets': {},
            'task_coverage': {},
            'priority_order': []
        }
        
        # Calculate samples and priority for each dataset
        for dataset_name in recommended_datasets:
            dataset_info = self.registry.get_dataset_info(dataset_name)
            base_samples = sample_counts.get(dataset_name, 100)
            adjusted_samples = max(10, int(base_samples * multiplier))
            
            plan['datasets'][dataset_name] = {
                'task_type': dataset_info.task_type,
                'sample_count': adjusted_samples,
                'evaluation_type': dataset_info.evaluation_type,
                'estimated_time_minutes': self._estimate_evaluation_time(dataset_name, adjusted_samples),
                'priority': self._get_dataset_priority(dataset_name, model_capabilities)
            }
            
            plan['estimated_samples'] += adjusted_samples
            
            # Update task coverage
            task_type = dataset_info.task_type
            if task_type not in plan['task_coverage']:
                plan['task_coverage'][task_type] = 0
            plan['task_coverage'][task_type] += 1
        
        # Sort datasets by priority
        plan['priority_order'] = sorted(
            recommended_datasets,
            key=lambda x: plan['datasets'][x]['priority'],
            reverse=True
        )
        
        return plan
    
    def _estimate_evaluation_time(self, dataset_name: str, sample_count: int) -> int:
        """Estimate evaluation time in minutes for a dataset"""
        dataset_info = self.registry.get_dataset_info(dataset_name)
        
        # Base time per sample in seconds (rough estimates)
        time_per_sample = {
            'coding': 30,  # Code generation and execution takes time
            'reasoning': 15,  # Mathematical reasoning
            'qa': 10,  # Question answering
            'function_calling': 20,  # Function calling complexity
            'instruction_following': 15,  # Instruction evaluation
            'biomedical_qa': 12,  # Medical domain complexity
            'relation_extraction': 8,  # Text processing
            'biomedical_ner': 8,  # NER tasks
            'scientific_ner': 8,  # Scientific NER
            'sequence_classification': 5,  # Classification tasks
            'summarization': 20,  # Text generation
            'document_vqa': 25,  # Visual reasoning
            'safety_classification': 5  # Classification
        }
        
        base_time = time_per_sample.get(dataset_info.task_type, 10)
        total_seconds = sample_count * base_time
        return max(1, total_seconds // 60)  # Convert to minutes
    
    def _get_dataset_priority(self, dataset_name: str, model_capabilities: List[str]) -> int:
        """Get priority score for a dataset based on model capabilities"""
        dataset_info = self.registry.get_dataset_info(dataset_name)
        
        # Base priority scores
        base_priorities = {
            'humaneval': 100,  # Always high priority
            'gsm8k': 95,  # Core reasoning
            'mbpp': 90,  # Core coding
            'bioasq': 85,  # Important biomedical
            'mt_bench': 80,  # Instruction following
            'bigcodebench': 75,  # Advanced coding
        }
        
        priority = base_priorities.get(dataset_name, 50)
        
        # Boost priority if dataset matches model capabilities
        if dataset_info.task_type in model_capabilities:
            priority += 20
        
        # Boost for implemented datasets
        if dataset_info.implemented:
            priority += 10
        
        return priority
    
    def analyze_dataset_gaps(self) -> Dict[str, Any]:
        """Analyze gaps in dataset coverage"""
        all_datasets = self.registry.get_all_datasets()
        implemented = self.registry.get_implemented_datasets()
        
        analysis = {
            'coverage_by_task_type': {},
            'missing_datasets': [],
            'implementation_gaps': [],
            'recommendations': []
        }
        
        # Analyze coverage by task type
        task_types = set(info.task_type for info in all_datasets.values())
        
        for task_type in task_types:
            task_datasets = self.registry.get_datasets_by_task_type(task_type)
            implemented_count = sum(1 for info in task_datasets.values() if info.implemented)
            
            analysis['coverage_by_task_type'][task_type] = {
                'total': len(task_datasets),
                'implemented': implemented_count,
                'coverage_ratio': implemented_count / len(task_datasets) if task_datasets else 0
            }
        
        # Identify missing implementations
        for name, info in all_datasets.items():
            if not info.implemented:
                analysis['implementation_gaps'].append({
                    'dataset': name,
                    'task_type': info.task_type,
                    'importance': 'high' if name in ['math', 'mmlu', 'mt_bench'] else 'medium'
                })
        
        # Generate recommendations
        low_coverage_tasks = [
            task for task, stats in analysis['coverage_by_task_type'].items()
            if stats['coverage_ratio'] < 0.5
        ]
        
        if low_coverage_tasks:
            analysis['recommendations'].append(
                f"Implement more datasets for: {', '.join(low_coverage_tasks)}"
            )
        
        if len(implemented) < 10:
            analysis['recommendations'].append(
                "Consider implementing more core evaluation datasets"
            )
        
        return analysis
    
    def get_dataset_compatibility_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Get compatibility matrix showing which datasets work with which evaluation types"""
        all_datasets = self.registry.get_all_datasets()
        evaluation_types = set(info.evaluation_type for info in all_datasets.values())
        
        matrix = {}
        
        for eval_type in evaluation_types:
            matrix[eval_type] = {}
            for dataset_name, dataset_info in all_datasets.items():
                matrix[eval_type][dataset_name] = dataset_info.evaluation_type == eval_type
        
        return matrix