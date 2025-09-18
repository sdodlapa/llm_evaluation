"""
Dataset Processor - Sample preparation and evaluation strategies
Handles sample preparation, evaluation strategies, and data processing
"""

from typing import Dict, Any, List
import logging

try:
    from .dataset_registry import DatasetInfo, dataset_registry
except ImportError:
    from evaluation.dataset_registry import DatasetInfo, dataset_registry

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Handles dataset sample preparation and evaluation strategies"""
    
    def __init__(self):
        self.registry = dataset_registry
    
    def get_evaluation_strategy(self, dataset_name: str) -> str:
        """Get the evaluation strategy for a dataset"""
        dataset_info = self.registry.get_dataset_info(dataset_name)
        return dataset_info.evaluation_type
    
    def prepare_evaluation_sample(self, dataset_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a sample for evaluation based on dataset type
        
        Args:
            dataset_name: Name of the dataset
            sample: Raw sample from dataset
            
        Returns:
            Processed sample ready for evaluation
        """
        dataset_info = self.registry.get_dataset_info(dataset_name)
        
        # Start with base sample
        prepared_sample = {
            'dataset_name': dataset_name,
            'task_type': dataset_info.task_type,
            'evaluation_type': dataset_info.evaluation_type,
            'original_sample': sample.copy()
        }
        
        # Task-specific preparation
        if dataset_info.task_type == 'coding':
            prepared_sample.update({
                'prompt': sample.get('problem', sample.get('prompt', sample.get('description', ''))),
                'expected_output': sample.get('solution', sample.get('canonical_solution', sample.get('answer', ''))),
                'test_cases': sample.get('test_cases', sample.get('tests', [])),
                'entry_point': sample.get('entry_point', 'main'),
                'timeout': sample.get('timeout', 10)
            })
        
        elif dataset_info.task_type == 'reasoning':
            prepared_sample.update({
                'question': sample.get('question', sample.get('problem', sample.get('prompt', ''))),
                'expected_answer': sample.get('answer', sample.get('solution', '')),
                'reasoning_steps': sample.get('reasoning', sample.get('steps', [])),
                'multiple_choice': sample.get('choices', sample.get('options', []))
            })
        
        elif dataset_info.task_type == 'qa':
            prepared_sample.update({
                'question': sample.get('question', sample.get('prompt', '')),
                'expected_answer': sample.get('answer', sample.get('target', '')),
                'context': sample.get('context', sample.get('passage', '')),
                'choices': sample.get('choices', sample.get('options', []))
            })
        
        elif dataset_info.task_type == 'function_calling':
            prepared_sample.update({
                'function_description': sample.get('function_description', sample.get('api_description', '')),
                'user_query': sample.get('query', sample.get('question', '')),
                'expected_function_call': sample.get('expected_call', sample.get('function_call', {})),
                'available_functions': sample.get('functions', sample.get('tools', []))
            })
        
        elif dataset_info.task_type == 'instruction_following':
            prepared_sample.update({
                'instruction': sample.get('instruction', sample.get('prompt', '')),
                'expected_behavior': sample.get('expected_behavior', sample.get('target', '')),
                'constraints': sample.get('constraints', [])
            })
        
        elif dataset_info.task_type == 'biomedical_qa':
            prepared_sample.update({
                'question': sample.get('question', sample.get('prompt', '')),
                'expected_answer': sample.get('answer', sample.get('target', '')),
                'context': sample.get('context', sample.get('abstract', sample.get('passage', ''))),
                'mesh_terms': sample.get('mesh_terms', []),
                'pmid': sample.get('pmid', sample.get('pubmed_id', ''))
            })
        
        elif dataset_info.task_type == 'relation_extraction':
            prepared_sample.update({
                'text': sample.get('text', sample.get('sentence', '')),
                'expected_relations': sample.get('relations', sample.get('labels', [])),
                'entities': sample.get('entities', []),
                'relation_types': sample.get('relation_types', [])
            })
        
        elif dataset_info.task_type in ['biomedical_ner', 'scientific_ner']:
            prepared_sample.update({
                'text': sample.get('text', sample.get('sentence', '')),
                'expected_entities': sample.get('entities', sample.get('labels', [])),
                'entity_types': sample.get('entity_types', [])
            })
        
        elif dataset_info.task_type == 'sequence_classification':
            prepared_sample.update({
                'sequence': sample.get('sequence', sample.get('protein_sequence', sample.get('text', ''))),
                'expected_label': sample.get('label', sample.get('function', '')),
                'sequence_type': sample.get('sequence_type', 'protein'),
                'label_categories': sample.get('categories', [])
            })
        
        elif dataset_info.task_type == 'summarization':
            prepared_sample.update({
                'text': sample.get('text', sample.get('article', sample.get('document', ''))),
                'expected_summary': sample.get('summary', sample.get('abstract', '')),
                'max_length': sample.get('max_length', 150),
                'min_length': sample.get('min_length', 50)
            })
        
        elif dataset_info.task_type == 'document_vqa':
            prepared_sample.update({
                'image_path': sample.get('image', sample.get('image_path', '')),
                'question': sample.get('question', sample.get('prompt', '')),
                'expected_answer': sample.get('answer', sample.get('target', '')),
                'ocr_text': sample.get('ocr_text', '')
            })
        
        elif dataset_info.task_type == 'safety_classification':
            prepared_sample.update({
                'text': sample.get('text', sample.get('content', '')),
                'expected_label': sample.get('label', sample.get('toxicity', '')),
                'confidence_threshold': sample.get('threshold', 0.5),
                'safety_categories': sample.get('categories', [])
            })
        
        return prepared_sample
    
    def get_evaluation_metrics(self, task_type: str) -> List[str]:
        """Get appropriate evaluation metrics for a task type
        
        Args:
            task_type: Type of task
            
        Returns:
            List of metric names
        """
        metric_mapping = {
            'coding': ['pass_at_k', 'code_execution_accuracy', 'syntax_correctness'],
            'reasoning': ['exact_match', 'numerical_accuracy', 'reasoning_score'],
            'qa': ['exact_match', 'f1_score', 'bleu_score'],
            'function_calling': ['function_call_accuracy', 'parameter_accuracy', 'execution_success'],
            'instruction_following': ['instruction_compliance', 'output_quality', 'constraint_satisfaction'],
            'biomedical_qa': ['exact_match', 'f1_score', 'medical_accuracy'],
            'relation_extraction': ['precision', 'recall', 'f1_score'],
            'biomedical_ner': ['entity_f1', 'precision', 'recall'],
            'scientific_ner': ['entity_f1', 'precision', 'recall'],
            'sequence_classification': ['accuracy', 'precision', 'recall', 'f1_score'],
            'summarization': ['rouge_score', 'bleu_score', 'semantic_similarity'],
            'document_vqa': ['exact_match', 'f1_score', 'visual_reasoning_score'],
            'safety_classification': ['accuracy', 'precision', 'recall', 'auc_roc']
        }
        return metric_mapping.get(task_type, ['accuracy', 'f1_score'])
    
    def prepare_prompt_template(self, dataset_name: str, sample: Dict[str, Any]) -> str:
        """Generate appropriate prompt template for a dataset sample
        
        Args:
            dataset_name: Name of the dataset
            sample: Prepared sample
            
        Returns:
            Formatted prompt string
        """
        dataset_info = self.registry.get_dataset_info(dataset_name)
        task_type = dataset_info.task_type
        
        # Task-specific prompt templates
        if task_type == 'coding':
            template = f"""Please solve the following coding problem:

Problem: {sample.get('prompt', '')}

Write a Python function that solves this problem. Make sure your solution is correct and handles all edge cases.

```python
def solution():
    # Your code here
    pass
```"""
        
        elif task_type == 'reasoning':
            if sample.get('multiple_choice'):
                choices_text = '\n'.join([f"{i+1}. {choice}" for i, choice in enumerate(sample.get('multiple_choice', []))])
                template = f"""Question: {sample.get('question', '')}

{choices_text}

Please answer with the number of the correct choice and provide your reasoning."""
            else:
                template = f"""Question: {sample.get('question', '')}

Please provide a detailed answer with step-by-step reasoning."""
        
        elif task_type == 'qa':
            context = sample.get('context', '')
            context_text = f"\n\nContext: {context}" if context else ""
            template = f"""Question: {sample.get('question', '')}{context_text}

Please provide a clear and accurate answer."""
        
        elif task_type == 'function_calling':
            functions_text = str(sample.get('available_functions', []))
            template = f"""Available Functions: {functions_text}

User Query: {sample.get('user_query', '')}

Please determine which function to call and provide the appropriate parameters in JSON format."""
        
        elif task_type == 'biomedical_qa':
            context = sample.get('context', '')
            context_text = f"\n\nContext: {context}" if context else ""
            template = f"""Medical Question: {sample.get('question', '')}{context_text}

Please provide a medically accurate answer based on the available information."""
        
        elif task_type == 'relation_extraction':
            template = f"""Text: {sample.get('text', '')}

Please identify and extract all relationships between entities in the given text. Format your response as a list of relationships."""
        
        elif task_type in ['biomedical_ner', 'scientific_ner']:
            template = f"""Text: {sample.get('text', '')}

Please identify and extract all named entities from the text. Classify each entity by type."""
        
        elif task_type == 'sequence_classification':
            template = f"""Sequence: {sample.get('sequence', '')}

Please classify this sequence and predict its function or category."""
        
        elif task_type == 'summarization':
            max_length = sample.get('max_length', 150)
            template = f"""Text: {sample.get('text', '')}

Please provide a concise summary of the above text (maximum {max_length} words)."""
        
        elif task_type == 'safety_classification':
            template = f"""Text: {sample.get('text', '')}

Please classify this text for safety concerns. Identify any potentially harmful, toxic, or inappropriate content."""
        
        else:
            # Generic template
            template = f"""Task: {sample.get('prompt', sample.get('question', sample.get('text', '')))}

Please provide an appropriate response."""
        
        return template
    
    def validate_model_output(self, dataset_name: str, model_output: str, expected_output: Any) -> Dict[str, Any]:
        """Validate model output against expected output
        
        Args:
            dataset_name: Name of the dataset
            model_output: Output from the model
            expected_output: Expected output
            
        Returns:
            Validation results
        """
        dataset_info = self.registry.get_dataset_info(dataset_name)
        
        validation_result = {
            'dataset_name': dataset_name,
            'task_type': dataset_info.task_type,
            'model_output': model_output,
            'expected_output': expected_output,
            'valid': True,
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Task-specific validation
            if dataset_info.task_type == 'coding':
                # Check if output contains code
                if 'def ' not in model_output and 'function' not in model_output.lower():
                    validation_result['valid'] = False
                    validation_result['issues'].append("Output does not contain recognizable code")
            
            elif dataset_info.task_type == 'function_calling':
                # Check if output contains valid function call format
                if '{' not in model_output or '}' not in model_output:
                    validation_result['valid'] = False
                    validation_result['issues'].append("Output does not contain valid function call format")
            
            # Add more task-specific validations as needed
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
        
        return validation_result