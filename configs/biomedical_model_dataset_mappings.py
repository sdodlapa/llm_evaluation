# Biomedical Model-Dataset Sub-Mapping Configuration
# Optimizes evaluation by matching models to their most suitable datasets

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """Configuration for a biomedical dataset"""
    name: str
    path: str
    format: str
    size: int
    domain_focus: str
    difficulty_level: str
    evaluation_metrics: List[str]
    required_capabilities: List[str]

@dataclass
class ModelDatasetMapping:
    """Maps models to their optimal datasets with performance expectations"""
    model_id: str
    primary_datasets: List[str]
    secondary_datasets: List[str]
    expected_performance: Dict[str, float]
    optimal_settings: Dict[str, any]

# Dataset Configurations
BIOMEDICAL_DATASETS = {
    "pubmedqa": DatasetConfig(
        name="PubMedQA",
        path="datasets/biomedical/pubmedqa",
        format="json",
        size=211300,
        domain_focus="pubmed_abstracts",
        difficulty_level="medium",
        evaluation_metrics=["accuracy", "f1_score", "exact_match"],
        required_capabilities=["reading_comprehension", "biomedical_knowledge"]
    ),
    
    "medqa": DatasetConfig(
        name="MedQA (USMLE)",
        path="datasets/biomedical/medqa", 
        format="json",
        size=12723,
        domain_focus="clinical_knowledge",
        difficulty_level="high",
        evaluation_metrics=["accuracy", "multiple_choice_accuracy"],
        required_capabilities=["medical_reasoning", "clinical_knowledge", "multiple_choice"]
    ),
    
    "bc5cdr": DatasetConfig(
        name="BC5CDR Chemical-Disease Relations",
        path="datasets/biomedical/bc5cdr",
        format="bioc",
        size=1500,
        domain_focus="relation_extraction",
        difficulty_level="high",
        evaluation_metrics=["precision", "recall", "f1_score"],
        required_capabilities=["named_entity_recognition", "relation_extraction"]
    ),
    
    "ddi": DatasetConfig(
        name="DDI Drug-Drug Interactions",
        path="datasets/biomedical/ddi",
        format="xml",
        size=1025,
        domain_focus="drug_interactions",
        difficulty_level="high", 
        evaluation_metrics=["precision", "recall", "f1_score"],
        required_capabilities=["named_entity_recognition", "relation_extraction"]
    ),
    
    "bioasq": DatasetConfig(
        name="BioASQ Semantic QA",
        path="datasets/biomedical/bioasq",
        format="json",
        size=3000,  # Varies by task
        domain_focus="semantic_qa",
        difficulty_level="very_high",
        evaluation_metrics=["gmap", "mrr", "rouge", "expert_evaluation"],
        required_capabilities=["information_retrieval", "qa", "summarization"]
    )
}

# Model-Dataset Optimization Mappings
BIOMEDICAL_MODEL_MAPPINGS = {
    "biomedlm_7b": ModelDatasetMapping(
        model_id="biomedlm_7b",
        primary_datasets=["pubmedqa", "medqa"],
        secondary_datasets=["bioasq"],
        expected_performance={
            "medqa": 0.503,  # Benchmark performance from paper
            "pubmedqa": 0.75,  # Estimated based on PubMed training
            "bioasq": 0.65    # Estimated
        },
        optimal_settings={
            "temperature": 0.1,
            "max_tokens": 512,
            "batch_size": 8,
            "context_utilization": 0.8  # Uses 80% of 1024 context
        }
    ),
    
    "biogpt": ModelDatasetMapping(
        model_id="biogpt",
        primary_datasets=["bc5cdr", "ddi", "pubmedqa"],
        secondary_datasets=["bioasq"],
        expected_performance={
            "bc5cdr": 0.4498,  # Benchmark F1 from paper
            "ddi": 0.4076,     # Benchmark F1 from paper
            "pubmedqa": 0.782,  # Benchmark accuracy from paper
            "bioasq": 0.70     # Estimated
        },
        optimal_settings={
            "temperature": 0.1,
            "max_tokens": 256,  # Good for relation extraction
            "batch_size": 8,
            "context_utilization": 0.9
        }
    ),
    
    "medalpaca_7b": ModelDatasetMapping(
        model_id="medalpaca_7b",
        primary_datasets=["medqa", "pubmedqa"],
        secondary_datasets=["bioasq"],
        expected_performance={
            "medqa": 0.45,     # Estimated based on instruction tuning
            "pubmedqa": 0.70,  # Estimated
            "bioasq": 0.60     # Estimated
        },
        optimal_settings={
            "temperature": 0.1,
            "max_tokens": 512,
            "batch_size": 6,
            "context_utilization": 0.85
        }
    ),
    
    "biomistral_7b": ModelDatasetMapping(
        model_id="biomistral_7b",
        primary_datasets=["pubmedqa", "medqa"],
        secondary_datasets=["bioasq"],
        expected_performance={
            "pubmedqa": 0.68,  # Estimated AWQ performance
            "medqa": 0.42,     # Estimated
            "bioasq": 0.58     # Estimated
        },
        optimal_settings={
            "temperature": 0.1,
            "max_tokens": 512,
            "batch_size": 8,
            "context_utilization": 0.9,
            "quantization_aware": True
        }
    ),
    
    "biomistral_7b_unquantized": ModelDatasetMapping(
        model_id="biomistral_7b_unquantized",
        primary_datasets=["pubmedqa", "medqa"],
        secondary_datasets=["bioasq"],
        expected_performance={
            "pubmedqa": 0.72,  # Higher than quantized version
            "medqa": 0.45,     # Higher than quantized version
            "bioasq": 0.62     # Higher than quantized version
        },
        optimal_settings={
            "temperature": 0.1,
            "max_tokens": 512,
            "batch_size": 4,   # Smaller batch due to memory
            "context_utilization": 0.9
        }
    ),
    
    "bio_clinicalbert": ModelDatasetMapping(
        model_id="bio_clinicalbert",
        primary_datasets=["bc5cdr"],  # Best for NER/classification tasks
        secondary_datasets=["ddi"],
        expected_performance={
            "bc5cdr": 0.85,    # Estimated for BERT-style classification
            "ddi": 0.80        # Estimated for BERT-style classification
        },
        optimal_settings={
            "temperature": 0.0,  # Not used for BERT embeddings
            "max_tokens": 512,   # BERT context limit
            "batch_size": 16,    # Can handle larger batches
            "task_type": "classification"  # Not generative
        }
    )
}

# Specialized Evaluation Strategies
EVALUATION_STRATEGIES = {
    "generative_qa": {
        "applicable_models": ["biomedlm_7b", "biogpt", "medalpaca_7b", "biomistral_7b", "biomistral_7b_unquantized"],
        "datasets": ["pubmedqa", "medqa", "bioasq"],
        "metrics": ["accuracy", "bleu", "rouge", "exact_match"],
        "sampling_strategy": "stratified",
        "sample_sizes": {"pubmedqa": 1000, "medqa": 500, "bioasq": 200}
    },
    
    "relation_extraction": {
        "applicable_models": ["biogpt", "bio_clinicalbert"],
        "datasets": ["bc5cdr", "ddi"],
        "metrics": ["precision", "recall", "f1_score", "auc"],
        "sampling_strategy": "balanced",
        "sample_sizes": {"bc5cdr": 500, "ddi": 300}
    },
    
    "clinical_reasoning": {
        "applicable_models": ["biomedlm_7b", "medalpaca_7b", "biomistral_7b", "biomistral_7b_unquantized"],
        "datasets": ["medqa"],
        "metrics": ["accuracy", "confidence_calibration"],
        "sampling_strategy": "difficulty_stratified",
        "sample_sizes": {"medqa": 800}
    }
}

# Performance Benchmarks and Targets
PERFORMANCE_TARGETS = {
    "tier_1": {  # High-performance models
        "models": ["biomedlm_7b", "biomistral_7b_unquantized"],
        "targets": {
            "pubmedqa": 0.70,
            "medqa": 0.45,
            "bioasq": 0.60
        }
    },
    
    "tier_2": {  # Efficient models
        "models": ["biomistral_7b", "medalpaca_7b"],
        "targets": {
            "pubmedqa": 0.65,
            "medqa": 0.40,
            "bioasq": 0.55
        }
    },
    
    "tier_3": {  # Specialized models
        "models": ["biogpt", "bio_clinicalbert"],
        "targets": {
            "bc5cdr": 0.45,
            "ddi": 0.40,
            "pubmedqa": 0.75
        }
    }
}

def get_optimal_datasets_for_model(model_id: str) -> List[str]:
    """Get the optimal datasets for a given model"""
    if model_id in BIOMEDICAL_MODEL_MAPPINGS:
        mapping = BIOMEDICAL_MODEL_MAPPINGS[model_id]
        return mapping.primary_datasets + mapping.secondary_datasets
    return []

def get_expected_performance(model_id: str, dataset: str) -> Optional[float]:
    """Get expected performance for a model on a specific dataset"""
    if model_id in BIOMEDICAL_MODEL_MAPPINGS:
        mapping = BIOMEDICAL_MODEL_MAPPINGS[model_id]
        return mapping.expected_performance.get(dataset)
    return None

def get_evaluation_strategy(model_id: str) -> str:
    """Determine the best evaluation strategy for a model"""
    if model_id == "bio_clinicalbert":
        return "relation_extraction"
    elif "biomistral" in model_id or "biomedlm" in model_id or "medalpaca" in model_id:
        return "generative_qa"
    elif model_id == "biogpt":
        return "relation_extraction"
    else:
        return "generative_qa"  # Default

# Export configuration for use in evaluation scripts
__all__ = [
    'BIOMEDICAL_DATASETS',
    'BIOMEDICAL_MODEL_MAPPINGS', 
    'EVALUATION_STRATEGIES',
    'PERFORMANCE_TARGETS',
    'get_optimal_datasets_for_model',
    'get_expected_performance',
    'get_evaluation_strategy'
]