"""
Model Categories Definition
===========================

Defines model categories and their characteristics for systematic evaluation.
Starting with Coding Specialists, will expand to other categories.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ModelCategory:
    """Represents a category of models with their associated datasets and evaluation strategy"""
    
    name: str
    description: str
    models: List[str]
    primary_datasets: List[str]
    optional_datasets: List[str]
    evaluation_metrics: List[str]
    category_config: Dict[str, Any]
    priority: str = "HIGH"
    
    def get_all_datasets(self) -> List[str]:
        """Get all datasets (primary + optional) for this category"""
        return self.primary_datasets + self.optional_datasets
    
    def is_model_in_category(self, model_name: str) -> bool:
        """Check if a model belongs to this category"""
        return model_name.lower() in [m.lower() for m in self.models]
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get category-specific evaluation configuration"""
        return self.category_config


# ================================
# CODING SPECIALISTS CATEGORY
# ================================

CODING_SPECIALISTS = {
    'models': [
        'qwen3_8b',
        'qwen3_14b',
        'codestral_22b',
        'qwen3_coder_30b',
        'deepseek_coder_16b'
    ],
    'primary_datasets': [
        "humaneval",
        "mbpp", 
        "bigcodebench"
    ],
    'optional_datasets': [
        "codecontests",
        "apps",
        "advanced_coding_sample",
        "advanced_coding_extended"
    ],
    'evaluation_metrics': [
        "code_execution",
        "pass_at_k", 
        "functional_correctness",
        "compilation_success",
        "test_case_pass_rate"
    ],
    'category_config': {
        "default_sample_limit": 100,
        "timeout_per_sample": 30,
        "max_tokens": 2048,
        "temperature": 0.1,  # Low temperature for consistent code generation
        "top_p": 0.9,
        "stop_sequences": ["```", "def ", "class ", "\n\n\n"],
        "code_execution_timeout": 10,
        "enable_syntax_validation": True,
        "enable_test_execution": True,
        "save_generated_code": True
    },
    'priority': "HIGH"
}


# ================================
# BIOMEDICAL SPECIALISTS CATEGORY
# ================================

BIOMEDICAL_SPECIALISTS = {
    'models': [
        'biomistral_7b',              # BioMistral specialized (AWQ quantized)
        'biomistral_7b_unquantized',  # BioMistral unquantized for comparison
        'biomedlm_7b',                # Stanford BioMedLM 2.7B - PubMed trained
        'medalpaca_7b',               # MedAlpaca 7B - LLaMA medical instruction tuned
        'biogpt',                     # Microsoft BioGPT - biomedical generation
        'bio_clinicalbert',           # Bio_ClinicalBERT - MIMIC-III clinical BERT
        'medalpaca_13b',              # Medical domain instruction-tuned (larger)
        'clinical_camel_70b',         # Clinical domain fine-tuned Llama (large)
        'pubmedbert_large',           # PubMed domain BERT-style model
        'biogpt_large'                # Biomedical text generation model (if exists)
    ],
    'primary_datasets': [
        "bioasq",
        "pubmedqa", 
        "mediqa"
    ],
    'optional_datasets': [
        "biomedical_sample",
        "biomedical_extended",
        "scierc"
    ],
    'evaluation_metrics': [
        "biomedical_qa_accuracy",
        "medical_entity_recognition", 
        "clinical_relevance_score",
        "pubmed_domain_accuracy",
        "biomedical_reasoning"
    ],
    'category_config': {
        "default_sample_limit": 50,
        "timeout_per_sample": 45,
        "max_tokens": 1024,
        "temperature": 0.1,  # Low temperature for medical accuracy
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "\n\n\n"],
        "enable_medical_validation": True,
        "enable_entity_extraction": True,
        "save_medical_reasoning": True,
        "require_evidence_citing": True
    },
    'priority': "HIGH"
}


# ================================
# MATHEMATICAL REASONING CATEGORY
# ================================

MATHEMATICAL_REASONING = {
    'models': [
        'qwen25_math_7b',    # Specialized Qwen math model - working ✅  
        'deepseek_math_7b',  # DeepSeek specialized math model
        'wizardmath_70b',    # WizardMath large model with AWQ quantization ✅
        'metamath_70b',      # MetaMath large model for comparison
        'qwen25_7b'          # General model with strong math capabilities ✅
    ],
    'primary_datasets': [
        "gsm8k",
        "enhanced_math_fixed"
    ],
    'optional_datasets': [
        "advanced_math_sample"
    ],
    'evaluation_metrics': [
        "mathematical_accuracy",
        "problem_solving_steps", 
        "numerical_correctness",
        "reasoning_clarity",
        "solution_completeness"
    ],
    'category_config': {
        "default_sample_limit": 50,
        "timeout_per_sample": 45,
        "max_tokens": 1024,
        "temperature": 0.1,  # Low temperature for consistent mathematical reasoning
        "top_p": 0.9,
        "stop_sequences": ["Problem:", "Solution:", "\n\n\n"],
        "enable_step_validation": True,
        "enable_numerical_verification": True,
        "save_reasoning_steps": True,
        "require_final_answer": True
    },
    'priority': "HIGH"
}


# ================================
# MULTIMODAL PROCESSING CATEGORY
# ================================

MULTIMODAL_PROCESSING = {
    'models': [
        'qwen2_vl_7b',
        'donut_base',
        'layoutlmv3_base'
    ],
    'primary_datasets': [
        "docvqa",
        "multimodal_sample"
    ],
    'optional_datasets': [
        "chartqa",
        "scienceqa"
    ],
    'evaluation_metrics': [
        "multimodal_accuracy",
        "visual_reasoning_score",
        "document_understanding",
        "text_extraction_accuracy",
        "question_answering_precision"
    ],
    'category_config': {
        "default_sample_limit": 25,  # Smaller batches for multimodal
        "timeout_per_sample": 60,   # Longer timeout for complex processing
        "max_tokens": 512,
        "temperature": 0.2,         # Low temperature for precise understanding
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "\n\n"],
        "enable_visual_processing": True,
        "enable_document_analysis": True,
        "save_visual_attention": False,  # Would require additional processing
        "require_confident_answers": True
    },
    'priority': "HIGH",
    'phase': "2"
}


# ================================
# SCIENTIFIC RESEARCH CATEGORY
# ================================

SCIENTIFIC_RESEARCH = {
    'models': [
        'scibert_base',
        'specter2_base', 
        'longformer_large'
    ],
    'primary_datasets': [
        "scientific_papers",
        "scierc"
    ],
    'optional_datasets': [
        "pubmed_abstracts"
    ],
    'evaluation_metrics': [
        "scientific_accuracy",
        "citation_relevance",
        "domain_knowledge_score",
        "technical_comprehension",
        "research_quality_assessment"
    ],
    'category_config': {
        "default_sample_limit": 20,  # Smaller batches for complex scientific texts
        "timeout_per_sample": 90,   # Longer timeout for complex scientific reasoning
        "max_tokens": 1024,
        "temperature": 0.1,         # Low temperature for precise scientific understanding
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "Conclusion:", "\n\n\n"],
        "enable_technical_validation": True,
        "enable_citation_analysis": True,
        "save_scientific_reasoning": True,
        "require_evidence_based_answers": True
    },
    'priority': "HIGH",
    'phase': "2"
}


# ================================
# CATEGORY REGISTRY
# ================================

# Registry of all available categories (will expand as we add more)
CATEGORY_REGISTRY = {
    "coding_specialists": CODING_SPECIALISTS,
    "mathematical_reasoning": MATHEMATICAL_REASONING,
    "biomedical_specialists": BIOMEDICAL_SPECIALISTS,
    "multimodal_processing": MULTIMODAL_PROCESSING,
    "scientific_research": SCIENTIFIC_RESEARCH
}

# Alias for compatibility with different import patterns
MODEL_CATEGORIES = {
    "CODING_SPECIALISTS": CODING_SPECIALISTS['models'],
    "MATHEMATICAL_REASONING": MATHEMATICAL_REASONING['models'],
    "BIOMEDICAL_SPECIALISTS": BIOMEDICAL_SPECIALISTS['models'],
    "MULTIMODAL_PROCESSING": MULTIMODAL_PROCESSING['models'],
    "SCIENTIFIC_RESEARCH": SCIENTIFIC_RESEARCH['models']
}


# ================================
# HELPER FUNCTIONS
# ================================

def get_all_categories() -> Dict[str, Dict[str, Any]]:
    """Get all available model categories"""
    return CATEGORY_REGISTRY.copy()


def get_category_for_model(model_name: str) -> Optional[str]:
    """Find which category a model belongs to"""
    model_lower = model_name.lower()
    
    for category_name, category in CATEGORY_REGISTRY.items():
        if model_lower in [m.lower() for m in category['models']]:
            return category_name
    
    return None


def get_models_in_category(category_name: str) -> List[str]:
    """Get all models in a specific category"""
    category = CATEGORY_REGISTRY.get(category_name.lower())
    return category['models'] if category else []


def get_datasets_for_category(category_name: str, include_optional: bool = True) -> List[str]:
    """Get all datasets for a specific category"""
    category = CATEGORY_REGISTRY.get(category_name.lower())
    if not category:
        return []
    
    datasets = category['primary_datasets'].copy()
    if include_optional:
        datasets.extend(category['optional_datasets'])
    
    return datasets


def is_valid_model_dataset_pair(model_name: str, dataset_name: str) -> bool:
    """Check if a model-dataset combination is valid based on category mapping"""
    category_name = get_category_for_model(model_name)
    if not category_name:
        return False  # Model not in any category
    
    category = CATEGORY_REGISTRY.get(category_name.lower())
    if not category:
        return False
    
    all_datasets = category['primary_datasets'] + category['optional_datasets']
    return dataset_name.lower() in [d.lower() for d in all_datasets]


def get_category_evaluation_config(category_name: str) -> Dict[str, Any]:
    """Get evaluation configuration for a category"""
    category = CATEGORY_REGISTRY.get(category_name.lower())
    return category['category_config'] if category else {}


# ================================
# FUTURE CATEGORIES (PLACEHOLDERS)
# ================================

# These will be implemented in subsequent phases:

# MATHEMATICAL_REASONING = ModelCategory(
#     name="mathematical_reasoning",
#     description="Models specialized in mathematical problem solving",
#     models=["qwen25_math_7b", "wizardmath_70b", "granite_3_1_8b"],
#     primary_datasets=["gsm8k", "math", "minerva_math"],
#     optional_datasets=["aime", "enhanced_math_fixed", "advanced_math_sample"],
#     evaluation_metrics=["numerical_accuracy", "step_by_step_reasoning"],
#     category_config={...},
#     priority="HIGH"
# )

# BIOMEDICAL_SPECIALISTS = ModelCategory(
#     name="biomedical_specialists", 
#     description="Models for biomedical and clinical tasks",
#     models=["biomistral_7b", "biogpt_large", "clinical_t5_large", "qwen25_1_5b_genomic"],
#     primary_datasets=["bioasq", "pubmedqa", "mediqa"],
#     optional_datasets=["genomics_ner", "protein_function", "chemprot"],
#     evaluation_metrics=["biomedical_qa_accuracy", "ner_f1", "clinical_relevance"],
#     category_config={...},
#     priority="HIGH"
# )