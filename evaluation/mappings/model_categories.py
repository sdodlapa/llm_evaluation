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

CODING_SPECIALISTS = ModelCategory(
    name="coding_specialists",
    description="Models optimized for code generation, debugging, and programming tasks",
    models=[
        "qwen3_8b",
        "qwen3_14b", 
        "qwen25_7b",
        "qwen3_coder_30b",
        "deepseek_coder_16b"
    ],
    primary_datasets=[
        "humaneval",
        "mbpp", 
        "bigcodebench"
    ],
    optional_datasets=[
        "codecontests",
        "apps",
        "advanced_coding_sample",
        "advanced_coding_extended"
    ],
    evaluation_metrics=[
        "code_execution",
        "pass_at_k", 
        "functional_correctness",
        "compilation_success",
        "test_case_pass_rate"
    ],
    category_config={
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
    priority="HIGH"
)


# ================================
# CATEGORY REGISTRY
# ================================

# Registry of all available categories (will expand as we add more)
CATEGORY_REGISTRY = {
    "coding_specialists": CODING_SPECIALISTS
}


# ================================
# HELPER FUNCTIONS
# ================================

def get_all_categories() -> Dict[str, ModelCategory]:
    """Get all available model categories"""
    return CATEGORY_REGISTRY.copy()


def get_category_for_model(model_name: str) -> Optional[ModelCategory]:
    """Find which category a model belongs to"""
    model_lower = model_name.lower()
    
    for category in CATEGORY_REGISTRY.values():
        if category.is_model_in_category(model_name):
            return category
    
    return None


def get_models_in_category(category_name: str) -> List[str]:
    """Get all models in a specific category"""
    category = CATEGORY_REGISTRY.get(category_name.lower())
    return category.models if category else []


def get_datasets_for_category(category_name: str, include_optional: bool = True) -> List[str]:
    """Get all datasets for a specific category"""
    category = CATEGORY_REGISTRY.get(category_name.lower())
    if not category:
        return []
    
    datasets = category.primary_datasets.copy()
    if include_optional:
        datasets.extend(category.optional_datasets)
    
    return datasets


def is_valid_model_dataset_pair(model_name: str, dataset_name: str) -> bool:
    """Check if a model-dataset combination is valid based on category mapping"""
    category = get_category_for_model(model_name)
    if not category:
        return False  # Model not in any category
    
    all_datasets = category.get_all_datasets()
    return dataset_name.lower() in [d.lower() for d in all_datasets]


def get_category_evaluation_config(category_name: str) -> Dict[str, Any]:
    """Get evaluation configuration for a category"""
    category = CATEGORY_REGISTRY.get(category_name.lower())
    return category.get_evaluation_config() if category else {}


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