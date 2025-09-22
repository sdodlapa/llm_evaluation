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
        'deepseek_coder_16b',
        'starcoder2_15b'  # Added: StarCoder2 15B
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
        "advanced_coding_extended",
        "repobench",         # Added: For StarCoder2 evaluation
        "repo_bench",        # NEW: Repository-level coding tasks (synthetic)
        "code_contests"      # NEW: Real competitive programming (DeepMind)
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
        "mediqa",
        "medqa"  # Added: Medical QA from USMLE-style questions
    ],
    'optional_datasets': [
        "biomedical_sample",
        "biomedical_extended",
        "scierc",
        "bc5cdr",    # Added: Chemical-disease relation extraction
        "ddi",       # Added: Drug-drug interaction extraction
        "chemprot",  # Added: Chemical-protein interaction extraction
        "genomics_ner"  # Added: Genomics named entity recognition
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
        "advanced_math_sample",
        "math_competition"     # NEW: Competition-level mathematics problems
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
        'layoutlmv3_base',
        'qwen25_vl_7b',
        'minicpm_v_26',
        'llava_next_vicuna_7b',
        'internvl2_8b',
        'llama32_vision_90b'  # Added: Llama 3.2 Vision 90B
    ],
    'primary_datasets': [
        "docvqa",
        "multimodal_sample",
        "ai2d",
        "scienceqa"
    ],
    'optional_datasets': [
        "chartqa",
        "textcqa"
    ],
    'evaluation_metrics': [
        "multimodal_accuracy",
        "visual_reasoning_score",
        "document_understanding",
        "text_extraction_accuracy",
        "question_answering_precision",
        "chart_comprehension",
        "scientific_diagram_understanding"
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
        "pubmed_abstracts",
        "chemprot",     # Added: Chemical-protein interaction extraction
        "genomics_ner", # Added: Genomics named entity recognition  
        "bioasq"        # Added: Biomedical semantic QA
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
# EFFICIENCY OPTIMIZED CATEGORY
# ================================

EFFICIENCY_OPTIMIZED = {
    'models': [
        'qwen25_0_5b',
        'qwen25_3b', 
        'phi35_mini'
    ],
    'primary_datasets': [
        "humaneval",      # Lighter coding tasks
        "gsm8k",          # Basic reasoning
        "arc_challenge"   # Simple QA
    ],
    'optional_datasets': [
        "hellaswag",
        "truthfulness_fixed"
    ],
    'evaluation_metrics': [
        "efficiency_score",
        "latency",
        "accuracy_per_parameter",
        "memory_efficiency",
        "throughput_optimization"
    ],
    'category_config': {
        "default_sample_limit": 50,  # Larger batches for efficient models
        "timeout_per_sample": 30,   # Faster timeout for lightweight models
        "max_tokens": 512,
        "temperature": 0.2,         # Balanced temperature for efficiency
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "\n\n"],
        "enable_efficiency_tracking": True,
        "enable_latency_monitoring": True,
        "save_performance_metrics": True,
        "optimize_for_throughput": True
    },
    'priority': "HIGH",
    'phase': "2"
}


# ================================
# GENERAL PURPOSE CATEGORY
# ================================

GENERAL_PURPOSE = {
    'models': [
        'llama31_8b',
        'mistral_7b',
        'mistral_nemo_12b',
        'olmo2_13b',
        'yi_9b',
        'yi_1_5_34b',
        'gemma2_9b',
        'llama31_70b',      # Added: Llama 3.1 70B 
        'llama33_70b',      # Added: Llama 3.3 70B (newer)
        'llama31_8b',       # Added: Llama 3.1 8B (faster)
        'gemma2_27b',       # Added: Gemma 2 27B
        'internlm2_20b'     # Added: InternLM2 20B
    ],
    'primary_datasets': [
        "arc_challenge",
        "hellaswag",
        "mt_bench",
        "mmlu"
    ],
    'optional_datasets': [
        "truthfulness_fixed",
        "gsm8k",  # Added: For general mathematical reasoning
        "humaneval"  # Added: For basic coding ability
    ],
    'evaluation_metrics': [
        "multiple_choice_accuracy",
        "llm_judge_score",
        "coherence",
        "general_reasoning",
        "knowledge_breadth"
    ],
    'category_config': {
        "default_sample_limit": 30,  # Moderate batches for general models
        "timeout_per_sample": 45,   # Standard timeout
        "max_tokens": 1024,
        "temperature": 0.3,         # Moderate temperature for balanced responses
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "Conclusion:", "\n\n"],
        "enable_general_reasoning": True,
        "enable_knowledge_assessment": True,
        "save_reasoning_traces": True,
        "require_coherent_responses": True
    },
    'priority': "MEDIUM",
    'phase': "3"
}


# ================================
# SAFETY ALIGNMENT CATEGORY
# ================================

SAFETY_ALIGNMENT = {
    'models': [
        'safety_bert',
        'biomistral_7b',  # Can be used for safety evaluation
        'qwen25_7b'       # General model for safety testing
    ],
    'primary_datasets': [
        "toxicity_detection",
        "truthfulqa",
        "hh_rlhf"
    ],
    'optional_datasets': [
        "safety_eval"
    ],
    'evaluation_metrics': [
        "safety_score",
        "toxicity_detection_f1",
        "bias_assessment",
        "harm_prevention",
        "ethical_alignment"
    ],
    'category_config': {
        "default_sample_limit": 25,  # Smaller batches for careful safety evaluation
        "timeout_per_sample": 60,   # Longer timeout for safety analysis
        "max_tokens": 512,
        "temperature": 0.1,         # Low temperature for consistent safety responses
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "Warning:", "\n\n"],
        "enable_safety_filtering": True,
        "enable_bias_detection": True,
        "save_safety_analysis": True,
        "require_safe_responses": True
    },
    'priority': "HIGH",
    'phase': "3"
}


# ================================
# H100-OPTIMIZED LARGE MODELS CATEGORY
# ================================

H100_OPTIMIZED_LARGE = {
    'models': [
        'qwen25_72b',            # Qwen2.5 72B Instruct (131K context)
        'llama31_70b_fp8',       # NVIDIA FP8 optimized Llama 3.1 70B
        'mixtral_8x22b',         # Mixtral 8x22B Instruct (efficient MoE)
        'dbrx_instruct',         # DBRX 132B Instruct (enterprise MoE)
        'deepseek_v3',           # DeepSeek-V3 (671B/37B active ultra-efficient)
        'xverse_65b'             # XVERSE-65B (Apache 2.0 alternative)
    ],
    'primary_datasets': [
        "mmlu_pro",              # Enhanced academic evaluation
        "bigbench_hard",         # Complex multi-step reasoning
        "longbench",             # Long-context understanding
        "mmlu",                  # Standard academic benchmark
        "hellaswag"              # Common sense reasoning
    ],
    'optional_datasets': [
        "mt_bench",              # Chat evaluation
        "truthfulness_fixed",    # Truthfulness assessment
        "enterprise_tasks",      # Business applications
        "advanced_reasoning",    # Complex problem solving
        "gsm8k",                 # Mathematical reasoning
        "humaneval"              # Basic coding ability
    ],
    'evaluation_metrics': [
        "complex_reasoning_accuracy",
        "long_context_coherence", 
        "multilingual_capability",
        "inference_efficiency",
        "h100_utilization_score",
        "memory_efficiency_ratio"
    ],
    'category_config': {
        "default_sample_limit": 10,  # Very small batches for huge models
        "timeout_per_sample": 180,  # Longer timeout for complex reasoning
        "max_tokens": 4096,
        "temperature": 0.1,         # Low temperature for consistency
        "top_p": 0.9,
        "stop_sequences": ["<|end_of_text|>", "\n\nHuman:", "\n\nUser:"],
        "enable_tensor_parallelism": True,
        "enable_h100_optimizations": True,
        "enable_performance_monitoring": True,
        "save_reasoning_traces": True,
        "require_detailed_analysis": True,
        "fp8_optimization": True,    # H100-specific
        "use_paged_kv_cache": True   # Memory optimization
    },
    'priority': "HIGHEST",
    'phase': "1"  # Priority implementation
}

# ================================
# ADVANCED CODE GENERATION CATEGORY (Updated)
# ================================

ADVANCED_CODE_GENERATION = {
    'models': [
        'granite_34b_code',          # IBM Granite 34B Code (Apache 2.0)
        'qwen3_coder_30b',          # Existing Qwen3 Coder 32B
        'codestral_22b',            # Existing Codestral 22B
        'deepseek_coder_16b',       # Existing DeepSeek Coder 16B
        'starcoder2_15b'            # Existing StarCoder2 15B
    ],
    'primary_datasets': [
        "swe_bench",                # Software engineering benchmark
        "livecodebench",            # Recent competitive programming
        "humaneval",                # Standard code evaluation
        "mbpp",                     # Python programming problems
        "bigcodebench"              # Complex coding tasks
    ],
    'optional_datasets': [
        "codecontests",
        "apps",
        "repo_bench",               # Repository-level tasks
        "code_contests"             # Competitive programming
    ],
    'evaluation_metrics': [
        "repository_level_accuracy",
        "competitive_programming_score",
        "code_execution_success_rate",
        "software_engineering_quality",
        "multifile_code_generation",
        "pass_at_k",
        "functional_correctness"
    ],
    'category_config': {
        "default_sample_limit": 50,
        "timeout_per_sample": 120,  # Longer for complex coding tasks
        "max_tokens": 4096,         # More tokens for complex code
        "temperature": 0.1,
        "top_p": 0.9,
        "stop_sequences": ["```", "def ", "class ", "\n\n\n"],
        "enable_repository_context": True,
        "enable_multifile_analysis": True,
        "save_generated_code": True,
        "require_compilation_check": True
    },
    'priority': "HIGH",
    'phase': "1"
}

# ================================
# ADVANCED MULTIMODAL CATEGORY (Updated)
# ================================

# ================================
# MIXTURE OF EXPERTS CATEGORY (Updated)
# ================================

MIXTURE_OF_EXPERTS = {
    'models': [
        'mixtral_8x7b',         # Existing: Mixtral 8x7B
        'mixtral_8x22b',        # New: Mixtral 8x22B (from ChatGPT recommendations)
        'dbrx_instruct',        # New: DBRX 132B Instruct
        'deepseek_v3'           # New: DeepSeek-V3 ultra-efficient MoE
    ],
    'primary_datasets': [
        "mmlu",
        "hellaswag", 
        "arc_challenge",
        "humaneval",
        "bigbench_hard"         # Added for advanced MoE evaluation
    ],
    'optional_datasets': [
        "gsm8k",
        "mt_bench",
        "truthfulness_fixed",
        "code_contests",
        "math_competition",
        "enterprise_tasks",     # Added for enterprise MoE models
        "longbench"            # Added for long-context MoE evaluation
    ],
    'evaluation_metrics': [
        "efficiency_per_active_param",
        "inference_speed",
        "multiple_choice_accuracy", 
        "reasoning_consistency",
        "multilingual_capability",
        "expert_utilization_efficiency",  # New: MoE-specific metric
        "memory_vs_quality_ratio"        # New: MoE efficiency metric
    ],
    'category_config': {
        "default_sample_limit": 20,  # Increased for better MoE evaluation
        "timeout_per_sample": 60,
        "max_tokens": 2048,          # Increased for complex MoE tasks
        "temperature": 0.1,          # Lower for consistency
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "\n\n"],
        "enable_expert_utilization_tracking": True,
        "enable_efficiency_metrics": True,
        "save_expert_activation_patterns": False,
        "optimize_for_throughput": True,
        "enable_moe_analysis": True,          # New: MoE-specific analysis
        "track_active_parameters": True       # New: Track active vs total params
    },
    'priority': "HIGHEST",  # Upgraded priority
    'phase': "1"           # Priority implementation
}


# ================================
# ADVANCED MULTIMODAL CATEGORY 
# ================================

ADVANCED_MULTIMODAL = {
    'models': [
        'llama32_vision_90b',       # Existing Llama 3.2 Vision 90B
        'internvl2_llama3_76b',     # New: InternVL2-Llama3-76B
        'qwen25_vl_7b',             # Existing Qwen2.5-VL 7B
        'qwen2_vl_7b',              # Existing Qwen2-VL 7B
        'minicpm_v_26',             # Existing MiniCPM-V 2.6
        'internvl2_8b'              # Existing InternVL2 8B
    ],
    'primary_datasets': [
        "mmmu",                     # Massive multimodal understanding
        "mathvista",                # Mathematical visual reasoning
        "docvqa",                   # Document visual QA
        "ai2d",                     # Diagram understanding
        "scienceqa"                 # Science visual QA
    ],
    'optional_datasets': [
        "chartqa",                  # Chart understanding
        "textvqa",                  # Text visual QA
        "multimodal_sample"         # Sample multimodal tasks
    ],
    'evaluation_metrics': [
        "advanced_visual_reasoning_score",
        "multimodal_academic_accuracy",
        "mathematical_visual_accuracy",
        "document_understanding_score",
        "chart_analysis_accuracy",
        "vision_language_coherence"
    ],
    'category_config': {
        "default_sample_limit": 15,  # Smaller batches for multimodal
        "timeout_per_sample": 120,  # Longer for visual processing
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "\n\n"],
        "enable_vision_processing": True,
        "enable_chart_analysis": True,
        "enable_document_understanding": True,
        "require_visual_attention": True,
        "multimodal_fusion": True
    },
    'priority': "HIGH",
    'phase': "1"
}


# ================================
# REASONING SPECIALIZED CATEGORY
# ================================

REASONING_SPECIALIZED = {
    'models': [
        'deepseek_r1_distill_llama_70b'  # First reasoning-distilled model
    ],
    'primary_datasets': [
        "gsm8k",
        "enhanced_math_fixed",
        "arc_challenge",
        "mmlu"
    ],
    'optional_datasets': [
        "advanced_math_sample",
        "logical_reasoning",
        "scientific_reasoning",
        "math_competition"     # NEW: Perfect fit for reasoning-specialized models
    ],
    'evaluation_metrics': [
        "chain_of_thought_quality",
        "reasoning_step_accuracy",
        "logical_consistency",
        "complex_problem_solving",
        "mathematical_reasoning_score"
    ],
    'category_config': {
        "default_sample_limit": 30,
        "timeout_per_sample": 60,  # Longer timeout for complex reasoning
        "max_tokens": 2048,        # More tokens for detailed reasoning
        "temperature": 0.1,        # Very low temperature for consistent reasoning
        "top_p": 0.9,
        "stop_sequences": ["Problem:", "Solution:", "Answer:", "\n\n\n"],
        "enable_reasoning_verification": True,
        "enable_step_by_step_analysis": True,
        "save_reasoning_chains": True,
        "require_justification": True
    },
    'priority': "HIGH",
    'phase': "2"
}


# ================================
# TEXT GEOSPATIAL CATEGORY
# ================================

TEXT_GEOSPATIAL = {
    'models': [
        'qwen25_7b',      # Strong geographic knowledge
        'qwen3_8b',       # Good general understanding
        'qwen3_14b',      # Complex spatial reasoning
        'mistral_nemo_12b' # Long context for complex queries
    ],
    'primary_datasets': [
        "spatial_reasoning",    # Custom spatial reasoning questions
        "coordinate_processing", # Coordinate math and conversions  
        "address_parsing",      # Address standardization
        "location_ner",         # Location entity recognition
        "ner_locations"         # Large-scale location NER
    ],
    'optional_datasets': [
        "geographic_features",  # Geographic knowledge
        "geographic_demand"     # Geographic analysis tasks
    ],
    'evaluation_metrics': [
        "spatial_reasoning_accuracy",
        "geographic_f1", 
        "coordinate_accuracy",
        "address_match_score",
        "qa_accuracy",
        "exact_match"
    ],
    'category_config': {
        "default_sample_limit": 20,
        "timeout_per_sample": 30,
        "max_tokens": 512,
        "temperature": 0.1,  # Low temperature for factual geographic information
        "top_p": 0.9,
        "stop_sequences": ["Question:", "Answer:", "\n\n"],
        "enable_coordinate_validation": True,
        "require_geographic_context": True,
        "save_spatial_reasoning": True,
        "enable_map_context": False  # Future feature
    },
    'priority': "HIGH",
    'phase': "4"  # Phase 4 addition
}


# ================================
# CATEGORY REGISTRY (Updated)
# ================================

# Registry of all available categories (expanded with H100-optimized models)
CATEGORY_REGISTRY = {
    "h100_optimized_large": H100_OPTIMIZED_LARGE,           # New: H100-optimized large models
    "advanced_code_generation": ADVANCED_CODE_GENERATION,   # Updated: Advanced coding models
    "advanced_multimodal": ADVANCED_MULTIMODAL,             # Updated: Advanced multimodal models
    "coding_specialists": CODING_SPECIALISTS,
    "mathematical_reasoning": MATHEMATICAL_REASONING,
    "biomedical_specialists": BIOMEDICAL_SPECIALISTS,
    "multimodal_processing": MULTIMODAL_PROCESSING,
    "scientific_research": SCIENTIFIC_RESEARCH,
    "efficiency_optimized": EFFICIENCY_OPTIMIZED,
    "general_purpose": GENERAL_PURPOSE,
    "safety_alignment": SAFETY_ALIGNMENT,
    "mixture_of_experts": MIXTURE_OF_EXPERTS,      # Updated to include new MoE models
    "reasoning_specialized": REASONING_SPECIALIZED,
    "text_geospatial": TEXT_GEOSPATIAL
}

# Alias for compatibility with different import patterns (updated)
MODEL_CATEGORIES = {
    "H100_OPTIMIZED_LARGE": H100_OPTIMIZED_LARGE['models'],           # New category
    "ADVANCED_CODE_GENERATION": ADVANCED_CODE_GENERATION['models'],   # Updated category
    "ADVANCED_MULTIMODAL": ADVANCED_MULTIMODAL['models'],             # Updated category
    "CODING_SPECIALISTS": CODING_SPECIALISTS['models'],
    "MATHEMATICAL_REASONING": MATHEMATICAL_REASONING['models'],
    "BIOMEDICAL_SPECIALISTS": BIOMEDICAL_SPECIALISTS['models'],
    "MULTIMODAL_PROCESSING": MULTIMODAL_PROCESSING['models'],
    "SCIENTIFIC_RESEARCH": SCIENTIFIC_RESEARCH['models'],
    "EFFICIENCY_OPTIMIZED": EFFICIENCY_OPTIMIZED['models'],
    "GENERAL_PURPOSE": GENERAL_PURPOSE['models'],
    "SAFETY_ALIGNMENT": SAFETY_ALIGNMENT['models'],
    "MIXTURE_OF_EXPERTS": MIXTURE_OF_EXPERTS['models'],
    "REASONING_SPECIALIZED": REASONING_SPECIALIZED['models'],
    "TEXT_GEOSPATIAL": TEXT_GEOSPATIAL['models']
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