"""
Scientific and biomedical model configurations
Specialized models for scientific research, biomedical analysis, and domain-specific tasks
"""

from typing import Dict
try:
    from .model_registry import ModelConfig
except ImportError:
    from configs.model_registry import ModelConfig

# Scientific & Biomedical Models
SCIENTIFIC_MODEL_CONFIGS = {
    # Biomedical Literature Models
    "biomistral_7b": ModelConfig(
        model_name="BioMistral 7B Instruct",
        huggingface_id="BioMistral/BioMistral-7B-Instruct",
        license="Apache 2.0",
        size_gb=7.0,
        context_window=32768,
        specialization_category="biomedical",
        specialization_subcategory="literature_qa",
        primary_use_cases=["biomedical_qa", "literature_summarization", "medical_reasoning"],
        preset="balanced",
        quantization_method="awq",
        max_model_len=16384,
        gpu_memory_utilization=0.85,
        priority="HIGH",
        agent_optimized=True,
        agent_temperature=0.1,
        max_function_calls_per_turn=6,
        evaluation_batch_size=8
    ),
    
    "biogpt_large": ModelConfig(
        model_name="BioGPT Large",
        huggingface_id="microsoft/BioGPT-Large",
        license="MIT",
        size_gb=1.6,
        context_window=1024,
        specialization_category="biomedical", 
        specialization_subcategory="pubmed_generation",
        primary_use_cases=["pubmed_qa", "relation_extraction", "biomedical_ner"],
        preset="balanced",
        quantization_method="none",  # Small enough
        max_model_len=1024,
        gpu_memory_utilization=0.90,
        priority="MEDIUM",
        agent_optimized=False,  # Specialized generation tool
        agent_temperature=0.2,
        evaluation_batch_size=16
    ),
    
    "clinical_t5_large": ModelConfig(
        model_name="Clinical T5 Large",
        huggingface_id="microsoft/clinical-t5-large",
        license="MIT",
        size_gb=3.0,
        context_window=512,
        specialization_category="biomedical",
        specialization_subcategory="clinical_text",
        primary_use_cases=["clinical_notes", "medical_summarization"],
        preset="balanced",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        agent_temperature=0.05,  # High precision for clinical
        evaluation_batch_size=12
    ),
    
    # Scientific Embedding Models
    "specter2_base": ModelConfig(
        model_name="SPECTER2 Scientific Paper Embeddings",
        huggingface_id="allenai/specter2_base",
        license="Apache 2.0",
        size_gb=0.4,
        context_window=512,
        specialization_category="scientific_embeddings",
        specialization_subcategory="paper_retrieval",
        primary_use_cases=["scientific_rag", "paper_similarity", "literature_search"],
        preset="performance",  # Optimized for throughput
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.95,
        priority="HIGH",
        agent_optimized=True,
        evaluation_batch_size=64  # High throughput for embeddings
    ),
    
    "scibert_base": ModelConfig(
        model_name="SciBERT Scientific Text Encoder",
        huggingface_id="allenai/scibert_scivocab_uncased",
        license="Apache 2.0",
        size_gb=0.4,
        context_window=512,
        specialization_category="scientific_embeddings",
        specialization_subcategory="scientific_text",
        primary_use_cases=["scientific_classification", "rag_retrieval", "concept_extraction"],
        preset="performance",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.95,
        priority="MEDIUM",
        agent_optimized=True,
        evaluation_batch_size=64
    ),
    
    # Document Understanding Models
    "donut_base": ModelConfig(
        model_name="Donut Document Understanding",
        huggingface_id="naver-clova-ix/donut-base",
        license="MIT",
        size_gb=0.8,
        context_window=1024,
        specialization_category="document_understanding",
        specialization_subcategory="ocr_free_vqa",
        primary_use_cases=["document_parsing", "form_understanding", "table_extraction"],
        preset="balanced",
        quantization_method="none",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        agent_temperature=0.1,
        evaluation_batch_size=16
    ),
    
    "layoutlmv3_base": ModelConfig(
        model_name="LayoutLMv3 Base",
        huggingface_id="microsoft/layoutlmv3-base",
        license="MIT",
        size_gb=0.5,
        context_window=512,
        specialization_category="document_understanding",
        specialization_subcategory="layout_analysis",
        primary_use_cases=["document_classification", "layout_analysis", "entity_extraction"],
        preset="balanced",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.90,
        priority="MEDIUM",
        agent_optimized=True,
        evaluation_batch_size=24
    ),
    
    # Long Context Specialist Models
    "longformer_large": ModelConfig(
        model_name="Longformer Large",
        huggingface_id="allenai/longformer-large-4096",
        license="Apache 2.0",
        size_gb=1.3,
        context_window=4096,
        specialization_category="long_context",
        specialization_subcategory="document_analysis",
        primary_use_cases=["long_document_qa", "research_paper_analysis"],
        preset="balanced",
        quantization_method="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        priority="MEDIUM",
        agent_optimized=True,
        evaluation_batch_size=8
    ),
    
    # Safety and Alignment Models
    "safety_bert": ModelConfig(
        model_name="Safety BERT Classifier",
        huggingface_id="unitary/toxic-bert",
        license="Apache 2.0",
        size_gb=0.4,
        context_window=512,
        specialization_category="safety_alignment",
        specialization_subcategory="toxicity_detection",
        primary_use_cases=["safety_classification", "bias_detection"],
        preset="performance",
        quantization_method="none",
        max_model_len=512,
        gpu_memory_utilization=0.95,
        priority="LOW",
        agent_optimized=False,  # Classification tool
        evaluation_batch_size=32
    )
}

def get_biomedical_models() -> Dict[str, ModelConfig]:
    """Get biomedical specialist models"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if v.specialization_category == "biomedical"}

def get_scientific_embedding_models() -> Dict[str, ModelConfig]:
    """Get scientific embedding models for RAG and retrieval"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if v.specialization_category == "scientific_embeddings"}

def get_document_understanding_models() -> Dict[str, ModelConfig]:
    """Get document understanding and analysis models"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if v.specialization_category == "document_understanding"}

def get_long_context_models() -> Dict[str, ModelConfig]:
    """Get long context specialist models"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if v.specialization_category == "long_context"}

def get_safety_alignment_models() -> Dict[str, ModelConfig]:
    """Get safety and alignment specialist models"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if v.specialization_category == "safety_alignment"}

def get_all_scientific_models() -> Dict[str, ModelConfig]:
    """Get all scientific and biomedical specialist models"""
    return SCIENTIFIC_MODEL_CONFIGS

def get_models_by_use_case(use_case: str) -> Dict[str, ModelConfig]:
    """Get models that support a specific use case"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if use_case in v.primary_use_cases}

def get_embedding_models() -> Dict[str, ModelConfig]:
    """Get all embedding models for retrieval and similarity tasks"""
    embedding_categories = ["scientific_embeddings"]
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if v.specialization_category in embedding_categories}

def get_clinical_models() -> Dict[str, ModelConfig]:
    """Get models specialized for clinical applications"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if "clinical" in v.specialization_subcategory or "clinical" in v.primary_use_cases}

def get_literature_models() -> Dict[str, ModelConfig]:
    """Get models specialized for literature analysis"""
    return {k: v for k, v in SCIENTIFIC_MODEL_CONFIGS.items() if "literature" in v.specialization_subcategory or any("literature" in use_case for use_case in v.primary_use_cases)}