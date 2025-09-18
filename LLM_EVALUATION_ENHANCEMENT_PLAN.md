# LLM Evaluation Pipeline Enhancement Plan
## Model Category-Dataset Mapping & Systematic Evaluation Implementation

**Document Version:** 1.0  
**Created:** September 18, 2025  
**Status:** Planning Phase  
**Implementation Priority:** HIGH  

---

## 🎯 **EXECUTIVE SUMMARY**

This document outlines a systematic approach to enhance our existing LLM evaluation pipeline with proper model category-dataset mapping. The goal is to ensure each model category is evaluated on appropriate, specialized datasets while maintaining architectural integrity.

**Key Objectives:**
1. Create formal model category-dataset mapping system
2. Implement category-based evaluation workflows
3. Ensure scalable, maintainable architecture
4. Provide three evaluation scenarios: single model, category-based, and comprehensive

---

## 📊 **CURRENT PIPELINE ANALYSIS**

### **Assets Available:**
- ✅ **32+ Model Configurations** - All major model families represented
- ✅ **40+ Dataset Files** - Much more than the 13 currently registered
- ✅ **Working Orchestrator** - `orchestrator.py` with `run_single_evaluation()`
- ✅ **Model Registry** - 36 models registered and validated
- ✅ **Modular Architecture** - Clean separation of concerns

### **Issues to Address:**
- ❌ **Dataset Discovery Gap** - Only 13/40+ datasets registered as "ready"
- ❌ **No Formal Mapping** - Models evaluated on arbitrary datasets
- ❌ **CLI Interface Broken** - Import issues in `run_evaluation.py`
- ❌ **Category Organization** - No systematic grouping of models by specialization

---

## 🏗️ **TECHNICAL ARCHITECTURE PLAN**

### **Phase 1: Model Category Definition & Mapping System**

#### **1.1 Model Categories** (Based on Specialization)

```yaml
MODEL_CATEGORIES:
  coding_specialists:
    description: "Models optimized for code generation, debugging, and programming tasks"
    models: 
      - qwen3_8b
      - qwen3_14b  
      - qwen25_7b
      - qwen3_coder_30b
      - deepseek_coder_16b
    datasets:
      - humaneval
      - mbpp
      - bigcodebench
      - codecontests
      - apps
    evaluation_metrics: ["code_execution", "pass_at_k", "functional_correctness"]
    
  mathematical_reasoning:
    description: "Models specialized in mathematical problem solving and quantitative reasoning"
    models:
      - qwen25_math_7b
      - wizardmath_70b
      - granite_3_1_8b
    datasets:
      - gsm8k
      - math
      - minerva_math
      - aime
    evaluation_metrics: ["numerical_accuracy", "step_by_step_reasoning", "proof_correctness"]
    
  biomedical_specialists:
    description: "Models trained for biomedical literature, clinical tasks, and life sciences"
    models:
      - biomistral_7b
      - biogpt_large
      - clinical_t5_large
      - qwen25_1_5b_genomic
      - qwen25_72b_genomic
    datasets:
      - bioasq
      - pubmedqa
      - mediqa
      - genomics_ner
      - protein_function
      - chemprot
    evaluation_metrics: ["biomedical_qa_accuracy", "ner_f1", "clinical_relevance"]
    
  multimodal_processing:
    description: "Models capable of processing multiple modalities (text, images, documents)"
    models:
      - qwen2_vl_7b
      - donut_base
      - layoutlmv3_base
    datasets:
      - docvqa
      - chartqa
      - multimodal_sample
      - scienceqa
    evaluation_metrics: ["multimodal_accuracy", "visual_reasoning", "ocr_accuracy"]
    
  general_purpose:
    description: "General-purpose models for diverse language understanding tasks"
    models:
      - llama31_8b
      - mistral_7b
      - mistral_nemo_12b
      - olmo2_13b
      - yi_9b
      - yi_1_5_34b
      - gemma2_9b
    datasets:
      - arc_challenge
      - hellaswag
      - mt_bench
      - mmlu
      - truthfulqa
    evaluation_metrics: ["multiple_choice_accuracy", "llm_judge_score", "coherence"]
    
  efficiency_optimized:
    description: "Lightweight models optimized for resource efficiency"
    models:
      - qwen25_0_5b
      - qwen25_3b
      - phi35_mini
    datasets:
      - humaneval      # Lighter coding tasks
      - gsm8k          # Basic reasoning
      - arc_challenge  # Simple QA
    evaluation_metrics: ["efficiency_score", "latency", "accuracy_per_parameter"]
    
  scientific_research:
    description: "Models specialized for scientific literature and research tasks"
    models:
      - scibert_base
      - specter2_base
      - longformer_large
    datasets:
      - scientific_papers
      - scierc
      - pubmed_abstracts
    evaluation_metrics: ["scientific_accuracy", "citation_relevance", "domain_knowledge"]
    
  safety_alignment:
    description: "Models focused on safety, toxicity detection, and ethical AI"
    models:
      - safety_bert
      - claude_sonnet    # If available
    datasets:
      - toxicity_detection
      - truthfulqa
      - safety_eval
    evaluation_metrics: ["safety_score", "toxicity_detection_f1", "bias_assessment"]
```

#### **1.2 Implementation Files Structure**

```
evaluation/
├── mappings/
│   ├── __init__.py
│   ├── model_categories.py          # Model category definitions
│   ├── dataset_categories.py        # Dataset category definitions  
│   ├── category_mappings.py         # Category-to-category mappings
│   └── evaluation_strategies.py     # Category-specific evaluation logic
├── category_evaluation/
│   ├── __init__.py
│   ├── coding_evaluator.py          # Coding-specific evaluation logic
│   ├── math_evaluator.py            # Math-specific evaluation logic
│   ├── biomedical_evaluator.py      # Biomedical-specific evaluation logic
│   ├── multimodal_evaluator.py      # Multimodal-specific evaluation logic
│   └── base_category_evaluator.py   # Base class for category evaluators
├── enhanced_dataset_discovery/
│   ├── __init__.py
│   ├── dataset_scanner.py           # Auto-discover datasets from files
│   ├── dataset_validator.py         # Validate dataset format and content
│   └── dataset_metadata_extractor.py # Extract metadata from datasets
└── cli/
    ├── __init__.py
    ├── evaluation_cli.py            # Enhanced CLI interface
    └── category_commands.py         # Category-specific commands
```

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Foundation Setup (Week 1)**

#### **Task 1.1: Create Model Category Mapping System**
```python
# evaluation/mappings/model_categories.py
@dataclass
class ModelCategory:
    name: str
    description: str
    models: List[str]
    primary_datasets: List[str]
    optional_datasets: List[str]
    evaluation_metrics: List[str]
    category_specific_config: Dict[str, Any]

# evaluation/mappings/category_mappings.py  
class CategoryMappingManager:
    def get_datasets_for_model(self, model_name: str) -> List[str]
    def get_category_for_model(self, model_name: str) -> ModelCategory
    def get_models_in_category(self, category_name: str) -> List[str]
    def validate_model_dataset_compatibility(self, model: str, dataset: str) -> bool
```

#### **Task 1.2: Enhanced Dataset Discovery**
```python
# evaluation/enhanced_dataset_discovery/dataset_scanner.py
class DatasetScanner:
    def scan_evaluation_data_directory(self) -> Dict[str, DatasetInfo]
    def auto_register_discovered_datasets(self) -> int
    def validate_dataset_files(self) -> List[ValidationResult]
    
    # Goal: Discover all 40+ datasets, not just 13
```

### **Phase 2: Category-Specific Evaluation (Week 2)**

#### **Task 2.1: Coding Specialists Implementation**
- **Models**: qwen3_8b, qwen3_14b, qwen25_7b, qwen3_coder_30b, deepseek_coder_16b
- **Datasets**: humaneval, mbpp, bigcodebench, codecontests, apps
- **Metrics**: code_execution, pass_at_k, functional_correctness

#### **Task 2.2: Mathematical Reasoning Implementation**  
- **Models**: qwen25_math_7b, wizardmath_70b, granite_3_1_8b
- **Datasets**: gsm8k, math, minerva_math, aime
- **Metrics**: numerical_accuracy, step_by_step_reasoning

#### **Task 2.3: Biomedical Specialists Implementation**
- **Models**: biomistral_7b, biogpt_large, clinical_t5_large, qwen25_*_genomic
- **Datasets**: bioasq, pubmedqa, mediqa, genomics_ner, protein_function, chemprot
- **Metrics**: biomedical_qa_accuracy, ner_f1, clinical_relevance

### **Phase 3: Advanced Categories (Week 3)**

#### **Task 3.1: Multimodal Processing**
- **Models**: qwen2_vl_7b, donut_base, layoutlmv3_base
- **Datasets**: docvqa, chartqa, multimodal_sample, scienceqa

#### **Task 3.2: General Purpose & Efficiency Models**
- **General**: llama31_8b, mistral_7b, mistral_nemo_12b, etc.
- **Efficiency**: qwen25_0_5b, qwen25_3b, phi35_mini

### **Phase 4: Integration & CLI (Week 4)**

#### **Task 4.1: Enhanced CLI Interface**
```bash
# Three evaluation scenarios you requested:

# Scenario 1: Single model + single dataset
python evaluate.py --model qwen3_8b --dataset humaneval --samples 5

# Scenario 2: Single model + all category datasets
python evaluate.py --model qwen3_8b --category-datasets --samples 5

# Scenario 3: Category-based evaluation
python evaluate.py --category coding_specialists --samples 5

# Scenario 4: Full comprehensive evaluation
python evaluate.py --all-categories --samples 5

# Additional utilities:
python evaluate.py --list-categories
python evaluate.py --list-models-in-category coding_specialists
python evaluate.py --list-datasets-for-model qwen3_8b
python evaluate.py --validate-mappings
```

---

## 🔧 **TECHNICAL IMPLEMENTATION DETAILS**

### **Core Configuration File**
```yaml
# evaluation/mappings/evaluation_config.yaml
categories:
  coding_specialists:
    models:
      qwen3_8b:
        preset: "balanced"
        priority: "high"
        specialization_weight: 0.95
      qwen3_14b:
        preset: "performance" 
        priority: "high"
        specialization_weight: 0.98
      deepseek_coder_16b:
        preset: "balanced"
        priority: "medium"
        specialization_weight: 0.90
    
    datasets:
      primary:
        - humaneval
        - mbpp
        - bigcodebench
      secondary:
        - codecontests
        - apps
    
    evaluation_config:
      default_sample_limit: 100
      timeout_per_sample: 30
      metrics: ["pass_at_1", "pass_at_10", "functional_correctness"]
      
  mathematical_reasoning:
    # Similar structure...
```

### **Enhanced Dataset Registration**
```python
# evaluation/enhanced_dataset_discovery/dataset_scanner.py
class EnhancedDatasetScanner:
    def __init__(self, base_path: str = "evaluation_data"):
        self.base_path = Path(base_path)
        self.discovered_datasets = {}
        
    def discover_all_datasets(self) -> Dict[str, DatasetInfo]:
        """Scan all subdirectories and discover datasets"""
        dataset_files = list(self.base_path.rglob("*.json"))
        
        # Filter out metadata, summaries, logs
        valid_datasets = [
            f for f in dataset_files 
            if not any(exclude in f.name.lower() 
                      for exclude in ["meta", "summary", "log", "download"])
        ]
        
        for dataset_file in valid_datasets:
            dataset_info = self.extract_dataset_info(dataset_file)
            if dataset_info:
                self.discovered_datasets[dataset_info.name] = dataset_info
        
        return self.discovered_datasets
```

### **Category-Specific Evaluation Logic**
```python
# evaluation/category_evaluation/coding_evaluator.py
class CodingEvaluator(BaseCategoryEvaluator):
    def __init__(self):
        super().__init__("coding_specialists")
        
    def evaluate_model_on_dataset(self, model_name: str, dataset_name: str, samples: int) -> EvaluationResult:
        """Coding-specific evaluation with code execution"""
        # Load model with coding-optimized configuration
        model = self.load_model_with_category_config(model_name)
        
        # Load dataset samples
        dataset_samples = self.load_dataset_samples(dataset_name, samples)
        
        # Run coding-specific evaluation
        results = []
        for sample in dataset_samples:
            generated_code = model.generate_code(sample["prompt"])
            execution_result = self.execute_code_safely(generated_code, sample.get("test_cases"))
            results.append({
                "sample_id": sample["id"],
                "generated_code": generated_code,
                "execution_result": execution_result,
                "pass_at_1": execution_result.success
            })
        
        return self.compute_category_metrics(results)
```

---

## 📊 **EXPECTED OUTCOMES**

### **Immediate Benefits:**
1. **Systematic Evaluation** - Each model tested on appropriate datasets
2. **Scalable Architecture** - Easy to add new models/datasets/categories
3. **Comprehensive Coverage** - All 40+ datasets utilized effectively
4. **Category Insights** - Deep analysis of specialization performance

### **Performance Metrics:**
- **Dataset Utilization**: 40+ datasets (vs current 13)
- **Model Coverage**: 100% of 32 models properly categorized
- **Evaluation Accuracy**: Category-specific metrics and validation
- **CLI Usability**: Three evaluation scenarios fully supported

### **Quality Assurance:**
- **Validation Pipeline** - Ensure model-dataset compatibility
- **Automated Testing** - Category evaluation correctness
- **Performance Monitoring** - Track evaluation quality over time
- **Documentation** - Complete mapping documentation

---

## 🎯 **IMPLEMENTATION STRATEGY**

### **Recommended Approach:**

**Week 1: Foundation** 
- Start with **Coding Specialists** (most models, clear datasets)
- Implement category mapping system
- Fix dataset discovery to find all 40+ datasets

**Week 2: Core Categories**
- Add **Mathematical Reasoning** and **Biomedical Specialists**
- Validate category-specific evaluation logic
- Create CLI interface

**Week 3: Advanced Categories**
- Add **Multimodal**, **General Purpose**, **Efficiency**
- Implement category-specific metrics
- Add validation and testing

**Week 4: Integration & Polish**
- Complete CLI with all three scenarios
- Add comprehensive documentation
- Performance optimization and testing

### **Alternative Rapid Implementation:**
If you prefer faster implementation, we could:
1. **Start with existing working components** (orchestrator + 13 datasets)
2. **Add category mapping layer** on top
3. **Incrementally discover and add datasets**
4. **Expand categories iteratively**

---

## ❓ **DECISION POINTS**

**Questions for Implementation Direction:**

1. **Phase Approach**: Start with comprehensive Week 1-4 plan OR rapid iterative approach?

2. **Category Priority**: Begin with Coding Specialists (largest group) OR your preferred category?

3. **Dataset Discovery**: Fix all 40+ datasets first OR start with working 13 and expand?

4. **CLI Interface**: Create new enhanced CLI OR fix existing `run_evaluation.py`?

5. **Configuration Format**: YAML config files OR Python-based configuration?

**Which approach and starting point would you prefer for implementation?**

---

## 📋 **NEXT STEPS**

Once you confirm the approach, I'll implement:

1. **Model category mapping system** with your preferred starting category
2. **Enhanced dataset discovery** to utilize all available datasets  
3. **Category-specific evaluation logic** 
4. **CLI interface** for the three evaluation scenarios
5. **Validation and testing framework**

**Ready to proceed with implementation once you confirm the direction! 🚀**