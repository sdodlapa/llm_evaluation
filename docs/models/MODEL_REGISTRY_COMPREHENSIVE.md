# 🤖 LLM Evaluation Framework - Model Registry Documentation

**Total Models**: 43 models across 3 categories  
**Last Updated**: September 19, 2025  
**Status**: Phase 1 Complete - 3 Categories Operational  

---

## 📊 **Model Overview by Category**

### **✅ Operational Categories (43 models)**

#### **🔧 CODING_SPECIALISTS (5 models)**
- qwen3_8b, qwen3_14b, codestral_22b, qwen3_coder_30b, deepseek_coder_16b
- **Datasets**: humaneval, mbpp, bigcodebench
- **Status**: ✅ Fully operational and tested

#### **🧮 MATHEMATICAL_REASONING (5 models)**  
- qwen25_math_7b, deepseek_math_7b, wizardmath_70b, metamath_70b, qwen25_7b
- **Datasets**: gsm8k, enhanced_math_fixed
- **Status**: ✅ Fully operational and tested

#### **🩺 BIOMEDICAL_SPECIALISTS (10 models)**
- biomistral_7b, biomistral_7b_unquantized, biomedlm_7b, medalpaca_7b, biogpt
- bio_clinicalbert, medalpaca_13b, clinical_camel_70b, pubmedbert_large, biogpt_large
- **Datasets**: bioasq, pubmedqa, mediqa
- **Status**: ✅ Fully operational, BioMistral 7B tested (~28 tokens/second)

#### **🌐 GENERAL_PURPOSE (23 models)**
- Various general models including Qwen, Llama, Mistral, Yi, Gemma families
- **Status**: Configured but not categorized yet

---

## 🎯 **Model Specialization Categories**

### **By Model Size Distribution**
- **Large (30B+)**: 5 models (qwen3_coder_30b, clinical_camel_70b, etc.)
- **Medium (14-16B)**: 4 models (qwen3_14b, deepseek_coder_16b, etc.)
- **Small-Medium (7-8B)**: 17 models (majority category)
- **Small (3-4B)**: 3 models (efficient deployment)
- **Tiny (<3B)**: 14 models (resource-constrained environments)

### **By License Distribution**
- **Apache 2.0**: 22 models (majority - open source)
- **MIT**: 8 models (permissive licensing)
- **Custom/Commercial**: 8 models (various restrictions)
- **Other Open**: 5 models (various open licenses)

---

## ⭐ **Key Model Highlights**

### **🏆 Top Performing Models (by category)**

#### **Coding Excellence**
- **qwen3_coder_30b**: 30B specialized coding model
- **deepseek_coder_16b**: 16B advanced code completion
- **codestral_22b**: Mistral's coding specialist

#### **Mathematical Reasoning**
- **qwen25_math_7b**: Specialized Qwen math model
- **wizardmath_70b**: Large-scale mathematical reasoning
- **deepseek_math_7b**: Efficient mathematical problem solving

#### **Biomedical Applications**
- **biomistral_7b**: Primary biomedical model (AWQ optimized, tested ✅)
- **biomedlm_7b**: Stanford's PubMed specialist (50.3% MedQA)
- **biogpt**: Microsoft's biomedical generation (proven benchmarks)

### **🔥 Qwen Model Family (14 models)**
- **qwen25_7b**: Latest general model with 128K context
- **qwen3_8b** & **qwen3_14b**: Main instruction models (tested ✅)
- **qwen2_vl_7b**: Multimodal vision-language model
- **qwen25_math_7b**: Mathematical reasoning specialist
- **qwen25_0_5b** & **qwen25_3b**: Efficiency-optimized models

---

## 🚀 **Performance Validation Status**

### **✅ Tested and Validated**
- **BioMistral 7B AWQ**: ~28 tokens/second, 3.88GB memory
- **Qwen3 8B**: Comprehensive testing complete
- **Qwen3 14B**: Framework validated
- **vLLM Integration**: Working with AWQ quantization

### **🎯 AWQ Quantization Support**
- **Working**: BioMistral 7B (proven), WizardMath 70B
- **Available**: Pre-quantized models from HuggingFace
- **Memory Savings**: ~75% reduction (4-bit quantization)

### **📊 Context Window Capabilities**
- **128K tokens**: qwen25_7b, qwen3_8b, qwen3_14b
- **32K tokens**: Most models (standard long context)
- **160K tokens**: HyenaDNA (removed with genomics category)
- **Standard (2-8K)**: Older and specialized models

---

## 🔧 **Configuration Presets**

### **Performance Preset**
- GPU Memory Utilization: 90%
- Focus: Maximum throughput
- Best for: Production deployment

### **Balanced Preset** 
- GPU Memory Utilization: 85%
- Focus: Stability and efficiency
- Best for: Development and testing

### **Memory Optimized Preset**
- GPU Memory Utilization: 75%
- Focus: Minimal memory usage
- Best for: Resource-constrained environments

---

## 📋 **Model Registry Structure**

### **Core Configuration Fields**
```python
@dataclass
class ModelConfig:
    model_name: str
    huggingface_id: str
    license: str
    size_gb: float
    context_window: int
    preset: str
    specialization_category: str
    specialization_subcategory: str
    primary_use_cases: List[str]
    quantization_method: str
    max_model_len: int
    gpu_memory_utilization: float
    priority: str
    agent_optimized: bool
    agent_temperature: float
    max_function_calls_per_turn: int
    evaluation_batch_size: int
```

### **Specialization Categories**
- **code_generation**: Programming and software development
- **mathematics**: Mathematical reasoning and problem solving
- **biomedical**: Healthcare, clinical, and life sciences
- **general**: Broad language understanding tasks
- **multimodal**: Text + image processing (future category)
- **efficiency**: Resource-optimized models (future category)

---

## 🎯 **Next Phase Models (Planned)**

### **Phase 2: Additional Categories**
- **Multimodal Processing**: qwen2_vl_7b, donut_base, layoutlmv3_base
- **Scientific Research**: scibert_base, specter2_base, longformer_large  
- **Efficiency Optimized**: qwen25_0_5b, qwen25_3b, phi35_mini
- **Safety Alignment**: Safety-focused models

### **Future Considerations**
- **Function Calling**: Models with tool use capabilities
- **Instruction Following**: Enhanced instruction adherence
- **Long Context**: Ultra-long context specialists (>128K)

---

## 📚 **Usage Commands**

### **Model Information**
```bash
# View all models
python show_models.py

# View by category
python show_models.py coding
python show_models.py math  
python show_models.py biomedical

# Model-dataset mappings
python show_models.py mapping
```

### **Evaluation Commands**
```bash
# List available categories
python category_evaluation.py --list-categories

# Single model evaluation
python category_evaluation.py --model biomistral_7b --samples 3

# Category evaluation (future)
python category_evaluation.py --category biomedical_specialists --samples 5
```

---

## 📊 **Performance Benchmarks**

### **Established Benchmarks**
- **BioMedLM**: 50.3% on MedQA (published)
- **BioGPT**: 44.98% F1 on BC5CDR, 78.2% on PubMedQA  
- **BioMistral**: ~28 tokens/second with AWQ quantization
- **Qwen3 Series**: Comprehensive framework validation complete

### **Memory Usage (H100 80GB)**
- **Large Models (30B+)**: 15-40GB (AWQ: ~10-15GB)
- **Medium Models (14-16B)**: 7-15GB (AWQ: ~4-8GB)
- **Small-Medium (7-8B)**: 3-8GB (AWQ: ~2-4GB)
- **Small Models (<4B)**: 1-4GB (minimal quantization needed)

---

**Documentation Maintained By**: LLM Evaluation Framework Team  
**Next Update**: After Phase 2 category implementations  
**Reference**: See `configs/model_registry.py` for complete technical specifications