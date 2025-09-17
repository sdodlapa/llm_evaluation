# 🎯 **OPTIMAL MODEL-DATASET COMBINATIONS (EXPANDED)**

## Overview
This document provides the recommended model-dataset combinations for efficient evaluation of our **17-model ecosystem**: 12 specialized Qwen models + 5 strategic non-Qwen comparisons. Instead of testing every model on every dataset, we focus on strategic combinations that showcase each model's specialized capabilities and provide meaningful architectural comparisons.

## 🏗️ **MODEL ARCHITECTURE OVERVIEW**

### **Qwen Family (12 Models)**
| Model | Size | Specialization | Primary Use | Context |
|-------|------|---------------|-------------|---------|
| `qwen25_math_7b` | 7B | Mathematics | GSM8K, enhanced_math | 128K |
| `qwen3_coder_30b` | 30B | Coding | HumanEval, advanced_coding | 128K |
| `qwen2_vl_7b` | 8.5B | Multimodal | multimodal_sample | 128K |
| `qwen25_7b` | 7B | General Purpose | MMLU, Any dataset | 128K |
| `qwen25_3b` | 3B | Efficiency Balance | General tasks | 128K |
| `qwen25_0_5b` | 0.5B | Ultra-Efficiency | Basic tasks | 128K |
| `qwen25_1_5b_genomic` | 1.5B | Genomics | biomedical_extended | 128K |
| `qwen25_72b_genomic` | 72B | Complex Genomics | biomedical_extended | 128K |
| `qwen3_8b` | 8B | Balanced General | Mixed datasets | 128K |
| `qwen3_14b` | 14B | Performance | Complex tasks | 128K |

### **Strategic Non-Qwen Additions (5 Models)**
| Model | Size | Strategic Value | Comparison Point | Context |
|-------|------|----------------|------------------|---------|
| `mistral_nemo_12b` | 12B | Long Context Specialist | vs Qwen 128K performance | 128K |
| `granite_3_1_8b` | 8B | Production Enterprise | vs Qwen production readiness | 128K |
| `olmo2_13b_research` | 13B | Research Transparency | Academic benchmark standard | 4K |
| `yi_1_5_34b` | 34B | Multilingual Powerhouse | vs Qwen multilingual capability | 32K |
| `phi35_mini_efficiency` | 3.8B | Ultra-Efficiency Champion | vs Qwen small models | 128K |

---

## 🔢 **MATHEMATICS SPECIALIST**

### **Primary Model: `qwen25_math_7b`**
**Specialization**: Advanced mathematical reasoning  
**Configuration**: Ultra-precise temperature (0.05), extended function calls (6)

#### **Recommended Datasets:**
1. **GSM8K** ⭐⭐⭐ (Primary validation)
   - **Purpose**: Validate mathematical reasoning capabilities
   - **Expected**: Should significantly outperform general models
   - **Samples**: 1,319 grade school math problems

2. **Enhanced Math Fixed** ⭐⭐ (Advanced validation)
   - **Purpose**: Test complex mathematical problem solving
   - **Expected**: Should excel at multi-step reasoning
   - **Samples**: 5 intermediate-level problems

#### **Comparison Tests:**
- `qwen25_math_7b` vs `qwen25_7b` on GSM8K (specialist vs general)
- `qwen25_math_7b` vs `qwen3_8b` on GSM8K (specialist vs baseline)

---

## 💻 **CODING SPECIALIST**

### **Primary Model: `qwen3_coder_30b`**
**Specialization**: Software development and complex algorithms  
**Configuration**: Precise temperature (0.05), extended function calls (8), MoE architecture

#### **Recommended Datasets:**
1. **HumanEval** ⭐⭐⭐ (Primary validation)
   - **Purpose**: Hand-written programming problems
   - **Expected**: Should achieve highest pass rates
   - **Samples**: 164 coding challenges

2. **MBPP** ⭐⭐⭐ (Comprehensive validation)
   - **Purpose**: Mostly basic Python problems
   - **Expected**: Should excel at Python programming
   - **Samples**: 974 problems

3. **Advanced Coding Extended** ⭐⭐ (Expert validation)
   - **Purpose**: Complex algorithms and system design
   - **Expected**: Should handle advanced programming concepts
   - **Samples**: 8 expert-level problems

#### **Comparison Tests:**
- `qwen3_coder_30b` vs `qwen25_7b` on HumanEval (specialist vs general)
- `qwen3_coder_30b` vs `qwen3_8b` on MBPP (specialist vs baseline)

#### **Secondary Coding Models:**
- **`qwen25_7b`**: Test on HumanEval for general coding capability
- **`qwen3_8b`**: Test on MBPP for baseline coding performance

---

## 🧬 **GENOMIC SPECIALISTS**

### **Primary Models: `qwen25_1_5b_genomic`, `qwen25_72b_genomic`**
**Specialization**: Biomedical and genomic data analysis  
**Configuration**: Ultra-precise temperature (0.01), extended context (65K tokens)

#### **Recommended Datasets:**
1. **Biomedical Extended** ⭐⭐⭐ (Primary validation)
   - **Purpose**: Genetics, molecular biology, gene editing
   - **Expected**: Should demonstrate domain expertise
   - **Samples**: 10 comprehensive biomedical questions

#### **Model Comparison:**
- `qwen25_1_5b_genomic` vs `qwen25_72b_genomic` on biomedical_extended (efficiency vs capability)
- `qwen25_1_5b_genomic` vs `qwen25_7b` on biomedical_extended (specialist vs general)

---

## 👁️ **MULTIMODAL SPECIALIST**

### **Primary Model: `qwen2_vl_7b`**
**Specialization**: Vision-language understanding  
**Configuration**: Multimodal processing, chart interpretation

#### **Recommended Datasets:**
1. **Multimodal Sample** ⭐⭐ (Primary validation)
   - **Purpose**: Chart and visualization interpretation
   - **Expected**: Should excel at visual data analysis
   - **Samples**: 3 multimodal tasks

#### **Note**: This model currently has limited evaluation due to text-only datasets. Future expansion needed for comprehensive multimodal evaluation.

---

## 🎓 **GENERAL KNOWLEDGE EVALUATION**

### **All Models Testing**
**Purpose**: Establish baseline capabilities across all models

#### **Core Dataset: MMLU** ⭐⭐⭐
**All models should be tested on MMLU for general comparison:**

| **Model** | **Expected Performance** | **Purpose** |
|-----------|-------------------------|-------------|
| `qwen25_7b` | High (baseline) | General knowledge standard |
| `qwen3_8b` | High (baseline) | Original model comparison |
| `qwen3_14b` | Highest | Largest general model |
| `qwen25_math_7b` | High+ (math subjects) | Math domain boost |
| `qwen3_coder_30b` | High+ (CS subjects) | Technical knowledge |
| `qwen25_3b` | Good | Efficiency baseline |
| `qwen25_0_5b` | Moderate | Ultra-efficiency test |

---

## 🎯 **TRUTHFULNESS EVALUATION**

### **All Models Testing**
**Purpose**: Evaluate factual accuracy and misconception avoidance

#### **Core Dataset: Truthfulness Fixed** ⭐⭐
**Priority models for truthfulness testing:**

1. **`qwen25_7b`** - General model truthfulness
2. **`qwen25_math_7b`** - Mathematical fact accuracy  
3. **`qwen25_1_5b_genomic`** - Scientific fact accuracy
4. **`qwen3_coder_30b`** - Technical fact accuracy

---

## ⚡ **EFFICIENCY EVALUATION**

### **Efficiency Models Performance Testing**
**Purpose**: Validate performance vs resource trade-offs

#### **Recommended Tests:**

| **Model** | **Test Dataset** | **Purpose** | **Expected** |
|-----------|-----------------|-------------|--------------|
| `qwen25_0_5b` | HumanEval | Ultra-efficiency coding | Lower but acceptable |
| `qwen25_3b` | HumanEval | Balanced efficiency | Good performance |
| `qwen25_3b` | GSM8K | Balanced math | Reasonable math ability |
| `qwen25_0_5b` | MMLU | Ultra-efficiency knowledge | Basic but functional |

---

## 📋 **STRATEGIC EVALUATION MATRIX**

### **Phase 1: Specialist Validation** (High Priority)
```bash
# Validate each specialist excels in their domain
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset gsm8k
python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset humaneval  
python evaluation/run_evaluation.py --model qwen25_1_5b_genomic --dataset biomedical_extended
```

### **Phase 2: Cross-Specialist Comparison** (Medium Priority)
```bash
# Compare specialists vs general models on same tasks
python evaluation/run_evaluation.py --model qwen25_7b --dataset gsm8k
python evaluation/run_evaluation.py --model qwen25_7b --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_7b --dataset biomedical_extended
```

### **Phase 3: General Knowledge Baseline** (Medium Priority)
```bash
# Establish MMLU performance across key models
python evaluation/run_evaluation.py --model qwen25_7b --dataset mmlu
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset mmlu
python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset mmlu
python evaluation/run_evaluation.py --model qwen25_3b --dataset mmlu
```

### **Phase 4: Efficiency Analysis** (Lower Priority)
```bash
# Test efficiency models on core tasks
python evaluation/run_evaluation.py --model qwen25_3b --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_0_5b --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_3b --dataset gsm8k
```

### **Phase 5: Truthfulness Evaluation** (Lower Priority)
```bash
# Test truthfulness across model types
python evaluation/run_evaluation.py --model qwen25_7b --dataset truthfulness_fixed
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset truthfulness_fixed
python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset truthfulness_fixed
```

---

## 🌍 **STRATEGIC NON-QWEN COMPARISONS**

### **Phase 6: Architecture Diversity Testing** (New Priority)

#### **Long Context Comparison: Mistral-NeMo vs Qwen**
```bash
# Compare 128K context handling
python evaluation/run_evaluation.py --model mistral_nemo_12b --dataset mmlu
python evaluation/run_evaluation.py --model qwen25_7b --dataset mmlu
# Direct size comparison (12B vs 7B, both 128K context)
```

#### **Production Enterprise: Granite vs Qwen**
```bash
# Test production-ready enterprise focus
python evaluation/run_evaluation.py --model granite_3_1_8b --dataset humaneval
python evaluation/run_evaluation.py --model qwen3_8b --dataset humaneval
# Similar sizes (8B), compare enterprise vs general optimization
```

#### **Research Transparency: OLMo vs Qwen**
```bash
# Academic benchmark standard
python evaluation/run_evaluation.py --model olmo2_13b_research --dataset mmlu
python evaluation/run_evaluation.py --model qwen3_14b --dataset mmlu
# Compare research transparency vs commercial development
```

#### **Multilingual Power: Yi vs Qwen**
```bash
# Multilingual capability comparison
python evaluation/run_evaluation.py --model yi_1_5_34b --dataset mmlu
python evaluation/run_evaluation.py --model qwen3_14b --dataset mmlu
# Large model comparison (34B vs 14B)
```

#### **Ultra-Efficiency Champion: Phi vs Qwen**
```bash
# Small model efficiency battle
python evaluation/run_evaluation.py --model phi35_mini_efficiency --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_3b --dataset humaneval
# Similar sizes (3.8B vs 3B), test efficiency architectures
```

### **Phase 7: Cross-Architecture Validation** (Lower Priority)
```bash
# Test architectural diversity on same task
python evaluation/run_evaluation.py --model granite_3_1_8b --dataset gsm8k
python evaluation/run_evaluation.py --model yi_1_5_34b --dataset advanced_coding_extended
python evaluation/run_evaluation.py --model phi35_mini_efficiency --dataset mmlu
```

---

## 🚫 **AVOID THESE COMBINATIONS**

### **Inefficient/Unnecessary Tests:**
- ❌ `qwen25_0_5b` on `advanced_coding_extended` (too complex for ultra-small model)
- ❌ `qwen25_math_7b` on `multimodal_sample` (not multimodal capable)
- ❌ `qwen2_vl_7b` on `gsm8k` (text-only math, not utilizing multimodal strength)
- ❌ All models on `biomedical_extended` (only genomic specialists need this)
- ❌ `qwen25_72b_genomic` on basic datasets (overkill, use efficient models)

### **Resource-Intensive Combinations to Defer:**
- ⚠️ `qwen25_72b_genomic` on any dataset (39.2GB VRAM - test only if resources available)
- ⚠️ `qwen3_coder_30b` on `mbpp` (18GB VRAM for basic problems - use smaller model first)

---

## 🎯 **EXPECTED OUTCOMES**

### **Specialist Model Advantages:**
- **Math Specialist**: 15-25% better on GSM8K vs general models
- **Coding Specialist**: 20-30% better pass rates on HumanEval vs general models  
- **Genomic Specialist**: Significantly better biomedical reasoning vs general models
- **Efficiency Models**: 80%+ performance at 50% resource usage

### **Success Criteria:**
1. **Specialists outperform general models** in their domains
2. **General models maintain competitiveness** across broad tasks
3. **Efficiency models provide acceptable performance** at reduced resources
4. **All models demonstrate truthfulness** in factual questions

---

## 📊 **EVALUATION PRIORITY SUMMARY**

### **🔥 Must-Run (Critical Validation)**
1. `qwen25_math_7b` + `gsm8k` (math specialist validation)
2. `qwen3_coder_30b` + `humaneval` (coding specialist validation)
3. `qwen25_7b` + `mmlu` (general baseline)

### **⭐ Should-Run (Important Comparison)**
4. `qwen25_7b` + `gsm8k` (general vs specialist comparison)
5. `qwen25_7b` + `humaneval` (general vs specialist comparison)
6. `qwen25_1_5b_genomic` + `biomedical_extended` (genomic validation)

### **✅ Nice-to-Run (Comprehensive Analysis)**
7. `qwen25_3b` + `humaneval` (efficiency analysis)
8. `qwen3_coder_30b` + `advanced_coding_extended` (expert coding)
9. `qwen25_math_7b` + `truthfulness_fixed` (specialist truthfulness)

### **⏰ Run-When-Time-Permits (Full Coverage)**
10. Remaining efficiency comparisons
11. Full truthfulness evaluation
12. Advanced coding problems across models

**Total Strategic Evaluations: 15-20 targeted runs (vs 140 possible combinations)**

This focused approach provides maximum insight with minimal computational waste!