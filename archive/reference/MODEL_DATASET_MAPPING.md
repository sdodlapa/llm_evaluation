# Model-to-Dataset Mapping Guide
*Strategic Mapping of Model Categories to Optimal Dataset Groups*

## 🎯 Executive Summary

This guide maps our **25+ models** across **6 specialization categories** to our **26 datasets** across **9 task types** for optimal evaluation strategies.

## 📊 Model Category ↔ Dataset Category Mapping

### 💻 **Coding Specialists → Coding + Programming Datasets**

| Model Category | Recommended Datasets | Priority | Samples |
|----------------|---------------------|----------|---------|
| **Coding Models** | **Primary Coding Group** | High | 19,664 |
| - qwen3_coder_30b | ✅ humaneval, mbpp, bigcodebench | Essential | 1,164 |
| - qwen25_7b | 🔄 codecontests, apps | Expansion | 18,500 |
| - qwen3_8b | **All 5 coding datasets** | Complete | 19,664 |
| - deepseek_coder_16b |  |  |  |

**Evaluation Strategy**: Focus on code generation, algorithm implementation, competitive programming

---

### 🧮 **Mathematical Reasoning → Mathematics + Reasoning Datasets**

| Model Category | Recommended Datasets | Priority | Samples |
|----------------|---------------------|----------|---------|
| **Math Models** | **Mathematics + Reasoning Group** | High | 55,705 |
| - qwen25_math_7b | ✅ gsm8k | Essential | 1,319 |
| - wizardmath_70b | 🔄 mathqa, math_competition, aime | Expansion | 37,577 |
| - qwen25_7b | ✅ hellaswag + 🔄 math, winogrande | Complete | 16,809 |

**Evaluation Strategy**: Grade school to competition-level mathematics, logical reasoning

---

### 📝 **Text Generation → General Purpose Datasets**

| Model Category | Recommended Datasets | Priority | Samples |
|----------------|---------------------|----------|---------|
| **Text Models** | **Broad Coverage Group** | Medium | 50,000+ |
| - qwen25_7b | ✅ All 7 ready datasets | Baseline | 13,777 |
| - qwen3_8b | ✅ mt_bench, hellaswag, arc_challenge | Core | 11,294 |
| - qwen3_14b | 🔄 mmlu, ifeval | Expansion | 14,542 |
| - llama31_8b | **Comparison across all categories** | Validation | All |

**Evaluation Strategy**: Instruction following, reasoning, general knowledge, conversation

---

### 🖼️ **Multimodal → Vision-Language Datasets**

| Model Category | Recommended Datasets | Priority | Samples |
|----------------|---------------------|----------|---------|
| **Multimodal Models** | **Multimodal Group** | Critical | 481,234 |
| - qwen2_vl_7b | 🔄 scienceqa, vqa_v2, chartqa | Essential | 481,234 |

**Evaluation Strategy**: Visual question answering, chart interpretation, science reasoning with images

---

### ⚡ **Efficiency Models → All Dataset Types (Resource Testing)**

| Model Category | Recommended Datasets | Priority | Samples |
|----------------|---------------------|----------|---------|
| **Efficiency Models** | **Performance Testing Group** | High | Variable |
| - qwen25_0_5b | ✅ Small samples from all categories | Speed | 2-5 per dataset |
| - qwen25_3b | ✅ Medium samples across categories | Efficiency | 10-20 per dataset |
| - qwen25_1_5b_genomic | 🔄 efficiency_bench, mobile_benchmark | Specialized | 3,000 |

**Evaluation Strategy**: Speed vs accuracy tradeoffs, resource utilization, mobile deployment

---

### 🧬 **Genomics Specialists → Genomics + Scientific Datasets**

| Model Category | Recommended Datasets | Priority | Samples |
|----------------|---------------------|----------|---------|
| **Genomics Models** | **Genomics + Science Group** | Specialized | 515,234 |
| - qwen25_1_5b_genomic | 🔄 genomics_benchmark, protein_sequences | Core | 31,000 |
| - qwen25_72b_genomic | 🔄 bioasq + scienceqa | Advanced | 22,206 |
| - qwen25_math_7b | ✅ gsm8k + 🔄 mathqa | Math foundation | 31,156 |

**Evaluation Strategy**: Sequence analysis, protein prediction, bioinformatics, scientific reasoning

---

## 🎛️ Evaluation Matrix

### **Ready-to-Use Combinations (✅)**

| Model Type | Dataset Group | Models | Datasets | Total Samples |
|------------|---------------|--------|----------|---------------|
| **Coding** | Coding Ready | 4 models | 3 datasets | 1,164 |
| **Math** | Reasoning Ready | 3 models | 2 datasets | 11,361 |
| **General** | All Ready | 7 models | 7 datasets | 13,777 |
| **Multimodal** | - | 1 model | 0 datasets | 0 |
| **Efficiency** | Sample Testing | 3 models | 7 datasets | Variable |
| **Genomics** | Basic Testing | 4 models | 1 dataset | 1,319 |

### **Expansion Opportunities (🔄)**

| Model Type | Dataset Group | Models | Datasets | Potential Samples |
|------------|---------------|--------|----------|-------------------|
| **Coding** | Full Coding | 4 models | 5 datasets | 19,664 |
| **Math** | Full Mathematics | 3 models | 6 datasets | 55,705 |
| **Multimodal** | Vision-Language | 1 model | 3 datasets | 481,234 |
| **Genomics** | Domain-Specific | 4 models | 6 datasets | 515,234 |
| **General** | Comprehensive | 11 models | 26 datasets | 614,397 |

## 🚀 Strategic Evaluation Plans

### **Plan A: Quick Validation (Ready Datasets)**
```bash
# Coding specialists on coding tasks
python run_evaluation.py --models qwen3_coder_30b,qwen3_8b --datasets humaneval,mbpp,bigcodebench

# Math specialists on reasoning tasks  
python run_evaluation.py --models qwen25_math_7b --datasets gsm8k,hellaswag

# Efficiency testing
python run_evaluation.py --models qwen25_0_5b,qwen25_3b --datasets humaneval --samples 5
```

### **Plan B: Comprehensive Evaluation (All Categories)**
```bash
# Implement pending datasets first, then:
python run_evaluation.py --models qwen3_coder_30b --datasets codecontests,apps
python run_evaluation.py --models qwen25_math_7b --datasets mathqa,math_competition
python run_evaluation.py --models qwen2_vl_7b --datasets scienceqa,vqa_v2
```

### **Plan C: Domain-Specific Deep Dive**
```bash
# Genomics specialization
python run_evaluation.py --models qwen25_1_5b_genomic,qwen25_72b_genomic --datasets genomics_benchmark,protein_sequences,bioasq

# Multimodal capabilities
python run_evaluation.py --models qwen2_vl_7b --datasets scienceqa,chartqa,vqa_v2
```

## 🎯 Optimization Recommendations

### **High-Impact Implementations (Priority 1)**
1. **Mathematics datasets** → Enable proper math model evaluation
2. **Multimodal datasets** → Unlock vision-language model testing
3. **Additional coding datasets** → Expand coding model coverage

### **Domain-Specific Implementations (Priority 2)**  
1. **Genomics datasets** → Enable specialized domain evaluation
2. **Efficiency datasets** → Support mobile/edge model testing
3. **Function calling datasets** → Complete agent evaluation

### **Cross-Category Validation (Priority 3)**
1. **Test math models on coding tasks** → Evaluate transfer learning
2. **Test coding models on reasoning** → Assess general capability  
3. **Test efficiency models across all domains** → Performance profiling

---

## 📊 Usage Statistics Prediction

### **Current Usage (7 Ready Datasets)**
- **Coding Models**: 3 datasets × 4 models = 12 evaluations
- **Math Models**: 2 datasets × 3 models = 6 evaluations  
- **General Models**: 7 datasets × 7 models = 49 evaluations
- **Total**: ~67 immediate evaluation combinations

### **Full Implementation (26 Datasets)**
- **Coding Models**: 5 datasets × 4 models = 20 evaluations
- **Math Models**: 6 datasets × 3 models = 18 evaluations
- **Multimodal Models**: 3 datasets × 1 model = 3 evaluations
- **Genomics Models**: 6 datasets × 4 models = 24 evaluations
- **Efficiency Models**: 26 datasets × 3 models = 78 evaluations
- **General Models**: 26 datasets × 11 models = 286 evaluations
- **Total**: ~429 comprehensive evaluation combinations

---

*This mapping enables strategic, efficient evaluation across the full spectrum of model capabilities and specialized domains.*