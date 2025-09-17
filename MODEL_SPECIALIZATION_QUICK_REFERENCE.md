# 🚀 **QUICK MODEL SPECIALIZATION REFERENCE**

## 📋 **By Use Case - What Model Should I Use?**

### **🔤 TEXT GENERATION & WRITING**
| Task | Small (3-8B) | Medium (12-16B) | Large (30B+) |
|------|---------------|-----------------|-------------|
| **General Writing** | `qwen25_7b` | `mistral_nemo_12b` | `yi_1_5_34b` |
| **Long Documents** | `qwen3_8b` | `mistral_nemo_12b` | - |
| **Enterprise/Business** | `granite_3_1_8b` | - | - |
| **Research Papers** | `olmo2_13b_research` | - | - |
| **Multilingual** | `qwen25_7b` | - | `yi_1_5_34b` |

### **💻 CODE GENERATION**
| Task | Small (7-16B) | Large (30B+) | Best Choice |
|------|---------------|-------------|-------------|
| **General Programming** | `qwen3_coder_30b` | `codellama_34b_instruct` | ⭐ `qwen3_coder_30b` |
| **Python Specific** | `deepseek_coder_16b` | `codellama_34b_python` | ⭐ `codellama_34b_python` |
| **Advanced Algorithms** | `starcoder2_15b` | `codellama_34b_instruct` | ⭐ `codellama_34b_instruct` |
| **Production Code** | `deepseek_coder_v2_advanced` | `codellama_34b_instruct` | ⭐ `deepseek_coder_v2_advanced` |
| **Multi-Language** | `starcoder2_15b` | `codellama_34b_instruct` | ⭐ `starcoder2_15b` |

### **📊 DATA SCIENCE & ANALYTICS**
| Task | Model | Why This Model? |
|------|-------|----------------|
| **Python Data Analysis** | `codellama_34b_python` | ⭐ Specialized for Pandas, NumPy |
| **Statistical Analysis** | `qwen25_math_7b` | Mathematical foundation |
| **Business Intelligence** | `granite_3_1_8b` | Enterprise focus |
| **Machine Learning** | `codellama_34b_python` | ML library expertise |
| **Data Visualization** | `codellama_34b_python` | Python plotting libraries |

### **🧮 MATHEMATICS & REASONING**
| Task | Model | Specialization |
|------|-------|---------------|
| **Grade School Math** | `qwen25_math_7b` | ⭐ Specialized math training |
| **Advanced Mathematics** | `wizardmath_70b` | Complex proofs, theorems |
| **Applied Mathematics** | `qwen25_math_7b` | Practical problem solving |
| **General Math** | `qwen25_7b` | Baseline mathematical capability |

### **🧬 BIOINFORMATICS & GENOMICS**
| Task | Model | Specialization |
|------|-------|---------------|
| **DNA/RNA Analysis** | `qwen25_1_5b_genomic` | ⭐ Genomic sequence analysis |
| **Complex Genomics** | `qwen25_72b_genomic` | Protein folding, pathways |
| **Genomic Visualization** | `qwen2_vl_7b` | Visual biological data |
| **Basic Biology** | `qwen25_7b` | General biological knowledge |

### **🖼️ MULTIMODAL (Vision + Text)**
| Task | Model | Capability |
|------|-------|------------|
| **Image Analysis** | `qwen2_vl_7b` | ⭐ Only multimodal option |
| **Visual Reasoning** | `qwen2_vl_7b` | Vision-language tasks |
| **Scientific Diagrams** | `qwen2_vl_7b` | Technical image analysis |

### **⚡ EFFICIENCY & MOBILE**
| Task | Ultra-Small (0.5-2B) | Small (3-4B) | Best For |
|------|----------------------|-------------|----------|
| **Edge Devices** | `qwen25_0_5b` | `qwen25_3b` | ⭐ `qwen25_0_5b` |
| **Mobile Apps** | `qwen25_3b` | `phi35_mini_efficiency` | ⭐ `phi35_mini_efficiency` |
| **Quick Inference** | `qwen25_0_5b` | `qwen25_3b` | ⭐ `qwen25_0_5b` |
| **Embedded Systems** | `qwen25_3b` | - | ⭐ `qwen25_3b` |

---

## 🎯 **SPECIALIZATION MATRIX**

### **By Size and Specialization**
| Size | Text Gen | Coding | Data Science | Math | Genomics | Efficiency |
|------|----------|--------|--------------|------|----------|------------|
| **0.5-2B** | - | - | - | - | `qwen25_1_5b_genomic` | `qwen25_0_5b` |
| **3-8B** | `qwen25_7b` | `deepseek_coder_16b` | `granite_3_1_8b` | `qwen25_math_7b` | - | `qwen25_3b` |
| **12-16B** | `mistral_nemo_12b` | `starcoder2_15b` | - | - | - | - |
| **30-40B** | `yi_1_5_34b` | `codellama_34b_instruct` | `codellama_34b_python` | - | - | - |
| **70B+** | - | - | - | `wizardmath_70b` | `qwen25_72b_genomic` | - |

### **By License Type**
| License | Models Available | Best For |
|---------|------------------|----------|
| **Apache 2.0** | 15 models | ⭐ Commercial use, redistribution |
| **MIT** | 1 model (`phi35_mini_efficiency`) | ⭐ Most permissive license |
| **Custom/Commercial** | 6 models (CodeLlama, DeepSeek, etc.) | Research, specific use cases |

---

## 🚀 **QUICK START COMMANDS**

### **Test Each Specialization**
```bash
# Text Generation
python evaluation/run_evaluation.py --model qwen25_7b --dataset mmlu

# Code Generation  
python evaluation/run_evaluation.py --model codellama_34b_instruct --dataset humaneval

# Data Science
python evaluation/run_evaluation.py --model codellama_34b_python --dataset data_science_tasks

# Mathematics
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset gsm8k

# Genomics
python evaluation/run_evaluation.py --model qwen25_1_5b_genomic --dataset biomedical_extended

# Multimodal
python evaluation/run_evaluation.py --model qwen2_vl_7b --dataset multimodal_sample

# Efficiency
python evaluation/run_evaluation.py --model phi35_mini_efficiency --dataset humaneval
```

### **Compare Specialists vs Generalists**
```bash
# Coding: Specialist vs General
python evaluation/run_evaluation.py --model codellama_34b_instruct --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_7b --dataset humaneval

# Math: Specialist vs General  
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset gsm8k
python evaluation/run_evaluation.py --model qwen25_7b --dataset gsm8k
```

---

## 💡 **DECISION TREE**

### **"Which model should I use?"**

**1. What's your primary task?**
- **General text/writing** → `qwen25_7b` or `mistral_nemo_12b`
- **Programming** → Go to step 2
- **Data analysis** → Go to step 3  
- **Mathematics** → `qwen25_math_7b`
- **Biology/genomics** → `qwen25_1_5b_genomic`
- **Image analysis** → `qwen2_vl_7b`
- **Resource constrained** → Go to step 4

**2. What type of programming?**
- **General coding** → `qwen3_coder_30b`
- **Python/Data Science** → `codellama_34b_python`
- **Complex algorithms** → `codellama_34b_instruct`
- **Production code** → `deepseek_coder_v2_advanced`

**3. What type of data analysis?**
- **Python/Pandas** → `codellama_34b_python`
- **Statistics** → `qwen25_math_7b`
- **Business analytics** → `granite_3_1_8b`

**4. How constrained are your resources?**
- **Very limited (mobile)** → `qwen25_0_5b`
- **Somewhat limited** → `qwen25_3b` or `phi35_mini_efficiency`
- **Moderate resources** → `qwen25_7b`

---

## 🏆 **TOP RECOMMENDATIONS BY CATEGORY**

| Category | #1 Choice | #2 Choice | #3 Choice |
|----------|-----------|-----------|-----------|
| **General Purpose** | `qwen25_7b` | `mistral_nemo_12b` | `granite_3_1_8b` |
| **Code Generation** | `codellama_34b_instruct` | `qwen3_coder_30b` | `starcoder2_15b` |
| **Data Science** | `codellama_34b_python` | `qwen25_math_7b` | `granite_3_1_8b` |
| **Mathematics** | `qwen25_math_7b` | `wizardmath_70b` | `qwen25_7b` |
| **Genomics** | `qwen25_1_5b_genomic` | `qwen25_72b_genomic` | `qwen2_vl_7b` |
| **Efficiency** | `phi35_mini_efficiency` | `qwen25_3b` | `qwen25_0_5b` |
| **Enterprise** | `granite_3_1_8b` | `qwen25_7b` | `mistral_nemo_12b` |

**Total Framework**: 22 models across 8 specializations for comprehensive LLM evaluation