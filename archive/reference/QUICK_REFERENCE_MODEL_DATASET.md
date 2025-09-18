# 📋 **QUICK REFERENCE: MODEL-DATASET LOOKUP (EXPANDED)**

## Primary Model Specializations

### **Qwen Family (12 Models)**
| **Model** | **Best For** | **Primary Dataset** | **Secondary Dataset** | **Avoid** |
|-----------|--------------|--------------------|-----------------------|-----------|
| `qwen25_math_7b` | Mathematics | GSM8K | enhanced_math_fixed | multimodal_sample |
| `qwen3_coder_30b` | Coding | HumanEval | advanced_coding_extended | biomedical_extended |
| `qwen25_1_5b_genomic` | Genomics | biomedical_extended | - | advanced_coding_extended |
| `qwen25_72b_genomic` | Complex Genomics | biomedical_extended | - | basic datasets |
| `qwen2_vl_7b` | Multimodal | multimodal_sample | - | text-only datasets |
| `qwen25_7b` | General Purpose | MMLU | Any dataset | - |
| `qwen25_3b` | Efficiency | HumanEval | GSM8K | advanced_coding_extended |
| `qwen25_0_5b` | Ultra-Efficiency | HumanEval | MMLU | complex datasets |

### **Strategic Non-Qwen Additions (5 Models)**
| **Model** | **Strategic Value** | **Primary Dataset** | **Compare Against** | **Architectural Focus** |
|-----------|--------------------|--------------------|---------------------|------------------------|
| `mistral_nemo_12b` | Long Context Specialist | MMLU | qwen25_7b, qwen3_14b | 128K Context Architecture |
| `granite_3_1_8b` | Production Enterprise | HumanEval | qwen3_8b | Enterprise Optimization |
| `olmo2_13b_research` | Research Transparency | MMLU | qwen3_14b | Academic Standard |
| `yi_1_5_34b` | Multilingual Power | MMLU | qwen3_coder_30b | Dense vs MoE, Cultural |
| `phi35_mini_efficiency` | Ultra-Efficiency | HumanEval | qwen25_3b | Microsoft vs Alibaba |

## Dataset Difficulty Levels

| **Dataset** | **Difficulty** | **Best Models** | **Avoid Models** |
|-------------|---------------|-----------------|------------------|
| GSM8K | Intermediate | math_7b, 25_7b, 3_8b | 0_5b |
| HumanEval | Hard | coder_30b, 25_7b, 25_3b | 0_5b |
| advanced_coding_extended | Expert | coder_30b | 25_3b, 0_5b |
| biomedical_extended | Intermediate | genomic models, 25_7b | efficiency models |
| MMLU | Intermediate | All models | - |
| truthfulness_fixed | Basic | All models | - |

## Resource Usage Guide

### **Qwen Models**
| **Model** | **VRAM** | **Suitable For** | **Not Suitable For** |
|-----------|----------|------------------|---------------------|
| `qwen25_0_5b` | 2.3GB | Quick tests, basic eval | Complex problems |
| `qwen25_3b` | 3.5GB | Efficiency analysis | Expert-level tasks |
| `qwen25_7b` | 5.6GB | General evaluation | - |
| `qwen3_coder_30b` | 18.0GB | Complex coding | Basic problems (overkill) |
| `qwen25_72b_genomic` | 39.2GB | Complex genomics only | General tasks (overkill) |

### **Strategic Non-Qwen Models**
| **Model** | **VRAM** | **Suitable For** | **Strategic Comparison** |
|-----------|----------|------------------|-------------------------|
| `mistral_nemo_12b` | 9.5GB | Long context tasks | vs Qwen 128K performance |
| `granite_3_1_8b` | 6.0GB | Production scenarios | vs Qwen enterprise readiness |
| `olmo2_13b_research` | 10.2GB | Academic benchmarks | vs Qwen research applications |
| `yi_1_5_34b` | 21.5GB | Multilingual tasks | vs Qwen cultural diversity |
| `phi35_mini_efficiency` | 2.8GB | Efficiency testing | vs Qwen small models |

## Quick Commands by Use Case

### **🎯 Validate Specialists (Qwen Framework)**
```bash
# Core validation (3 commands)
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset gsm8k
python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_1_5b_genomic --dataset biomedical_extended
```

### **🔄 Cross-Architecture Validation (NEW)**
```bash
# Strategic architectural comparisons (5 commands)
python evaluation/run_evaluation.py --model mistral_nemo_12b --dataset mmlu
python evaluation/run_evaluation.py --model granite_3_1_8b --dataset humaneval
python evaluation/run_evaluation.py --model olmo2_13b_research --dataset mmlu
python evaluation/run_evaluation.py --model yi_1_5_34b --dataset mmlu
python evaluation/run_evaluation.py --model phi35_mini_efficiency --dataset humaneval
```

### **📊 Baseline Comparison**  
```bash
# General model on specialist tasks (3 commands)
python evaluation/run_evaluation.py --model qwen25_7b --dataset gsm8k
python evaluation/run_evaluation.py --model qwen25_7b --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_7b --dataset mmlu
```

### **⚡ Efficiency Testing**
```bash
# Efficiency vs performance (4 commands - now includes cross-architecture)
python evaluation/run_evaluation.py --model qwen25_3b --dataset humaneval
python evaluation/run_evaluation.py --model qwen25_0_5b --dataset humaneval
python evaluation/run_evaluation.py --model phi35_mini_efficiency --dataset humaneval
python evaluation/run_evaluation.py --model phi35_mini_efficiency --dataset mmlu
```

### **🧪 Advanced Testing**
```bash
# Expert-level evaluation (3 commands)
python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset advanced_coding_extended
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset enhanced_math_fixed
python evaluation/run_evaluation.py --model yi_1_5_34b --dataset advanced_coding_extended
```

**Total Strategic Tests: 18 commands cover comprehensive evaluation with architectural diversity**

## 🏆 **Framework Summary**

- **Total Models**: 17 (12 Qwen + 5 Strategic)
- **Total Datasets**: 14 with 14,860+ samples
- **Architectural Diversity**: 7 organizations, 5 training philosophies
- **Strategic Tests**: 15-20 focused evaluations (not 238 exhaustive combinations)
- **License Coverage**: Apache 2.0, MIT, Custom research
- **Size Range**: 0.5B to 72B parameters
- **Specializations**: Math, Coding, Genomics, Multimodal, Enterprise, Research, Efficiency