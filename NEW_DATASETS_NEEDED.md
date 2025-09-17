# New Datasets Needed for Specialized Qwen Models

Based on the new specialized Qwen models we added, here are the **new datasets we need to download** to fully utilize their capabilities:

## 📊 **Current Dataset Status**

### ✅ **Already Available** (Downloaded)
- **Coding**: HumanEval (164 samples), MBPP (974 samples)
- **Reasoning**: GSM8K (1,319 samples), ARC-Challenge (1,172 samples), HellaSwag (10,042 samples)
- **Instruction**: MT-Bench (160 samples)

### ❌ **Missing Datasets** (Empty folders)
- **QA**: MMLU, TruthfulQA (folders exist but empty)
- **Function Calling**: BFCL, ToolLLaMA (folders exist but empty)

---

## 🆕 **NEW DATASETS NEEDED FOR SPECIALIZED MODELS**

### 🔢 **Mathematics Specialist** (`qwen25_math_7b`)

#### **High Priority - Advanced Math**
1. **MATH Dataset** ⭐⭐⭐
   - **Description**: Competition-level mathematics problems
   - **Samples**: 12,500 problems across algebra, geometry, calculus
   - **Source**: `hendrycks/competition_math`
   - **Size**: ~25MB
   - **Why needed**: GSM8K is too basic for qwen25_math_7b specialist

2. **MathQA** ⭐⭐
   - **Description**: Mathematical reasoning with multiple choice
   - **Samples**: 37,000 problems
   - **Source**: `math_qa`
   - **Size**: ~15MB
   - **Why needed**: Tests mathematical reasoning chains

#### **Medium Priority**
3. **TheoremQA** ⭐⭐
   - **Description**: STEM theorem application
   - **Samples**: 800 problems
   - **Source**: `TIGER-Lab/TheoremQA`
   - **Size**: ~5MB

### 🧬 **Genomic Specialists** (`qwen25_1_5b_genomic`, `qwen25_72b_genomic`)

#### **High Priority - Biomedical**
4. **PubMedQA** ⭐⭐⭐
   - **Description**: Biomedical question answering
   - **Samples**: 1,000 expert-annotated questions
   - **Source**: `pubmed_qa`
   - **Size**: ~10MB
   - **Why needed**: Tests genomic/medical knowledge

5. **MedQA** ⭐⭐⭐
   - **Description**: Medical exam questions (USMLE style)
   - **Samples**: 12,723 questions
   - **Source**: `bigbio/med_qa`
   - **Size**: ~20MB
   - **Why needed**: Medical reasoning for genomic context

#### **Custom Genomic Datasets** (Need to create)
6. **DNA Sequence Analysis** ⭐⭐⭐
   - **Description**: DNA/RNA sequence interpretation tasks
   - **Samples**: 500 sequences with analysis questions
   - **Custom dataset**: Protein folding, gene expression analysis
   - **Size**: ~5MB

7. **Genomic Variant Interpretation** ⭐⭐
   - **Description**: SNP/variant effect prediction
   - **Samples**: 300 variant interpretation tasks
   - **Custom dataset**: Clinical genomics scenarios
   - **Size**: ~3MB

### 👁️ **Multimodal Model** (`qwen2_vl_7b`)

#### **High Priority - Vision-Language**
8. **VQAv2** ⭐⭐⭐
   - **Description**: Visual question answering
   - **Samples**: 1,105 validation questions
   - **Source**: `HuggingFace VQAv2`
   - **Size**: ~50MB (images + text)
   - **Why needed**: Tests vision-language capabilities

9. **ScienceQA** ⭐⭐⭐
   - **Description**: Science questions with diagrams/charts
   - **Samples**: 21,208 multimodal questions
   - **Source**: `derek-thomas/ScienceQA`
   - **Size**: ~100MB
   - **Why needed**: Scientific chart/diagram interpretation

#### **Medium Priority**
10. **ChartQA** ⭐⭐
    - **Description**: Chart and graph understanding
    - **Samples**: 9,608 questions
    - **Source**: `HuggingFace ChartQA`
    - **Size**: ~30MB

### 💻 **Advanced Coding** (`qwen3_coder_30b`)

#### **High Priority - Complex Coding**
11. **CodeContests** ⭐⭐⭐
    - **Description**: Programming competition problems
    - **Samples**: 13,328 problems
    - **Source**: `deepmind/code_contests`
    - **Size**: ~50MB
    - **Why needed**: Tests advanced algorithmic thinking

12. **APPS Dataset** ⭐⭐⭐
    - **Description**: Python programming problems
    - **Samples**: 10,000 problems (difficulty levels)
    - **Source**: `codeparrot/apps`
    - **Size**: ~40MB
    - **Why needed**: More complex than HumanEval/MBPP

#### **Medium Priority**
13. **CodeXGLUE** ⭐⭐
    - **Description**: Code understanding tasks
    - **Samples**: Various subtasks
    - **Source**: `microsoft/CodeXGLUE`
    - **Size**: ~100MB

### 🎯 **General Enhancement**

#### **Missing Core Datasets**
14. **MMLU** ⭐⭐⭐ (Currently missing)
    - **Description**: Multitask language understanding
    - **Samples**: 14,042 questions across 57 subjects
    - **Source**: `cais/mmlu`
    - **Size**: ~200MB
    - **Why needed**: Standard benchmark for all models

15. **BFCL** ⭐⭐⭐ (Currently missing)
    - **Description**: Function calling benchmark
    - **Samples**: 2,000 function calling scenarios
    - **Source**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
    - **Size**: ~50MB
    - **Why needed**: Agent capabilities testing

---

## 🎯 **RECOMMENDED DOWNLOAD PRIORITY**

### **Phase 1: Immediate (High Priority)**
```bash
# Mathematics for qwen25_math_7b
1. MATH Dataset (competition math) - 25MB
2. MathQA (mathematical reasoning) - 15MB

# Biomedical for genomic models  
3. PubMedQA (biomedical QA) - 10MB
4. MedQA (medical reasoning) - 20MB

# Vision-Language for qwen2_vl_7b
5. VQAv2 (visual QA) - 50MB
6. ScienceQA (science + diagrams) - 100MB

# Advanced coding for qwen3_coder_30b
7. CodeContests (programming contests) - 50MB
8. APPS (advanced Python problems) - 40MB

# Missing core datasets
9. MMLU (multitask understanding) - 200MB
10. BFCL (function calling) - 50MB
```
**Total Phase 1: ~560MB**

### **Phase 2: Enhancement (Medium Priority)**
```bash
# Additional specialized datasets
11. TheoremQA (STEM theorems) - 5MB
12. ChartQA (chart understanding) - 30MB  
13. CodeXGLUE (code understanding) - 100MB
14. TruthfulQA (truthfulness) - 5MB
```
**Total Phase 2: ~140MB**

### **Phase 3: Custom Datasets (Create Later)**
```bash
# Custom genomic datasets (need to create)
15. DNA Sequence Analysis - 5MB
16. Genomic Variant Interpretation - 3MB
```
**Total Phase 3: ~8MB**

---

## 📋 **DOWNLOAD COMMANDS**

### **Automated Download Script**
```python
# Add to manage_datasets.py or create new script
priority_datasets = [
    "competition_math",        # MATH dataset
    "math_qa",                # MathQA
    "pubmed_qa",              # PubMedQA  
    "bigbio/med_qa",          # MedQA
    "HuggingFace VQAv2",      # VQAv2
    "derek-thomas/ScienceQA", # ScienceQA
    "deepmind/code_contests", # CodeContests
    "codeparrot/apps",        # APPS
    "cais/mmlu",              # MMLU
    "gorilla-llm/Berkeley-Function-Calling-Leaderboard"  # BFCL
]
```

### **Manual HuggingFace Downloads**
```bash
# In Python environment
from datasets import load_dataset

# Mathematics
math_dataset = load_dataset("hendrycks/competition_math")
mathqa = load_dataset("math_qa") 

# Biomedical
pubmedqa = load_dataset("pubmed_qa")
medqa = load_dataset("bigbio/med_qa")

# Vision-Language
vqav2 = load_dataset("HuggingFace/VQAv2")
scienceqa = load_dataset("derek-thomas/ScienceQA")

# Advanced Coding  
codecontests = load_dataset("deepmind/code_contests")
apps = load_dataset("codeparrot/apps")

# Core Missing
mmlu = load_dataset("cais/mmlu")
bfcl = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")
```

---

## 🎯 **EXPECTED IMPACT**

### **Model-Dataset Alignment**
- **qwen25_math_7b**: MATH + MathQA → Advanced mathematical reasoning
- **qwen25_*_genomic**: PubMedQA + MedQA → Biomedical knowledge
- **qwen2_vl_7b**: VQAv2 + ScienceQA → Multimodal understanding  
- **qwen3_coder_30b**: CodeContests + APPS → Advanced programming
- **All models**: MMLU + BFCL → Comprehensive evaluation

### **Evaluation Enhancement**
- **From**: 6 datasets → **To**: 16+ datasets
- **Coverage**: Basic tasks → **Specialized expert-level tasks**
- **Model utilization**: 60% → **95% of specialized capabilities**

**Total Additional Storage**: ~708MB for comprehensive specialized evaluation