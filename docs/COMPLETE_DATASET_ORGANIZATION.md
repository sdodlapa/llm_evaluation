# Complete Dataset Organization & Structure

**LLM Evaluation System Dataset Repository**  
**Total Categories:** 15 major categories + subdirectories  
**Last Updated:** September 22, 2025  

## 📊 **Dataset Organization Overview**

Your evaluation system contains a comprehensive collection of datasets organized in a hybrid structure that supports both categorical evaluation and legacy formats. The organization follows both domain-specific categorization and evaluation-ready formats.

---

## 🗂️ **Primary Dataset Categories**

### 1. **📚 General Knowledge** (7 datasets)
**Location:** `evaluation_data/general_knowledge/`
```
├── arc_challenge          # AI2 Reasoning Challenge (MCQ)
├── boolq                  # Boolean Questions (reading comprehension)
├── hellaswag              # Common sense reasoning
├── mmlu                   # Massive Multitask Language Understanding (57 subjects)
├── piqa                   # Physical Interaction QA
├── truthfulqa_mc          # TruthfulQA Multiple Choice format
└── winogrande             # Winograd Schema Challenge
```

### 2. **🧮 Mathematical Reasoning** (2 datasets)  
**Location:** `evaluation_data/mathematical_reasoning/`
```
├── gsm8k                  # Grade School Math 8K problems
└── math_competition       # MATH Competition problems (advanced)
```

### 3. **📐 Mathematics (Specialized)** (3 datasets)
**Location:** `evaluation_data/mathematics/`
```
├── aime                   # American Invitational Mathematics Examination
├── math_competition       # MATH Competition dataset
└── mathqa                 # Mathematical Question Answering
```

### 4. **🧠 Reasoning Specialized** (2 datasets)
**Location:** `evaluation_data/reasoning_specialized/`
```
├── bigbench_hard          # BIG-bench Hard tasks (challenging reasoning)
└── gpqa                   # Graduate-level Google-Proof Q&A (STEM)
```

### 5. **💻 Coding Specialists** (5 datasets)
**Location:** `evaluation_data/coding_specialists/`
```
├── apps                   # Automated Programming Progress Standard
├── bigcodebench          # BigCodeBench comprehensive evaluation
├── code_contests         # CodeContests programming competitions
├── humaneval             # HumanEval Python code generation
└── mbpp                  # Mostly Basic Python Problems
```

### 6. **🧬 Biomedical** (8 datasets)
**Location:** `evaluation_data/biomedical/`
```
├── bc5cdr                # BioCreative V CDR task
├── ddi                   # Drug-Drug Interaction extraction
├── medqa                 # Medical Question Answering
├── bioasq.json           # BioASQ challenge questions
├── mediqa.json           # Medical Question Answering dataset
├── pubmedqa_full.json    # PubMedQA full dataset
├── pubmedqa.json         # PubMedQA standard
└── pubmedqa_sample.json  # PubMedQA sample subset
```

### 7. **🖼️ Multimodal Processing** (5 datasets)
**Location:** `evaluation_data/multimodal_processing/`
```
├── chartqa               # Chart Question Answering
├── docvqa                # Document Visual QA
├── mmmu                  # Massive Multi-discipline Multimodal Understanding
├── scienceqa             # Science Question Answering with diagrams
└── textcaps              # Text understanding in images
```

### 8. **🗺️ Text Geospatial** (9 datasets)
**Location:** `evaluation_data/text_geospatial/`
```
├── address_parsing       # Address parsing and standardization
├── coordinate_processing # Geographic coordinate processing
├── geographic_demand     # Geographic demand analysis
├── geographic_features   # Geographic feature recognition
├── location_ner          # Location Named Entity Recognition
├── ner_locations         # NER for geographic locations
├── spatial_reasoning     # Spatial relationship reasoning
├── wikidata_geographic   # Wikidata geographic entities
└── download_summary.json # Geospatial download metadata
```

### 9. **🔧 Function Calling** (1 dataset)
**Location:** `evaluation_data/function_calling/`
```
└── bfcl                  # Berkeley Function-Calling Leaderboard
```

### 10. **🔬 Scientific** (2 datasets)
**Location:** `evaluation_data/scientific/`
```
├── scientific_papers.json # Scientific paper analysis
└── scierc.json            # Scientific entity and relation corpus
```

### 11. **🛡️ Safety** (1 dataset)
**Location:** `evaluation_data/safety/`
```
└── toxicity_detection.json # Toxicity and harmful content detection
```

---

## 📋 **Legacy & Alternative Organization**

### **⚙️ Coding (Legacy)** 
**Location:** `evaluation_data/coding/`
- Contains JSON format versions of coding datasets
- Includes metadata and configuration files
- Maintains backward compatibility

### **📋 Instruction Following**
**Location:** `evaluation_data/instruction_following/`
```
└── mt_bench.json         # MT-Bench conversation evaluation
```

### **❓ QA (General)**
**Location:** `evaluation_data/qa/`
```
├── biomedical_extended.json
├── biomedical_sample.json
├── mmlu.json
├── multimodal_sample.json
└── truthfulness_fixed.json
```

### **🧮 Reasoning (General)**
**Location:** `evaluation_data/reasoning/`
```
├── advanced_math_sample.json
├── arc_challenge.json
├── enhanced_math_fixed.json
├── gsm8k.json
└── hellaswag.json
```

---

## 📁 **Standalone Dataset Directories**

### **Individual Dataset Folders**
```
├── ai2d/                 # AI2 Diagram understanding (standalone)
├── chartqa/              # Chart QA (legacy location)
├── scienceqa/            # Science QA (legacy location)
├── textvqa/              # Text VQA (standalone)
└── general/              # General purpose datasets
```

---

## 🗄️ **Structured Datasets Collection**

**Location:** `evaluation_data/datasets/`

This provides an organized view by domain:

### **By Domain Structure:**
```
datasets/
├── biomedical/           # All biomedical datasets in JSON format
├── coding/               # All coding datasets in JSON format
├── efficiency/           # Efficiency-focused dataset subset
├── general/              # General-purpose evaluation datasets
├── geospatial/          # Geospatial and geographic datasets
├── math/                # Mathematical reasoning datasets
├── multimodal/          # Multimodal datasets in JSON format
├── safety/              # Safety and alignment datasets
└── scientific/          # Scientific research datasets
```

---

## 📊 **Metadata & Registry System**

### **Metadata Storage**
**Location:** `evaluation_data/meta/`
- Contains metadata files for all major datasets
- Includes configuration and preprocessing information
- Tracks dataset versions and updates

### **Registry Files**
**Location:** `evaluation_data/registry/`
- Central registry for dataset availability
- Compatibility matrices
- Evaluation configuration templates

---

## 📈 **Dataset Statistics & Summary**

### **By Category Count:**
- **General Knowledge:** 7 datasets
- **Mathematical/Reasoning:** 7 datasets (across categories)
- **Coding:** 10+ datasets (multiple formats)
- **Biomedical:** 8 datasets
- **Multimodal:** 5+ datasets
- **Geospatial:** 9 datasets
- **Scientific:** 2 datasets
- **Safety:** 1 dataset
- **Function Calling:** 1 dataset

### **Total Dataset Inventory:**
- **Primary Categories:** 15 major categories
- **Dataset Files:** 100+ individual dataset files
- **JSON Configurations:** 60+ configuration files
- **Metadata Files:** 20+ metadata files
- **Format Support:** HuggingFace datasets, JSON, custom formats

### **Storage Organization:**
- **Categorical Structure:** Domain-specific organization for evaluation
- **Legacy Support:** Backward compatibility with existing scripts
- **Dual Format:** Both HuggingFace datasets and JSON formats
- **Metadata Rich:** Comprehensive metadata and configuration tracking

---

## 🚀 **Dataset Access Patterns**

### **For Evaluation Scripts:**
1. **Primary Access:** Use categorical structure (`general_knowledge/`, `coding_specialists/`, etc.)
2. **Legacy Access:** Use JSON files in domain folders (`datasets/coding/`, `qa/`, etc.)
3. **Metadata Access:** Use `meta/` for dataset information and configuration

### **For New Integrations:**
1. **Standard Datasets:** Access via categorical structure
2. **Custom Datasets:** Add to appropriate domain category
3. **Evaluation Configuration:** Update metadata and registry

### **For Research & Analysis:**
1. **Cross-Domain Studies:** Use `datasets/` organized view
2. **Domain-Specific:** Use categorical structure
3. **Performance Analysis:** Use metadata for dataset characteristics

---

## 🔧 **Maintenance & Updates**

### **Recent Additions:**
- Enhanced coding datasets with repository-level tasks
- Advanced multimodal datasets for large vision-language models
- Specialized reasoning datasets for complex problem solving
- Function calling datasets for tool-use evaluation

### **Organizational Benefits:**
- **Scalable Structure:** Easy to add new datasets and categories
- **Multiple Access Patterns:** Supports different evaluation workflows
- **Rich Metadata:** Comprehensive tracking and configuration
- **Format Flexibility:** Supports multiple dataset formats and evaluation harnesses

This organization provides comprehensive coverage for evaluating language models across all major domains while maintaining flexibility for different evaluation approaches and backward compatibility with existing systems.