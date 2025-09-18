# Biomedical & Genomics Category Analysis

## Overview
This document provides a comprehensive analysis of the biomedical specialist models in our LLM evaluation framework, focusing on optimal model-dataset mappings for genomics, healthcare, bioinformatics, and computational biology applications.

## Model Inventory & Specifications

### 1. BioMistral 7B (AWQ Quantized)
- **Model ID**: `BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM`
- **Size**: 7B parameters (3.88GB memory footprint)
- **Context Window**: 2048 tokens
- **Specialization**: Medical reasoning, clinical applications
- **Training Data**: Biomedical literature, clinical texts
- **Quantization**: AWQ (4-bit weights)
- **Performance**: ~28 tokens/second
- **Strengths**: Memory efficient, good for clinical QA
- **Optimal Datasets**: PubMedQA, MedQA, clinical reasoning tasks

### 2. BioMistral 7B (Unquantized)
- **Model ID**: `BioMistral/BioMistral-7B`
- **Size**: 7B parameters (13.5GB memory footprint)
- **Context Window**: 2048 tokens
- **Specialization**: Medical reasoning, clinical applications
- **Training Data**: Biomedical literature, clinical texts
- **Quantization**: None (full precision)
- **Performance**: ~138 tokens/second
- **Strengths**: Higher accuracy, full model capacity
- **Optimal Datasets**: Complex medical reasoning, clinical decision support

### 3. Stanford BioMedLM 2.7B
- **Model ID**: `stanford-crfm/BioMedLM`
- **Size**: 2.7B parameters
- **Context Window**: 1024 tokens
- **Specialization**: Biomedical text generation and mining
- **Training Data**: PubMed abstracts and papers from The Pile
- **Performance Benchmark**: 50.3% accuracy on MedQA
- **Custom Features**: Biomedical-optimized tokenizer
- **Strengths**: Excellent PubMed domain knowledge, efficient size
- **Optimal Datasets**: PubMedQA, biomedical literature QA, MedQA

### 4. MedAlpaca 7B
- **Model ID**: `medalpaca/medalpaca-7b`
- **Size**: 7B parameters
- **Context Window**: 2048 tokens
- **Specialization**: Medical instruction following
- **Training Data**: 
  - ChatDoctor (200K QA pairs)
  - Wikidoc medical content
  - Anki medical flashcards
  - StackExchange (Academia, Bioinformatics, Biology, Fitness, Health)
- **Base Model**: LLaMA
- **Strengths**: Strong instruction following, diverse medical training
- **Optimal Datasets**: Medical QA, patient education, clinical guidelines

### 5. Microsoft BioGPT
- **Model ID**: `microsoft/biogpt`
- **Size**: ~1.5B parameters
- **Context Window**: 1024 tokens
- **Specialization**: Biomedical text generation
- **Training Data**: Large-scale biomedical literature
- **Performance Benchmarks**:
  - BC5CDR relation extraction: 44.98% F1
  - KD-DTI relation extraction: 38.42% F1
  - DDI relation extraction: 40.76% F1
  - PubMedQA: 78.2% accuracy
- **Strengths**: Strong relation extraction, biomedical generation
- **Optimal Datasets**: Relation extraction tasks, PubMedQA, biomedical NER

### 6. Bio_ClinicalBERT
- **Model ID**: `emilyalsentzer/Bio_ClinicalBERT`
- **Size**: ~110M parameters (1.3GB)
- **Context Window**: 512 tokens
- **Specialization**: Clinical text understanding
- **Training Data**: MIMIC-III electronic health records (~880M words)
- **Base Model**: BioBERT (PubMed + PMC initialization)
- **Architecture**: BERT (encoder-only)
- **Strengths**: Clinical domain expertise, EHR understanding
- **Optimal Datasets**: Clinical NER, MIMIC-III tasks, clinical classification
- **Note**: Primarily for embeddings/classification, not generation

## Dataset Requirements Analysis

### Core Biomedical Datasets Needed

#### 1. PubMedQA
- **Purpose**: Biomedical question answering
- **Content**: PubMed abstract-based questions
- **Size**: ~211K questions
- **Format**: Question-Answer pairs with PubMed abstracts
- **Best Models**: BioMedLM, BioGPT, BioMistral variants
- **Download Status**: ❌ Need to download

#### 2. MedQA (USMLE)
- **Purpose**: Medical knowledge assessment
- **Content**: USMLE-style multiple choice questions
- **Size**: ~12K training, ~1.7K test questions
- **Format**: Multiple choice with explanations
- **Best Models**: BioMedLM (50.3% benchmark), MedAlpaca, BioMistral
- **Download Status**: ❌ Need to download

#### 3. BioASQ
- **Purpose**: Biomedical semantic indexing and QA
- **Content**: PubMed/MEDLINE based questions
- **Size**: Multiple tasks and years
- **Format**: Factoid, list, yes/no, and summary questions
- **Best Models**: All generative models
- **Download Status**: ❌ Need to download

#### 4. MEDIQA
- **Purpose**: Medical question answering and summarization
- **Content**: Consumer health questions
- **Size**: Various subtasks
- **Format**: Question answering and text summarization
- **Best Models**: MedAlpaca, BioMistral variants
- **Download Status**: ❌ Need to download

### Specialized Clinical Datasets

#### 5. MIMIC-III Clinical Notes
- **Purpose**: Clinical text processing
- **Content**: De-identified hospital records
- **Access**: Requires credentialing
- **Best Models**: Bio_ClinicalBERT, BioMistral variants
- **Note**: Access restrictions may apply

#### 6. Clinical NER Datasets
- **Datasets**: i2b2, n2c2 challenges
- **Purpose**: Named entity recognition in clinical text
- **Content**: Clinical notes with entity annotations
- **Best Models**: Bio_ClinicalBERT, BioGPT
- **Download Status**: ❌ Need to research access requirements

### Relation Extraction Datasets

#### 7. BC5CDR (Chemical-Disease Relations)
- **Purpose**: Chemical-disease relation extraction
- **Content**: PubMed abstracts with chemical-disease annotations
- **Best Models**: BioGPT (44.98% F1 benchmark)
- **Download Status**: ❌ Need to download

#### 8. DDI (Drug-Drug Interactions)
- **Purpose**: Drug interaction detection
- **Content**: Biomedical texts with drug interaction annotations
- **Best Models**: BioGPT (40.76% F1 benchmark)
- **Download Status**: ❌ Need to download

### Genomics & Computational Biology Datasets

#### 9. GenomeQA
- **Purpose**: Genomics question answering
- **Content**: Genetics and genomics questions
- **Size**: Research required
- **Best Models**: Models with genomics training (need to research)
- **Priority**: HIGH (user's primary interest)

#### 10. Protein Function Prediction
- **Datasets**: UniProt, GO annotations
- **Purpose**: Protein sequence and function analysis
- **Content**: Protein sequences with functional annotations
- **Best Models**: Need specialized protein language models
- **Priority**: HIGH (computational biology focus)

## Model-Dataset Optimization Matrix

### High Performance Combinations

| Model | Primary Datasets | Secondary Datasets | Performance Focus |
|-------|-----------------|-------------------|-------------------|
| BioMedLM 2.7B | MedQA, PubMedQA | BioASQ | Medical knowledge, PubMed expertise |
| BioGPT | BC5CDR, DDI, PubMedQA | Relation extraction | Text generation, relations |
| MedAlpaca 7B | MEDIQA, MedQA | Clinical guidelines | Instruction following |
| BioMistral (AWQ) | PubMedQA, Clinical QA | MedQA | Memory-efficient clinical reasoning |
| BioMistral (Full) | Complex MedQA, Clinical | Advanced reasoning | Full-capacity medical analysis |
| Bio_ClinicalBERT | Clinical NER, MIMIC | i2b2, n2c2 | Clinical text understanding |

### Specialized Genomics Evaluation

#### Current Gap Analysis
- **Genomics Models**: Limited in current inventory
- **Genomics Datasets**: Need comprehensive research
- **Computational Biology**: Requires specialized models

#### Recommended Additions
1. **DNA/RNA Language Models**: Research DNABERT, GenomeGPT variants
2. **Protein Language Models**: ESM-2, ProtBERT, ChemBERTa
3. **Bioinformatics LLMs**: Models trained on biological sequence data

## Implementation Strategy

### Phase 1: Core Dataset Integration
1. Download and validate PubMedQA, MedQA, BioASQ
2. Test current biomedical models on standard datasets
3. Establish baseline performance metrics

### Phase 2: Specialized Dataset Integration
1. Research access requirements for clinical datasets
2. Download relation extraction datasets (BC5CDR, DDI)
3. Implement specialized evaluation metrics

### Phase 3: Genomics Expansion
1. Research and identify genomics-specific models
2. Download genomics and computational biology datasets
3. Develop genomics evaluation framework

### Phase 4: Model-Dataset Sub-mapping
1. Implement fine-grained model-dataset assignments
2. Optimize evaluation strategies per model type
3. Create specialized evaluation pipelines

## Next Actions

### Immediate Priorities
1. ✅ Document current model inventory
2. 🔄 Research and download core biomedical datasets
3. ❌ Test model-dataset compatibility
4. ❌ Implement sub-mapping system

### Research Priorities (Genomics Focus)
1. ❌ Identify genomics-specific language models
2. ❌ Research DNA/protein sequence datasets
3. ❌ Explore computational biology evaluation frameworks
4. ❌ Investigate bioinformatics-specific benchmarks

## Dataset Download Commands & Access Information

### Immediately Available Datasets

#### 1. PubMedQA
```bash
# Direct GitHub download
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/pubmedqa_train_set.json
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/pubmedqa_dev_set.json
wget https://github.com/pubmedqa/pubmedqa/raw/master/data/pubmedqa_test_set.json

# Alternative: Python download
python -c "from datasets import load_dataset; load_dataset('pubmed_qa', 'pqa_labeled')"
```
- **Size**: 1K expert labeled + 61.2K unlabeled + 211.3K artificially generated
- **Format**: Yes/No/Maybe questions with PubMed abstracts
- **Leaderboard Performance**: GPT-4 (Medprompt): 82.0% accuracy
- **Best Models**: BioMedLM (50.3% benchmark), BioGPT, BioMistral variants

#### 2. MedQA (USMLE-style)
```bash
# HuggingFace datasets
python -c "from datasets import load_dataset; load_dataset('openlifescienceai/medqa')"
```
- **Size**: 12,723 questions total
- **Format**: Multiple choice medical questions with explanations  
- **Source**: USMLE-style questions
- **Best Models**: BioMedLM (proven 50.3% performance), MedAlpaca, BioMistral

### Registration-Required Datasets

#### 3. BioASQ Challenge Data
- **Access**: http://participants-area.bioasq.org/ (registration required)
- **Tasks Available**:
  - Task B: Biomedical Semantic QA (IR, QA, Summarization)
  - Task Synergy: QA for developing biomedical issues
  - Task MultiClinSum: Multilingual clinical summarization
  - Task BioNNE-L: Nested named entity linking
- **Format**: Questions, relevant articles, snippets, exact answers, ideal answers
- **Evaluation**: Both automatic (GMAP, MRR, ROUGE) and manual expert evaluation
- **Best Models**: All generative models, particularly BioGPT and BioMedLM

#### 4. Clinical Datasets (Restricted Access)
```bash
# MIMIC-III (requires credentialing)
# Access: https://physionet.org/content/mimiciii/
# Note: Requires completion of CITI training and data use agreement

# i2b2/n2c2 Clinical NER Challenges
# Access: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
# Note: Registration and approval required
```

### Relation Extraction Datasets

#### 5. BC5CDR (Chemical-Disease Relations)
```bash
# PubTator Central access
wget ftp://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/BC5CDR_corpus.zip
```
- **Performance Benchmark**: BioGPT achieves 44.98% F1 score
- **Content**: PubMed abstracts with chemical-disease relation annotations
- **Best Models**: BioGPT (proven performance), Bio_ClinicalBERT

#### 6. DDI (Drug-Drug Interactions)
```bash
# DDI Extraction 2013 Challenge Data
wget https://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/datasets/ddi-corpus-v1.0.zip
```
- **Performance Benchmark**: BioGPT achieves 40.76% F1 score
- **Content**: Drug interaction detection from biomedical texts
- **Best Models**: BioGPT, potentially MedAlpaca

## License and Access Considerations

### Open Access Datasets
- PubMedQA: Open access
- BioASQ: Registration required
- BC5CDR: Open access through PubTator

### Restricted Access Datasets
- MIMIC-III: Requires credentialing and training
- Clinical datasets: May require IRB approval
- Some medical datasets: Healthcare data restrictions

### Model Licensing
- Most models: Open source with attribution
- MedGemma: Health AI Developer Foundation terms
- Clinical models: May have usage restrictions

---

*Document created: September 18, 2025*  
*Purpose: Comprehensive biomedical/genomics category analysis for LLM evaluation framework*  
*Focus: Genomics, healthcare, bioinformatics, computational biology applications*