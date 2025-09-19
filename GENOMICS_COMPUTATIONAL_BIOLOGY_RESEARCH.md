# 🧬 Genomics & Computational Biology Models Research

## Specialized DNA/Protein Language Models

### 1. **DNABERT-2** 
- **Model**: `zhihan1996/DNABERT-2-117M`
- **Focus**: DNA sequence understanding, genomic analysis
- **Training**: 1B DNA sequences, optimized for biological relevance  
- **Capabilities**: Gene prediction, variant analysis, sequence classification
- **Context**: Up to 1024 nucleotides
- **Use Cases**: SNP analysis, regulatory element detection, pathogenicity prediction

### 2. **ProteinBERT**
- **Model**: `Rostlab/prot_bert_bfd`
- **Focus**: Protein sequence analysis, structure prediction
- **Training**: UniRef100 database, 2.1B protein sequences
- **Capabilities**: Secondary structure, solubility, membrane prediction
- **Context**: Up to 1024 amino acids
- **Use Cases**: Drug discovery, protein engineering, function prediction

### 3. **Nucleotide Transformer**
- **Model**: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
- **Focus**: Multi-species genomic sequences
- **Training**: 850+ species, 3.2B nucleotide sequences
- **Capabilities**: Cross-species analysis, evolutionary patterns
- **Context**: Up to 6000 nucleotides
- **Use Cases**: Comparative genomics, conservation analysis

### 4. **ESM-2** (Evolutionary Scale Modeling)
- **Model**: `facebook/esm2_t33_650M_UR50D`
- **Focus**: Protein language modeling
- **Training**: UniRef50 database, evolutionary relationships
- **Capabilities**: Contact prediction, fold classification, functional annotation
- **Context**: Up to 1024 residues
- **Use Cases**: Structure prediction, homology detection, enzyme classification

### 5. **GenSLMs** (Genome-Scale Language Models)
- **Model**: `ORNL/GenSLMs` (Argonne National Lab)
- **Focus**: Large-scale genomic analysis
- **Training**: 280B+ nucleotides across prokaryotes
- **Capabilities**: Gene finding, antimicrobial resistance prediction
- **Context**: Variable length sequences
- **Use Cases**: Microbiome analysis, pathogen detection

### 6. **HyenaDNA**
- **Model**: `LongSafari/hyenadna-medium-160k-seqlen`
- **Focus**: Long genomic sequences
- **Training**: Human reference genome, 160k context
- **Capabilities**: Long-range interactions, regulatory analysis
- **Context**: Up to 160,000 nucleotides
- **Use Cases**: Chromosome-scale analysis, structural variant detection

### 7. **CaLM** (Codon and Amino acid Language Model)
- **Model**: `tattabio/CaLM`
- **Focus**: Codon optimization, translation efficiency
- **Training**: Codon-aware protein sequences
- **Capabilities**: Translation optimization, expression prediction
- **Context**: Paired DNA/protein sequences
- **Use Cases**: Synthetic biology, protein expression optimization

## Specialized Datasets for Genomics Evaluation

### Sequence Analysis Datasets
1. **Human Genome Variants**: 1000 Genomes Project SNPs
2. **Protein Function**: UniProt functional annotations
3. **Gene Expression**: GTEx tissue-specific expression
4. **Regulatory Elements**: ENCODE ChIP-seq peaks
5. **Disease Variants**: ClinVar pathogenic mutations

### Benchmark Tasks
1. **Variant Effect Prediction**: Predict pathogenicity of mutations
2. **Gene Expression Prediction**: Tissue-specific expression levels
3. **Protein Folding**: Secondary/tertiary structure prediction
4. **Regulatory Prediction**: Promoter/enhancer identification
5. **Drug-Target Interaction**: Molecular binding prediction

## Integration Strategy

### Model Registry Updates
```python
# Add to configs/model_registry.py

"dnabert2_117m": ModelConfig(
    model_name="DNABERT-2 117M",
    huggingface_id="zhihan1996/DNABERT-2-117M",
    specialization_category="genomics",
    specialization_subcategory="dna_analysis",
    context_window=1024,
    size_gb=0.5,
    license="MIT"
),

"proteinbert_bfd": ModelConfig(
    model_name="ProteinBERT BFD",
    huggingface_id="Rostlab/prot_bert_bfd", 
    specialization_category="genomics",
    specialization_subcategory="protein_analysis",
    context_window=1024,
    size_gb=1.7,
    license="MIT"
),

"hyenadna_160k": ModelConfig(
    model_name="HyenaDNA 160k",
    huggingface_id="LongSafari/hyenadna-medium-160k-seqlen",
    specialization_category="genomics", 
    specialization_subcategory="long_genomic_sequences",
    context_window=160000,
    size_gb=0.8,
    license="Apache-2.0"
)
```

### Genomics Category Definition
```python
# Add to evaluation/mappings/model_categories.py

GENOMICS_COMPUTATIONAL_BIOLOGY = ModelCategory(
    name="genomics_computational_biology",
    description="Specialized models for DNA/RNA/protein sequence analysis",
    models=[
        "dnabert2_117m", "proteinbert_bfd", "hyenadna_160k",
        "nucleotide_transformer", "esm2_650m", "genslms", "calm"
    ],
    primary_datasets=["genomic_variants", "protein_function", "regulatory_elements"],
    evaluation_focus=["sequence_classification", "variant_prediction", "function_annotation"]
)
```

## Research Priorities for Your Work

### High-Priority Models (Immediate Integration)
1. **DNABERT-2**: Essential for genomic variant analysis
2. **ProteinBERT**: Critical for protein function prediction  
3. **HyenaDNA**: Unique long-context genomic analysis

### Medium-Priority Models (Future Expansion)
1. **ESM-2**: Advanced protein modeling
2. **Nucleotide Transformer**: Multi-species comparisons
3. **GenSLMs**: Microbiome/pathogen analysis

### Research Applications Alignment
- **Variant Effect Prediction**: DNABERT-2 + clinical variant datasets
- **Protein Engineering**: ProteinBERT + functional annotation tasks
- **Regulatory Analysis**: HyenaDNA + ENCODE regulatory datasets
- **Drug Discovery**: ESM-2 + protein-drug interaction datasets

## Next Implementation Steps

1. **Add 3-5 top genomics models** to model registry
2. **Create genomics datasets** (variant analysis, protein function)
3. **Build genomics evaluation category** with specialized metrics
4. **Test genomics models** on computational biology tasks
5. **Create genomics documentation** and usage guides

This positions your evaluation framework as the most comprehensive platform for genomics/computational biology model assessment!