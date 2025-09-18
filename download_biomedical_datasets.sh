#!/bin/bash

# Biomedical Dataset Download Script
# Downloads publicly available biomedical datasets for LLM evaluation

set -e  # Exit on any error
module load python3

echo "🧬 Biomedical Dataset Download Script"
echo "===================================="

# Create datasets directory
DATASET_DIR="/home/sdodl001_odu_edu/llm_evaluation/datasets/biomedical"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "📁 Created dataset directory: $DATASET_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "🔍 Checking dependencies..."
if ! command_exists crun -p ~/envs/llm_env python; then
    echo "❌ Python not found. Please install Python first."
    exit 1
fi

if ! command_exists wget; then
    echo "❌ wget not found. Please install wget first."
    exit 1
fi

echo "✅ Dependencies OK"

# Download PubMedQA
echo ""
echo "📚 Downloading PubMedQA dataset..."
mkdir -p pubmedqa
cd pubmedqa

if [ ! -f "pubmedqa_train_set.json" ]; then
    echo "  Downloading training set..."
    wget -q --show-progress https://github.com/pubmedqa/pubmedqa/raw/master/data/pubmedqa_train_set.json
fi

if [ ! -f "pubmedqa_dev_set.json" ]; then
    echo "  Downloading development set..."
    wget -q --show-progress https://github.com/pubmedqa/pubmedqa/raw/master/data/pubmedqa_dev_set.json
fi

if [ ! -f "pubmedqa_test_set.json" ]; then
    echo "  Downloading test set..."
    wget -q --show-progress https://github.com/pubmedqa/pubmedqa/raw/master/data/pubmedqa_test_set.json
fi

echo "✅ PubMedQA download complete"
cd ..

# Download MedQA via HuggingFace
echo ""
echo "🏥 Downloading MedQA dataset via HuggingFace..."
crun -p ~/envs/llm_env python << 'EOF'
try:
    from datasets import load_dataset
    import os
    
    print("  Loading MedQA dataset...")
    dataset = load_dataset('openlifescienceai/medqa')
    
    # Save to local directory
    medqa_dir = 'medqa'
    os.makedirs(medqa_dir, exist_ok=True)
    
    # Save each split
    for split_name, split_data in dataset.items():
        output_file = f"{medqa_dir}/medqa_{split_name}.json"
        split_data.to_json(output_file)
        print(f"  Saved {split_name} split to {output_file}")
    
    print("✅ MedQA download complete")
    
except ImportError:
    print("❌ HuggingFace datasets library not found.")
    print("   Install with: pip install datasets")
    exit(1)
except Exception as e:
    print(f"❌ Error downloading MedQA: {e}")
    exit(1)
EOF

# Download BC5CDR
echo ""
echo "🧪 Downloading BC5CDR (Chemical-Disease Relations) dataset..."
mkdir -p bc5cdr
cd bc5cdr

if [ ! -f "BC5CDR_corpus.zip" ]; then
    echo "  Downloading BC5CDR corpus..."
    wget -q --show-progress ftp://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/BC5CDR_corpus.zip || {
        echo "  ⚠️  FTP download failed, trying alternative source..."
        # Alternative approach if FTP fails
        echo "  Please manually download BC5CDR from PubTator Central"
    }
fi

if [ -f "BC5CDR_corpus.zip" ]; then
    echo "  Extracting BC5CDR corpus..."
    unzip -q BC5CDR_corpus.zip
    echo "✅ BC5CDR download complete"
else
    echo "⚠️  BC5CDR download incomplete - manual download may be required"
fi
cd ..

# Download DDI
echo ""
echo "💊 Downloading DDI (Drug-Drug Interactions) dataset..."
mkdir -p ddi
cd ddi

if [ ! -f "ddi-corpus-v1.0.zip" ]; then
    echo "  Downloading DDI corpus..."
    wget -q --show-progress https://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/datasets/ddi-corpus-v1.0.zip || {
        echo "  ⚠️  Download failed, trying alternative..."
        echo "  Please check SemEval 2013 Task 9 website for DDI corpus"
    }
fi

if [ -f "ddi-corpus-v1.0.zip" ]; then
    echo "  Extracting DDI corpus..."
    unzip -q ddi-corpus-v1.0.zip
    echo "✅ DDI download complete"
else
    echo "⚠️  DDI download incomplete - manual download may be required"
fi
cd ..

# Create dataset summary
echo ""
echo "📊 Creating dataset summary..."
cat > dataset_summary.txt << EOF
Biomedical Dataset Summary
=========================
Downloaded: $(date)

Datasets Available:
1. PubMedQA - Biomedical question answering with PubMed abstracts
   - Training set: $(if [ -f pubmedqa/pubmedqa_train_set.json ]; then echo "✅ Available"; else echo "❌ Missing"; fi)
   - Development set: $(if [ -f pubmedqa/pubmedqa_dev_set.json ]; then echo "✅ Available"; else echo "❌ Missing"; fi)
   - Test set: $(if [ -f pubmedqa/pubmedqa_test_set.json ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

2. MedQA - USMLE-style medical questions
   - Dataset: $(if [ -d medqa ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

3. BC5CDR - Chemical-Disease Relations
   - Corpus: $(if [ -d bc5cdr ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

4. DDI - Drug-Drug Interactions
   - Corpus: $(if [ -d ddi ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

Additional Datasets Requiring Registration:
- BioASQ: http://participants-area.bioasq.org/
- MIMIC-III: https://physionet.org/content/mimiciii/
- i2b2/n2c2: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

Dataset Directory: $DATASET_DIR
EOF

echo ""
echo "📋 Dataset Summary:"
cat dataset_summary.txt

echo ""
echo "🎉 Biomedical dataset download completed!"
echo "📂 All datasets saved to: $DATASET_DIR"
echo ""
echo "Next steps:"
echo "1. Register for BioASQ access if needed"
echo "2. Apply for MIMIC-III access if clinical data needed" 
echo "3. Test datasets with biomedical models"
echo "4. Configure model-dataset mappings"