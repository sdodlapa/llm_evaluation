#!/usr/bin/env python3
"""
Download new specialized datasets for enhanced Qwen models.
Prioritizes datasets that match our new specialized model capabilities.
"""

import sys
import os
from pathlib import Path
import json
from datasets import load_dataset
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_mathematics_datasets():
    """Download datasets for qwen25_math_7b specialist."""
    print("\n🔢 DOWNLOADING MATHEMATICS DATASETS")
    print("="*50)
    
    datasets_info = {}
    
    try:
        # 1. MATH Dataset (Competition Mathematics)
        print("\n📚 Downloading MATH dataset (competition mathematics)...")
        math_dataset = load_dataset("hendrycks/competition_math", "main")
        
        # Save test split for evaluation
        test_data = []
        for item in math_dataset['test']:
            test_data.append({
                'problem': item['problem'],
                'solution': item['solution'],
                'level': item['level'],
                'type': item['type']
            })
        
        math_path = Path("evaluation_data/reasoning/math_competition.json")
        with open(math_path, 'w') as f:
            json.dump(test_data[:500], f, indent=2)  # Limit to 500 for now
        
        datasets_info['math_competition'] = {
            'name': 'MATH Competition',
            'samples': len(test_data[:500]),
            'description': 'Competition-level mathematics problems',
            'path': str(math_path),
            'model_target': 'qwen25_math_7b'
        }
        print(f"✅ Saved {len(test_data[:500])} MATH problems")
        
    except Exception as e:
        print(f"❌ Error downloading MATH dataset: {e}")
    
    try:
        # 2. MathQA Dataset
        print("\n🧮 Downloading MathQA dataset...")
        mathqa_dataset = load_dataset("math_qa")
        
        # Save test data
        test_data = []
        for item in mathqa_dataset['test']:
            test_data.append({
                'question': item['Problem'],
                'options': item['options'],
                'correct_answer': item['correct'],
                'rationale': item['Rationale']
            })
        
        mathqa_path = Path("evaluation_data/reasoning/mathqa.json")
        with open(mathqa_path, 'w') as f:
            json.dump(test_data[:300], f, indent=2)  # Limit to 300
        
        datasets_info['mathqa'] = {
            'name': 'MathQA',
            'samples': len(test_data[:300]),
            'description': 'Mathematical reasoning with multiple choice',
            'path': str(mathqa_path),
            'model_target': 'qwen25_math_7b'
        }
        print(f"✅ Saved {len(test_data[:300])} MathQA problems")
        
    except Exception as e:
        print(f"❌ Error downloading MathQA dataset: {e}")
    
    return datasets_info

def download_biomedical_datasets():
    """Download datasets for genomic specialist models."""
    print("\n🧬 DOWNLOADING BIOMEDICAL DATASETS")
    print("="*50)
    
    datasets_info = {}
    
    try:
        # 1. PubMedQA
        print("\n🏥 Downloading PubMedQA dataset...")
        pubmedqa_dataset = load_dataset("pubmed_qa", "pqa_labeled")
        
        test_data = []
        for item in pubmedqa_dataset['test']:
            test_data.append({
                'question': item['question'],
                'context': item['context'],
                'answer': item['final_decision'],
                'long_answer': item['long_answer']
            })
        
        pubmed_path = Path("evaluation_data/qa/pubmedqa.json")
        pubmed_path.parent.mkdir(exist_ok=True)
        with open(pubmed_path, 'w') as f:
            json.dump(test_data[:200], f, indent=2)  # Limit to 200
        
        datasets_info['pubmedqa'] = {
            'name': 'PubMedQA',
            'samples': len(test_data[:200]),
            'description': 'Biomedical question answering',
            'path': str(pubmed_path),
            'model_target': 'qwen25_1_5b_genomic, qwen25_72b_genomic'
        }
        print(f"✅ Saved {len(test_data[:200])} PubMedQA questions")
        
    except Exception as e:
        print(f"❌ Error downloading PubMedQA: {e}")
    
    return datasets_info

def download_advanced_coding_datasets():
    """Download datasets for qwen3_coder_30b specialist."""
    print("\n💻 DOWNLOADING ADVANCED CODING DATASETS")
    print("="*50)
    
    datasets_info = {}
    
    try:
        # 1. APPS Dataset (Advanced Python Problems)
        print("\n🐍 Downloading APPS dataset...")
        apps_dataset = load_dataset("codeparrot/apps", "all")
        
        test_data = []
        for item in apps_dataset['test']:
            if len(test_data) >= 100:  # Limit to 100 complex problems
                break
            test_data.append({
                'problem': item['question'],
                'solutions': item['solutions'],
                'input_output': item['input_output'],
                'difficulty': item['difficulty'],
                'url': item.get('url', '')
            })
        
        apps_path = Path("evaluation_data/coding/apps_advanced.json")
        with open(apps_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        datasets_info['apps_advanced'] = {
            'name': 'APPS Advanced',
            'samples': len(test_data),
            'description': 'Advanced Python programming problems',
            'path': str(apps_path),
            'model_target': 'qwen3_coder_30b'
        }
        print(f"✅ Saved {len(test_data)} APPS problems")
        
    except Exception as e:
        print(f"❌ Error downloading APPS dataset: {e}")
    
    return datasets_info

def download_core_missing_datasets():
    """Download core datasets that are currently missing."""
    print("\n📋 DOWNLOADING CORE MISSING DATASETS")
    print("="*50)
    
    datasets_info = {}
    
    try:
        # 1. MMLU (Massive Multitask Language Understanding)
        print("\n🎓 Downloading MMLU dataset...")
        mmlu_dataset = load_dataset("cais/mmlu", "all")
        
        test_data = []
        for item in mmlu_dataset['test']:
            if len(test_data) >= 1000:  # Limit to 1000 questions
                break
            test_data.append({
                'question': item['question'],
                'choices': item['choices'],
                'answer': item['answer'],
                'subject': item['subject']
            })
        
        mmlu_path = Path("evaluation_data/qa/mmlu.json")
        with open(mmlu_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        datasets_info['mmlu'] = {
            'name': 'MMLU',
            'samples': len(test_data),
            'description': 'Massive Multitask Language Understanding',
            'path': str(mmlu_path),
            'model_target': 'All models (general evaluation)'
        }
        print(f"✅ Saved {len(test_data)} MMLU questions")
        
    except Exception as e:
        print(f"❌ Error downloading MMLU: {e}")
    
    return datasets_info

def create_dataset_summary(all_datasets_info):
    """Create summary of all downloaded datasets."""
    summary = {
        'download_timestamp': datetime.now().isoformat(),
        'total_new_datasets': len(all_datasets_info),
        'datasets': all_datasets_info,
        'specialized_model_coverage': {
            'qwen25_math_7b': ['math_competition', 'mathqa'],
            'qwen25_1_5b_genomic': ['pubmedqa'],
            'qwen25_72b_genomic': ['pubmedqa'],
            'qwen3_coder_30b': ['apps_advanced'],
            'all_models': ['mmlu']
        }
    }
    
    summary_path = Path("evaluation_data/new_datasets_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main download function."""
    print("🚀 SPECIALIZED DATASET DOWNLOADER")
    print("="*60)
    print("Downloading datasets for new specialized Qwen models")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Ensure directories exist
    Path("evaluation_data/reasoning").mkdir(parents=True, exist_ok=True)
    Path("evaluation_data/qa").mkdir(parents=True, exist_ok=True)
    Path("evaluation_data/coding").mkdir(parents=True, exist_ok=True)
    
    all_datasets = {}
    
    try:
        # Download datasets by category
        print("Phase 1: Mathematics datasets for qwen25_math_7b")
        math_datasets = download_mathematics_datasets()
        all_datasets.update(math_datasets)
        
        print("\nPhase 2: Biomedical datasets for genomic models")
        bio_datasets = download_biomedical_datasets()
        all_datasets.update(bio_datasets)
        
        print("\nPhase 3: Advanced coding datasets for qwen3_coder_30b")
        coding_datasets = download_advanced_coding_datasets()
        all_datasets.update(coding_datasets)
        
        print("\nPhase 4: Core missing datasets")
        core_datasets = download_core_missing_datasets()
        all_datasets.update(core_datasets)
        
        # Create summary
        summary = create_dataset_summary(all_datasets)
        
        print(f"\n✅ DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"📊 Total datasets downloaded: {len(all_datasets)}")
        print(f"📁 Summary saved to: evaluation_data/new_datasets_summary.json")
        print()
        
        print("📋 Downloaded Datasets:")
        for name, info in all_datasets.items():
            print(f"  • {info['name']}: {info['samples']} samples → {info['model_target']}")
        
        print(f"\n🎯 Next Steps:")
        print("1. Run evaluation with new datasets:")
        print("   python evaluation/run_evaluation.py --model qwen25_math_7b --dataset math_competition")
        print("2. Test genomic models:")
        print("   python evaluation/run_evaluation.py --model qwen25_1_5b_genomic --dataset pubmedqa")
        print("3. Test advanced coding:")
        print("   python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset apps_advanced")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()