#!/usr/bin/env python3
"""
Quick Evaluation Script for Working Models
==========================================

This script runs a quick evaluation on the 3 working models to validate the system
while we fix the quantization issues with the other models.
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from category_evaluation import CategoryEvaluationCLI

# Working models (without AWQ quantization issues)
WORKING_MODELS = [
    "qwen3_8b",      # Working ✅
    "qwen3_14b",     # Working ✅  
    "qwen25_7b"      # Working ✅
]

# All coding datasets 
ALL_CODING_DATASETS = [
    "humaneval",
    "mbpp", 
    "bigcodebench",
    "codecontests",
    "apps",
    "advanced_coding_sample",
    "advanced_coding_extended"
]

def main():
    """Run quick evaluation with working models only"""
    
    print("🚀 Starting Quick Evaluation with Working Models")
    print("=" * 60)
    print(f"Models to test: {WORKING_MODELS}")
    print(f"Datasets: {ALL_CODING_DATASETS}")
    print(f"Sample size: 5 per dataset")
    print("=" * 60)
    
    # Create temporary evaluation arguments
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])  # Empty args
    
    # Override args with our settings
    args.category = "coding_specialists"
    args.preset = "balanced"
    args.samples = 5
    args.include_optional = True
    args.models = WORKING_MODELS  # Override to working models only
    args.verbose = True
    args.output_dir = "quick_results"
    
    print("⚡ Running evaluation...")
    
    try:
        # Import and run the category evaluation with our overrides
        from category_evaluation import main as run_evaluation
        
        # Temporarily modify sys.argv to pass our arguments
        original_argv = sys.argv.copy()
        sys.argv = [
            "category_evaluation.py",
            "--category", "coding_specialists", 
            "--preset", "balanced",
            "--samples", "5",
            "--include-optional",
            "--verbose",
            "--output-dir", "quick_results"
        ]
        
        # Add model restrictions
        for model in WORKING_MODELS:
            sys.argv.extend(["--models", model])
            
        result = run_evaluation()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("\n🎉 Quick evaluation completed!")
        return result
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()