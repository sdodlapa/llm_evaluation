#!/usr/bin/env python3
"""
Quick validation of HellaSwag and MT-Bench fixed datasets
This script will check that datasets are properly formatted with real content
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_dataset_content(dataset_name, dataset_path, task_type):
    """Validate that a dataset has proper content (not empty fields)"""
    logger.info(f"Validating {dataset_name} dataset...")
    
    try:
        # Load dataset to verify it's properly formatted
        with open(dataset_path) as f:
            dataset_data = json.load(f)
        
        samples = dataset_data.get('samples', [])
        logger.info(f"Dataset loaded: {len(samples)} samples")
        
        if not samples:
            logger.error(f"No samples found in {dataset_name}")
            return False
        
        # Check first few samples for content
        validation_samples = samples[:5]
        
        for i, sample in enumerate(validation_samples, 1):
            logger.info(f"Sample {i} format: {list(sample.keys())}")
            
            # Check for empty content based on task type
            if task_type == "reasoning":  # HellaSwag
                question = sample.get('question', '').strip()
                choices = sample.get('choices', [])
                context = sample.get('context', '').strip()
                
                logger.info(f"  Question length: {len(question)}")
                logger.info(f"  Number of choices: {len(choices)}")
                logger.info(f"  Context length: {len(context)}")
                
                if not question and not context:
                    logger.error(f"Sample {i} has empty question and context!")
                    return False
                
                if len(choices) == 0:
                    logger.error(f"Sample {i} has no choices!")
                    return False
                
                # Show content preview
                logger.info(f"  Question preview: {question[:100]}...")
                if context:
                    logger.info(f"  Context preview: {context[:100]}...")
                logger.info(f"  Choices: {choices}")
                
            elif task_type == "instruction_following":  # MT-Bench
                instruction = sample.get('instruction', '').strip()
                category = sample.get('category', '').strip()
                
                logger.info(f"  Instruction length: {len(instruction)}")
                logger.info(f"  Category: {category}")
                
                if not instruction:
                    logger.error(f"Sample {i} has empty instruction!")
                    return False
                
                # Show content preview
                logger.info(f"  Instruction preview: {instruction[:100]}...")
        
        logger.info(f"✅ {dataset_name} validation passed - real content found!")
        return True
            
    except Exception as e:
        logger.error(f"Failed to validate {dataset_name}: {e}")
        return False

def main():
    """Main execution function"""
    logger.info("🧪 Validating fixed datasets content...")
    
    # Check HellaSwag
    logger.info("=" * 60)
    hellaswag_success = validate_dataset_content(
        "HellaSwag",
        "evaluation_data/reasoning/hellaswag.json",
        "reasoning"
    )
    
    # Check MT-Bench
    logger.info("=" * 60)
    mt_bench_success = validate_dataset_content(
        "MT-Bench",
        "evaluation_data/instruction_following/mt_bench.json",
        "instruction_following"
    )
    
    logger.info("=" * 60)
    logger.info("🎯 Validation Results Summary:")
    logger.info(f"  HellaSwag: {'✅ CONTENT VALID' if hellaswag_success else '❌ STILL EMPTY/CORRUPTED'}")
    logger.info(f"  MT-Bench: {'✅ CONTENT VALID' if mt_bench_success else '❌ STILL EMPTY/CORRUPTED'}")
    
    if hellaswag_success and mt_bench_success:
        logger.info("🎉 Both datasets have proper content! Ready for evaluation.")
        return True
    else:
        logger.warning("⚠️ Some datasets still have content issues")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("✅ SUCCESS: Dataset content validated!")
        else:
            print("⚠️ ISSUES: Some datasets still have problems")
            sys.exit(1)
    except Exception as e:
        print(f"❌ SCRIPT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)