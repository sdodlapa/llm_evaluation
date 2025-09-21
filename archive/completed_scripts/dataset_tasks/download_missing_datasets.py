#!/usr/bin/env python3
"""
Modern Dataset Downloader for Missing Coding Datasets
Uses the Hugging Face datasets library properly instead of broken URL scraping
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_codecontests(output_dir: str = "evaluation_data/coding") -> Dict[str, Any]:
    """Download CodeContests dataset using the datasets library"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    codecontests_file = output_path / "codecontests.json"
    
    try:
        logger.info("Loading CodeContests dataset from deepmind/code_contests...")
        ds = load_dataset("deepmind/code_contests")
        
        # Use the training split for evaluation samples
        train_data = ds["train"]
        logger.info(f"Loaded {len(train_data)} training samples")
        
        # Convert to our expected format
        samples = []
        for i, item in enumerate(train_data):
            if i >= 1000:  # Limit to first 1000 for evaluation
                break
                
            # Extract the problem and solution
            problem_text = item.get("description", "")
            solutions = item.get("solutions", {})
            
            # Get the first Python solution if available
            python_solution = None
            if solutions:
                for lang, sol_list in solutions.items():
                    if lang.lower() in ["python", "python3"] and sol_list:
                        python_solution = sol_list[0]
                        break
                
                # If no Python solution, try any solution
                if not python_solution:
                    for lang, sol_list in solutions.items():
                        if sol_list:
                            python_solution = sol_list[0]
                            break
            
            if problem_text and python_solution:
                samples.append({
                    "problem": problem_text,
                    "solution": python_solution,
                    "difficulty": item.get("difficulty", "unknown"),
                    "source": "codecontests",
                    "metadata": {
                        "cf_rating": item.get("cf_rating"),
                        "cf_tags": item.get("cf_tags", []),
                        "time_limit": item.get("time_limit"),
                        "memory_limit_bytes": item.get("memory_limit_bytes")
                    }
                })
        
        # Save to JSON
        with open(codecontests_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"Saved {len(samples)} CodeContests problems to {codecontests_file}")
        
        return {
            "success": True,
            "dataset": "codecontests",
            "samples": len(samples),
            "file": str(codecontests_file)
        }
        
    except Exception as e:
        logger.error(f"Error downloading CodeContests: {e}")
        return {
            "success": False,
            "dataset": "codecontests", 
            "error": str(e)
        }

def find_apps_alternative() -> Dict[str, Any]:
    """Find an alternative to the deprecated APPS dataset"""
    alternatives = [
        "microsoft/CodeXGLUE",
        "bigcode/apps",
        "openai/openai_humaneval",
        "microsoft/APPS",
        "THUDM/apps"
    ]
    
    logger.info("Searching for APPS alternatives...")
    
    for alt in alternatives:
        try:
            logger.info(f"Trying {alt}...")
            ds = load_dataset(alt)
            logger.info(f"Found working alternative: {alt}")
            logger.info(f"Splits: {list(ds.keys())}")
            for split, data in ds.items():
                logger.info(f"  {split}: {len(data)} samples")
            return {
                "success": True,
                "alternative": alt,
                "dataset": ds
            }
        except Exception as e:
            logger.debug(f"Failed to load {alt}: {e}")
            continue
    
    return {
        "success": False,
        "message": "No working APPS alternatives found"
    }

def download_alternative_coding_dataset(output_dir: str = "evaluation_data/coding") -> Dict[str, Any]:
    """Download an alternative coding dataset to replace APPS"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Try to get a good coding competition/interview dataset
    alternatives = [
        ("nyu-mll/APPS", "apps_alternative"),
        ("codeparrot/github-code-clean", "github_code"),
        ("microsoft/CodeSearchNet", "codesearchnet"),
    ]
    
    for dataset_name, local_name in alternatives:
        try:
            logger.info(f"Trying to download {dataset_name}...")
            ds = load_dataset(dataset_name)
            
            # Use first available split
            split_name = list(ds.keys())[0]
            data = ds[split_name]
            
            logger.info(f"Successfully loaded {dataset_name} with {len(data)} samples")
            
            # Convert to our format and save
            samples = []
            for i, item in enumerate(data):
                if i >= 500:  # Limit samples
                    break
                
                # Try to extract problem and solution based on dataset structure
                if "problem" in item and "solution" in item:
                    samples.append({
                        "problem": item["problem"],
                        "solution": item["solution"],
                        "source": local_name,
                        "difficulty": item.get("difficulty", "unknown")
                    })
                elif "text" in item:
                    # For code datasets without explicit problem/solution
                    samples.append({
                        "problem": f"Complete this code: {item['text'][:200]}...",
                        "solution": item["text"],
                        "source": local_name,
                        "difficulty": "unknown"
                    })
            
            if samples:
                apps_alt_file = output_path / f"{local_name}.json"
                with open(apps_alt_file, 'w') as f:
                    json.dump(samples, f, indent=2)
                
                logger.info(f"Saved {len(samples)} samples to {apps_alt_file}")
                return {
                    "success": True,
                    "dataset": local_name,
                    "original": dataset_name,
                    "samples": len(samples),
                    "file": str(apps_alt_file)
                }
        
        except Exception as e:
            logger.debug(f"Failed to load {dataset_name}: {e}")
            continue
    
    return {
        "success": False,
        "message": "No suitable APPS alternative found"
    }

def main():
    """Download missing coding datasets"""
    print("🔍 Downloading Missing Coding Datasets")
    print("=" * 50)
    
    results = []
    
    # Download CodeContests
    print("\n1. Downloading CodeContests...")
    cc_result = download_codecontests()
    results.append(cc_result)
    
    if cc_result["success"]:
        print(f"✅ CodeContests: {cc_result['samples']} samples")
    else:
        print(f"❌ CodeContests failed: {cc_result['error']}")
    
    # Try to find APPS alternative
    print("\n2. Finding APPS alternative...")
    apps_result = download_alternative_coding_dataset()
    results.append(apps_result)
    
    if apps_result["success"]:
        print(f"✅ APPS Alternative ({apps_result['dataset']}): {apps_result['samples']} samples")
    else:
        print(f"❌ APPS alternative failed: {apps_result['message']}")
    
    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    
    successful = [r for r in results if r["success"]]
    print(f"✅ Successfully downloaded: {len(successful)}/{len(results)} datasets")
    
    for result in successful:
        print(f"  - {result['dataset']}: {result['samples']} samples")
    
    if len(successful) < len(results):
        print(f"\n❌ Failed downloads: {len(results) - len(successful)}")

if __name__ == "__main__":
    main()