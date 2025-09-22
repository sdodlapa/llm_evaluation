#!/usr/bin/env python3
"""
Large Model Dataset Downloader
Download advanced datasets required for 70B+ parameter model evaluation
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LargeModelDatasetDownloader:
    """Download datasets specifically for large model evaluation"""
    
    def __init__(self, base_path: str = "/home/sdodl001_odu_edu/llm_evaluation/evaluation_data"):
        self.base_path = Path(base_path)
        self.datasets_path = self.base_path / "datasets"
        self.download_log_path = self.base_path / "download_logs"
        
        # Create directories
        for path in [self.datasets_path, self.download_log_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Category directories
        self.categories = {
            "advanced_coding": self.datasets_path / "coding",
            "advanced_reasoning": self.datasets_path / "reasoning", 
            "enterprise": self.datasets_path / "enterprise",
            "advanced_multimodal": self.datasets_path / "multimodal",
            "long_context": self.datasets_path / "long_context",
            "multilingual": self.datasets_path / "multilingual"
        }
        
        for category_path in self.categories.values():
            category_path.mkdir(parents=True, exist_ok=True)
    
    def download_bigbench_hard(self) -> Dict[str, Any]:
        """Download BigBench-Hard for complex reasoning evaluation"""
        logger.info("Downloading BigBench-Hard dataset...")
        try:
            # Load the BigBench-Hard dataset
            dataset = load_dataset("lukaemon/bbh")
            
            # Process test split
            test_data = dataset["test"]
            samples = []
            
            for i, item in enumerate(test_data):
                if i >= 3000:  # Limit for evaluation
                    break
                    
                samples.append({
                    'id': f"bbh_{i}",
                    'task': item.get('task', ''),
                    'input': item.get('input', ''),
                    'target': item.get('target', ''),
                    'examples': item.get('examples', ''),
                    'metadata': {
                        'source': 'BigBench-Hard',
                        'task_type': 'multi_step_reasoning',
                        'difficulty': 'hard'
                    }
                })
            
            # Save to file
            output_file = self.categories["advanced_reasoning"] / "bigbench_hard.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"BigBench-Hard downloaded: {len(samples)} samples")
            return {
                "name": "bigbench_hard",
                "samples": len(samples),
                "task_type": "complex_reasoning",
                "description": "Challenging multi-step reasoning tasks",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to download BigBench-Hard: {e}")
            return {"error": str(e)}
    
    def download_mmlu_pro(self) -> Dict[str, Any]:
        """Download MMLU-Pro for enhanced academic evaluation"""
        logger.info("Downloading MMLU-Pro dataset...")
        try:
            # Load MMLU-Pro dataset
            dataset = load_dataset("TIGER-Lab/MMLU-Pro")
            
            # Process test split
            test_data = dataset["test"]
            samples = []
            
            for i, item in enumerate(test_data):
                if i >= 5000:  # Reasonable limit for large model evaluation
                    break
                    
                samples.append({
                    'id': f"mmlu_pro_{i}",
                    'question': item.get('question', ''),
                    'options': item.get('options', []),
                    'answer': item.get('answer', ''),
                    'answer_index': item.get('answer_index', 0),
                    'cot_content': item.get('cot_content', ''),
                    'category': item.get('category', ''),
                    'src': item.get('src', ''),
                    'metadata': {
                        'source': 'MMLU-Pro',
                        'task_type': 'enhanced_multiple_choice',
                        'has_reasoning': bool(item.get('cot_content', ''))
                    }
                })
            
            # Save to file
            output_file = self.categories["advanced_reasoning"] / "mmlu_pro.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"MMLU-Pro downloaded: {len(samples)} samples")
            return {
                "name": "mmlu_pro",
                "samples": len(samples),
                "task_type": "enhanced_academic_qa",
                "description": "Enhanced MMLU with reasoning chains",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to download MMLU-Pro: {e}")
            return {"error": str(e)}
    
    def download_longbench(self) -> Dict[str, Any]:
        """Download LongBench for long-context evaluation"""
        logger.info("Downloading LongBench dataset...")
        try:
            # Load LongBench dataset  
            dataset = load_dataset("THUDM/LongBench", "narrativeqa")  # Start with one subset
            
            # Process test split
            test_data = dataset["test"]
            samples = []
            
            for i, item in enumerate(test_data):
                if i >= 1000:  # Long context samples are expensive to evaluate
                    break
                    
                samples.append({
                    'id': f"longbench_{i}",
                    'input': item.get('input', ''),
                    'context': item.get('context', ''),
                    'answers': item.get('answers', []),
                    'length': item.get('length', 0),
                    'dataset': item.get('dataset', ''),
                    'language': item.get('language', 'en'),
                    'metadata': {
                        'source': 'LongBench',
                        'task_type': 'long_context_qa',
                        'context_length': item.get('length', 0),
                        'requires_long_context': True
                    }
                })
            
            # Save to file
            output_file = self.categories["long_context"] / "longbench.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"LongBench downloaded: {len(samples)} samples")
            return {
                "name": "longbench",
                "samples": len(samples),
                "task_type": "long_context_understanding",
                "description": "Long-context understanding tasks (32K+ tokens)",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to download LongBench: {e}")
            return {"error": str(e)}
    
    def download_swe_bench(self) -> Dict[str, Any]:
        """Download SWE-bench for software engineering evaluation"""
        logger.info("Downloading SWE-bench dataset...")
        try:
            # Load SWE-bench dataset
            dataset = load_dataset("princeton-nlp/SWE-bench")
            
            # Process test split
            test_data = dataset["test"]
            samples = []
            
            for i, item in enumerate(test_data):
                if i >= 500:  # Software engineering tasks are complex and time-consuming
                    break
                    
                samples.append({
                    'id': f"swe_bench_{i}",
                    'instance_id': item.get('instance_id', ''),
                    'problem_statement': item.get('problem_statement', ''),
                    'hints_text': item.get('hints_text', ''),
                    'created_at': item.get('created_at', ''),
                    'patch': item.get('patch', ''),
                    'test_patch': item.get('test_patch', ''),
                    'repo': item.get('repo', ''),
                    'base_commit': item.get('base_commit', ''),
                    'environment_setup_commit': item.get('environment_setup_commit', ''),
                    'metadata': {
                        'source': 'SWE-bench',
                        'task_type': 'software_engineering',
                        'requires_repository_context': True,
                        'difficulty': 'expert'
                    }
                })
            
            # Save to file
            output_file = self.categories["advanced_coding"] / "swe_bench.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"SWE-bench downloaded: {len(samples)} samples")
            return {
                "name": "swe_bench",
                "samples": len(samples),
                "task_type": "software_engineering",
                "description": "Real-world software engineering tasks",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to download SWE-bench: {e}")
            return {"error": str(e)}
    
    def download_mmmu(self) -> Dict[str, Any]:
        """Download MMMU for advanced multimodal understanding"""
        logger.info("Downloading MMMU dataset...")
        try:
            # Load MMMU dataset
            dataset = load_dataset("MMMU/MMMU", "validation")  # Start with validation split
            
            # Process validation data
            val_data = dataset["validation"]
            samples = []
            
            for i, item in enumerate(val_data):
                if i >= 2000:  # Multimodal evaluation is resource-intensive
                    break
                    
                # Handle images if present
                images = []
                for img_key in ['image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7']:
                    if img_key in item and item[img_key] is not None:
                        images.append(f"image_{len(images) + 1}")
                
                samples.append({
                    'id': f"mmmu_{i}",
                    'question': item.get('question', ''),
                    'options': item.get('options', []),
                    'answer': item.get('answer', ''),
                    'question_type': item.get('question_type', ''),
                    'subfield': item.get('subfield', ''),
                    'subject': item.get('subject', ''),
                    'language': item.get('language', 'en'),
                    'num_images': len(images),
                    'images': images,
                    'metadata': {
                        'source': 'MMMU',
                        'task_type': 'multimodal_academic_qa',
                        'multimodal': True,
                        'difficulty': 'college_level'
                    }
                })
            
            # Save to file
            output_file = self.categories["advanced_multimodal"] / "mmmu.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"MMMU downloaded: {len(samples)} samples")
            return {
                "name": "mmmu",
                "samples": len(samples),
                "task_type": "advanced_multimodal_qa",
                "description": "Massive multi-discipline multimodal understanding",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to download MMMU: {e}")
            return {"error": str(e)}
    
    def download_mathvista(self) -> Dict[str, Any]:
        """Download MathVista for mathematical visual reasoning"""
        logger.info("Downloading MathVista dataset...")
        try:
            # Load MathVista dataset
            dataset = load_dataset("AI4Math/MathVista", "testmini")  # Use testmini split
            
            # Process test data
            test_data = dataset["testmini"]
            samples = []
            
            for i, item in enumerate(test_data):
                if i >= 1500:  # Mathematical visual reasoning is complex
                    break
                    
                samples.append({
                    'id': f"mathvista_{i}",
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'question_type': item.get('question_type', ''),
                    'answer_type': item.get('answer_type', ''),
                    'language': item.get('language', 'en'),
                    'split': item.get('split', ''),
                    'source': item.get('source', ''),
                    'task': item.get('task', ''),
                    'context': item.get('context', ''),
                    'grade': item.get('grade', ''),
                    'img_width': item.get('img_width', 0),
                    'img_height': item.get('img_height', 0),
                    'metadata': {
                        'source': 'MathVista',
                        'task_type': 'mathematical_visual_reasoning',
                        'multimodal': True,
                        'domain': 'mathematics',
                        'requires_visual_analysis': True
                    }
                })
            
            # Save to file
            output_file = self.categories["advanced_multimodal"] / "mathvista.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"MathVista downloaded: {len(samples)} samples")
            return {
                "name": "mathvista",
                "samples": len(samples),
                "task_type": "mathematical_visual_reasoning",
                "description": "Mathematical visual reasoning and problem solving",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to download MathVista: {e}")
            return {"error": str(e)}
    
    def download_livecodebench(self) -> Dict[str, Any]:
        """Download LiveCodeBench for recent coding challenges"""
        logger.info("Downloading LiveCodeBench dataset...")
        try:
            # Load LiveCodeBench dataset
            dataset = load_dataset("livecodebench/code_generation_lite")
            
            # Process test split
            test_data = dataset["test"] 
            samples = []
            
            for i, item in enumerate(test_data):
                if i >= 800:  # Coding challenges are time-intensive
                    break
                    
                samples.append({
                    'id': f"livecodebench_{i}",
                    'question_title': item.get('question_title', ''),
                    'question_content': item.get('question_content', ''),
                    'platform': item.get('platform', ''),
                    'difficulty': item.get('difficulty', ''),
                    'programming_language': item.get('programming_language', ''),
                    'contest_date': item.get('contest_date', ''),
                    'contest_id': item.get('contest_id', ''),
                    'problem_id': item.get('problem_id', ''),
                    'starter_code': item.get('starter_code', ''),
                    'function_name': item.get('function_name', ''),
                    'test_code': item.get('test_code', ''),
                    'metadata': {
                        'source': 'LiveCodeBench',
                        'task_type': 'competitive_programming',
                        'live_contest': True,
                        'language': item.get('programming_language', 'python')
                    }
                })
            
            # Save to file
            output_file = self.categories["advanced_coding"] / "livecodebench.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"LiveCodeBench downloaded: {len(samples)} samples")
            return {
                "name": "livecodebench",
                "samples": len(samples),
                "task_type": "live_competitive_programming",
                "description": "Recent competitive programming challenges",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to download LiveCodeBench: {e}")
            return {"error": str(e)}
    
    def create_enterprise_sample_dataset(self) -> Dict[str, Any]:
        """Create sample enterprise dataset for business applications"""
        logger.info("Creating enterprise sample dataset...")
        try:
            # Create sample enterprise tasks
            samples = [
                {
                    'id': 'enterprise_1',
                    'task_type': 'business_analysis',
                    'prompt': 'Analyze quarterly sales data and identify key trends for executive summary.',
                    'context': 'Q1-Q4 sales figures show 15% growth in enterprise segment, 8% decline in consumer segment.',
                    'expected_output_type': 'executive_summary',
                    'complexity': 'high',
                    'domain': 'business_intelligence'
                },
                {
                    'id': 'enterprise_2', 
                    'task_type': 'technical_documentation',
                    'prompt': 'Create API documentation for the user authentication endpoint.',
                    'context': 'POST /api/auth/login accepts email, password, returns JWT token or error.',
                    'expected_output_type': 'api_documentation',
                    'complexity': 'medium',
                    'domain': 'software_documentation'
                },
                {
                    'id': 'enterprise_3',
                    'task_type': 'risk_assessment', 
                    'prompt': 'Evaluate cybersecurity risks in the proposed cloud migration plan.',
                    'context': 'Moving customer data from on-premise servers to AWS cloud infrastructure.',
                    'expected_output_type': 'risk_analysis',
                    'complexity': 'high',
                    'domain': 'cybersecurity'
                }
            ] * 100  # Duplicate to create larger sample
            
            # Add unique IDs
            for i, sample in enumerate(samples):
                sample['id'] = f"enterprise_{i + 1}"
            
            # Save to file
            output_file = self.categories["enterprise"] / "enterprise_tasks.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"Enterprise sample dataset created: {len(samples)} samples")
            return {
                "name": "enterprise_tasks",
                "samples": len(samples),
                "task_type": "business_applications",
                "description": "Enterprise business task evaluation samples",
                "file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to create enterprise dataset: {e}")
            return {"error": str(e)}
    
    def download_all_datasets(self) -> Dict[str, Any]:
        """Download all datasets for large model evaluation"""
        logger.info("Starting large model dataset download process...")
        
        download_functions = [
            ("BigBench-Hard", self.download_bigbench_hard),
            ("MMLU-Pro", self.download_mmlu_pro),
            ("LongBench", self.download_longbench),
            ("SWE-bench", self.download_swe_bench),
            ("MMMU", self.download_mmmu),
            ("MathVista", self.download_mathvista),
            ("LiveCodeBench", self.download_livecodebench),
            ("Enterprise Tasks", self.create_enterprise_sample_dataset)
        ]
        
        results = {}
        successful_downloads = 0
        total_samples = 0
        
        for name, download_func in download_functions:
            logger.info(f"\n--- Downloading {name} ---")
            try:
                result = download_func()
                results[name] = result
                
                if 'error' not in result:
                    successful_downloads += 1
                    total_samples += result.get('samples', 0)
                    logger.info(f"✅ {name}: {result.get('samples', 0)} samples")
                else:
                    logger.error(f"❌ {name}: {result['error']}")
                    
                # Brief pause between downloads
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Failed to download {name}: {e}")
                results[name] = {"error": str(e)}
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": len(download_functions),
            "successful_downloads": successful_downloads,
            "failed_downloads": len(download_functions) - successful_downloads,
            "total_samples": total_samples,
            "datasets": results
        }
        
        # Save summary
        summary_file = self.download_log_path / "large_model_datasets_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n--- Download Complete ---")
        logger.info(f"Successful: {successful_downloads}/{len(download_functions)} datasets")
        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return summary

def main():
    """Main execution function"""
    logger.info("🚀 LARGE MODEL DATASET DOWNLOADER")
    logger.info("=" * 60)
    logger.info("Downloading advanced datasets for 70B+ parameter model evaluation")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info()
    
    try:
        downloader = LargeModelDatasetDownloader()
        summary = downloader.download_all_datasets()
        
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD SUMMARY")
        print(f"✅ Successful: {summary['successful_downloads']}")
        print(f"❌ Failed: {summary['failed_downloads']}")
        print(f"📈 Total samples: {summary['total_samples']:,}")
        print(f"📁 Data location: {downloader.datasets_path}")
        
        if summary['successful_downloads'] > 0:
            print("\n🎯 Ready for large model evaluation!")
        else:
            print("\n⚠️ No datasets downloaded successfully. Check logs for errors.")
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()