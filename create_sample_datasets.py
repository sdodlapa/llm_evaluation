#!/usr/bin/env python3
"""
Quick download script for the most critical missing datasets.
Focuses on high-impact datasets for specialized Qwen model evaluation.
"""

import json
from pathlib import Path
from datetime import datetime

def create_sample_datasets():
    """Create sample datasets for immediate testing while we work on downloads."""
    print("🎯 CREATING SAMPLE DATASETS FOR IMMEDIATE TESTING")
    print("="*60)
    
    # Ensure directories exist
    Path("evaluation_data/reasoning").mkdir(parents=True, exist_ok=True)
    Path("evaluation_data/qa").mkdir(parents=True, exist_ok=True)
    Path("evaluation_data/coding").mkdir(parents=True, exist_ok=True)
    
    datasets_created = {}
    
    # 1. Sample advanced math problems for qwen25_math_7b
    print("📐 Creating sample advanced math dataset...")
    advanced_math = [
        {
            "problem": "Find the number of positive integers n ≤ 2023 such that n and n+1 have the same number of positive divisors.",
            "solution": "We need to find when d(n) = d(n+1) where d(x) is the number of divisors of x.",
            "level": "Level 5",
            "type": "Number Theory",
            "difficulty": "hard"
        },
        {
            "problem": "Let f(x) = x³ - 3x² + 2x + 1. Find all real values of a such that f(a) = f'(a).",
            "solution": "First find f'(x) = 3x² - 6x + 2, then solve x³ - 3x² + 2x + 1 = 3x² - 6x + 2.",
            "level": "Level 4", 
            "type": "Algebra",
            "difficulty": "medium"
        },
        {
            "problem": "A regular octagon is inscribed in a circle of radius 1. What is the area of the octagon?",
            "solution": "Split into 8 triangles, each with central angle π/4. Area = 8 × (1/2) × 1² × sin(π/4) = 2√2.",
            "level": "Level 3",
            "type": "Geometry", 
            "difficulty": "medium"
        },
        {
            "problem": "Evaluate ∫₀^π sin⁶(x) dx using reduction formulas.",
            "solution": "Use integration by parts and reduction formula for sin^n(x).",
            "level": "Level 5",
            "type": "Calculus",
            "difficulty": "hard"
        },
        {
            "problem": "Find the coefficient of x¹⁰ in the expansion of (1+x+x²+x³)⁵.",
            "solution": "Use generating functions. (1+x+x²+x³)⁵ = ((1-x⁴)/(1-x))⁵.",
            "level": "Level 4",
            "type": "Combinatorics", 
            "difficulty": "medium"
        }
    ]
    
    math_path = Path("evaluation_data/reasoning/advanced_math_sample.json")
    with open(math_path, 'w') as f:
        json.dump(advanced_math, f, indent=2)
    
    datasets_created['advanced_math_sample'] = {
        'name': 'Advanced Math Sample',
        'samples': len(advanced_math),
        'path': str(math_path),
        'target_model': 'qwen25_math_7b',
        'description': 'Competition-level mathematics problems'
    }
    print(f"✅ Created {len(advanced_math)} advanced math problems")
    
    # 2. Sample biomedical questions for genomic models
    print("🧬 Creating sample biomedical dataset...")
    biomedical_qa = [
        {
            "question": "What is the primary function of the BRCA1 gene?",
            "context": "BRCA1 is a tumor suppressor gene involved in DNA repair mechanisms.",
            "answer": "DNA repair and tumor suppression",
            "type": "genetics",
            "difficulty": "intermediate"
        },
        {
            "question": "Which enzyme is responsible for unwinding DNA during replication?",
            "context": "DNA replication requires several enzymes to separate and copy strands.",
            "answer": "DNA helicase",
            "type": "molecular_biology",
            "difficulty": "basic"
        },
        {
            "question": "What is the significance of CpG islands in gene regulation?",
            "context": "CpG islands are regions with high frequency of cytosine-guanine dinucleotides.",
            "answer": "CpG islands are important for gene expression regulation through methylation",
            "type": "epigenetics",
            "difficulty": "advanced"
        },
        {
            "question": "How does CRISPR-Cas9 achieve targeted gene editing?",
            "context": "CRISPR-Cas9 is a genome editing tool derived from bacterial immune systems.",
            "answer": "Guide RNA directs Cas9 nuclease to specific DNA sequences for cutting",
            "type": "gene_editing",
            "difficulty": "intermediate"
        },
        {
            "question": "What is the role of telomeres in cellular aging?",
            "context": "Telomeres are protective DNA-protein structures at chromosome ends.",
            "answer": "Telomeres shorten with each cell division, limiting cellular lifespan",
            "type": "cell_biology",
            "difficulty": "intermediate"
        }
    ]
    
    bio_path = Path("evaluation_data/qa/biomedical_sample.json")
    with open(bio_path, 'w') as f:
        json.dump(biomedical_qa, f, indent=2)
    
    datasets_created['biomedical_sample'] = {
        'name': 'Biomedical QA Sample',
        'samples': len(biomedical_qa),
        'path': str(bio_path),
        'target_model': 'qwen25_1_5b_genomic, qwen25_72b_genomic',
        'description': 'Genomics and molecular biology questions'
    }
    print(f"✅ Created {len(biomedical_qa)} biomedical questions")
    
    # 3. Sample advanced coding problems for qwen3_coder_30b
    print("💻 Creating sample advanced coding dataset...")
    advanced_coding = [
        {
            "problem": "Implement a function to find the longest increasing subsequence in O(n log n) time.",
            "solution": "def longest_increasing_subsequence(arr):\n    from bisect import bisect_left\n    tails = []\n    for num in arr:\n        pos = bisect_left(tails, num)\n        if pos == len(tails):\n            tails.append(num)\n        else:\n            tails[pos] = num\n    return len(tails)",
            "difficulty": "hard",
            "topics": ["dynamic_programming", "binary_search"]
        },
        {
            "problem": "Design a data structure that supports insert, delete, search, and getRandom operations in O(1) time.",
            "solution": "class RandomizedSet:\n    def __init__(self):\n        self.nums = []\n        self.positions = {}\n    \n    def insert(self, val):\n        if val in self.positions:\n            return False\n        self.positions[val] = len(self.nums)\n        self.nums.append(val)\n        return True",
            "difficulty": "medium",
            "topics": ["hash_table", "array", "design"]
        },
        {
            "problem": "Find the minimum number of merge operations to make an array palindromic.",
            "solution": "def min_merge_palindrome(arr):\n    left, right = 0, len(arr) - 1\n    merges = 0\n    while left < right:\n        if arr[left] == arr[right]:\n            left += 1\n            right -= 1\n        elif arr[left] < arr[right]:\n            arr[left + 1] += arr[left]\n            left += 1\n            merges += 1\n        else:\n            arr[right - 1] += arr[right]\n            right -= 1\n            merges += 1\n    return merges",
            "difficulty": "hard",
            "topics": ["two_pointers", "greedy"]
        },
        {
            "problem": "Implement a trie (prefix tree) with word search functionality including wildcard '.' support.",
            "solution": "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_word = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    \n    def insert(self, word):\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.is_word = True",
            "difficulty": "medium",
            "topics": ["trie", "tree", "design"]
        },
        {
            "problem": "Find the shortest path in a weighted graph using Dijkstra's algorithm with custom priority queue.",
            "solution": "import heapq\n\ndef dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    \n    while pq:\n        current_dist, current = heapq.heappop(pq)\n        \n        if current_dist > distances[current]:\n            continue\n            \n        for neighbor, weight in graph[current].items():\n            distance = current_dist + weight\n            \n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n                heapq.heappush(pq, (distance, neighbor))\n    \n    return distances",
            "difficulty": "hard",
            "topics": ["graph", "shortest_path", "heap"]
        }
    ]
    
    coding_path = Path("evaluation_data/coding/advanced_coding_sample.json")
    with open(coding_path, 'w') as f:
        json.dump(advanced_coding, f, indent=2)
    
    datasets_created['advanced_coding_sample'] = {
        'name': 'Advanced Coding Sample',
        'samples': len(advanced_coding),
        'path': str(coding_path),
        'target_model': 'qwen3_coder_30b',
        'description': 'Complex algorithmic programming problems'
    }
    print(f"✅ Created {len(advanced_coding)} advanced coding problems")
    
    # 4. Sample multimodal tasks (text descriptions for now)
    print("👁️ Creating sample multimodal dataset...")
    multimodal_sample = [
        {
            "question": "Describe what information you can extract from a scatter plot showing correlation between gene expression levels.",
            "context": "Chart analysis for scientific data interpretation",
            "answer": "Correlation strength, outliers, clustering patterns, relationship direction",
            "type": "chart_interpretation",
            "domain": "scientific"
        },
        {
            "question": "What trends would you look for in a time series plot of protein concentration over 24 hours?",
            "context": "Biological data visualization analysis",  
            "answer": "Circadian rhythms, peak expression times, baseline levels, periodic patterns",
            "type": "time_series_analysis",
            "domain": "biological"
        },
        {
            "question": "How would you interpret a heatmap showing gene expression across different tissue types?",
            "context": "Genomic data visualization",
            "answer": "Tissue-specific expression patterns, co-expressed gene clusters, functional groupings",
            "type": "heatmap_analysis", 
            "domain": "genomics"
        }
    ]
    
    multimodal_path = Path("evaluation_data/qa/multimodal_sample.json")
    with open(multimodal_path, 'w') as f:
        json.dump(multimodal_sample, f, indent=2)
    
    datasets_created['multimodal_sample'] = {
        'name': 'Multimodal Analysis Sample',
        'samples': len(multimodal_sample),
        'path': str(multimodal_path),
        'target_model': 'qwen2_vl_7b',
        'description': 'Chart and visualization interpretation tasks'
    }
    print(f"✅ Created {len(multimodal_sample)} multimodal tasks")
    
    # Create summary
    summary = {
        'creation_timestamp': datetime.now().isoformat(),
        'type': 'sample_datasets',
        'purpose': 'Immediate testing of specialized models',
        'total_datasets': len(datasets_created),
        'datasets': datasets_created,
        'next_steps': [
            'Test each specialized model with its sample dataset',
            'Download full datasets using download_specialized_datasets.py',
            'Run comprehensive evaluation once full datasets are available'
        ]
    }
    
    summary_path = Path("evaluation_data/sample_datasets_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return datasets_created, summary

def main():
    """Create sample datasets for immediate testing."""
    print("🚀 SAMPLE DATASET CREATOR")
    print("="*60)
    print("Creating sample datasets for immediate specialized model testing")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        datasets, summary = create_sample_datasets()
        
        print(f"\n✅ SAMPLE DATASETS CREATED!")
        print("="*60)
        print(f"📊 Total sample datasets: {len(datasets)}")
        print()
        
        print("📋 Created Sample Datasets:")
        for name, info in datasets.items():
            print(f"  • {info['name']}: {info['samples']} samples → {info['target_model']}")
        
        print(f"\n🎯 Immediate Testing Commands:")
        print("# Test math specialist")
        print("python evaluation/run_evaluation.py --model qwen25_math_7b --dataset advanced_math_sample")
        print("\n# Test genomic specialist") 
        print("python evaluation/run_evaluation.py --model qwen25_1_5b_genomic --dataset biomedical_sample")
        print("\n# Test coding specialist")
        print("python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset advanced_coding_sample")
        print("\n# Test multimodal model")
        print("python evaluation/run_evaluation.py --model qwen2_vl_7b --dataset multimodal_sample")
        
        print(f"\n📁 Next Steps:")
        print("1. Test sample datasets with specialized models (above commands)")
        print("2. Run: python download_specialized_datasets.py (for full datasets)")
        print("3. Validate results and expand to comprehensive evaluation")
        
    except Exception as e:
        print(f"❌ Error creating sample datasets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()