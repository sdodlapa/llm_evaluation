# Enhanced LLM Evaluation Framework Plan
**Comprehensive Multi-Model, Multi-Dataset Evaluation with Live Performance Monitoring**

## 🎯 Objective
Execute comprehensive evaluation across **all 6 models × all 12 datasets** with **100-200 samples per dataset**, capturing **live GPU/memory metrics** instead of hardcoded preset scores, with proper **logging, predictions, and metrics organization**.

---

## 📊 Current State Analysis

### Models to Evaluate (6 models × 3 presets = 18 configurations)
```
8B Models: qwen2.5_8b, qwen3_8b, qwen_8b
14B Models: qwen2.5_14b, qwen3_14b, qwen_14b  
Presets: balanced, performance, memory_optimized
```

### Datasets to Evaluate (12 datasets → expanded from 4)
```
CURRENT (4 datasets, 25 samples):
✅ humaneval (coding) - 164 samples available
✅ gsm8k (reasoning) - 1319 samples available  
✅ hellaswag (reasoning) - 10042 samples available
✅ mt_bench (instruction_following) - 160 samples available

EXPANSION (8 additional datasets):
🆕 bfcl (function_calling) - 2000 samples available
🆕 toolllama (function_calling) - 1500 samples available
🆕 mbpp (coding) - 974 samples available
🆕 codet5 (coding) - 5000 samples available
🆕 arc_challenge (reasoning) - 1172 samples available
🆕 alpaca_eval (instruction_following) - 805 samples available
🆕 mmlu (qa) - 14042 samples available
🆕 truthfulqa (qa) - 817 samples available
```

---

## 🔧 Required Code Changes

### 1. Enhanced Performance Monitoring (`performance_monitor.py`)
**Location**: `evaluation/performance_monitor.py`

```python
class LivePerformanceMonitor:
    """Captures real-time GPU/memory metrics during model evaluation"""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.memory_tracker = MemoryTracker()
        self.throughput_calculator = ThroughputCalculator()
    
    def start_monitoring(self, model_name: str, preset: str):
        """Start real-time monitoring session"""
        
    def capture_metrics(self) -> Dict[str, float]:
        """Capture current GPU/memory state"""
        return {
            'gpu_utilization_percent': self.gpu_monitor.get_utilization(),
            'gpu_memory_used_gb': self.gpu_monitor.get_memory_used(),
            'gpu_memory_total_gb': self.gpu_monitor.get_memory_total(),
            'gpu_temperature_c': self.gpu_monitor.get_temperature(),
            'throughput_tokens_per_second': self.throughput_calculator.get_current_rate(),
            'avg_latency_ms': self.throughput_calculator.get_avg_latency(),
            'peak_memory_gb': self.memory_tracker.get_peak_usage(),
            'memory_efficiency': self.memory_tracker.calculate_efficiency()
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics"""
```

### 2. Enhanced Dataset Manager (`dataset_manager.py`)
**Location**: `evaluation/dataset_manager.py`

```python
class EnhancedDatasetManager(EvaluationDatasetManager):
    """Extended dataset manager with configurable sample sizes"""
    
    def get_evaluation_samples(self, dataset_name: str, 
                             num_samples: int = 150) -> List[Dict]:
        """Get specified number of samples from dataset"""
        
    def get_all_datasets_for_evaluation(self, 
                                      samples_per_dataset: int = 150) -> Dict[str, List]:
        """Get all 12 datasets with specified sample counts"""
        return {
            # Function Calling (critical for agents)
            'bfcl': self.get_evaluation_samples('bfcl', min(samples_per_dataset, 200)),
            'toolllama': self.get_evaluation_samples('toolllama', min(samples_per_dataset, 200)),
            
            # Coding (expanded coverage)
            'humaneval': self.get_evaluation_samples('humaneval', min(samples_per_dataset, 164)),
            'mbpp': self.get_evaluation_samples('mbpp', min(samples_per_dataset, 200)),
            'codet5': self.get_evaluation_samples('codet5', min(samples_per_dataset, 200)),
            
            # Reasoning (comprehensive)
            'gsm8k': self.get_evaluation_samples('gsm8k', min(samples_per_dataset, 200)),
            'arc_challenge': self.get_evaluation_samples('arc_challenge', min(samples_per_dataset, 200)),
            'hellaswag': self.get_evaluation_samples('hellaswag', min(samples_per_dataset, 200)),
            
            # Instruction Following
            'alpaca_eval': self.get_evaluation_samples('alpaca_eval', min(samples_per_dataset, 200)),
            'mt_bench': self.get_evaluation_samples('mt_bench', min(samples_per_dataset, 160)),
            
            # Knowledge & QA
            'mmlu': self.get_evaluation_samples('mmlu', min(samples_per_dataset, 200)),
            'truthfulqa': self.get_evaluation_samples('truthfulqa', min(samples_per_dataset, 200))
        }
```

### 3. Enhanced Evaluation Runner (`comprehensive_evaluation.py`)
**Location**: `comprehensive_evaluation.py`

```python
class ComprehensiveEvaluationRunner:
    """Runs evaluation across all models and datasets with live monitoring"""
    
    def __init__(self, samples_per_dataset: int = 150):
        self.samples_per_dataset = samples_per_dataset
        self.performance_monitor = LivePerformanceMonitor()
        self.dataset_manager = EnhancedDatasetManager()
        self.results_organizer = ResultsOrganizer()
        
    def run_full_evaluation(self):
        """Run comprehensive evaluation: 6 models × 12 datasets × 3 presets"""
        
        models = ['qwen2.5_8b', 'qwen3_8b', 'qwen_8b', 
                 'qwen2.5_14b', 'qwen3_14b', 'qwen_14b']
        presets = ['balanced', 'performance', 'memory_optimized']
        datasets = self.dataset_manager.get_all_datasets_for_evaluation(self.samples_per_dataset)
        
        total_combinations = len(models) * len(presets) * len(datasets)
        print(f"🚀 Starting comprehensive evaluation: {total_combinations} combinations")
        
        for model in models:
            for preset in presets:
                for dataset_name, samples in datasets.items():
                    self.evaluate_model_dataset_combination(
                        model, preset, dataset_name, samples
                    )
```

### 4. Results Organization System (`results_organizer.py`)
**Location**: `evaluation/results_organizer.py`

```python
class ResultsOrganizer:
    """Organizes evaluation outputs into structured directories"""
    
    def __init__(self, base_dir: str = "comprehensive_evaluation_results"):
        self.base_dir = Path(base_dir)
        self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """Create organized directory structure"""
        directories = [
            "logs/model_evaluation",
            "logs/performance_monitoring", 
            "logs/dataset_processing",
            "predictions/by_model",
            "predictions/by_dataset", 
            "predictions/by_preset",
            "metrics/aggregated",
            "metrics/detailed",
            "metrics/performance",
            "reports/model_comparison",
            "reports/dataset_analysis",
            "reports/scaling_analysis",
            "cache/models",
            "cache/datasets"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def save_evaluation_result(self, model: str, preset: str, 
                              dataset: str, results: Dict):
        """Save results to appropriate locations"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        pred_file = f"{model}_{preset}_{dataset}_{timestamp}.json"
        
        # Save to multiple organizational schemes
        paths = [
            self.base_dir / "predictions" / "by_model" / model / pred_file,
            self.base_dir / "predictions" / "by_dataset" / dataset / pred_file,
            self.base_dir / "predictions" / "by_preset" / preset / pred_file
        ]
        
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
```

---

## 📂 Directory Structure Plan

```
comprehensive_evaluation_results/
├── logs/
│   ├── model_evaluation/           # Model loading and evaluation logs
│   ├── performance_monitoring/     # Real-time GPU/memory monitoring logs  
│   └── dataset_processing/         # Dataset loading and processing logs
├── predictions/
│   ├── by_model/                   # Organized by model name
│   │   ├── qwen2.5_8b/
│   │   ├── qwen3_8b/
│   │   └── ...
│   ├── by_dataset/                 # Organized by dataset
│   │   ├── humaneval/
│   │   ├── bfcl/
│   │   └── ...
│   └── by_preset/                  # Organized by configuration preset
│       ├── balanced/
│       ├── performance/
│       └── memory_optimized/
├── metrics/
│   ├── aggregated/                 # Summary metrics across runs
│   ├── detailed/                   # Individual evaluation metrics
│   └── performance/                # Live performance monitoring data
├── reports/
│   ├── model_comparison/           # Cross-model analysis
│   ├── dataset_analysis/           # Cross-dataset insights  
│   └── scaling_analysis/           # Performance scaling patterns
└── cache/
    ├── models/                     # Model cache
    └── datasets/                   # Dataset cache
```

---

## 🔍 Enhanced Metrics Collection

### Live Performance Metrics (Real-time capture)
```python
{
    "performance_metrics": {
        "gpu_utilization_percent": float,      # Real GPU usage %
        "gpu_memory_used_gb": float,           # Actual memory consumption  
        "gpu_memory_peak_gb": float,           # Peak memory during evaluation
        "gpu_temperature_c": float,            # GPU temperature monitoring
        "throughput_tokens_per_second": float, # Measured throughput
        "avg_latency_ms": float,               # Average response latency
        "memory_efficiency": float,            # Memory utilization efficiency
        "inference_batch_size": int,           # Actual batch size used
        "context_length_avg": float,           # Average context length processed
        "total_tokens_processed": int          # Total tokens in evaluation
    }
}
```

### Dataset Evaluation Metrics
```python
{
    "dataset_metrics": {
        "accuracy": float,                     # Task-specific accuracy
        "exact_match": float,                  # Exact string matching
        "bleu_score": float,                   # For generation tasks
        "rouge_scores": Dict[str, float],      # ROUGE-1, ROUGE-2, ROUGE-L
        "function_calling_success_rate": float, # For agent tasks
        "code_execution_success_rate": float,  # For coding tasks
        "reasoning_step_accuracy": float,      # For multi-step reasoning
        "response_quality_score": float        # Human-like response quality
    }
}
```

---

## 🚀 Execution Plan

### Phase 1: Framework Enhancement (1-2 hours)
1. **Create Enhanced Performance Monitor** (`evaluation/performance_monitor.py`)
2. **Extend Dataset Manager** with configurable sample sizes
3. **Build Results Organizer** for proper output structure
4. **Create Comprehensive Evaluation Runner**

### Phase 2: Validation Testing (30 minutes)
1. **Test single model-dataset combination** with live monitoring
2. **Validate performance metric capture** 
3. **Verify results organization** and file structure
4. **Confirm GPU monitoring accuracy**

### Phase 3: Full Evaluation Execution (4-6 hours)
1. **Run comprehensive evaluation**: 6 models × 12 datasets × 3 presets = 216 combinations
2. **Monitor progress** with intermediate checkpoints
3. **Capture all metrics** and predictions in organized structure
4. **Generate summary reports** for immediate analysis

### Phase 4: Analysis and Reporting (1 hour)
1. **Aggregate performance metrics** across all combinations
2. **Identify optimal model-dataset-preset combinations**
3. **Generate scaling analysis** and efficiency insights
4. **Create comprehensive evaluation report**

---

## 🎯 Expected Outputs

### Immediate Deliverables
- **Live performance metrics** for all 216 model-dataset-preset combinations
- **Organized predictions** in multiple organizational schemes  
- **Comprehensive accuracy metrics** across all task types
- **GPU/memory utilization patterns** for optimization insights

### Analysis Insights
- **Best performing models** per task type (coding, reasoning, function calling)
- **Optimal preset configurations** for different use cases
- **Memory efficiency patterns** across model sizes
- **Scaling behavior** from 8B to 14B parameter models
- **Dataset difficulty rankings** based on model performance

---

## 📊 Resource Estimation

### Computational Requirements
- **Total GPU time**: ~6 hours for 216 combinations @ 1.5 min average per combination
- **Storage requirements**: ~5GB for all predictions and metrics
- **Memory peak**: ~50GB GPU memory for largest 14B models

### Sample Size Strategy
- **High-value datasets** (function calling, coding): 200 samples
- **Large datasets** (mmlu, hellaswag): 150 samples  
- **Smaller datasets** (humaneval, mt_bench): Use full available samples
- **Total evaluation samples**: ~2000 samples across all datasets

This plan will provide comprehensive insights into model performance across all task types with accurate, live performance monitoring, enabling data-driven decisions for production deployment.