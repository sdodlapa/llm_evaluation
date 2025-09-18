# 🚀 LLM Evaluation Framework

A comprehensive, scalable framework for evaluating Large Language Models across multiple benchmarks and specialized tasks.

## ✨ Key Features

- **22+ Models**: Complete Qwen series + strategic alternatives (Llama, Phi, DeepSeek)
- **26 Datasets**: Coding, mathematics, multimodal, genomics, efficiency, reasoning, QA, function calling, instruction following
- **Real-time Monitoring**: GPU utilization, memory usage, throughput tracking
- **Specialized Models**: Math, coding, genomics, efficiency-optimized variants
- **Clean Architecture**: Consolidated, maintainable, scalable structure
- **Smart Recommendations**: Enhanced dataset selection system

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Evaluation
```bash
# Quick validation (5 samples)
python evaluation/run_evaluation.py --model qwen3_8b --dataset humaneval --samples 5

# Comprehensive evaluation (200 samples)
python evaluation/run_evaluation.py --model qwen3_8b --dataset humaneval --samples 200

# Multiple models and datasets
python evaluation/run_evaluation.py --model qwen3_8b,qwen3_14b --dataset humaneval,gsm8k
```

### 3. Specialized Evaluations
```bash
# Math specialist validation
python evaluation/run_evaluation.py --model qwen25_math_7b --dataset gsm8k

# Coding specialist validation  
python evaluation/run_evaluation.py --model qwen3_coder_30b --dataset humaneval

# Efficiency comparison
python evaluation/run_evaluation.py --model qwen25_0_5b,qwen25_3b --dataset humaneval
```

### 4. Dataset Summary
```bash
# View all datasets
python show_datasets.py

# View specific category
python show_datasets.py coding
python show_datasets.py mathematics
python show_datasets.py multimodal
```

## 📊 Dataset Overview

**26 Total Datasets** across 9 categories with **614,397+ evaluation samples**

- **📋 Complete Catalog**: See [DATASET_SUMMARY_TABLE.md](DATASET_SUMMARY_TABLE.md)
- **🔢 7 Ready Datasets**: 13,777 samples available immediately
- **⏳ 19 Pending Datasets**: 600,620 samples (infrastructure ready)

| Category | Datasets | Ready | Samples |
|----------|----------|-------|---------|
| **Coding** | 5 | 3 | 19,664 |
| **Mathematics** | 3 | 0 | 37,577 |
| **Multimodal** | 3 | 0 | 481,234 |
| **Reasoning** | 4 | 2 | 18,128 |
| **Genomics** | 3 | 0 | 34,000 |
| **Function Calling** | 2 | 0 | 5,000 |
| **QA** | 2 | 1 | 15,214 |
| **Instruction Following** | 2 | 1 | 580 |
| **Efficiency** | 2 | 0 | 3,000 |

## 📁 Project Structure

```
llm_evaluation/
├── 📁 evaluation/              # Core evaluation system
│   ├── run_evaluation.py       # Main entry point
│   ├── dataset_manager.py      # Dataset handling (26 datasets)
│   ├── performance_monitor.py  # Real-time monitoring
│   ├── comprehensive_runner.py # Advanced evaluation orchestrator
│   └── metrics.py             # Evaluation metrics
├── 📁 models/                  # Model implementations
│   ├── registry.py            # Model discovery/loading
│   ├── base_model.py          # Abstract base class
│   └── qwen_implementation.py # Qwen-specific implementation
├── 📁 configs/                 # Configuration management
│   └── model_configs.py       # Model definitions (22+ models)
├── 📁 evaluation_data/         # Dataset storage
├── 📁 results/                 # Evaluation outputs
├── 📁 docs/                    # Documentation
└── 📁 archive/                 # Historical/redundant files
```

# Multi-model comparison
python evaluation/run_evaluation.py --models qwen3_8b,qwen3_14b --preset performance

# Specific datasets
python evaluation/run_evaluation.py --datasets humaneval,mbpp --sample-limit 50
```

## 📊 Supported Models

| Model | Size | Context | Optimization | Status |
|-------|------|---------|--------------|--------|
| Qwen-3 8B | 7.5GB | 128K | H100 optimized | ✅ Production |
| Qwen-3 14B | 14GB | 128K | AWQ quantized | ✅ Production |
| Qwen 8B | 8GB | 32K | Framework validated | ✅ Ready |
| Qwen 14B | 14GB | 32K | Framework validated | ✅ Ready |
| Qwen 2.5 8B | 8GB | 32K | Framework validated | ✅ Ready |
| Qwen 2.5 14B | 14GB | 32K | Framework validated | ✅ Ready |

## 📚 Evaluation Datasets

**26 Comprehensive Datasets** - See [DATASET_SUMMARY_TABLE.md](DATASET_SUMMARY_TABLE.md) for complete details

### ✅ Ready for Evaluation (7 datasets)
- **HumanEval**: 164 Python programming problems
- **MBPP**: 500 Python programming tasks  
- **BigCodeBench**: 500 comprehensive coding challenges
- **GSM8K**: 1,319 grade school math problems
- **ARC-Challenge**: 1,172 science reasoning questions
- **HellaSwag**: 10,042 commonsense reasoning scenarios
- **MT-Bench**: 80 multi-turn conversation evaluations

### 🔄 Implementation Ready (19 datasets)
**Coding**: CodeContests, APPS  
**Mathematics**: MATH Competition, MathQA, AIME  
**Multimodal**: ScienceQA, VQA v2.0, ChartQA  
**Genomics**: Genomics Benchmark, Protein Sequences, BioASQ  
**Function Calling**: BFCL, ToolLLaMA  
**Efficiency**: Efficiency Bench, Mobile Benchmark  
**QA/Reasoning**: MMLU, Math, Winogrande, IFEval  

### 🎯 Usage Examples
```bash
# All ready datasets
python show_datasets.py

# Category-specific
python show_datasets.py coding
python show_datasets.py mathematics

# Smart recommendations
from evaluation.dataset_manager import EnhancedDatasetManager
dm = EnhancedDatasetManager()
recommended = dm.get_recommended_datasets()
```

## ⚙️ Configuration Presets

### Performance Preset
- **Focus**: Maximum throughput (119+ tok/s)
- **GPU Memory**: 90% utilization
- **Features**: Flash Attention, CUDA graphs, prefix caching

### Balanced Preset (Default)
- **Focus**: Optimal speed/memory balance
- **GPU Memory**: 85% utilization  
- **Features**: Prefix caching, optimized batch sizes

### Memory Optimized Preset
- **Focus**: Minimal memory footprint
- **GPU Memory**: 70% utilization
- **Features**: Conservative settings, smaller batches

## 📈 Performance Results

### Qwen-3 8B (H100 Optimized)
- **Throughput**: 119.3 tokens/second
- **Memory Usage**: 14.25GB (18% H100 utilization)
- **Dataset Performance**: GSM8K validated, coding pipelines operational

### Qwen-3 14B (AWQ-Marlin)
- **Throughput**: 126.70 tokens/second (with quantization)
- **Memory Usage**: 9.38GB (66% memory savings)
- **Optimization**: AWQ-Marlin kernel for optimal performance

## 🔧 Advanced Features

### vLLM Integration
- **Flash Attention**: Optimized attention mechanism
- **Block Management**: Optimized for block_size=16
- **Quantization**: AWQ-Marlin support for memory efficiency
- **Prefix Caching**: Accelerated repeated evaluations

### Infrastructure Hardening
- **Compatibility Fixes**: Resolved vLLM v0.10.2 compatibility issues
- **Configuration Validation**: 18 model/preset combinations tested
- **Error Recovery**: Graceful degradation and detailed logging
- **Resource Management**: Automatic cleanup and memory optimization

## 📖 Documentation

- **[QWEN_EVALUATION_TRACKER.md](QWEN_EVALUATION_TRACKER.md)**: Comprehensive evaluation results and progress tracking
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)**: Complete documentation navigation
- **[manage_datasets.py](manage_datasets.py)**: Dataset management CLI tool

## 🧪 Testing & Validation

The framework includes comprehensive testing:
- **Configuration Testing**: All 18 model/preset combinations validated
- **Pipeline Testing**: Complete evaluation workflows verified
- **Performance Testing**: Throughput and memory benchmarks confirmed
- **Infrastructure Testing**: vLLM compatibility and optimization validated

## 🤝 Contributing

### Adding New Models
1. Create model implementation in `models/`
2. Register in `models/registry.py`
3. Add configuration in `configs/model_configs.py`
4. Test with validation scripts

### Adding New Datasets
1. Place dataset in appropriate `evaluation_data/` subdirectory
2. Update `evaluation/dataset_manager.py`
3. Add metadata file with dataset information
4. Validate integration with test scripts

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Qwen team for excellent model implementations
- vLLM team for high-performance inference engine
- HuggingFace for dataset hosting and model distribution
````