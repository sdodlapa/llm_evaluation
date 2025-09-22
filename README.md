# 🚀 LLM Evaluation Framework

A comprehensive, scalable framework for evaluating Large Language Models across multiple benchmarks and specialized tasks.

## ✨ Key Features

- **61 Models**: Complete model coverage including Qwen, Llama 3.1/3.2, Mixtral MoE, StarCoder2, Gemma 2, and specialized models
- **11 Categories**: Coding, mathematical reasoning, biomedical, multimodal, scientific research, efficiency, general purpose, safety, mixture of experts, reasoning specialized, geospatial
- **26 Datasets**: Coding, mathematics, multimodal, genomics, efficiency, reasoning, QA, function calling, instruction following
- **Real-time Monitoring**: GPU utilization, memory usage, throughput tracking
- **Specialized Models**: Math, coding, genomics, efficiency-optimized variants, mixture of experts, reasoning-distilled models
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
│   └── model_configs.py       # Model definitions (61 models)
├── 📁 evaluation_data/         # Dataset storage
├── 📁 results/                 # Evaluation outputs
├── 📁 docs/                    # Documentation
└── 📁 archive/                 # Historical/redundant files
```

## 🤖 Model Coverage (61 Models)

### 🆕 Recently Added Models (Sept 2025)
| Model | Size | License | Category | Key Capabilities |
|-------|------|---------|----------|------------------|
| **Llama 3.1 70B** | 140GB | Llama 3.1 | General Purpose | Large-scale reasoning, 128K context |
| **Mixtral 8x7B** | 93GB | Apache 2.0 | Mixture of Experts | Efficient inference, multilingual |
| **DeepSeek-R1-Distill 70B** | 140GB | MIT | Reasoning Specialized | Advanced reasoning, distilled knowledge |
| **StarCoder2 15B** | 32GB | BigCode | Coding Specialists | Repository-level code understanding |
| **Gemma 2 27B** | 54GB | Gemma Terms | General Purpose | Instruction following, balanced performance |
| **InternLM2 20B** | 40GB | Apache 2.0 | General Purpose | Multilingual chat, research-grade |
| **Llama 3.2 Vision 90B** | 180GB | Llama 3.2 | Multimodal | Large-scale vision-language understanding |

### 📋 Model Categories (11 Total)
1. **Coding Specialists** (6 models): Advanced code generation and understanding
2. **Mathematical Reasoning** (5 models): Specialized mathematical problem solving  
3. **Biomedical Specialists** (10 models): Medical and biological domain expertise
4. **Multimodal Processing** (8 models): Vision-language understanding
5. **Scientific Research** (3 models): Academic and research applications
6. **Efficiency Optimized** (3 models): Resource-constrained deployments
7. **General Purpose** (10 models): Versatile language understanding
8. **Safety Alignment** (3 models): Responsible AI and safety evaluation
9. **🆕 Mixture of Experts** (1 model): Efficient sparse expert routing
10. **🆕 Reasoning Specialized** (1 model): Advanced logical reasoning
11. **Text Geospatial** (4 models): Location and spatial understanding

### 🎯 Usage Examples
```bash
# Test new large models
python category_evaluation.py --category general_purpose --samples 5 --preset balanced

# Test mixture of experts
python category_evaluation.py --category mixture_of_experts --dry-run

# Test coding with StarCoder2
python category_evaluation.py --model starcoder2_15b --dataset humaneval --samples 10
```

## 📊 Legacy Model Support

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