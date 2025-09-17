````markdown
# LLM Evaluation Framework

A comprehensive evaluation framework for Large Language Models, optimized for NVIDIA H100 GPUs and focused on Qwen model families.

## 🎯 Overview

This framework provides production-ready LLM evaluation capabilities with:
- **Multi-model support**: 6 Qwen variants across different sizes and versions
- **H100 optimization**: Advanced GPU utilization with 119+ tokens/sec performance
- **Comprehensive datasets**: 12+ evaluation datasets covering coding, reasoning, and function calling
- **Modular architecture**: Extensible design for easy addition of new models and benchmarks
- **Production ready**: Robust error handling, comprehensive logging, and automated reporting

## 🏗️ Architecture

### Core Components
```
llm_evaluation/
├── configs/                      # Model configurations and presets
├── models/                       # Model implementations with registry pattern
├── evaluation/                   # Core evaluation pipeline
├── evaluation_data/              # Standardized dataset storage
└── results/                      # Organized evaluation outputs
```

### Design Principles
- **Modularity**: Each component has single responsibility
- **Extensibility**: Easy addition of new models via registry pattern
- **Performance**: Optimized for H100 with Flash Attention and CUDA graphs
- **Reliability**: Comprehensive error handling and resource management

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd llm_evaluation

# Install dependencies
pip install -r requirements.txt

# Download recommended datasets
python manage_datasets.py --download-recommended
```

### Basic Usage
```bash
# Single model evaluation
python evaluation/run_evaluation.py --models qwen3_8b --preset balanced

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

### Coding & Programming
- **HumanEval**: 164 hand-written programming problems
- **MBPP**: 974 mostly basic Python programming tasks
- **CodeT5**: Multi-language code understanding and generation

### Reasoning & Problem Solving
- **GSM8K**: 1,319 grade school math word problems
- **ARC-Challenge**: 1,172 science reasoning questions
- **HellaSwag**: 10,042 commonsense reasoning scenarios

### Function Calling & Agents
- **BFCL**: Berkeley Function Calling Leaderboard
- **ToolLLaMA**: Multi-tool coordination and usage

### Instruction Following
- **AlpacaEval**: 805 instruction-following evaluations
- **MT-Bench**: 160 multi-turn conversation scenarios

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