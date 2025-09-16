# Getting Started with LLM Individual Model Evaluation

This guide helps you quickly start evaluating individual LLM models on H100 GPU for agentic system development.

## Quick Start (5 minutes)

### 1. Setup Environment
```bash
cd /home/sdodl001_odu_edu/methyl_savanna/llm_evaluation
./setup_environment.sh
```

### 2. Activate Environment
```bash
conda activate llm_eval
```

### 3. Test Configuration
```bash
python configs/model_configs.py
```

Expected output:
```
=== High Priority Models ===
qwen3_8b: Qwen-3 8B Instruct
  Size: 7.5GB, License: Apache 2.0
  Estimated VRAM: 12.3GB (15.4%)

qwen3_14b: Qwen-3 14B Instruct
  Size: 14.0GB, License: Apache 2.0
  Estimated VRAM: 18.7GB (23.4%)
```

### 4. Quick Test (Single Model)
```bash
python evaluation/run_evaluation.py --quick-test
```

### 5. Full Evaluation (High Priority Models)
```bash
python evaluation/run_evaluation.py --priority-only
```

## What You Get

### Individual Model Results
Each model evaluation produces:
- **Performance metrics**: Memory usage, throughput, latency
- **Agent capabilities**: Function calling, reasoning, instruction following
- **Detailed analysis**: JSON outputs with all test results

### Comparative Analysis
- **Performance comparison** across all tested models
- **Agent capability rankings** 
- **Memory efficiency analysis**
- **Commercial viability assessment** (license-based)

### Recommendations
- **Best overall model** for agentic systems
- **Most memory-efficient** option
- **Best Apache-licensed** model (commercial safe)

## Example Results Structure

```
results/
├── performance/
│   ├── qwen3_8b_results.json
│   ├── qwen3_14b_results.json
│   └── deepseek_coder_16b_results.json
├── comparisons/
│   └── comparison_20250916_143022.json
└── reports/
    └── summary_20250916_143022.md
```

## Next Steps

1. **Review Results**: Check `results/reports/` for summary
2. **Select Best Model**: Based on your priorities (performance vs efficiency vs license)
3. **Design Agent Architecture**: Use findings to plan your agentic system
4. **Implement Agent Framework**: Start with top-performing model

## Advanced Usage

### Test Specific Models
```bash
python evaluation/run_evaluation.py --models qwen3_8b deepseek_coder_16b
```

### Custom Output Directory
```bash
python evaluation/run_evaluation.py --output-dir /path/to/results
```

### Use Custom Cache
```bash
python evaluation/run_evaluation.py --cache-dir /path/to/model/cache
```

## Key Metrics to Watch

### For Agentic Systems:
1. **Function Calling Accuracy** > 0.85
2. **Instruction Following** > 0.80
3. **Multi-turn Coherence** > 0.75
4. **Memory Usage** < 70GB (H100 safety margin)
5. **Throughput** > 50 tokens/second

### Commercial Considerations:
- **Apache 2.0 License**: ✅ Safe for commercial use
- **Meta License**: ⚠️ Read terms carefully
- **Custom Licenses**: ⚠️ Evaluate case-by-case

## Troubleshooting

### GPU Memory Issues
- Reduce `max_model_len` in configs
- Lower `gpu_memory_utilization`
- Enable more aggressive quantization

### Loading Failures
- Check CUDA installation: `nvidia-smi`
- Verify model permissions on Hugging Face
- Check internet connectivity for downloads

### Performance Issues
- Monitor GPU utilization: `nvtop`
- Check for CPU bottlenecks: `htop`
- Verify fast storage for model cache

## Support

- 📖 **Full Documentation**: `LLM_INDIVIDUAL_MODEL_IMPLEMENTATION_PLAN.md`
- 🔧 **Configuration**: `configs/model_configs.py`
- 🏃 **Evaluation Runner**: `evaluation/run_evaluation.py`
- 📊 **Results Analysis**: `results/reports/`