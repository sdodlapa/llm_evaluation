#!/bin/bash

# Create timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_PREFIX="logs/qwen3_8b_balanced_${TIMESTAMP}"

# Ensure logs directory exists
mkdir -p logs

echo "Starting Qwen-3 8B balanced preset evaluation..."
echo "Logs will be saved to:"
echo "  Output: ${LOG_PREFIX}.out"
echo "  Errors: ${LOG_PREFIX}.err"
echo "  Combined: ${LOG_PREFIX}.log"
echo "======================================"

# Run the evaluation with output redirection
{
    echo "Job Start Time: $(date)"
    echo "Working Directory: $(pwd)"
    echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
    echo "Command: module load python3 && crun -p ~/envs/llm_env python evaluation/run_evaluation.py --model qwen3_8b --preset balanced"
    echo "======================================"
    
    module load python3 && crun -p ~/envs/llm_env python evaluation/run_evaluation.py --model qwen3_8b --preset balanced
    
    exit_code=$?
    echo "======================================"
    echo "Job End Time: $(date)"
    echo "Exit Code: $exit_code"
    
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: Evaluation completed successfully"
    else
        echo "ERROR: Evaluation failed with exit code $exit_code"
    fi
    
    exit $exit_code
    
} > "${LOG_PREFIX}.log" 2>&1

# Also create separate .out and .err files by splitting the combined log
grep -v "ERROR\|WARN\|Exception\|Traceback" "${LOG_PREFIX}.log" > "${LOG_PREFIX}.out" 2>/dev/null || cp "${LOG_PREFIX}.log" "${LOG_PREFIX}.out"
grep -E "ERROR|WARN|Exception|Traceback" "${LOG_PREFIX}.log" > "${LOG_PREFIX}.err" 2>/dev/null || touch "${LOG_PREFIX}.err"

echo ""
echo "Evaluation completed. Check log files:"
echo "  Combined log: ${LOG_PREFIX}.log"
echo "  Output log: ${LOG_PREFIX}.out" 
echo "  Error log: ${LOG_PREFIX}.err"
echo ""
echo "To monitor real-time output:"
echo "  tail -f ${LOG_PREFIX}.log"