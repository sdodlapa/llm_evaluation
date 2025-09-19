#!/bin/bash
#SBATCH --job-name=backend_test
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_jobs/logs/backend_test_%j.out
#SBATCH --error=slurm_jobs/logs/backend_test_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "BACKEND FIXES TEST - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Testing backend fixes with small biomedical model subset..."
echo "This should resolve vLLM import and TransformersWrapper errors"

# Test with just 2 models, 2 samples each for quick validation
crun -p ~/envs/llm_env python category_evaluation.py \
    --category biomedical_specialists \
    --samples 2 \
    --preset balanced \
    --exclude-models clinical_camel_70b,medalpaca_13b,biogpt_large,pubmedbert_large

exit_code=$?

echo "========================================="
echo "BACKEND FIXES TEST COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Testing vLLM import and TransformersWrapper fixes"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ Backend fixes test completed successfully"
    echo "vLLM import and TransformersWrapper errors should be resolved"
else
    echo "❌ Backend fixes test failed with exit code: $exit_code"
    echo "Check logs for remaining issues"
fi

exit $exit_code