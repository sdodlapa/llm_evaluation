#!/bin/bash
#SBATCH --job-name=fixed_comprehensive_eval
#SBATCH --output=slurm_jobs/logs/fixed_comprehensive_%j.out
#SBATCH --error=slurm_jobs/logs/fixed_comprehensive_%j.err
#SBATCH --partition=aimlab
#SBATCH --account=aimlab
#SBATCH --qos=aimlab_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H100:4
#SBATCH --mem=400G
#SBATCH --time=02:00:00

echo "============================================"
echo "Fixed Comprehensive Model Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "============================================"

# Setup environment using our script
echo "Setting up environment..."
source /home/sdodl001_odu_edu/llm_evaluation/scripts/slurm_env_setup.sh

# Additional environment verification
echo "Environment verification:"
echo "  HF_TOKEN set: $([ -n "$HF_TOKEN" ] && echo 'Yes' || echo 'No')"
echo "  Working directory: $(pwd)"
echo "  Python path: $PYTHONPATH"

# Load conda environment
echo "Loading conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env

echo "Environment setup complete"
echo "============================================"

# Run the fixed evaluation
echo "Starting fixed evaluation..."
python run_fixed_evaluation.py

EXIT_CODE=$?

echo "============================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
fi

echo "============================================"

exit $EXIT_CODE