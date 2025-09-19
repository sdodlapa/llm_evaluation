#!/bin/bash
#SBATCH --job-name=biomedical_eval
#SBATCH --partition=h100dualflex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=75GB
#SBATCH --time=5:00:00
#SBATCH --output=slurm_jobs/logs/biomedical_%j.out
#SBATCH --error=slurm_jobs/logs/biomedical_%j.err

# Biomedical Specialists Category Evaluation
# Models: 10 (largest category)
# Datasets: 6 (3 primary + 3 optional)
# Priority: HIGH - 5 hour time limit

echo "========================================="
echo "BIOMEDICAL SPECIALISTS EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Load environment
module load python3/2025.1-py312

# Set working directory
cd /home/sdodl001_odu_edu/llm_evaluation

# Create results directory for this job
RESULTS_DIR="results/biomedical_specialists_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sdodl001_odu_edu/llm_evaluation:$PYTHONPATH"

echo "Starting Biomedical Specialists evaluation..."
echo "Results will be saved to: $RESULTS_DIR"

# Run category evaluation using existing category_evaluation.py
crun -p ~/envs/llm_env python category_evaluation.py \
    --category biomedical_specialists \
    --samples 15 \
    --preset balanced \
    --output_dir "$RESULTS_DIR" \
    --verbose

EXIT_CODE=$?

echo "========================================="
echo "BIOMEDICAL SPECIALISTS EVALUATION COMPLETE"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "Results location: $RESULTS_DIR"
echo "========================================="

# Generate summary report
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: $RESULTS_DIR"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE