#!/bin/bash
#SBATCH --job-name=geospatial_eval
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=slurm_jobs/logs/text_geospatial_%j.out
#SBATCH --error=slurm_jobs/logs/text_geospatial_%j.err

# Load environment
module load python3/2025.1-py312

echo "========================================="
echo "TEXT GEOSPATIAL EVALUATION - Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

echo "Starting Text Geospatial evaluation..."
echo "Results will be saved to: results/text_geospatial_$(date +%Y%m%d_%H%M%S)"

# Run evaluation with multi-backend support (geospatial-aware language models)
crun -p ~/envs/llm_env python category_evaluation.py \
    --category text_geospatial \
    --samples 6 \
    --preset balanced

exit_code=$?

echo "========================================="
echo "TEXT GEOSPATIAL EVALUATION COMPLETE"
echo "Exit code: $exit_code"
echo "Finished at: $(date)"
echo "Results location: results/text_geospatial_*"
echo "========================================="

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Check results in: results/text_geospatial_*"
else
    echo "Evaluation failed with exit code: $exit_code"
    echo "Check logs for details"
fi

exit $exit_code