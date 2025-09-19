#!/bin/bash
#SBATCH --job-name=submit_all_evaluations
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm_jobs/logs/submit_all_%j.out
#SBATCH --error=slurm_jobs/logs/submit_all_%j.err

# Master submission script for all category evaluations
# This submits all 8 category evaluation jobs with multi-backend support

echo "========================================="
echo "SUBMITTING ALL CATEGORY EVALUATIONS"
echo "Started at: $(date)"
echo "========================================="

# Create logs directory if it doesn't exist
mkdir -p slurm_jobs/logs

# Submit all category evaluation jobs
echo "Submitting Coding Specialists evaluation..."
job1=$(sbatch slurm_jobs/coding_specialists_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job1"

echo "Submitting Biomedical Specialists evaluation..."
job2=$(sbatch slurm_jobs/biomedical_specialists_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job2"

echo "Submitting General Purpose evaluation..."
job3=$(sbatch slurm_jobs/general_purpose_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job3"

echo "Submitting Mathematical Reasoning evaluation..."
job4=$(sbatch slurm_jobs/mathematical_reasoning_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job4"

echo "Submitting Multimodal Processing evaluation..."
job5=$(sbatch slurm_jobs/multimodal_processing_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job5"

echo "Submitting Scientific Research evaluation..."
job6=$(sbatch slurm_jobs/scientific_research_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job6"

echo "Submitting Efficiency Optimized evaluation..."
job7=$(sbatch slurm_jobs/efficiency_optimized_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job7"

echo "Submitting Safety Alignment evaluation..."
job8=$(sbatch slurm_jobs/safety_alignment_multibackend.slurm | awk '{print $4}')
echo "Job ID: $job8"

echo "========================================="
echo "ALL JOBS SUBMITTED SUCCESSFULLY"
echo "Job IDs: $job1 $job2 $job3 $job4 $job5 $job6 $job7 $job8"
echo "Total estimated runtime: ~20-25 hours across all categories"
echo "========================================="

echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: slurm_jobs/logs/"
echo "Results will be in: results/<category>_<timestamp>/"

echo "Category breakdown:"
echo "  - Coding Specialists: 3h (models: CodeLlama, WizardCoder, StarCoder)"
echo "  - Biomedical: 2.5h (models: BioGPT, PubMedBERT, BioBERT)"  
echo "  - General Purpose: 4h (models: Qwen, Llama3, Mistral)"
echo "  - Mathematical: 3h (models: MetaMath, WizardMath, DeepSeek-Math)"
echo "  - Multimodal: 3.5h (models: Qwen2-VL, LayoutLM)"
echo "  - Scientific: 3.5h (models: Galactica, SciBERT)"
echo "  - Efficiency: 2.5h (models: MobileVLM, MobileLLaMA)"
echo "  - Safety: 2h (models: RLHF-tuned variants)"

exit 0