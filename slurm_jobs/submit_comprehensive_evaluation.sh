#!/bin/bash

# Comprehensive script to submit all category evaluation jobs with full testing
echo "========================================="
echo "COMPREHENSIVE LLM EVALUATION BATCH SUBMISSION"
echo "Started at: $(date)"
echo "========================================="

echo "Submitting all category evaluation jobs with comprehensive testing:"
echo "- All models in each category"
echo "- All datasets for each category"
echo "- 10 samples per dataset"
echo "- Enhanced resource allocations"
echo ""

# Create logs directory if it doesn't exist
mkdir -p slurm_jobs/logs

# Track submitted jobs
job_ids=()

echo "Submitting jobs..."

# Submit all category jobs
echo "1. Biomedical Specialists..."
job1=$(sbatch slurm_jobs/biomedical_specialists_multibackend.slurm)
job_ids+=("$job1")

echo "2. Coding Specialists..."
job2=$(sbatch slurm_jobs/coding_specialists_multibackend.slurm)
job_ids+=("$job2")

echo "3. Efficiency Optimized..."
job3=$(sbatch slurm_jobs/efficiency_optimized_multibackend.slurm)
job_ids+=("$job3")

echo "4. General Purpose..."
job4=$(sbatch slurm_jobs/general_purpose_multibackend.slurm)
job_ids+=("$job4")

echo "5. Mathematical Reasoning..."
job5=$(sbatch slurm_jobs/mathematical_reasoning_multibackend.slurm)
job_ids+=("$job5")

echo "6. Multimodal Processing..."
job6=$(sbatch slurm_jobs/multimodal_processing_multibackend.slurm)
job_ids+=("$job6")

echo "7. Safety Alignment..."
job7=$(sbatch slurm_jobs/safety_alignment_multibackend.slurm)
job_ids+=("$job7")

echo "8. Scientific Research..."
job8=$(sbatch slurm_jobs/scientific_research_multibackend.slurm)
job_ids+=("$job8")

echo "9. Text Geospatial..."
job9=$(sbatch slurm_jobs/text_geospatial_multibackend.slurm)
job_ids+=("$job9")

echo ""
echo "========================================="
echo "ALL 9 CATEGORY JOBS SUBMITTED SUCCESSFULLY"
echo "========================================="

# Display submitted job IDs
echo "Submitted job IDs:"
for job in "${job_ids[@]}"; do
    echo "  $job"
done

echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER --format=\"%.10i %.20j %.8u %.2t %.10M %.6D %R\""
echo ""
echo "Check logs in: slurm_jobs/logs/"
echo "Results will be saved in: category_evaluation_results/"
echo ""
echo "Estimated total completion time: 4-6 hours"
echo "========================================="