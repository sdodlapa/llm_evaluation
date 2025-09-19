#!/bin/bash

# Master script to submit all category evaluation jobs
# This script submits each category as a separate SLURM job and tracks job IDs

echo "========================================="
echo "COMPREHENSIVE CATEGORY EVALUATION SUITE"
echo "Started at: $(date)"
echo "========================================="

# Create logs directory if it doesn't exist
mkdir -p slurm_jobs/logs

# Array to store job IDs
declare -a job_ids=()
declare -a job_names=()

# Function to submit a job and capture job ID
submit_job() {
    local script_name=$1
    local category_name=$2
    
    echo "Submitting $category_name evaluation..."
    
    # Submit job and capture job ID
    job_output=$(sbatch slurm_jobs/$script_name)
    
    if [[ $job_output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        job_id=${BASH_REMATCH[1]}
        job_ids+=($job_id)
        job_names+=("$category_name")
        echo "  ✓ Job ID: $job_id ($category_name)"
    else
        echo "  ✗ Failed to submit $category_name job"
        echo "  Error: $job_output"
    fi
}

# Submit all category jobs
echo "Submitting category evaluation jobs..."
echo ""

# High Priority - Fast models first
submit_job "efficiency_optimized.slurm" "Efficiency Optimized"
submit_job "mathematical_reasoning.slurm" "Mathematical Reasoning"

# Medium Priority - Balanced workload
submit_job "coding_specialists.slurm" "Coding Specialists"
submit_job "general_purpose.slurm" "General Purpose"
submit_job "safety_alignment.slurm" "Safety & Alignment"
submit_job "scientific_research.slurm" "Scientific Research"

# Lower Priority - Resource intensive
submit_job "biomedical_specialists.slurm" "Biomedical Specialists"
submit_job "multimodal_processing.slurm" "Multimodal Processing"

echo ""
echo "========================================="
echo "SUBMISSION SUMMARY"
echo "========================================="
echo "Total jobs submitted: ${#job_ids[@]}"
echo ""

# Display job summary
for i in "${!job_ids[@]}"; do
    echo "Job ${job_ids[$i]}: ${job_names[$i]}"
done

echo ""
echo "========================================="
echo "MONITORING COMMANDS"
echo "========================================="
echo "Monitor all jobs:     squeue -u \$USER"
echo "Check specific job:   squeue -j JOB_ID"
echo "Cancel all jobs:      scancel -u \$USER"
echo ""
echo "Job status monitoring:"
for i in "${!job_ids[@]}"; do
    echo "  ${job_names[$i]}: squeue -j ${job_ids[$i]}"
done

echo ""
echo "========================================="
echo "LOG LOCATIONS"
echo "========================================="
echo "Job logs will be saved to: slurm_jobs/logs/"
echo "Results will be saved to: results/"
echo ""
echo "To check progress:"
echo "  tail -f slurm_jobs/logs/CATEGORY_NAME_*.out"
echo ""

# Create job tracking file
cat > slurm_jobs/job_tracker.txt << EOF
# Comprehensive Category Evaluation - Job Tracker
# Submitted at: $(date)
# Total jobs: ${#job_ids[@]}

EOF

for i in "${!job_ids[@]}"; do
    echo "JOB_ID=${job_ids[$i]} CATEGORY=\"${job_names[$i]}\"" >> slurm_jobs/job_tracker.txt
done

echo "Job tracking information saved to: slurm_jobs/job_tracker.txt"
echo ""

# Provide immediate status check
echo "Current job queue status:"
squeue -u $USER

echo ""
echo "========================================="
echo "NEXT STEPS"
echo "========================================="
echo "1. Monitor job progress: squeue -u \$USER"
echo "2. Check individual logs: tail -f slurm_jobs/logs/CATEGORY_*.out"
echo "3. Wait for all jobs to complete"
echo "4. Review results in respective results/ subdirectories"
echo "5. Run analysis script after all jobs finish"
echo ""
echo "Estimated completion time: 2-6 hours (depending on category)"
echo "========================================="