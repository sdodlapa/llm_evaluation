#!/bin/bash

# Enhanced script to submit all lightweight category evaluation jobs
echo "========================================"
echo "LIGHTWEIGHT LLM EVALUATION BATCH SUBMIT"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p slurm_jobs/logs

# Track submitted jobs
SUBMITTED_JOBS=()

echo "Submitting lightweight evaluation jobs..."

# Submit each category with status tracking
echo "[1/9] Submitting biomedical specialists..."
JOB1=$(sbatch slurm_jobs/biomedical_specialists_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB1)
echo "  → Job ID: $JOB1"

echo "[2/9] Submitting coding specialists..."
JOB2=$(sbatch slurm_jobs/coding_specialists_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB2)
echo "  → Job ID: $JOB2"

echo "[3/9] Submitting efficiency optimized..."
JOB3=$(sbatch slurm_jobs/efficiency_optimized_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB3)
echo "  → Job ID: $JOB3"

echo "[4/9] Submitting general purpose..."
JOB4=$(sbatch slurm_jobs/general_purpose_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB4)
echo "  → Job ID: $JOB4"

echo "[5/9] Submitting mathematical reasoning..."
JOB5=$(sbatch slurm_jobs/mathematical_reasoning_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB5)
echo "  → Job ID: $JOB5"

echo "[6/9] Submitting multimodal processing..."
JOB6=$(sbatch slurm_jobs/multimodal_processing_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB6)
echo "  → Job ID: $JOB6"

echo "[7/9] Submitting safety alignment..."
JOB7=$(sbatch slurm_jobs/safety_alignment_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB7)
echo "  → Job ID: $JOB7"

echo "[8/9] Submitting scientific research..."
JOB8=$(sbatch slurm_jobs/scientific_research_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB8)
echo "  → Job ID: $JOB8"

echo "[9/9] Submitting text geospatial..."
JOB9=$(sbatch slurm_jobs/text_geospatial_multibackend.slurm | grep -o '[0-9]*')
SUBMITTED_JOBS+=($JOB9)
echo "  → Job ID: $JOB9"

echo ""
echo "========================================"
echo "BATCH SUBMISSION COMPLETE"
echo "========================================"
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""

# Save job tracking info
echo "# Lightweight LLM Evaluation Jobs - $(date)" > slurm_jobs/job_tracker.txt
echo "# All jobs use 2 samples, 1-hour timeout, specific working models" >> slurm_jobs/job_tracker.txt
echo "SUBMITTED_JOBS=(${SUBMITTED_JOBS[*]})" >> slurm_jobs/job_tracker.txt
echo "" >> slurm_jobs/job_tracker.txt

echo "Job tracking info saved to: slurm_jobs/job_tracker.txt"
echo ""

echo "Monitor job status with:"
echo "  squeue -u \$USER"
echo "  squeue -j ${SUBMITTED_JOBS[*]// /,}"
echo ""

echo "Check job logs in: slurm_jobs/logs/"
echo "Expected completion time: ~1 hour each"
echo ""

echo "To cancel all jobs if needed:"
echo "  scancel ${SUBMITTED_JOBS[*]}"