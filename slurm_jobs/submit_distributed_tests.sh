#!/bin/bash

# Distributed Engine Testing - Submission Script
# This script submits all distributed engine tests in the correct order

echo "=========================================="
echo "Distributed Engine Testing - Job Submission"
echo "Start Time: $(date)"
echo "=========================================="

# Check if we're in the correct directory
if [ ! -f "distributed_validation.slurm" ]; then
    echo "❌ Error: Must run from slurm_jobs directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to submit job and capture job ID
submit_job() {
    local job_file=$1
    local job_name=$2
    
    echo ""
    echo "Submitting $job_name..."
    echo "File: $job_file"
    
    if [ ! -f "$job_file" ]; then
        echo "❌ Error: Job file $job_file not found"
        return 1
    fi
    
    # Submit job and capture output
    submit_output=$(sbatch "$job_file" 2>&1)
    submit_status=$?
    
    if [ $submit_status -eq 0 ]; then
        # Extract job ID from output
        job_id=$(echo "$submit_output" | grep -o '[0-9]\+')
        echo "✅ $job_name submitted successfully"
        echo "   Job ID: $job_id"
        echo "   Command: sbatch $job_file"
        
        # Add to job tracker
        echo "$(date): $job_name (ID: $job_id) - $job_file" >> job_tracker.txt
        
        echo "$job_id"  # Return job ID
    else
        echo "❌ Failed to submit $job_name"
        echo "   Error: $submit_output"
        return 1
    fi
}

# Function to check job status
check_job_status() {
    local job_id=$1
    local job_name=$2
    
    if [ -n "$job_id" ]; then
        status=$(squeue -j "$job_id" --noheader --format="%T" 2>/dev/null)
        if [ -n "$status" ]; then
            echo "   Status: $status"
        else
            echo "   Status: COMPLETED or NOT FOUND"
        fi
    fi
}

echo ""
echo "Starting distributed engine test submissions..."

# Step 1: Submit Validation Job (prerequisite)
echo ""
echo "=========================================="
echo "STEP 1: Basic Validation"
echo "=========================================="

VALIDATION_JOB_ID=$(submit_job "distributed_validation.slurm" "Distributed Engine Validation")
validation_status=$?

if [ $validation_status -ne 0 ]; then
    echo "❌ Critical Error: Validation job submission failed"
    echo "   Cannot proceed with other tests"
    exit 1
fi

echo ""
echo "Validation job submitted. Waiting 30 seconds before checking status..."
sleep 30

check_job_status "$VALIDATION_JOB_ID" "Validation"

# Step 2: Submit Large Model Testing (depends on validation)
echo ""
echo "=========================================="
echo "STEP 2: Large Model Testing"
echo "=========================================="

echo "Submitting large model testing job..."
echo "Note: This job tests 4-GPU tensor parallelism with 30B-70B models"

LARGE_MODEL_JOB_ID=$(submit_job "distributed_large_models.slurm" "Large Model Distributed Testing")
large_model_status=$?

if [ $large_model_status -eq 0 ]; then
    check_job_status "$LARGE_MODEL_JOB_ID" "Large Model Testing"
else
    echo "⚠️  Warning: Large model testing job submission failed"
fi

# Step 3: Submit Hybrid Parallelism Testing (advanced)
echo ""
echo "=========================================="
echo "STEP 3: Advanced Hybrid Parallelism"
echo "=========================================="

echo "Submitting hybrid parallelism testing job..."
echo "Note: This job tests 8-GPU hybrid tensor+pipeline parallelism"

HYBRID_JOB_ID=$(submit_job "distributed_hybrid_parallelism.slurm" "Hybrid Parallelism Testing")
hybrid_status=$?

if [ $hybrid_status -eq 0 ]; then
    check_job_status "$HYBRID_JOB_ID" "Hybrid Parallelism"
else
    echo "⚠️  Warning: Hybrid parallelism testing job submission failed"
fi

# Summary
echo ""
echo "=========================================="
echo "SUBMISSION SUMMARY"
echo "=========================================="
echo "Submitted Jobs:"

if [ $validation_status -eq 0 ]; then
    echo "✅ Validation: Job ID $VALIDATION_JOB_ID"
else
    echo "❌ Validation: FAILED"
fi

if [ $large_model_status -eq 0 ]; then
    echo "✅ Large Model Testing: Job ID $LARGE_MODEL_JOB_ID"
else
    echo "❌ Large Model Testing: FAILED"
fi

if [ $hybrid_status -eq 0 ]; then
    echo "✅ Hybrid Parallelism: Job ID $HYBRID_JOB_ID"
else
    echo "❌ Hybrid Parallelism: FAILED"
fi

# Show current queue status
echo ""
echo "Current job queue status:"
squeue -u $USER --format="%.10i %.20j %.10T %.15R %.8M"

# Instructions
echo ""
echo "=========================================="
echo "MONITORING INSTRUCTIONS"
echo "=========================================="
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $VALIDATION_JOB_ID,$LARGE_MODEL_JOB_ID,$HYBRID_JOB_ID"
echo ""
echo "Check logs in real-time:"
echo "  tail -f logs/distributed_validation_${VALIDATION_JOB_ID}.out"
echo "  tail -f logs/distributed_large_models_${LARGE_MODEL_JOB_ID}.out"
echo "  tail -f logs/distributed_hybrid_parallelism_${HYBRID_JOB_ID}.out"
echo ""
echo "Check results:"
echo "  ls -la distributed_test_results/"
echo ""
echo "Cancel jobs if needed:"
echo "  scancel $VALIDATION_JOB_ID $LARGE_MODEL_JOB_ID $HYBRID_JOB_ID"

# Calculate total estimated time
echo ""
echo "=========================================="
echo "ESTIMATED COMPLETION TIMES"
echo "=========================================="
echo "Validation (2 GPUs, 4 hours): ~4 hours"
echo "Large Models (4 GPUs, 8 hours): ~8 hours" 
echo "Hybrid Parallelism (8 GPUs, 12 hours): ~12 hours"
echo ""
echo "Total estimated time: ~12 hours (jobs may run in parallel)"

echo ""
echo "=========================================="
echo "Job submission completed at: $(date)"
echo "=========================================="