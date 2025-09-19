#!/bin/bash
# SLURM Job Recovery and Monitoring Script
# Generated: September 19, 2025 - 03:29 UTC
# Use this script to resume monitoring from session end state

echo "========================================"
echo "LLM EVALUATION SESSION RECOVERY"
echo "========================================"
echo "Session End Time: September 19, 2025 - 03:29 UTC"
echo "Jobs Submitted: 8 category evaluations + 1 test job"
echo "========================================"

# Job IDs from this session
declare -A SESSION_JOBS=(
    ["1591"]="Test Job - Efficiency Optimized (RUNNING at session end)"
    ["1592"]="Efficiency Optimized"
    ["1593"]="Mathematical Reasoning" 
    ["1594"]="Coding Specialists"
    ["1595"]="General Purpose"
    ["1596"]="Safety & Alignment"
    ["1597"]="Scientific Research"
    ["1598"]="Biomedical Specialists"
    ["1599"]="Multimodal Processing"
)

echo "=== CHECKING SESSION JOBS STATUS ==="
echo ""

# Check status of all session jobs
for job_id in "${!SESSION_JOBS[@]}"; do
    job_status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
    
    if [[ -n "$job_status" ]]; then
        echo "Job $job_id (${SESSION_JOBS[$job_id]}): $job_status"
    else
        # Check if completed
        job_info=$(sacct -j "$job_id" -n -o "State" --parsable2 2>/dev/null | head -1)
        if [[ "$job_info" == "COMPLETED" ]]; then
            echo "Job $job_id (${SESSION_JOBS[$job_id]}): COMPLETED ✅"
        elif [[ "$job_info" == "FAILED" ]] || [[ "$job_info" == "CANCELLED" ]]; then
            echo "Job $job_id (${SESSION_JOBS[$job_id]}): FAILED ❌"
        else
            echo "Job $job_id (${SESSION_JOBS[$job_id]}): NOT FOUND (may have completed)"
        fi
    fi
done

echo ""
echo "=== CURRENT USER JOBS ==="
squeue -u $USER

echo ""
echo "=== RECENT RESULTS DIRECTORIES ==="
find results/ -maxdepth 1 -type d -name "*$(date +%Y%m%d)*" 2>/dev/null | sort

echo ""
echo "=== AVAILABLE LOG FILES ==="
ls -la slurm_jobs/logs/*.out slurm_jobs/logs/*.err 2>/dev/null | tail -10

echo ""
echo "========================================"
echo "QUICK MONITORING COMMANDS"
echo "========================================"
echo "Check all user jobs:        squeue -u \$USER"
echo "Monitor live logs:          tail -f slurm_jobs/logs/CATEGORY_*.out"
echo "Check specific job:         squeue -j JOB_ID"
echo "Job completion history:     sacct -j JOB_ID"
echo "Advanced monitoring:        ./slurm_jobs/submit_all_categories_advanced.sh --status"
echo ""
echo "========================================"
echo "RESULT ANALYSIS COMMANDS"
echo "========================================"
echo "List results:               ls -la results/"
echo "Latest efficiency results:  ls -la results/efficiency_optimized_*/"
echo "Check evaluation logs:      grep -r 'EVALUATION COMPLETE' slurm_jobs/logs/"
echo "Error analysis:             grep -r 'ERROR\\|FAILED' slurm_jobs/logs/"
echo ""
echo "SESSION STATE: $(date)"