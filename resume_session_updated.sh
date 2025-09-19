#!/bin/bash
# Updated Job Recovery Script after Preset Fixes
# Generated: September 19, 2025 - 03:35 UTC

echo "========================================"
echo "LLM EVALUATION SESSION RECOVERY (UPDATED)"
echo "========================================"
echo "CRITICAL UPDATE: Preset issues found and fixed!"
echo "Some jobs were resubmitted with corrected presets."
echo "========================================"

# Updated job IDs after fixes
declare -A SESSION_JOBS=(
    ["1591"]="Test Job - Efficiency Optimized (RUNNING at session end)"
    ["1592"]="Efficiency Optimized" 
    ["1602"]="Mathematical Reasoning (RESUBMITTED - was 1593)"
    ["1594"]="Coding Specialists"
    ["1595"]="General Purpose (COMPLETED)"
    ["1600"]="Safety & Alignment (RESUBMITTED - was 1596)"
    ["1603"]="Scientific Research (RESUBMITTED - was 1597)"
    ["1598"]="Biomedical Specialists"
    ["1601"]="Multimodal Processing (RESUBMITTED - was 1599)"
)

echo "=== CHECKING CURRENT SESSION JOBS ==="
echo ""

# Check status of all current session jobs
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
echo "=== PRESET ISSUE RESOLUTION SUMMARY ==="
echo "❌ FAILED (invalid presets): 1596 (safety), 1599 (multimodal)"
echo "❌ CANCELLED (invalid presets): 1593 (math), 1597 (scientific)"  
echo "✅ RESUBMITTED with valid presets:"
echo "   1596 → 1600 (Safety & Alignment)"
echo "   1599 → 1601 (Multimodal Processing)"
echo "   1593 → 1602 (Mathematical Reasoning)"
echo "   1597 → 1603 (Scientific Research)"
echo ""
echo "✅ COMPLETED: 1595 (General Purpose)"
echo "✅ RUNNING: 1591, 1592, 1594, 1598, 1600, 1601"
echo "⏳ PENDING: 1602, 1603"

echo ""
echo "=== CURRENT USER JOBS ==="
squeue -u $USER

echo ""
echo "=== AVAILABLE RESULTS ===" 
find results/ -maxdepth 1 -type d -name "*$(date +%Y%m%d)*" 2>/dev/null | sort

echo ""
echo "========================================"
echo "PRESET VALIDATION"
echo "========================================"
echo "All SLURM scripts now use ONLY valid presets:"
echo "- balanced (most jobs)"
echo "- performance (efficiency_optimized)"
echo "- memory_optimized (available but not used)"
echo ""
echo "Invalid presets removed:"
echo "- analytical → balanced"
echo "- multimodal → balanced" 
echo "- ethical → balanced"
echo "- scientific → balanced"
echo ""
echo "SESSION STATE: CORRECTED - $(date)"