#!/bin/bash
"""
Master SLURM Job Orchestration Script
Submits all category evaluation jobs with dependency management and monitoring.
"""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SLURM_DIR="$PROJECT_DIR/slurm_jobs"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_DIR="$SLURM_DIR/logs"

# Ensure directories exist
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Job categories with priority and dependencies
declare -A CATEGORIES=(
    ["efficiency_optimized"]="HIGH:none"
    ["safety_alignment"]="HIGH:none"
    ["mathematical_reasoning"]="HIGH:none"
    ["scientific_research"]="HIGH:none"
    ["coding_specialists"]="HIGH:none"
    ["multimodal_processing"]="HIGH:none"
    ["biomedical_specialists"]="HIGH:none"
    ["general_purpose"]="MEDIUM:none"
)

# Track submitted jobs
declare -A SUBMITTED_JOBS=()
MASTER_LOG="$LOG_DIR/master_submission_$(date +%Y%m%d_%H%M%S).log"

# Function to check if SLURM is available
check_slurm() {
    if ! command -v sbatch &> /dev/null; then
        log_error "SLURM not available. Please ensure you're on a SLURM cluster."
        return 1
    fi
    
    if ! command -v squeue &> /dev/null; then
        log_error "squeue command not available."
        return 1
    fi
    
    log_info "SLURM environment detected"
    return 0
}

# Function to validate SLURM scripts
validate_slurm_scripts() {
    log_info "Validating SLURM scripts..."
    
    local validation_errors=0
    
    for category in "${!CATEGORIES[@]}"; do
        local script_file="$SLURM_DIR/${category}.slurm"
        
        if [[ ! -f "$script_file" ]]; then
            log_error "SLURM script not found: $script_file"
            ((validation_errors++))
        else
            # Check if script has required SBATCH directives
            if ! grep -q "#SBATCH --job-name" "$script_file"; then
                log_error "Missing job-name directive in: $script_file"
                ((validation_errors++))
            fi
            
            if ! grep -q "#SBATCH --gres=gpu" "$script_file"; then
                log_error "Missing GPU resource directive in: $script_file"
                ((validation_errors++))
            fi
        fi
    done
    
    if [[ $validation_errors -eq 0 ]]; then
        log_success "All SLURM scripts validated successfully"
        return 0
    else
        log_error "Found $validation_errors validation errors"
        return 1
    fi
}

# Function to submit a single category job
submit_category_job() {
    local category="$1"
    local script_file="$SLURM_DIR/${category}.slurm"
    
    log_info "Submitting job for category: $category"
    
    # Submit job and capture job ID
    local submit_output
    submit_output=$(sbatch "$script_file" 2>&1)
    local submit_status=$?
    
    if [[ $submit_status -eq 0 ]]; then
        # Extract job ID from sbatch output
        local job_id
        job_id=$(echo "$submit_output" | grep -oE '[0-9]+')
        
        if [[ -n "$job_id" ]]; then
            SUBMITTED_JOBS["$category"]="$job_id"
            log_success "Submitted $category job with ID: $job_id"
            
            # Log submission details
            echo "$(date '+%Y-%m-%d %H:%M:%S') - $category: Job ID $job_id submitted" >> "$MASTER_LOG"
            
            return 0
        else
            log_error "Could not extract job ID from sbatch output: $submit_output"
            return 1
        fi
    else
        log_error "Failed to submit $category job: $submit_output"
        return 1
    fi
}

# Function to submit all jobs
submit_all_jobs() {
    log_info "Starting batch submission of all category jobs..."
    
    local submitted_count=0
    local failed_count=0
    
    # Submit high priority jobs first
    log_info "Submitting HIGH priority jobs..."
    for category in "${!CATEGORIES[@]}"; do
        local priority_deps="${CATEGORIES[$category]}"
        local priority="${priority_deps%%:*}"
        
        if [[ "$priority" == "HIGH" ]]; then
            if submit_category_job "$category"; then
                ((submitted_count++))
                sleep 2  # Brief delay between submissions
            else
                ((failed_count++))
            fi
        fi
    done
    
    # Submit medium priority jobs
    log_info "Submitting MEDIUM priority jobs..."
    for category in "${!CATEGORIES[@]}"; do
        local priority_deps="${CATEGORIES[$category]}"
        local priority="${priority_deps%%:*}"
        
        if [[ "$priority" == "MEDIUM" ]]; then
            if submit_category_job "$category"; then
                ((submitted_count++))
                sleep 2
            else
                ((failed_count++))
            fi
        fi
    done
    
    log_info "Submission complete: $submitted_count successful, $failed_count failed"
    
    return $failed_count
}

# Function to monitor job status
monitor_jobs() {
    log_info "Monitoring submitted jobs..."
    
    local total_jobs=${#SUBMITTED_JOBS[@]}
    local completed_jobs=0
    local failed_jobs=0
    
    echo "Job Status Dashboard:"
    echo "===================="
    
    while [[ $completed_jobs -lt $total_jobs ]]; do
        local running_jobs=0
        local pending_jobs=0
        
        echo -e "\n$(date '+%Y-%m-%d %H:%M:%S') - Job Status Update:"
        
        for category in "${!SUBMITTED_JOBS[@]}"; do
            local job_id="${SUBMITTED_JOBS[$category]}"
            
            # Get job status from squeue
            local job_status
            job_status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
            
            if [[ -z "$job_status" ]]; then
                # Job not in queue, check if it completed successfully
                local job_info
                job_info=$(sacct -j "$job_id" -n -o "State" --parsable2 2>/dev/null | head -1)
                
                if [[ "$job_info" == "COMPLETED" ]]; then
                    echo -e "  ✓ $category (Job $job_id): ${GREEN}COMPLETED${NC}"
                    ((completed_jobs++))
                elif [[ "$job_info" == "FAILED" ]] || [[ "$job_info" == "CANCELLED" ]]; then
                    echo -e "  ✗ $category (Job $job_id): ${RED}FAILED${NC}"
                    ((failed_jobs++))
                    ((completed_jobs++))
                else
                    echo -e "  ? $category (Job $job_id): ${YELLOW}UNKNOWN${NC}"
                fi
            else
                case "$job_status" in
                    "RUNNING")
                        echo -e "  ⚡ $category (Job $job_id): ${BLUE}RUNNING${NC}"
                        ((running_jobs++))
                        ;;
                    "PENDING")
                        echo -e "  ⏳ $category (Job $job_id): ${YELLOW}PENDING${NC}"
                        ((pending_jobs++))
                        ;;
                    *)
                        echo -e "  ⚠ $category (Job $job_id): ${YELLOW}$job_status${NC}"
                        ;;
                esac
            fi
        done
        
        echo "Summary: $completed_jobs completed, $running_jobs running, $pending_jobs pending, $failed_jobs failed"
        
        if [[ $completed_jobs -lt $total_jobs ]]; then
            sleep 30  # Check every 30 seconds
        fi
    done
    
    log_success "All jobs completed. $completed_jobs total, $failed_jobs failed."
    
    return $failed_jobs
}

# Function to generate master summary
generate_master_summary() {
    log_info "Generating master evaluation summary..."
    
    local summary_file="$RESULTS_DIR/master_evaluation_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$summary_file" << EOF
MASTER LLM EVALUATION SUMMARY
=============================
Generated: $(date '+%Y-%m-%d %H:%M:%S')

SUBMITTED JOBS:
EOF
    
    for category in "${!SUBMITTED_JOBS[@]}"; do
        local job_id="${SUBMITTED_JOBS[$category]}"
        echo "  • $category: Job ID $job_id" >> "$summary_file"
    done
    
    cat >> "$summary_file" << EOF

RESULTS DIRECTORIES:
$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "*$(date +%Y%m%d)*" | sort)

LOG FILES:
$(find "$LOG_DIR" -name "*.out" -o -name "*.err" | sort)

NEXT STEPS:
1. Check individual category results in: $RESULTS_DIR
2. Review job logs in: $LOG_DIR
3. Generate consolidated analysis using the individual category reports

Master submission log: $MASTER_LOG
EOF
    
    log_success "Master summary saved to: $summary_file"
    
    # Display summary to console
    cat "$summary_file"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --submit-all        Submit all category evaluation jobs
    --monitor-only      Monitor previously submitted jobs
    --status            Show current job status
    --validate          Validate SLURM scripts only
    --help              Show this help message

EXAMPLES:
    $0 --submit-all     # Submit all evaluation jobs
    $0 --monitor-only   # Monitor existing jobs
    $0 --status         # Quick status check
    $0 --validate       # Validate scripts without submitting

CATEGORIES:
EOF
    
    for category in "${!CATEGORIES[@]}"; do
        local priority_deps="${CATEGORIES[$category]}"
        local priority="${priority_deps%%:*}"
        echo "    • $category ($priority priority)"
    done
}

# Function to show current status
show_status() {
    log_info "Checking current job status..."
    
    # Check for running jobs
    local my_jobs
    my_jobs=$(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null)
    
    if [[ -n "$my_jobs" ]]; then
        echo "Current jobs:"
        echo "$my_jobs" | while read -r job_id job_name job_state; do
            if [[ "$job_name" =~ eval$ ]]; then
                echo -e "  Job $job_id ($job_name): $job_state"
            fi
        done
    else
        echo "No active jobs found."
    fi
    
    # Check for recent results
    local recent_results
    recent_results=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "*$(date +%Y%m%d)*" 2>/dev/null)
    
    if [[ -n "$recent_results" ]]; then
        echo -e "\nRecent results directories:"
        echo "$recent_results"
    fi
}

# Main execution
main() {
    echo "============================================"
    echo "LLM Evaluation SLURM Job Orchestration"
    echo "============================================"
    echo "Project Directory: $PROJECT_DIR"
    echo "SLURM Scripts: $SLURM_DIR"
    echo "Results Directory: $RESULTS_DIR"
    echo "============================================"
    
    case "${1:-}" in
        --submit-all)
            check_slurm || exit 1
            validate_slurm_scripts || exit 1
            submit_all_jobs || exit 1
            log_info "Monitoring jobs... (Press Ctrl+C to stop monitoring)"
            monitor_jobs
            generate_master_summary
            ;;
        --monitor-only)
            check_slurm || exit 1
            monitor_jobs
            ;;
        --status)
            check_slurm || exit 1
            show_status
            ;;
        --validate)
            validate_slurm_scripts
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "Error: No action specified."
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"