# Quick Reference Card - LLM Evaluation Session
Generated: September 19, 2025 03:29 UTC

## IMMEDIATE STATUS CHECK
```bash
squeue -u $USER                    # Check all your jobs
./resume_session.sh                # Run recovery script
tail -f slurm_jobs/logs/*.out      # Monitor live logs
```

## JOB IDs TO TRACK
- 1591: Test Job (RUNNING)
- 1592-1599: Category Jobs (SUBMITTED)

## EXPECTED TIMELINE
- Next 2-3 hours: Efficiency, Math, Scientific complete
- Next 4-6 hours: General, Safety, Coding complete  
- Next 6-8 hours: Biomedical, Multimodal complete

## CRITICAL FILES
- SESSION_END_STATE.md: Complete session documentation
- resume_session.sh: Automated status checker
- session_state.json: Machine-readable state
- slurm_jobs/job_tracker.txt: Job audit trail

## IF JOBS FAIL
```bash
# Check specific failed job
sacct -j JOB_ID -o "JobID,JobName,State,ExitCode"

# View error logs  
cat slurm_jobs/logs/CATEGORY_NAME_JOB_ID.err

# Resubmit single category
sbatch slurm_jobs/CATEGORY_NAME.slurm
```

## RESULTS LOCATION
- results/CATEGORY_NAME_YYYYMMDD_HHMMSS/
- Each job creates timestamped results directory

## SUCCESS INDICATORS
- Job status: COMPLETED
- Log contains: "EVALUATION COMPLETE"
- Results directory has evaluation files
- No errors in .err log files

## PARTITION USAGE
- h100dualflex: 5 intensive jobs (coding, general, safety, biomedical, multimodal)
- h100flex: 3 lighter jobs (efficiency, math, scientific)

Session Status: ALL SYSTEMS GO ✅