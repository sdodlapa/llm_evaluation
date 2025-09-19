#!/bin/bash

# Simple script to submit all category evaluation jobs
echo "Submitting all category evaluation jobs..."

sbatch slurm_jobs/biomedical_specialists_multibackend.slurm
sbatch slurm_jobs/coding_specialists_multibackend.slurm
sbatch slurm_jobs/efficiency_optimized_multibackend.slurm
sbatch slurm_jobs/general_purpose_multibackend.slurm
sbatch slurm_jobs/mathematical_reasoning_multibackend.slurm
sbatch slurm_jobs/multimodal_processing_multibackend.slurm
sbatch slurm_jobs/safety_alignment_multibackend.slurm
sbatch slurm_jobs/scientific_research_multibackend.slurm

echo "All jobs submitted. Check status with: squeue -u \$USER"