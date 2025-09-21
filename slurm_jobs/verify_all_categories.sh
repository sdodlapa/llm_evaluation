#!/bin/bash

# Quick verification script to test all categories with lightweight settings
echo "========================================="
echo "QUICK CATEGORY VERIFICATION TEST"
echo "Testing all 9 categories with 1 sample each"
echo "Started at: $(date)"
echo "========================================="

categories=(
    "coding_specialists"
    "mathematical_reasoning" 
    "biomedical_specialists"
    "multimodal_processing"
    "scientific_research"
    "efficiency_optimized"
    "general_purpose"
    "safety_alignment"
    "text_geospatial"
)

for category in "${categories[@]}"; do
    echo ""
    echo "Testing $category..."
    
    # Use different exclude patterns based on category
    case $category in
        "biomedical_specialists")
            exclude_args="--exclude-models bio_clinicalbert pubmedbert_large clinical_camel_70b"
            ;;
        "scientific_research")
            exclude_args="--exclude-models scibert_base specter2_base"
            ;;
        "safety_alignment")
            exclude_args="--exclude-models safety_bert"
            ;;
        "general_purpose")
            exclude_args="--exclude-models llama31_8b mistral_7b"
            ;;
        "text_geospatial")
            exclude_args=""
            ;;
        "mathematical_reasoning")
            exclude_args="--exclude-models wizardmath_70b metamath_70b"
            ;;
        *)
            exclude_args=""
            ;;
    esac
    
    crun -p ~/envs/llm_env python category_evaluation.py \
        --category $category \
        --samples 1 \
        --preset balanced \
        $exclude_args
    
    if [ $? -eq 0 ]; then
        echo "✅ $category - SUCCESS"
    else
        echo "❌ $category - FAILED"
    fi
done

echo ""
echo "========================================="
echo "VERIFICATION COMPLETE at $(date)"
echo "========================================="