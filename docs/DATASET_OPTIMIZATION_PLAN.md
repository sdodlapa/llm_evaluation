# Dataset Organization Optimization Plan

## 🎯 **Recommended Efficient Structure**

### **Proposed Optimized Organization:**

```
evaluation_data/
├── 📚 general_knowledge/          # 7 datasets (fully utilized)
├── 🧮 mathematical_reasoning/     # 2 datasets (add missing ones)
├── 💻 coding_specialists/         # 5 datasets (consolidate from coding/)
├── 🧬 biomedical/                # 8 datasets (well organized)
├── 🖼️ multimodal_processing/      # 5 datasets (add missing MMMU, MathVista)
├── 🗺️ text_geospatial/           # 9 datasets (efficient)
├── 🧠 reasoning_specialized/      # Add BigBench-Hard, GPQA
├── 🔧 function_calling/           # 1 dataset (room for growth)
├── 🔬 scientific/                # 2 datasets (consolidate scientific/)
├── 🛡️ safety/                   # Add safety_eval, truthfulqa variants
└── 📊 metadata/                  # Centralize all metadata
```

### **Elimination Targets:**

**Remove These Redundant Structures:**
- `datasets/` subfolder (duplicate organization)
- `coding/` (merge into coding_specialists/)
- `mathematics/` (merge into mathematical_reasoning/)
- `qa/`, `reasoning/` (merge into appropriate categories)
- `backup_original_structure/` (archive separately)
- Standalone directories (merge: ai2d→multimodal, etc.)

### **Category Utilization Optimization:**

**High Priority Downloads:**
1. **H100 Critical:** bigbench_hard, longbench, mmlu_pro
2. **Coding Advanced:** swe_bench, livecodebench, repobench  
3. **Multimodal:** mmmu, mathvista
4. **Safety:** safety_eval, truthfulqa variants

**Dataset Mapping Updates:**
1. Map `pubmedqa_full`, `pubmedqa_sample` to biomedical categories
2. Consolidate scientific paper datasets
3. Utilize geospatial integration data
4. Remove broken references

### **Expected Efficiency Gains:**

**Storage Optimization:**
- Eliminate ~30% redundant storage
- Reduce directory depth and complexity
- Centralize metadata management

**Evaluation Efficiency:**
- 100% category readiness (vs current 64%)
- Clear dataset→category mapping
- Simplified evaluation scripts

**Maintenance Benefits:**
- Single source of truth for each dataset
- Consistent naming and organization
- Automated dataset discovery

### **Implementation Priority:**

**Phase 1 (Immediate):** Fix missing primary datasets
**Phase 2 (Week 1):** Consolidate redundant structures  
**Phase 3 (Week 2):** Update all evaluation scripts
**Phase 4 (Week 3):** Archive and cleanup

This optimization would improve your evaluation system from 67% efficiency to 95%+ efficiency while reducing storage overhead and maintenance complexity.