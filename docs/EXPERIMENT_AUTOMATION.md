# Experiment Automation System

## Overview

The Experiment Manager is a comprehensive automation system that replaces the manual, error-prone workflow of editing SLURM scripts and tracking results in Excel.

## What Problem Does It Solve?

### Before (Manual Workflow)
- ❌ Manually edit `.slurm` scripts for each experiment
- ❌ Risk of overwriting previous configurations
- ❌ Manually record results in Excel spreadsheet
- ❌ Incomplete tracking (missing git commits, exact hyperparameters)
- ❌ No audit trail of what parameters were actually used
- ❌ Tedious CSV parsing and data entry
- ❌ Easy to forget to record critical metadata

### After (Automated Workflow)
- ✅ Define experiments in YAML configs
- ✅ Auto-generate unique SLURM scripts for each experiment
- ✅ Automatic metadata tracking (git commit, timestamps, job IDs)
- ✅ Automatic result harvesting from evaluation outputs
- ✅ Master CSV export matching Excel format
- ✅ Complete experiment history with reproducibility
- ✅ Query and filter experiments by tags, status, etc.

## Key Features

### 1. Config-Driven Experiments
Define once in YAML, use everywhere:
```yaml
exp_id: b15
description: "Test higher learning rate"
agent: ijepa_planning_agent_v3
batch_size: 48
learning_rate: 0.0002  # Changed from 0.0001
...
```

### 2. Template System
Pre-configured templates for common experiments:
- `ijepa_mlp` - Standard I-JEPA V3 (best performing)
- `ijepa_mlp_v4` - V4 with enhanced preprocessing
- `vit_mlp` - ViT baseline (no I-JEPA pretraining)
- `ijepa_mlp_ablation` - For hyperparameter searches

### 3. Automatic Script Generation
Generates optimized SLURM scripts from configs:
- Correct resource allocation (GPUs, CPUs, memory)
- Proper environment setup (conda, paths, overlays)
- Multi-node DDP configuration
- Checkpoint registration
- Error handling

### 4. Metadata Tracking
Automatically records:
- Git commit hash and branch
- Whether working tree was dirty
- Submission timestamp
- Training and evaluation job IDs
- Checkpoint path
- All hyperparameters
- Custom tags and notes

### 5. Result Harvesting
Automatically extracts from evaluation outputs:
- PDM score (primary metric)
- No Collision rate
- Drivable Area Compliance
- Ego Progress
- Time to Collision
- Comfort
- Number of scenarios

### 6. Master CSV Export
Generates comprehensive CSV with all experiments:
- Matches your Excel tracking format
- Sortable by PDM score, status, date
- Filterable by tags
- Includes all metadata and results

## File Organization

```
scripts/
├── experiment_manager.py          # Main automation script
├── EXPERIMENT_MANAGER_GUIDE.md    # Detailed user guide
├── quick_start_example.sh         # Demo workflow
│
├── experiment_templates/          # YAML templates
│   ├── ijepa_mlp.yaml            # V3 standard
│   ├── ijepa_mlp_v4.yaml         # V4 enhanced
│   ├── vit_mlp.yaml              # ViT baseline
│   └── ijepa_mlp_ablation.yaml   # Ablation studies
│
├── experiment_configs/            # Generated configs
│   ├── b15.yaml                  # User experiment configs
│   ├── b16.yaml
│   └── experiments.json          # Metadata database
│
└── generated/                     # Auto-generated SLURM scripts
    ├── train_b15.slurm
    └── eval_b15.slurm
```

## Typical Workflow

### 1. Create Experiment
```bash
python scripts/experiment_manager.py new \
    --exp-id b15 \
    --template ijepa_mlp \
    --description "Test higher learning rate" \
    --learning-rate 0.0002 \
    --tags lr-ablation
```

### 2. Submit to SLURM
```bash
# Dry run first (recommended)
python scripts/experiment_manager.py submit b15 --dry-run

# Review generated scripts
cat scripts/generated/train_b15.slurm

# Submit for real
python scripts/experiment_manager.py submit b15
```

### 3. Monitor
```bash
# Check queue
squeue -u $USER

# View logs
tail -f /scratch/ah7072/experiments/logs/output/train_b15_*.out
```

### 4. Harvest Results
```bash
# After evaluation completes
python scripts/experiment_manager.py harvest b15
```

### 5. Export All Results
```bash
python scripts/experiment_manager.py export results_master.csv
```

## Common Use Cases

### Reproduce B11 (Best Experiment)
```bash
python scripts/experiment_manager.py new \
    --exp-id b11_reproduce \
    --template ijepa_mlp \
    --description "Reproduce B11 (PDM 0.8206)"

python scripts/experiment_manager.py submit b11_reproduce
```

### Hyperparameter Ablation
```bash
# Batch size ablation
for bs in 32 48 64; do
    python scripts/experiment_manager.py new \
        --exp-id b20_bs${bs} \
        --template ijepa_mlp \
        --description "Batch size ${bs}" \
        --batch-size ${bs} \
        --tags batch-size-ablation

    python scripts/experiment_manager.py submit b20_bs${bs}
done
```

### Compare V3 vs V4
```bash
# V3
python scripts/experiment_manager.py new \
    --exp-id b21_v3 \
    --template ijepa_mlp \
    --description "V3 baseline"

# V4
python scripts/experiment_manager.py new \
    --exp-id b21_v4 \
    --template ijepa_mlp_v4 \
    --description "V4 with same params"

# Submit both
python scripts/experiment_manager.py submit b21_v3
python scripts/experiment_manager.py submit b21_v4
```

## Querying Experiments

```bash
# List all
python scripts/experiment_manager.py list

# Filter by status
python scripts/experiment_manager.py list --status completed

# Filter by tags
python scripts/experiment_manager.py list --tags ablation

# Show details
python scripts/experiment_manager.py show b15
```

## Integration with Existing Tools

The experiment manager integrates seamlessly with your existing infrastructure:

- Uses existing `train_and_eval.sh` wrapper for job submission
- Works with existing SLURM partitions and resource configs
- Compatible with existing cache overlays
- Stores results in standard evaluation directories
- Generates scripts following existing conventions

## Benefits

### Reproducibility
- Every experiment has complete configuration saved
- Git commit tracked automatically
- Exact hyperparameters recorded
- Easy to recreate any experiment months later

### Organization
- Tag experiments by category (ablation, baseline, comparison)
- Filter and search experiment history
- Track experiment status (created, submitted, completed)
- Add custom notes to experiments

### Efficiency
- No more manual script editing
- No more copy-paste errors
- No more forgetting to record metadata
- Automatic result extraction
- One command to export all results

### Safety
- Never overwrite existing scripts
- Each experiment gets unique generated scripts
- Dry-run mode to preview before submitting
- Complete audit trail

## Next Steps

1. **Read the guide**: `scripts/EXPERIMENT_MANAGER_GUIDE.md`
2. **Run the example**: `bash scripts/quick_start_example.sh`
3. **Create your first real experiment**: Use templates for B14/B13 corrected runs
4. **Export existing results**: Harvest completed experiments into the system

## Technical Details

### Dependencies
- Python 3.7+
- PyYAML
- pandas (for result parsing and CSV export)
- subprocess (for SLURM interaction)

### Metadata Database
JSON-based (`experiments.json`) for simplicity and portability:
- No external database required
- Human-readable and editable
- Version control friendly
- Easy to backup

### Script Generation
Template-based with parameter substitution:
- Ensures consistency across experiments
- Easy to update all experiments by changing templates
- Type-safe configuration with dataclasses
- Validation before submission

## Troubleshooting

See `scripts/EXPERIMENT_MANAGER_GUIDE.md` for detailed troubleshooting section.

Quick tips:
- Always use `--dry-run` first
- Review generated scripts before submitting
- Check `experiments.json` for metadata issues
- Use `show` command to debug experiment state

## Future Enhancements

Potential additions:
- [ ] Automatic performance comparison plots
- [ ] Slack/email notifications on completion
- [ ] Integration with Weights & Biases for experiment tracking
- [ ] Automatic hyperparameter optimization
- [ ] Experiment dependency graphs
- [ ] Result visualization dashboard

## Credits

Created to automate the NavSim I-JEPA planning agent research workflow, replacing manual Excel tracking with a comprehensive experiment management system.
