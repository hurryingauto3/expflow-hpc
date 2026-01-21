# Experiment Manager Guide

Automated workflow for NavSim I-JEPA planning agent experiments.

## Overview

The Experiment Manager eliminates manual script editing and result tracking by:

1. **Defining experiments in YAML** - Config-driven approach with templates
2. **Generating SLURM scripts automatically** - No more manual editing errors
3. **Tracking all metadata** - Git commits, hyperparameters, job IDs, timestamps
4. **Harvesting results** - Automatic PDM score extraction from evaluation outputs
5. **Exporting to CSV** - Master results file matching your Excel format

## Quick Start

### 1. Create a New Experiment

```bash
# Using a template (recommended)
python scripts/experiment_manager.py new \
    --exp-id b15 \
    --template ijepa_mlp \
    --description "Test higher learning rate"

# With custom overrides
python scripts/experiment_manager.py new \
    --exp-id b16 \
    --template ijepa_mlp \
    --description "Ablation: 25% trainable encoder" \
    --trainable-fraction 0.25 \
    --tags ablation trainable-fraction
```

This creates `scripts/experiment_configs/b15.yaml` with all experiment parameters.

### 2. Submit the Experiment

```bash
# Submit both training and evaluation (chained)
python scripts/experiment_manager.py submit b15

# Dry run to see what would be submitted
python scripts/experiment_manager.py submit b15 --dry-run

# Train only
python scripts/experiment_manager.py submit b15 --train-only

# Eval only (requires existing checkpoint)
python scripts/experiment_manager.py submit b15 --eval-only
```

This:
- Generates `scripts/generated/train_b15.slurm` and `eval_b15.slurm`
- Logs experiment metadata (job IDs, timestamps, git commit)
- Submits jobs to SLURM queue
- Chains evaluation after training completion

### 3. Monitor Progress

```bash
# Check SLURM queue
squeue -u $USER

# View training logs
tail -f /scratch/ah7072/experiments/logs/output/train_b15_*.out

# View eval logs
tail -f /scratch/ah7072/experiments/logs/output/eval_b15_*.out
```

### 4. Harvest Results

```bash
# After evaluation completes
python scripts/experiment_manager.py harvest b15
```

This:
- Finds the evaluation directory
- Parses the CSV results
- Extracts PDM scores and metrics
- Updates experiment metadata

### 5. Export All Results

```bash
# Export to master CSV
python scripts/experiment_manager.py export results_master.csv

# View in terminal
column -t -s ',' results_master.csv | less -S
```

## Available Templates

### 1. `ijepa_mlp` (V3 - Best Performing)
- Agent: `ijepa_planning_agent_v3`
- Batch size: 48
- Learning rate: 1e-4
- Encoder LR: 3e-5
- Trainable: 50%
- Multi-camera: Yes (L0, F0, R0)
- **Use for**: Standard I-JEPA experiments, reproducing B11

### 2. `ijepa_mlp_v4` (V4 - Enhanced)
- Agent: `ijepa_planning_agent_v4`
- Same hyperparameters as V3
- Enhanced vision preprocessing
- **Use for**: Testing V4 improvements

### 3. `vit_mlp` (ViT Baseline)
- Agent: `ijepa_planning_agent_v4`
- Backbone: `google/vit-huge-patch14-224-in21k`
- No I-JEPA pretraining
- **Use for**: Baseline comparison, testing ViT without self-supervised pretraining

### 4. `ijepa_mlp_ablation`
- Template for ablation studies
- Comments indicate parameters to modify
- **Use for**: Hyperparameter searches, data efficiency tests

## Common Workflows

### Recreate B11 (Best Experiment)

```bash
# Create experiment
python scripts/experiment_manager.py new \
    --exp-id b11_reproduce \
    --template ijepa_mlp \
    --description "Reproduce B11 (0.8206 PDM)" \
    --tags reproduction baseline

# Submit
python scripts/experiment_manager.py submit b11_reproduce

# After completion
python scripts/experiment_manager.py harvest b11_reproduce
```

### Test New Learning Rate

```bash
python scripts/experiment_manager.py new \
    --exp-id b17 \
    --template ijepa_mlp \
    --description "Test LR=2e-4" \
    --learning-rate 0.0002 \
    --tags lr-ablation

python scripts/experiment_manager.py submit b17
```

### Compare V3 vs V4

```bash
# V3
python scripts/experiment_manager.py new \
    --exp-id b18_v3 \
    --template ijepa_mlp \
    --description "V3 baseline" \
    --tags v3 comparison

# V4
python scripts/experiment_manager.py new \
    --exp-id b18_v4 \
    --template ijepa_mlp_v4 \
    --description "V4 with same params" \
    --tags v4 comparison

# Submit both
python scripts/experiment_manager.py submit b18_v3
python scripts/experiment_manager.py submit b18_v4
```

### ViT Baseline (No I-JEPA Pretraining)

```bash
python scripts/experiment_manager.py new \
    --exp-id b19 \
    --template vit_mlp \
    --description "ViT baseline (no SSL pretraining)" \
    --tags vit baseline

python scripts/experiment_manager.py submit b19
```

### Data Efficiency Study

```bash
# 25% data
python scripts/experiment_manager.py new \
    --exp-id b20_25pct \
    --template ijepa_mlp \
    --description "Data efficiency: 25%" \
    --data-percent 25 \
    --tags data-efficiency

# 50% data
python scripts/experiment_manager.py new \
    --exp-id b20_50pct \
    --template ijepa_mlp \
    --description "Data efficiency: 50%" \
    --data-percent 50 \
    --tags data-efficiency

# 75% data
python scripts/experiment_manager.py new \
    --exp-id b20_75pct \
    --template ijepa_mlp \
    --description "Data efficiency: 75%" \
    --data-percent 75 \
    --tags data-efficiency

# Submit all three
python scripts/experiment_manager.py submit b20_25pct
python scripts/experiment_manager.py submit b20_50pct
python scripts/experiment_manager.py submit b20_75pct
```

## Listing and Querying Experiments

```bash
# List all experiments
python scripts/experiment_manager.py list

# Filter by status
python scripts/experiment_manager.py list --status completed
python scripts/experiment_manager.py list --status submitted

# Filter by tags
python scripts/experiment_manager.py list --tags ablation
python scripts/experiment_manager.py list --tags v4 comparison

# Show detailed info for one experiment
python scripts/experiment_manager.py show b15
```

## Understanding the File Structure

```
scripts/
├── experiment_manager.py          # Main script
├── experiment_templates/          # YAML templates
│   ├── ijepa_mlp.yaml
│   ├── ijepa_mlp_v4.yaml
│   ├── vit_mlp.yaml
│   └── ijepa_mlp_ablation.yaml
├── experiment_configs/            # Generated configs
│   ├── b15.yaml
│   ├── b16.yaml
│   └── experiments.json           # Metadata database
├── generated/                     # Auto-generated SLURM scripts
│   ├── train_b15.slurm
│   └── eval_b15.slurm
└── train_and_eval.sh             # Wrapper (existing)

/scratch/ah7072/experiments/
├── checkpoints/                   # Checkpoint registry
│   └── <run_id>.txt              # Checkpoint paths
├── training/                      # Training outputs
│   └── exp_b15_*/
├── evaluations/                   # Evaluation results
│   └── eval_navtest_exp_b15_*/
└── logs/                          # SLURM logs
    ├── output/
    └── error/
```

## Experiment Config YAML Format

```yaml
# Required fields
exp_id: b15
description: "Test higher learning rate"

# Agent
agent: ijepa_planning_agent_v3
backbone: ijepa
model_id: null  # or "google/vit-huge-patch14-224-in21k" for ViT

# Training hyperparameters
batch_size: 48
learning_rate: 0.0001
encoder_learning_rate: 0.00003
trainable_fraction: 0.5
epochs: 30
data_percent: 100

# Multi-camera
use_multi_camera: true
camera_views:
  - cam_l0
  - cam_f0
  - cam_r0
vision_mode: multi_per_view
image_size: [224, 224]

# Resources
partition: l40s_public
num_gpus: 4
num_nodes: 1
cpus_per_task: 16
time_limit: "48:00:00"
account: torch_pr_68_tandon_advanced

# Cache
cache_name: training_cache_ijepa_planning_agent_v3_v5
use_cache_overlay: true

# Evaluation
eval_split: navtest
eval_workers: 48

# Metadata (auto-populated)
created_at: "2026-01-21T12:00:00"
submitted_at: null
completed_at: null
train_job_id: null
eval_job_id: null
checkpoint_path: null
pdm_score: null
git_commit: "94c4c7e..."
git_branch: "main"
git_dirty: false

# Organization
tags:
  - ijepa
  - ablation
notes: ""
```

## Metadata Database (`experiments.json`)

The experiment manager maintains a JSON database with:

```json
{
  "b15": {
    "exp_id": "b15",
    "status": "completed",
    "config": { ... },
    "train_script_path": "scripts/generated/train_b15.slurm",
    "eval_script_path": "scripts/generated/eval_b15.slurm",
    "run_id": "20260121_120000_12345",
    "results": {
      "pdm_score": 0.8215,
      "no_collision": 0.95,
      "drivable_area_compliance": 0.98,
      "ego_progress": 0.87,
      "time_to_collision": 0.82,
      "comfort": 0.91,
      "num_scenarios": 12146,
      "harvested_at": "2026-01-21T14:30:00"
    }
  }
}
```

## Exported CSV Format

Matches your Excel tracking sheet:

```csv
exp_id,description,agent,backbone,data_percent,batch_size,learning_rate,encoder_lr,trainable_fraction,epochs,multi_camera,num_gpus,pdm_score,no_collision,drivable_area,ego_progress,status,train_job_id,eval_job_id,git_commit,created_at,submitted_at,completed_at,tags,notes
b15,Test higher LR,ijepa_planning_agent_v3,ijepa,100,48,0.0002,0.00003,0.5,30,True,4,0.8215,0.95,0.98,0.87,completed,12345,12346,94c4c7e,2026-01-21T12:00:00,2026-01-21T12:05:00,2026-01-21T14:30:00,"lr-ablation",""
```

## Advanced Features

### Custom Templates

Create your own template:

```bash
cp scripts/experiment_templates/ijepa_mlp.yaml scripts/experiment_templates/my_template.yaml
# Edit my_template.yaml
python scripts/experiment_manager.py new --exp-id b20 --template my_template --description "..."
```

### Editing Existing Configs

Before submission, you can manually edit the config:

```bash
vim scripts/experiment_configs/b15.yaml
python scripts/experiment_manager.py submit b15  # Uses updated config
```

### Re-running Evaluations

```bash
# Eval only mode with existing checkpoint
python scripts/experiment_manager.py submit b15 --eval-only
```

### Dry Run Testing

Always test with dry-run first:

```bash
python scripts/experiment_manager.py submit b15 --dry-run
# Review generated scripts
cat scripts/generated/train_b15.slurm
cat scripts/generated/eval_b15.slurm
# If looks good, submit for real
python scripts/experiment_manager.py submit b15
```

## Troubleshooting

### Experiment not found
```bash
# List all experiments
python scripts/experiment_manager.py list

# Show what configs exist
ls scripts/experiment_configs/
```

### Harvest fails
```bash
# Check if evaluation completed
ls /scratch/ah7072/experiments/evaluations/eval_navtest_exp_b15_*/

# Manually check results
cat /scratch/ah7072/experiments/evaluations/eval_navtest_exp_b15_*/*.csv
```

### Script generation issues
```bash
# Use dry-run to see generated scripts without submitting
python scripts/experiment_manager.py submit b15 --dry-run

# Check generated scripts
cat scripts/generated/train_b15.slurm
cat scripts/generated/eval_b15.slurm
```

### Git tracking disabled
If not in a git repo, git fields will be null. This is fine - metadata still tracks everything else.

## Benefits Over Manual Workflow

### Before (Manual)
1. ❌ Edit `ijepa_mlp_train_100pc.slurm` directly
2. ❌ Risk overwriting previous config
3. ❌ Manually copy experiment name
4. ❌ Manually record parameters in Excel
5. ❌ Forget to note git commit
6. ❌ Manually parse PDM scores from logs
7. ❌ Incomplete tracking

### After (Automated)
1. ✅ Define experiment in YAML
2. ✅ Auto-generate unique SLURM scripts
3. ✅ All metadata tracked automatically
4. ✅ Git commit/branch recorded
5. ✅ Results harvested automatically
6. ✅ Master CSV always up to date
7. ✅ Complete experiment history

## Integration with Existing Workflow

The experiment manager works with your existing `train_and_eval.sh` wrapper:

```bash
# Old way (still works)
./scripts/train_and_eval.sh --train scripts/train/ijepa_mlp_train_100pc.slurm --eval scripts/evaluate/pdm_score_ijepa_mlp.slurm

# New way (automated)
python scripts/experiment_manager.py new --exp-id b15 --template ijepa_mlp --description "..."
python scripts/experiment_manager.py submit b15
```

Both use the same SLURM infrastructure, the experiment manager just automates the config management and tracking.

## Next Steps

1. **Create your first experiment**:
   ```bash
   python scripts/experiment_manager.py new --exp-id b15 --template ijepa_mlp --description "Test experiment manager"
   python scripts/experiment_manager.py submit b15 --dry-run
   ```

2. **Review generated scripts**:
   ```bash
   cat scripts/generated/train_b15.slurm
   cat scripts/generated/eval_b15.slurm
   ```

3. **If satisfied, submit for real**:
   ```bash
   python scripts/experiment_manager.py submit b15
   ```

4. **After completion, harvest and export**:
   ```bash
   python scripts/experiment_manager.py harvest b15
   python scripts/experiment_manager.py export results_master.csv
   ```

## Questions?

Check the experiment metadata:
```bash
python scripts/experiment_manager.py show b15
```

List all experiments:
```bash
python scripts/experiment_manager.py list
```

View the code:
```bash
less scripts/experiment_manager.py
```
