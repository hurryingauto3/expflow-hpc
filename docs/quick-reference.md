# Experiment Manager - Quick Reference

## Essential Commands

### Create New Experiment
```bash
# From template
python experiment_manager.py new --exp-id b15 --template ijepa_mlp --description "Your description"

# With overrides
python experiment_manager.py new --exp-id b15 --template ijepa_mlp --description "..." --learning-rate 0.0002
```

### Submit Experiment
```bash
# Dry run (recommended first)
python experiment_manager.py submit b15 --dry-run

# Submit for real
python experiment_manager.py submit b15

# Train only
python experiment_manager.py submit b15 --train-only

# Eval only
python experiment_manager.py submit b15 --eval-only
```

### Harvest Results
```bash
python experiment_manager.py harvest b15
```

### Export All Results
```bash
python experiment_manager.py export results_master.csv
```

### List and Query
```bash
# List all
python experiment_manager.py list

# Filter
python experiment_manager.py list --status completed
python experiment_manager.py list --tags ablation

# Show details
python experiment_manager.py show b15
```

## Templates

- `ijepa_mlp` - Standard I-JEPA V3 (best performing, B11 config)
- `ijepa_mlp_v4` - V4 with enhanced preprocessing
- `vit_mlp` - ViT baseline (no I-JEPA pretraining)
- `ijepa_mlp_ablation` - For hyperparameter studies

## Common Overrides

```bash
--learning-rate 0.0002          # Change head LR
--encoder-learning-rate 0.00005 # Change encoder LR
--batch-size 64                 # Change batch size
--trainable-fraction 0.25       # Change % trainable encoder
--epochs 20                     # Change training epochs
--data-percent 50               # Use 50% of data
--partition h200_tandon         # Change partition
--num-gpus 2                    # Change GPU count
--tags tag1 tag2                # Add tags
```

## Workflow Examples

### Quick Test
```bash
python experiment_manager.py new --exp-id test1 --template ijepa_mlp --description "Quick test"
python experiment_manager.py submit test1 --dry-run
# Review scripts/generated/train_test1.slurm
python experiment_manager.py submit test1
```

### Reproduce B11
```bash
python experiment_manager.py new --exp-id b11_repro --template ijepa_mlp --description "Reproduce B11"
python experiment_manager.py submit b11_repro
```

### Learning Rate Sweep
```bash
for lr in 0.00005 0.0001 0.0002; do
    python experiment_manager.py new \
        --exp-id b_lr_${lr} \
        --template ijepa_mlp \
        --description "LR sweep: ${lr}" \
        --learning-rate ${lr} \
        --tags lr-sweep
    python experiment_manager.py submit b_lr_${lr}
done
```

### Compare V3 vs V4
```bash
python experiment_manager.py new --exp-id b_v3 --template ijepa_mlp --description "V3"
python experiment_manager.py new --exp-id b_v4 --template ijepa_mlp_v4 --description "V4"
python experiment_manager.py submit b_v3
python experiment_manager.py submit b_v4
```

## File Locations

```
scripts/
  experiment_manager.py           # Main script
  experiment_templates/*.yaml     # Templates
  experiment_configs/*.yaml       # Your experiments
  experiment_configs/experiments.json  # Metadata DB
  generated/*.slurm               # Auto-generated scripts

/scratch/ah7072/experiments/
  training/exp_<id>_*/            # Training outputs
  evaluations/eval_*_exp_<id>_*/  # Evaluation results
  checkpoints/<run_id>.txt        # Checkpoint registry
  logs/output/train_<id>_*.out    # Training logs
  logs/error/train_<id>_*.err     # Error logs
```

## Monitoring

```bash
# SLURM queue
squeue -u $USER

# Training log (live)
tail -f /scratch/ah7072/experiments/logs/output/train_b15_*.out

# Evaluation log (live)
tail -f /scratch/ah7072/experiments/logs/output/eval_b15_*.out

# Check experiment status
python experiment_manager.py show b15
```

## Results

```bash
# After evaluation completes
python experiment_manager.py harvest b15

# Export to CSV
python experiment_manager.py export results.csv

# View CSV
column -t -s ',' results.csv | less -S
```

## Troubleshooting

```bash
# List all experiments
python experiment_manager.py list

# Show experiment details
python experiment_manager.py show b15

# Check generated scripts
cat scripts/generated/train_b15.slurm
cat scripts/generated/eval_b15.slurm

# Check metadata
cat scripts/experiment_configs/experiments.json | jq '.b15'

# Check evaluation directory
ls /scratch/ah7072/experiments/evaluations/eval_navtest_exp_b15_*/
```

## Tips

1. **Always dry-run first**: `--dry-run` to preview
2. **Use tags**: Organize with `--tags ablation lr-sweep`
3. **Track git**: System auto-records commit hash
4. **Filter results**: Use `list --status completed --tags ablation`
5. **Export often**: Keep `results_master.csv` updated

## Help

```bash
# Main help
python experiment_manager.py --help

# Subcommand help
python experiment_manager.py new --help
python experiment_manager.py submit --help
```

## Full Documentation

- User Guide: `scripts/EXPERIMENT_MANAGER_GUIDE.md`
- Overview: `docs/EXPERIMENT_AUTOMATION.md`
- Example: `bash scripts/quick_start_example.sh`
