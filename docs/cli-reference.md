# CLI Reference

Command-line interface for ExpFlow.

## Installation

```bash
pip install git+https://github.com/YOUR_USERNAME/expflow-hpc.git
```

After installation, the `expflow` command is available globally.

## Commands

### `expflow init <project_name>`

Initialize a new HPC experiment project.

```bash
expflow init my-research
```

**Creates:**
- `/scratch/YOUR_ID/my-research/` - Project directory
- `.hpc_config.yaml` - Auto-detected configuration
- `experiment_configs/` - Experiment definitions
- `experiment_templates/` - Reusable templates
- `generated_scripts/` - Auto-generated SLURM scripts
- `experiments/` - Results and logs

**Example:**
```bash
$ expflow init deep-learning-research
Auto-detecting HPC environment...

 Detected HPC Environment:
  Cluster: greene
  Username: ab1234
  Scratch: /scratch/ab1234
  Default Account: your_account
  Default Partition: l40s_public

Creating project directories...
   /scratch/ab1234/deep-learning-research/experiments
   /scratch/ab1234/deep-learning-research/experiments/logs/output
  ...

 Configuration saved to .hpc_config.yaml
```

### `expflow info`

Show HPC environment information.

```bash
expflow info
```

**Output:**
```
======================================================================
HPC Environment Information
======================================================================
Username: ab1234
Home: /home/ab1234
Scratch: /scratch/ab1234
Cluster: greene
SLURM Accounts: torch_pr_68_tandon_advanced, general
Partitions: l40s_public, h200_tandon, a100_public, ...
======================================================================
```

### `expflow config [--project-root PATH]`

Show project configuration.

```bash
expflow config
expflow config --project-root /scratch/ab1234/my-project
```

### `expflow resources [OPTIONS]`

Show GPU resource status and recommendations.

**Options:**
- `--status` - Show current status only
- `--global-batch N` - Target global batch size (default: 192)
- `--use-gemini` - Use Gemini API for AI suggestions
- `--project-root PATH` - Project root directory

**Examples:**

```bash
# Check GPU availability
expflow resources --status

# Get recommendations
expflow resources

# Get AI-powered suggestions
expflow resources --use-gemini --global-batch 256
```

**Output:**
```
======================================================================
HPC Resource Status - NYU Greene Cluster
======================================================================
Updated: 2025-01-22 14:30:00

 L40S_PUBLIC
   GPU Type: L40S (Ada Lovelace, 48GB)
   Available: 12/40 GPUs
   Nodes: 3/10 available
   Queue: 0 jobs
   Wait Time:  Ready now
   Status: [READY] Good availability

 H200_TANDON
   GPU Type: H200 (Hopper, 141GB)
   Available: 0/10 GPUs
   Nodes: 0/5 available
   Queue: 8 jobs
   Wait Time: ~240 minutes
   Status:  No GPUs available

======================================================================

Recommendation: Use L40S×4 now
```

### `expflow template <name> [--force]`

Create an experiment template.

```bash
expflow template baseline
expflow template quick_test --force
```

**Creates:** `experiment_templates/<name>.yaml`

## Project-Specific Commands

After creating a custom manager (e.g., `my_manager.py`):

### Create Experiment

```bash
python my_manager.py new \
    --exp-id exp001 \
    --template baseline \
    --description "First experiment" \
    --tags test baseline
```

### Submit Experiment

```bash
# Dry run (no actual submission)
python my_manager.py submit exp001 --dry-run

# Submit training and evaluation
python my_manager.py submit exp001

# Training only
python my_manager.py submit exp001 --train-only

# Evaluation only (requires checkpoint)
python my_manager.py submit exp001 --eval-only
```

### Harvest Results

```bash
python my_manager.py harvest exp001
```

### List Experiments

```bash
# All experiments
python my_manager.py list

# Filter by status
python my_manager.py list --status completed

# Filter by tags
python my_manager.py list --tags baseline ablation
```

### Show Experiment Details

```bash
python my_manager.py show exp001
```

### Export Results

```bash
python my_manager.py export results.csv
```

## Examples

### Complete Workflow

```bash
# 1. Initialize project
expflow init image-classification
cd /scratch/YOUR_ID/image-classification

# 2. Check resources
expflow resources --status

# 3. Create template
expflow template resnet_baseline

# 4. Create experiment
python my_manager.py new \
    --exp-id exp001 \
    --template resnet_baseline \
    --description "ResNet50 baseline on ImageNet"

# 5. Submit
python my_manager.py submit exp001

# 6. Monitor
squeue -u $USER
tail -f experiments/logs/output/train_exp001_*.out

# 7. Harvest results
python my_manager.py harvest exp001

# 8. Export
python my_manager.py export results.csv
```

### Batch Experiments

```bash
# Learning rate sweep
for lr in 0.05 0.1 0.2; do
    python my_manager.py new \
        --exp-id exp_lr_${lr} \
        --template baseline \
        --learning-rate ${lr} \
        --tags lr-sweep
    python my_manager.py submit exp_lr_${lr}
done

# List sweep results
python my_manager.py list --tags lr-sweep
```

## Environment Variables

- `GEMINI_API_KEY` - For AI-powered suggestions
- `EXPFLOW_CONFIG` - Override config file location

## Exit Codes

- `0` - Success
- `1` - Error (invalid arguments, missing files, etc.)
- `2` - SLURM submission failed

## See Also

- [Getting Started](getting-started.md)
- [User Guide](user-guide.md)
- [API Reference](api-reference.md)
