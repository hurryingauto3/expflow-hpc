# Setup Guide - HPC Experiment Manager

**Complete setup instructions for NYU HPC researchers**

## Prerequisites

- Access to NYU Greene HPC cluster
- Basic familiarity with SLURM
- Python 3.8+
- Git (optional but recommended)

## Installation

### Option 1: Quick Install (Recommended)

```bash
# SSH into NYU Greene
ssh YOUR_NYU_ID@greene.hpc.nyu.edu

# Clone repository to your home directory
cd ~
git clone https://github.com/YOUR_USERNAME/hpc-experiment-manager.git
cd hpc-experiment-manager

# Make CLI executable
chmod +x hpcexp

# Add to PATH
echo 'export PATH="$HOME/hpc-experiment-manager:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install Python dependencies
pip install --user pyyaml pandas

# Optional: Install Gemini API for AI suggestions
pip install --user google-generativeai
export GEMINI_API_KEY="your_api_key_here"
```

### Option 2: Conda Environment

```bash
# Create dedicated environment
module load anaconda3/2025.06
conda create -n hpcexp python=3.10
conda activate hpcexp

# Install dependencies
pip install pyyaml pandas google-generativeai

# Clone and setup
cd ~
git clone https://github.com/YOUR_USERNAME/hpc-experiment-manager.git
cd hpc-experiment-manager
chmod +x hpcexp

# Add alias (add to ~/.bashrc)
alias hpcexp="conda run -n hpcexp python $HOME/hpc-experiment-manager/hpcexp"
```

## Verification

Test that everything is installed correctly:

```bash
# Test auto-detection
hpcexp info

# Expected output:
# ======================================================================
# HPC Environment Information
# ======================================================================
# Username: YOUR_NYU_ID
# Home: /home/YOUR_NYU_ID
# Scratch: /scratch/YOUR_NYU_ID
# Cluster: greene
# SLURM Accounts: [your accounts]
# Partitions: l40s_public, h200_tandon, a100_public, ...
# ======================================================================
```

## Quick Start Tutorial

### 1. Initialize Your First Project

```bash
# Example: Image classification project
hpcexp init imagenet_resnet

# Output:
# Auto-detecting HPC environment...
#
# ✓ Detected HPC Environment:
#   Cluster: greene
#   Username: ah7072
#   Scratch: /scratch/ah7072
#   Default Account: torch_pr_68_tandon_advanced
#   Default Partition: l40s_public
#   Available Partitions: l40s_public, h200_tandon, ...
#
# Creating project directories...
#   ✓ /scratch/ah7072/imagenet_resnet/experiments
#   ✓ /scratch/ah7072/imagenet_resnet/experiments/logs/output
#   ✓ /scratch/ah7072/imagenet_resnet/experiments/logs/error
#   ✓ /scratch/ah7072/imagenet_resnet/experiments/cache
#   ✓ /scratch/ah7072/imagenet_resnet/experiments/checkpoints
#
# ✓ Configuration saved to /scratch/ah7072/imagenet_resnet/.hpc_config.yaml
```

### 2. Inspect Auto-Detected Configuration

```bash
cd /scratch/YOUR_NYU_ID/imagenet_resnet
cat .hpc_config.yaml

# Output:
# username: YOUR_NYU_ID
# user_home: /home/YOUR_NYU_ID
# scratch_dir: /scratch/YOUR_NYU_ID
# project_name: imagenet_resnet
# project_root: /scratch/YOUR_NYU_ID/imagenet_resnet
# default_account: torch_pr_68_tandon_advanced
# default_partition: l40s_public
# default_time_limit: 48:00:00
# experiments_dir: /scratch/YOUR_NYU_ID/imagenet_resnet/experiments
# ...
```

Notice: **No hardcoded paths!** Everything is auto-detected based on YOUR NYU ID.

### 3. Copy Example Manager

```bash
cd /scratch/YOUR_NYU_ID/imagenet_resnet

# Copy the example image classification manager
cp ~/hpc-experiment-manager/examples/simple_image_classification.py ./my_manager.py

# Make it executable
chmod +x my_manager.py
```

### 4. Create Experiment Template

```bash
# Create template
cat > experiment_templates/resnet50_baseline.yaml << 'EOF'
# ResNet50 baseline configuration
description: "ResNet50 baseline on ImageNet"

# Model
model: resnet50
pretrained: true
dataset: imagenet
num_classes: 1000

# Training
batch_size: 256
learning_rate: 0.1
momentum: 0.9
weight_decay: 0.0001
epochs: 90
optimizer: sgd
scheduler: cosine
warmup_epochs: 5

# Augmentation
augmentation:
  - random_crop
  - random_flip
  - color_jitter

# Resources (auto-uses your account)
partition: l40s_public
num_gpus: 4
num_nodes: 1
cpus_per_task: 16
time_limit: "24:00:00"

# Evaluation
eval_batch_size: 512
test_split: val
EOF
```

### 5. Create Your First Experiment

```bash
# Create experiment from template
python my_manager.py new \
    --exp-id exp001 \
    --template resnet50_baseline \
    --description "Baseline ResNet50 - first run"

# Output:
# ✓ Created experiment: exp001
#   Config: /scratch/YOUR_NYU_ID/imagenet_resnet/experiment_configs/exp001.yaml
```

### 6. Check Resources Before Submitting

```bash
# See what GPUs are available
hpcexp resources --status

# Get smart recommendations
hpcexp resources

# With AI suggestions (optional)
hpcexp resources --use-gemini
```

### 7. Submit Experiment (Dry Run First)

```bash
# Dry run to see what would happen
python my_manager.py submit exp001 --dry-run

# Output:
# ✓ Generated training script: /scratch/YOUR_NYU_ID/imagenet_resnet/generated_scripts/train_exp001.slurm
# ✓ Generated evaluation script: /scratch/YOUR_NYU_ID/imagenet_resnet/generated_scripts/eval_exp001.slurm
#
# [DRY RUN] Would submit the following jobs:
#   Training: sbatch .../train_exp001.slurm
#   Evaluation: sbatch --dependency=afterok:TRAIN_JOB_ID .../eval_exp001.slurm
```

### 8. Inspect Generated Scripts

```bash
# Look at the auto-generated training script
cat generated_scripts/train_exp001.slurm

# Notice:
# - Your NYU ID is used (not hardcoded!)
# - Your SLURM account is used
# - Paths point to YOUR scratch directory
# - Git commit is tracked
```

### 9. Submit For Real

```bash
# Submit both training and evaluation (chained)
python my_manager.py submit exp001

# Output:
# ✓ Generated training script: .../train_exp001.slurm
# ✓ Generated evaluation script: .../eval_exp001.slurm
# ✓ Submitted training job: 12345678
# ✓ Submitted evaluation job: 12345679 (depends on 12345678)
```

### 10. Monitor Progress

```bash
# Check SLURM queue
squeue -u $USER

# Watch training logs
tail -f experiments/logs/output/train_exp001_*.out

# Check GPU utilization
ssh greene-1234  # Replace with your node
nvidia-smi
```

### 11. Harvest Results

```bash
# After evaluation completes
python my_manager.py harvest exp001

# Output:
# Parsing results from: .../results/exp001_results.json
#
# ✓ Results harvested for exp001
#   Top-1 Accuracy: 76.15%
#   Top-5 Accuracy: 92.87%
#   Test Loss: 1.234
```

### 12. Export All Results

```bash
# Export to CSV
python my_manager.py export results.csv

# Output:
# ✓ Exported 1 experiments to results.csv

# View results
column -t -s ',' results.csv | less -S
```

## Common Workflows

### Running Multiple Experiments

```bash
# Learning rate sweep
for lr in 0.05 0.1 0.2; do
    python my_manager.py new \
        --exp-id exp_lr_${lr} \
        --template resnet50_baseline \
        --description "LR sweep: ${lr}" \
        --learning-rate ${lr} \
        --tags lr-sweep

    python my_manager.py submit exp_lr_${lr}
done

# List all experiments in sweep
python my_manager.py list --tags lr-sweep
```

### Comparing Results

```bash
# Export all results
python my_manager.py export results.csv

# Filter and sort in Python
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
df = df[df['tags'].str.contains('lr-sweep', na=False)]
df = df.sort_values('top1_acc', ascending=False)
print(df[['exp_id', 'learning_rate', 'top1_acc', 'top5_acc']])
"
```

### Re-running Failed Experiments

```bash
# Check status
python my_manager.py list --status failed

# Re-submit
python my_manager.py submit exp_lr_0.05 --train-only
```

## Adapting for Your Research

### Custom Experiment Manager

See `examples/` for complete examples:
- `simple_image_classification.py` - Basic classification
- `llm_finetuning.py` - LLM fine-tuning with LoRA
- `reinforcement_learning.py` - RL training
- `navsim_planning.py` - Original NavSim implementation

Copy one of these and modify:

1. **Define your config class** (subclass `BaseExperimentConfig`)
2. **Implement `_generate_train_script()`** - Your training SLURM script
3. **Implement `_generate_eval_script()`** - Your evaluation SLURM script
4. **Implement `harvest_results()`** - Parse your result format

### Custom Templates

Create templates for common experiment types:

```bash
cd /scratch/YOUR_NYU_ID/your_project

# Create templates
cat > experiment_templates/quick_test.yaml << EOF
# Quick test on 1 GPU
partition: l40s_public
num_gpus: 1
epochs: 10
time_limit: "02:00:00"
EOF

cat > experiment_templates/full_run.yaml << EOF
# Full training run
partition: l40s_public
num_gpus: 8
num_nodes: 2
epochs: 100
time_limit: "48:00:00"
EOF
```

## Troubleshooting

### "Command not found: hpcexp"

```bash
# Check if it's in PATH
which hpcexp

# If not, add to ~/.bashrc
echo 'export PATH="$HOME/hpc-experiment-manager:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### "No HPC config found"

```bash
# Make sure you initialized a project
cd /scratch/YOUR_NYU_ID/your_project
hpcexp init your_project

# Or specify project root
cd ~/somewhere
hpcexp config --project-root /scratch/YOUR_NYU_ID/your_project
```

### "ImportError: No module named hpc_config"

```bash
# Make sure you're running from project directory
cd /scratch/YOUR_NYU_ID/your_project

# Or add framework to PYTHONPATH
export PYTHONPATH="$HOME/hpc-experiment-manager:$PYTHONPATH"
```

### SLURM Submission Fails

```bash
# Check your SLURM accounts
sacctmgr show user $USER

# Verify partition access
sinfo -p l40s_public

# Test with dry run
python my_manager.py submit exp001 --dry-run

# Manually test generated script
sbatch generated_scripts/train_exp001.slurm
```

## Tips & Best Practices

### 1. Use Templates for Common Configs
Create templates for your standard experiment types and override only what changes.

### 2. Always Dry Run First
Use `--dry-run` to check generated scripts before submitting.

### 3. Check Resources Before Submitting
Use `hpcexp resources --status` to see GPU availability.

### 4. Tag Your Experiments
Use tags for organization: `--tags ablation baseline sweep`

### 5. Commit Before Running
The framework tracks git commits. Commit your code before running experiments.

### 6. Use Meaningful Experiment IDs
Good: `exp_resnet50_lr0.1_bs256`
Bad: `exp001`, `test`, `final`

### 7. Document in Description
Be specific: "ResNet50, LR=0.1, warmup=5, cosine schedule"

## Next Steps

1. **Explore examples**: Check `examples/` for more use cases
2. **Customize**: Create your own experiment manager
3. **Share**: Contribute templates and examples back to the repo
4. **Get help**: Open issues on GitHub

## Support

- **NYU HPC Docs**: https://sites.google.com/nyu.edu/nyu-hpc/
- **SLURM Documentation**: https://slurm.schedmd.com/
- **GitHub Issues**: Report bugs and request features

---

Happy experimenting! 🚀
