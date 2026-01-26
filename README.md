# ExpFlow - HPC Experiment Manager

> Lightweight experiment tracking for HPC clusters. Stop manually editing SLURM scripts.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SLURM](https://img.shields.io/badge/HPC-SLURM-orange.svg)](https://slurm.schedmd.com/)

**ExpFlow** auto-detects your HPC environment (username, scratch paths, SLURM accounts) and automates experiment tracking - no hardcoded paths, no manual script editing, no Excel spreadsheets.

## Quick Start

```bash
# Install
pip install git+https://github.com/hurryingauto3/expflow-hpc.git

# Initialize with interactive setup
expflow init -i my-research

# Navigate to project
cd /scratch/YOUR_ID/my-research

# Create template
expflow template baseline

# Check resources and monitor experiments
expflow resources --status
expflow status
expflow logs exp001
```

## Key Features

- **Auto-Detection**: Automatically detects username, scratch directory, SLURM accounts, and partition access
- **Interactive Setup**: Menu-based initialization with intelligent account and GPU recommendations
- **Experiment Monitoring**: Built-in commands for status tracking, log viewing, and job management
- **Checkpoint Resumption**: Automatic checkpoint detection and experiment resume support (v0.6.0+)
- **Experiment Pruning**: Clean up duplicate runs and invalid experiments with safe archival
- **Resource Advisor**: Real-time GPU availability and smart recommendations
- **Partition Validation**: Automatic partition-account compatibility testing
- **YAML-Based Configs**: No more editing SLURM scripts manually
- **Complete Tracking**: Git commits, job IDs, timestamps, and results automatically logged
- **Extensible**: Subclass `BaseExperimentManager` for custom workflows

## Installation

### From GitHub (Recommended)
```bash
pip install git+https://github.com/hurryingauto3/expflow-hpc.git
```

### With Conda
```bash
conda create -n expflow python=3.10
conda activate expflow
pip install git+https://github.com/hurryingauto3/expflow-hpc.git
```

### From Source
```bash
git clone https://github.com/hurryingauto3/expflow-hpc.git
cd expflow-hpc
pip install -e .
```

## Getting Started

### 1. Initialize Your Project

**Interactive Mode (Recommended):**
```bash
expflow init -i my-research
```

Guided setup with:
- Account selection (prefers "general" accounts for broad access)
- GPU/Partition selection (H200, L40s, A100, RTX8000 categories)
- Time limit preferences (6h, 12h, 24h, 48h, 72h)
- Automatic partition access validation

**Quick Mode:**
```bash
expflow init -q my-research
```

Uses smart defaults without prompts.

### 2. Check Your Environment

```bash
expflow info
```

Output:
```
======================================================================
HPC Environment Information
======================================================================
Username: ah7072
Scratch: /scratch/ah7072
Cluster: greene
Accounts: torch_pr_68_general, torch_pr_68_tandon_advanced
Partitions: l40s_public, h200_public, rtx8000
```

### 3. Create Experiment Template

```bash
cd /scratch/YOUR_ID/my-research
expflow template baseline
```

Edit `experiment_templates/baseline.yaml`:
```yaml
description: "Baseline experiment"

# Your parameters
model: resnet50
dataset: imagenet
batch_size: 256
learning_rate: 0.1

# Resources (auto-detected defaults)
partition: l40s_public
account: torch_pr_68_general
num_gpus: 4
num_nodes: 1
cpus_per_task: 16
time_limit: "48:00:00"
```

### 4. Monitor and Manage Experiments

```bash
# View all experiments and running jobs
expflow status

# List experiments
expflow list
expflow list --status running

# View logs
expflow logs exp001              # Last 50 lines
expflow logs exp001 -n 100       # Last 100 lines
expflow logs exp001 --type eval  # Evaluation logs

# Follow logs in real-time
expflow tail exp001

# Cancel jobs
expflow cancel exp001
```

## Core Commands

### Initialization
```bash
expflow init -i <project>    # Interactive setup with menus
expflow init -q <project>    # Quick setup with defaults
expflow init <project>       # Legacy auto-detect mode
```

### Environment Info
```bash
expflow info                 # Show HPC environment details
expflow config               # Show project configuration
```

### Resource Management
```bash
expflow resources --status                     # Check GPU availability
expflow partitions                             # Show partition-account access map
expflow partitions --json                      # Export as JSON
```

### Experiment Monitoring
```bash
expflow status                                 # Show experiments and SLURM jobs
expflow list                                   # List all experiments
expflow list --status running                  # Filter by status
expflow logs <exp_id>                          # View experiment logs
expflow logs <exp_id> -n 100 --errors          # View last 100 lines of errors
expflow tail <exp_id>                          # Follow logs in real-time
expflow cancel <exp_id>                        # Cancel running jobs
expflow prune --dry-run                        # Preview cleanup of duplicate experiments
```

### Templates
```bash
expflow template <name>      # Create experiment template
```

## Why ExpFlow?

### Before ExpFlow
```bash
# Manually edit SLURM scripts
vim train.slurm

# Hardcoded paths that only work for you
#SBATCH --account=my_account        # Others can't use this!
export DATA=/scratch/myuser/data    # Hardcoded!

sbatch train.slurm

# Track experiments manually in Excel
# Forget git commits
# Lose hyperparameters
```

### With ExpFlow
```bash
# One-time setup (works for ANY user)
expflow init -i my-project

# Create experiment from template
python -m my_manager new --exp-id exp001 --template baseline

# Submit (auto-detects YOUR paths, account, GPUs)
python -m my_manager submit exp001

# Monitor in real-time
expflow status
expflow tail exp001

# Auto-harvest results
python -m my_manager harvest exp001
python -m my_manager export results.csv
```

**Time Saved:** ~80% reduction in experiment setup time

## Resource Advisor

Check GPU availability before submitting:

```bash
$ expflow resources --status

======================================================================
GPU Resource Status
======================================================================

L40S_PUBLIC
   Total: 40 GPUs
   Available: 12
   In Use: 28
   Queue: 3 jobs
   Status: AVAILABLE

H200_TANDON
   Total: 10 GPUs
   Available: 0
   In Use: 10
   Queue: 8 jobs
   Wait Time: ~4 hours
   Status: BUSY

Recommendation: Use l40s_public with 4 GPUs (best availability)
```

## Partition Management

View partition-account compatibility:

```bash
$ expflow partitions

======================================================================
Partition Access Map
======================================================================

h200_public (GPU: H200) [GPU Required]
  ✓ torch_pr_68_general
  ✓ torch_pr_68_tandon_advanced

l40s_public (GPU: L40s) [GPU Required]
  ✓ torch_pr_68_general

Account Access Summary:
  torch_pr_68_general → h200_public, l40s_public, rtx8000
  torch_pr_68_tandon_advanced → h200_public, h200_tandon
```

## Creating Custom Managers

For project-specific workflows, create a custom manager:

```python
from expflow import BaseExperimentManager

class MyManager(BaseExperimentManager):
    def _generate_train_script(self, config):
        """Generate SLURM training script"""
        return f'''#!/bin/bash
#SBATCH --gres=gpu:{config['num_gpus']}
#SBATCH --partition={config['partition']}
#SBATCH --account={config['account']}

python train.py --model {config['model']} ...
'''

    def _generate_eval_script(self, config):
        """Generate SLURM evaluation script"""
        return "#!/bin/bash\npython evaluate.py ..."

    def harvest_results(self, exp_id):
        """Parse experiment results"""
        return {"accuracy": 0.95, "loss": 0.12}
```

Use your manager:
```bash
python my_manager.py new --exp-id exp001 --template baseline
python my_manager.py submit exp001
python my_manager.py harvest exp001
python my_manager.py export results.csv
```

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user guide with examples
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and updates

## What's New in v0.6.0

**Checkpoint Resumption:**
- `manager.resume_experiment()` - Automatically resume from checkpoints
- Smart checkpoint detection (best, latest, epoch-specific)
- Support for PyTorch (.pth), PyTorch Lightning (.ckpt) formats
- Epoch tracking and resume metadata
- Config inheritance with overrides
- Perfect for recovering from time limits or continuing training

**Previous (v0.5.0) - Experiment Pruning:**
- `expflow prune` - Clean up duplicate runs and invalid experiments
- Safe archival to `.archive/` instead of permanent deletion
- Typical savings: 10-25 GB on systems with many duplicate runs

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## Requirements

- Python 3.8+
- SLURM-based HPC cluster
- Linux environment

## Use Cases

| Use Case | Status |
|----------|--------|
| Image Classification (ResNet, ViT) | ✓ Ready |
| LLM Fine-tuning (LLaMA, GPT) | ✓ Ready |
| Reinforcement Learning (PPO, SAC) | ✓ Ready |
| Computer Vision (Object Detection) | ✓ Ready |

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Share your experiment templates:
```bash
examples/templates/your_usecase.yaml
```

## License

MIT License - see [LICENSE](LICENSE)

## Support

- **Issues**: [GitHub Issues](https://github.com/hurryingauto3/expflow-hpc/issues)
- **Documentation**: [USER_GUIDE.md](USER_GUIDE.md)
- **NYU HPC**: [NYU HPC Wiki](https://sites.google.com/nyu.edu/nyu-hpc/)

## Acknowledgments

Built for the NYU HPC deep learning community. Works on any SLURM-based cluster.

**Maintained by:** [Ali Hamza](https://github.com/hurryingauto3)

---

**Stop fighting SLURM. Start doing research.**

For complete documentation, see [USER_GUIDE.md](USER_GUIDE.md).
