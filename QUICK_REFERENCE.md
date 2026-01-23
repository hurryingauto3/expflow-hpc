# ExpFlow Quick Reference

## Installation

```bash
pip install git+https://github.com/hurryingauto3/expflow-hpc.git
```

## Core Commands

### Project Setup
```bash
expflow init -i my-research          # Interactive setup (recommended)
expflow init -q my-research          # Quick setup with defaults
expflow init my-research             # Legacy auto-detect
```

### Environment
```bash
expflow info                         # Show HPC environment
expflow config                       # Show project config
```

### Resources
```bash
expflow resources --status           # GPU availability
expflow partitions                   # Partition-account map
expflow partitions --json            # Export as JSON
```

### Experiments
```bash
expflow status                       # Show all experiments + jobs
expflow list                         # List experiments
expflow list --status running        # Filter by status
```

### Logs
```bash
expflow logs exp001                  # View last 50 lines
expflow logs exp001 -n 100           # Last 100 lines
expflow logs exp001 --type eval      # Evaluation logs
expflow logs exp001 -e               # Error logs
expflow tail exp001                  # Follow in real-time
```

### Job Control
```bash
expflow cancel exp001                # Cancel all jobs
expflow cancel exp001 --type train   # Cancel specific job
```

### Templates
```bash
expflow template baseline            # Create template
```

## Interactive Init Flow

```bash
$ expflow init -i my-research

[1/4] Detecting environment...
[2/4] Selecting account...
[3/4] Selecting GPU partition...
[4/4] Additional settings...

Configuration Summary → Confirm → Done!
```

## Custom Manager Template

```python
from expflow import BaseExperimentManager

class MyManager(BaseExperimentManager):
    def _generate_train_script(self, config):
        return f'''#!/bin/bash
#SBATCH --gres=gpu:{config['num_gpus']}
python train.py --model {config['model']}
'''

    def _generate_eval_script(self, config):
        return "#!/bin/bash\npython evaluate.py ..."

    def harvest_results(self, exp_id):
        return {"accuracy": 0.95}
```

## Experiment Template (YAML)

```yaml
description: "My experiment"

# Parameters
model: resnet50
batch_size: 256
learning_rate: 0.1

# Resources (auto-filled from config)
partition: l40s_public
account: my_account
num_gpus: 4
time_limit: "48:00:00"
```

## Common Workflows

### Start New Project
```bash
expflow init -i my-research
cd /scratch/$USER/my-research
expflow template baseline
vim experiment_templates/baseline.yaml
```

### Check Before Submitting
```bash
expflow resources --status           # Check GPU availability
expflow partitions                   # Verify partition access
```

### Monitor Running Experiments
```bash
expflow status                       # Overview
expflow tail exp001                  # Follow logs
expflow logs exp001 -e               # Check errors
```

### Cancel and Restart
```bash
expflow cancel exp001
# Fix issues, then resubmit
python my_manager.py submit exp001
```

## GPU Partition Types

- **H200**: 96GB HBM3, newest, most powerful
- **L40s**: 48GB, great availability
- **A100**: 40-80GB, solid choice
- **RTX8000**: 48GB, older but available

## Account Types

- `*_general`: Broadest access (recommended)
- `*_public`: Public access
- `*_tandon`, `*_courant`: School-specific

## Status Values

- `created`: Experiment defined, not submitted
- `submitted`: Job submitted to SLURM
- `running`: Currently executing
- `completed`: Finished successfully
- `failed`: Job failed
- `cancelled`: Manually cancelled

## Quick Troubleshooting

### No partitions detected
```bash
# Check SLURM is available
sinfo --version

# Check accounts
sacctmgr show associations user=$USER format=Account -n

# Try legacy mode
expflow init my-research
```

### Partition access denied
```bash
# Check access map
expflow partitions

# Use recommended combinations
expflow init -i my-research
```

### Experiment not found
```bash
expflow list                         # See all experiments
cat experiment_configs/experiments.json
```

## Useful Aliases

Add to your `~/.bashrc`:

```bash
alias eflow='expflow'
alias es='expflow status'
alias el='expflow list'
alias er='expflow resources --status'
alias ep='expflow partitions'
```

## Documentation

- **README.md**: Project overview and quick start
- **USER_GUIDE.md**: Complete documentation with examples
- **CHANGELOG.md**: Version history
- **QUICK_REFERENCE.md**: This cheat sheet

## Support

- Issues: https://github.com/hurryingauto3/expflow-hpc/issues
- Docs: See USER_GUIDE.md
- NYU HPC: https://sites.google.com/nyu.edu/nyu-hpc/
