# ExpFlow - HPC Experiment Manager

> Lightweight experiment tracking for HPC clusters. Stop manually editing SLURM scripts.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SLURM](https://img.shields.io/badge/HPC-SLURM-orange.svg)](https://slurm.schedmd.com/)

**ExpFlow** auto-detects your HPC environment (username, scratch paths, SLURM accounts) and automates experiment tracking - no hardcoded paths, no manual script editing, no Excel spreadsheets.

```bash
# Install
pip install git+https://github.com/hurryingauto3/expflow-hpc.git

# Initialize project (auto-detects YOUR username and paths!)
expflow init my-research

# Create and run experiments
cd /scratch/YOUR_ID/my-research
python -m expflow.examples.simple new --exp-id exp001 --template baseline
python -m expflow.examples.simple submit exp001
```

## Features

- **Auto-detects** your username, scratch directory, SLURM accounts
- **YAML-based configs** - no more editing SLURM scripts
- **Resource advisor** - real-time GPU availability and recommendations
- **Complete tracking** - git commits, job IDs, timestamps, results
- **Reproducible** - warns about GPU/precision changes
- **Extensible** - subclass for your specific use case

## Quick Start

### 1. Install

**With pip:**
```bash
pip install git+https://github.com/hurryingauto3/expflow-hpc.git
```

**With conda:**
```bash
# Create environment
conda create -n expflow python=3.10
conda activate expflow

# Install
pip install git+https://github.com/hurryingauto3/expflow-hpc.git
```

**From source:**
```bash
git clone https://github.com/hurryingauto3/expflow-hpc.git
cd expflow-hpc

# With pip
pip install -e .

# With conda
conda create -n expflow python=3.10
conda activate expflow
pip install -e .
```

### 2. Initialize Your Project

```bash
# Auto-detects YOUR environment
expflow init my-research

# Output:
# [OK] Detected: greene cluster, user: YOUR_ID
# [OK] Created: /scratch/YOUR_ID/my-research
```

### 3. Create Experiment

```bash
cd /scratch/YOUR_ID/my-research

# Create from template
expflow template baseline

# Edit template
vim experiment_templates/baseline.yaml
```

### 4. Run

```bash
# Create experiment
python -m expflow.examples.simple new \
    --exp-id exp001 \
    --template baseline \
    --description "Baseline ResNet50"

# Check resources
expflow resources --status

# Submit
python -m expflow.examples.simple submit exp001

# Harvest results
python -m expflow.examples.simple harvest exp001

# Export to CSV
python -m expflow.examples.simple export results.csv
```

## Documentation

- **[Getting Started](docs/getting-started.md)** - Complete tutorial
- **[User Guide](docs/user-guide.md)** - Full documentation
- **[API Reference](docs/api-reference.md)** - For developers
- **[Examples](examples/)** - Image classification, LLM fine-tuning, RL

## Use Cases

| Use Case | Example | Status |
|----------|---------|--------|
| Image Classification | ResNet, ViT on ImageNet | Ready |
| LLM Fine-tuning | LLaMA, GPT with LoRA | Ready |
| Reinforcement Learning | PPO, SAC on Atari | Ready |
| NavSim Planning | I-JEPA planning agents | Ready |

See [`examples/`](examples/) for complete implementations.

## Why ExpFlow?

**Before:**
```bash
# Edit SLURM script manually
vim train.slurm

# Hardcoded paths
#SBATCH --account=torch_pr_68_tandon  # Only works for one user!
export DATA=/scratch/ah7072/data      # Hardcoded!

sbatch train.slurm
# → Manually track in Excel
# → Forget git commit
# → Lose hyperparameters
```

**After:**
```bash
# Define once in YAML (works for ANY user)
expflow init my-project

# Create experiment (auto-generates SLURM scripts)
python -m expflow.examples.simple new --exp-id exp001 --template baseline

# Submit (auto-detects YOUR paths, account, GPUs)
python -m expflow.examples.simple submit exp001

# Auto-harvest results
python -m expflow.examples.simple harvest exp001
python -m expflow.examples.simple export results.csv
```

**Time Saved:** ~80% reduction in experiment setup time

## For Your Research

Create a custom experiment manager in 3 steps:

```python
from expflow import BaseExperimentManager

class MyManager(BaseExperimentManager):
    def _generate_train_script(self, config):
        """Your training SLURM script"""
        return f'''#!/bin/bash
#SBATCH --gres=gpu:{config['num_gpus']}
python train.py --model {config['model']} ...
'''

    def _generate_eval_script(self, config):
        """Your evaluation script"""
        return "#!/bin/bash\npython evaluate.py ..."

    def harvest_results(self, exp_id):
        """Parse your results"""
        return {"accuracy": 0.95}
```

See **[Creating Custom Managers](docs/custom-managers.md)** for details.

## Highlights

### Auto-Detection
```bash
$ expflow info
Cluster: greene
Username: YOUR_ID (auto-detected!)
Scratch: /scratch/YOUR_ID (auto-detected!)
Account: your_slurm_account (auto-detected!)
Partitions: l40s_public, h200_tandon, ...
```

### Resource Advisor
```bash
$ expflow resources --status

L40S_PUBLIC
   Available: 12/40 GPUs
   Status: Ready now

H200_TANDON
   Available: 0/10 GPUs
   Queue: 8 jobs
   Wait: ~4 hours
   Status: Busy

Recommendation: Use L40S×4 now
```

### Reproducibility
```
WARNING: GPU type changed from L40S to H200
WARNING: Batch config: 96/GPU × 2 = 192 global (Matches)
WARNING: Consider locking precision mode (bf16)
```

## Contributing

Contributions welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md).

**Share your templates:**
```bash
examples/templates/your_usecase.yaml
```

## License

MIT License - see [`LICENSE`](LICENSE)

## Acknowledgments

Built for the NYU HPC deep learning community. Works on any SLURM-based cluster.

**Maintained by:** [Ali Hamza](https://github.com/hurryingauto3)

## Support

- **Issues:** [GitHub Issues](https://github.com/hurryingauto3/expflow-hpc/issues)
- **Docs:** [Full Documentation](docs/)
- **NYU HPC:** [NYU HPC Wiki](https://sites.google.com/nyu.edu/nyu-hpc/)

---

**Stop fighting SLURM. Start doing research.**
