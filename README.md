# HPC Experiment Manager

**A lightweight, generic experiment tracking framework for NYU HPC clusters (Greene)**

Stop manually editing SLURM scripts, losing track of experiments, and copying results to Excel. This framework automates your entire experiment workflow with zero hardcoded paths - it auto-detects your NYU ID, scratch directory, and SLURM configuration.

## 🎯 What This Solves

### Before (Manual Workflow)
- ❌ Manually edit `.slurm` scripts for each experiment
- ❌ Hardcoded paths like `/scratch/ah7072` everywhere
- ❌ Accidentally overwrite previous experiment scripts
- ❌ Manually track results in Excel (incomplete, error-prone)
- ❌ Forget which git commit you used
- ❌ No idea which GPU partition to use

### After (Automated Workflow)
- ✅ Define experiments in YAML configs
- ✅ Auto-generate unique SLURM scripts per experiment
- ✅ Auto-detect your username, scratch paths, SLURM accounts
- ✅ Automatic metadata tracking (git, timestamps, job IDs)
- ✅ Automatic result harvesting and CSV export
- ✅ Smart resource recommendations

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# On NYU Greene HPC
cd ~
git clone https://github.com/YOUR_USERNAME/hpc-experiment-manager.git
cd hpc-experiment-manager

# Make CLI executable
chmod +x hpcexp

# Add to PATH (optional)
echo 'export PATH="$HOME/hpc-experiment-manager:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Initialize Your Project

```bash
# Auto-detects your NYU ID, scratch dir, SLURM accounts
hpcexp init my_awesome_project

# Output:
# ✓ Detected HPC Environment:
#   Cluster: greene
#   Username: ab1234
#   Scratch: /scratch/ab1234
#   Default Account: torch_pr_68_tandon_advanced
#   Default Partition: l40s_public
```

This creates:
```
/scratch/YOUR_NYU_ID/my_awesome_project/
├── .hpc_config.yaml              # Auto-detected config
├── experiment_configs/           # Your experiment definitions
├── experiment_templates/         # Reusable templates
├── generated_scripts/            # Auto-generated SLURM scripts
└── experiments/                  # Outputs
    ├── logs/
    ├── checkpoints/
    └── cache/
```

### 3. Create Your Experiment Manager

Create `my_experiment_manager.py` in your project:

```python
#!/usr/bin/env python3
from pathlib import Path
from hpc_experiment_manager.hpcexp_core import BaseExperimentManager
from hpc_experiment_manager.hpc_config import load_project_config

class MyExperimentManager(BaseExperimentManager):
    """Custom experiment manager for your project"""

    def _generate_train_script(self, config):
        """Generate training SLURM script"""
        return f'''#!/bin/bash
#SBATCH --partition={config['partition']}
#SBATCH --gres=gpu:{config['num_gpus']}
#SBATCH --nodes={config['num_nodes']}
#SBATCH --cpus-per-task={config['cpus_per_task']}
#SBATCH --time={config['time_limit']}
#SBATCH --account={config['account']}
#SBATCH --job-name={config['exp_id']}_train
#SBATCH --output={self.logs_dir}/output/train_{config['exp_id']}_%j.out

# Your training command
python train.py \\
    --model {config['model']} \\
    --dataset {config['dataset']} \\
    --batch-size {config['batch_size']} \\
    --lr {config['learning_rate']} \\
    --epochs {config['epochs']}
'''

    def _generate_eval_script(self, config):
        """Generate evaluation SLURM script"""
        return f'''#!/bin/bash
#SBATCH --partition={config['partition']}
#SBATCH --cpus-per-task={config.get('eval_cpus', 32)}
#SBATCH --job-name={config['exp_id']}_eval

# Your evaluation command
python evaluate.py --checkpoint $CHECKPOINT
'''

    def harvest_results(self, exp_id):
        """Parse your specific result format"""
        results_file = self.experiments_dir / f"results_{exp_id}.json"
        if results_file.exists():
            import json
            with open(results_file) as f:
                results = json.load(f)

            # Update metadata
            self.metadata[exp_id]["results"] = results
            self.metadata[exp_id]["status"] = "completed"
            self._save_metadata()

            print(f"✓ Harvested results for {exp_id}")
            return results
        return {}

# CLI
if __name__ == "__main__":
    import argparse
    config = load_project_config()
    manager = MyExperimentManager(config)

    # Add standard commands: new, submit, harvest, list, export, show
    # (See examples/simple_image_classification.py for full implementation)
```

### 4. Create Experiment Template

```bash
cd /scratch/YOUR_NYU_ID/my_awesome_project

# Create template
cat > experiment_templates/resnet_baseline.yaml << EOF
# ResNet baseline template
description: "ResNet baseline"

# Model config
model: resnet50
dataset: imagenet
batch_size: 256
learning_rate: 0.1
epochs: 90

# Resources
partition: l40s_public
num_gpus: 4
num_nodes: 1
time_limit: "24:00:00"
EOF
```

### 5. Run Experiments

```bash
cd /scratch/YOUR_NYU_ID/my_awesome_project

# Create experiment
python my_experiment_manager.py new \
    --exp-id exp001 \
    --template resnet_baseline \
    --description "Baseline ResNet50"

# Check resources before submitting
hpcexp resources --status

# Submit experiment
python my_experiment_manager.py submit exp001

# Monitor
squeue -u $USER

# After completion, harvest results
python my_experiment_manager.py harvest exp001

# Export all results
python my_experiment_manager.py export results.csv
```

## 🎨 Features

### 1. Auto-Detection
- ✅ NYU ID / username
- ✅ Scratch directory (`/scratch/YOUR_ID`)
- ✅ Home directory
- ✅ SLURM accounts
- ✅ Available partitions
- ✅ Cluster name (Greene, etc.)

### 2. Resource Advisor
```bash
# Check current GPU availability
hpcexp resources --status

# Get smart recommendations
hpcexp resources --use-gemini  # Optional: AI-powered suggestions
```

Output:
```
📊 L40S_PUBLIC
   Available: 12/40 GPUs
   Status: 🟢 Good availability

📊 H200_TANDON
   Available: 0/10 GPUs
   Queue: 8 jobs
   Wait Time: ~240 minutes
   Status: 🔴 No GPUs available

Recommendation: Use L40S×4 now (ready immediately)
```

### 3. Git Integration
- Automatically tracks git commit, branch, dirty status
- Warns if working tree is dirty
- Links results to exact code version

### 4. Metadata Tracking
- Job IDs (training & evaluation)
- Timestamps (created, submitted, completed)
- Full configuration snapshot
- Resource allocation
- Results

### 5. Template System
- Reusable experiment configs
- Override any parameter
- Share templates across team

## 📚 Examples

See `examples/` directory for complete implementations:

- **`simple_image_classification.py`** - ResNet/ViT on ImageNet
- **`llm_finetuning.py`** - LLaMA/GPT fine-tuning with LoRA
- **`reinforcement_learning.py`** - PPO for Atari/MuJoCo
- **`navsim_planning.py`** - The original NavSim I-JEPA implementation

## 🏗️ Architecture

### Core Framework (`hpcexp_core.py`)
- `BaseExperimentManager` - Subclass this for your project
- Handles: config loading, SLURM submission, metadata tracking
- Abstract methods: `_generate_train_script()`, `_generate_eval_script()`, `harvest_results()`

### Configuration (`hpc_config.py`)
- `HPCConfig` - Auto-detected environment
- `HPCEnvironment` - Detection utilities
- `initialize_project()` - Setup new projects

### Resource Advisor (`resource_advisor.py`)
- Real-time SLURM queue analysis
- GPU availability checking
- Wait time estimation
- Reproducibility warnings (GPU type changes)
- Optional Gemini API integration

### CLI (`hpcexp`)
- Project initialization
- Environment inspection
- Resource recommendations
- Template creation

## 🔧 Advanced Usage

### Multi-Node Training
```yaml
# experiment_templates/distributed_training.yaml
num_nodes: 4
num_gpus: 16  # 4 GPUs per node
partition: l40s_public
```

### Custom Result Parsers
```python
def harvest_results(self, exp_id):
    """Parse custom result format"""
    # Your parsing logic here
    results_dir = self.experiments_dir / f"eval_{exp_id}"
    csv_files = list(results_dir.glob("*.csv"))

    import pandas as pd
    df = pd.read_csv(csv_files[0])

    results = {
        "accuracy": float(df["accuracy"].mean()),
        "loss": float(df["loss"].mean()),
        # ... your metrics
    }

    # Update metadata
    self.metadata[exp_id]["results"] = results
    self._save_metadata()

    return results
```

### Apptainer/Singularity Containers
```yaml
# In .hpc_config.yaml
container_image: /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
use_apptainer: true
```

Then in your script generator:
```python
def _generate_train_script(self, config):
    if self.hpc_config.use_apptainer:
        return f'''#!/bin/bash
#SBATCH ...

apptainer exec --nv \\
    {self.hpc_config.container_image} \\
    python train.py ...
'''
```

## 📊 Reproducibility

The framework helps maintain reproducibility when switching GPUs:

```bash
python my_experiment_manager.py submit exp001

# Output:
# ⚠ GPU type changed from L40S to H200
# ⚠ Batch config: 96/GPU × 2 = 192 global ✓ Matches
# ⚠ Consider locking precision mode (bf16)
```

Checklist automatically shown:
```
Reproducibility Checklist when switching GPUs:
  ✓ Lock global batch size (not per-GPU)
  ✓ Schedule LR by steps (not epochs)
  ✓ Set precision explicitly (bf16/fp16/fp32)
  ✓ Set TF32 mode explicitly
  ✓ Use deterministic settings if critical
  ✓ Run ≥3 seeds for statistical confidence
```

## 🤝 Contributing

This framework is designed for the NYU HPC community. Contributions welcome!

### Sharing Templates
Share your experiment templates:
```bash
# Add to examples/templates/
examples/templates/llm_lora_finetuning.yaml
examples/templates/stable_diffusion_training.yaml
```

### Adding Cluster Support
Currently supports:
- ✅ NYU Greene
- 🚧 NERSC Perlmutter (coming soon)
- 🚧 OLCF Summit (coming soon)

## 📝 License

MIT License - Free for academic and commercial use

## 🙏 Acknowledgments

Created by Ali (NYU Tandon) to solve experiment tracking hell.

Built on top of:
- SLURM Workload Manager
- NYU Greene HPC Cluster
- Optional: Google Gemini API for resource suggestions

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/hpc-experiment-manager/issues)
- **NYU HPC Docs**: [NYU HPC Wiki](https://sites.google.com/nyu.edu/nyu-hpc/)

---

**Made with ❤️ for the NYU deep learning research community**
