# ExpFlow User Guide

Complete guide to using ExpFlow for HPC experiment management.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Interactive Initialization](#interactive-initialization)
4. [Experiment Management](#experiment-management)
5. [Monitoring Experiments](#monitoring-experiments)
6. [Resource Management](#resource-management)
7. [Partition and Account Management](#partition-and-account-management)
8. [Cache Building](#cache-building)
9. [Creating Custom Managers](#creating-custom-managers)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)

---

## Installation

### Option 1: Install from GitHub (Recommended)

```bash
pip install git+https://github.com/hurryingauto3/expflow-hpc.git
```

### Option 2: Install with Conda

```bash
# Create environment
conda create -n expflow python=3.10
conda activate expflow

# Install
pip install git+https://github.com/hurryingauto3/expflow-hpc.git
```

### Option 3: Install from Source

```bash
git clone https://github.com/hurryingauto3/expflow-hpc.git
cd expflow-hpc
pip install -e .
```

### Verify Installation

```bash
expflow --help
```

---

## Quick Start

### 1. Initialize Your Project

ExpFlow offers three initialization modes:

#### Interactive Mode (Recommended)
```bash
expflow init -i my-research
```

Interactive mode walks you through:
- Account selection with intelligent recommendations
- GPU/Partition selection with categorization (H200, L40s, A100, RTX8000)
- Time limit preferences
- Configuration summary and confirmation

#### Quick Mode
```bash
expflow init -q my-research
```

Quick mode uses smart defaults:
- Prefers "general" accounts for broad access
- Selects l40s_public or h200_public partitions
- Sets 48-hour time limit

#### Legacy Mode
```bash
expflow init my-research
```

Auto-detects everything with minimal prompts.

### 2. Navigate to Your Project

```bash
cd /scratch/YOUR_ID/my-research
```

### 3. Create an Experiment Template

```bash
expflow template baseline
```

This creates `experiment_templates/baseline.yaml`:

```yaml
# Experiment template
description: "Baseline experiment"

# Add your project-specific parameters here
model: your_model
dataset: your_dataset
batch_size: 256
learning_rate: 0.1

# Resource configuration
partition: l40s_public
account: your_account
num_gpus: 4
num_nodes: 1
cpus_per_task: 16
time_limit: "48:00:00"

# Tags
tags:
  - baseline
```

### 4. View Environment Info

```bash
expflow info
```

Output:
```
======================================================================
HPC Environment Information
======================================================================
Username: ah7072
Home: /home/ah7072
Scratch: /scratch/ah7072
Cluster: greene
Accounts: torch_pr_68_general, torch_pr_68_tandon_advanced
Partitions: l40s_public, h200_public, rtx8000, ...
```

---

## Interactive Initialization

Interactive initialization provides the best setup experience with guided menus and intelligent recommendations.

### Step-by-Step Walkthrough

#### Step 1: Environment Detection

ExpFlow automatically detects:
- Cluster name (e.g., greene)
- Your username
- Scratch directory path

```
[1/4] Detecting HPC environment...
  Cluster: greene
  Username: ah7072
  Scratch: /scratch/ah7072
```

#### Step 2: Account Selection

ExpFlow shows all your SLURM accounts with recommendations:

```
[2/4] Selecting SLURM account...

  Available accounts (2):
    1. torch_pr_68_general [RECOMMENDED: Broadest access]
    2. torch_pr_68_tandon_advanced

  Select account [1-2] (default: 1):
```

**Recommendation Logic:**
- `*_general` accounts → Broadest access (RECOMMENDED)
- `*_public` accounts → Public access
- Other accounts → Specific access

#### Step 3: GPU/Partition Selection

ExpFlow analyzes partition access and categorizes by GPU type:

```
[3/4] Selecting default GPU partition...
  Analyzing GPU partition access...
Testing 7 GPU partitions with 2 accounts...

  Accessible partitions with account 'torch_pr_68_general':

  H200 GPUs:
    1. h200_public (H200) [RECOMMENDED: Powerful & available]

  L40s GPUs:
    2. l40s_public (L40s) [RECOMMENDED: Best availability]

  Select partition [1-2] (default: 2):
```

**GPU Categories:**
- **H200**: Newest, most powerful (96GB HBM3)
- **L40s**: Great availability, good performance (48GB)
- **A100**: Solid choice (40GB or 80GB)
- **RTX8000**: Older but available (48GB)

#### Step 4: Additional Settings

```
[4/4] Additional settings...

  Default time limit for jobs:
    1. 6 hours
    2. 12 hours
    3. 24 hours
    4. 48 hours [RECOMMENDED]
    5. 72 hours
    6. Custom

  Select time limit [1-6] (default: 4):
```

#### Step 5: Configuration Summary

```
======================================================================
Configuration Summary
======================================================================
  Project: my-research
  Location: /scratch/ah7072/my-research
  Account: torch_pr_68_general
  Default GPU: l40s_public
  Time Limit: 48:00:00
======================================================================

Proceed with this configuration? [Y/n]:
```

#### Step 6: Directory Creation

After confirmation, ExpFlow creates:
```
Creating project directories...
   /scratch/ah7072/my-research/experiments
   /scratch/ah7072/my-research/experiments/logs/output
   /scratch/ah7072/my-research/experiments/logs/error
   /scratch/ah7072/my-research/experiments/cache
   /scratch/ah7072/my-research/experiments/checkpoints

 Configuration saved to /scratch/ah7072/my-research/.hpc_config.yaml
```

---

## Experiment Management

ExpFlow v0.3.4+ includes built-in experiment monitoring commands.

### View Experiment Status

```bash
expflow status
```

Shows:
- Active SLURM jobs (running/pending)
- Recent experiments with job IDs
- Current status of each experiment

Output:
```
================================================================================
Experiment Status
================================================================================

Active Jobs (2):
Experiment      Type   JobID      State      Time         Node
--------------------------------------------------------------------------------
exp001          train  990410     RUNNING    1:23:45      gr001
exp002          eval   990411     PENDING    0:00:00

Recent Experiments:
ID              Status       Train Job    Eval Job     Description
--------------------------------------------------------------------------------
exp003          submitted    990412       990413       ResNet50 baseline
exp002          running      990411       -            Ablation study lr=0.01
exp001          running      990410       -            Baseline experiment
```

### List All Experiments

```bash
expflow list
```

Filter by status:
```bash
expflow list --status running
expflow list --status completed
expflow list --status failed
```

### View Experiment Logs

```bash
# View last 50 lines of training log
expflow logs exp001

# View last 100 lines
expflow logs exp001 -n 100

# View evaluation logs
expflow logs exp001 --type eval

# View error logs
expflow logs exp001 -e
expflow logs exp001 --errors
```

### Follow Logs in Real-Time

```bash
# Tail training logs (like tail -f)
expflow tail exp001

# Tail evaluation logs
expflow tail exp001 --type eval

# Tail error logs
expflow tail exp001 -e
```

Press `Ctrl+C` to stop tailing.

### Cancel Running Jobs

```bash
# Cancel all jobs for an experiment
expflow cancel exp001

# Cancel only training job
expflow cancel exp001 --type train

# Cancel only evaluation job
expflow cancel exp001 --type eval
```

---

## Resource Management

### Check GPU Availability

```bash
expflow resources --status
```

Output:
```
======================================================================
GPU Resource Status
======================================================================

L40S_PUBLIC
   Total GPUs: 40
   Available: 12
   In Use: 28
   Queue: 3 jobs
   Status: AVAILABLE

H200_TANDON
   Total GPUs: 10
   Available: 0
   In Use: 10
   Queue: 8 jobs
   Wait Time: ~4 hours
   Status: BUSY

======================================================================
Recommendations
======================================================================

For your workload (global_batch=192):
  Recommended: l40s_public with 4 GPUs
  Reason: Best availability right now
```

### Get Resource Recommendations

```bash
expflow resources --per-gpu-batch 48 --global-batch 192
```

Calculates:
- Number of GPUs needed
- Batch size distribution
- Current availability per partition

### Advanced Resource Planning

```bash
# Use Gemini AI for recommendations (requires GEMINI_API_KEY)
expflow resources --use-gemini

# Custom batch sizes
expflow resources --per-gpu-batch 64 --global-batch 256
```

---

## Partition and Account Management

### View Partition-Account Access Map

```bash
expflow partitions
```

Output:
```
======================================================================
Partition Access Map
======================================================================

h200_public (GPU: H200) [GPU Required]
  ✓ torch_pr_68_general
  ✓ torch_pr_68_tandon_advanced

l40s_public (GPU: L40s) [GPU Required]
  ✓ torch_pr_68_general

rtx8000 (GPU: RTX8000) [GPU Required]
  ✓ torch_pr_68_general

======================================================================
Account Access Summary
======================================================================

torch_pr_68_general
  Can access: h200_public, l40s_public, rtx8000

torch_pr_68_tandon_advanced
  Can access: h200_public, h200_tandon
```

### Export as JSON

```bash
expflow partitions --json
```

Output:
```json
{
  "h200_public": ["torch_pr_68_general", "torch_pr_68_tandon_advanced"],
  "l40s_public": ["torch_pr_68_general"],
  "rtx8000": ["torch_pr_68_general"]
}
```

### Understanding Partition Access

ExpFlow uses `sbatch --test-only` to validate partition-account combinations without submitting jobs.

**Common Partition Types:**
- `*_public`: Public access, usually available to all accounts
- `*_general`: General purpose, broadest access
- `*_tandon`, `*_courant`, etc.: School/department specific
- `*_advanced`: May require special access or quotas

---

## Cache Building

ExpFlow provides a generic framework for building and managing data caches on HPC. This is useful for preprocessing datasets, extracting features, or precomputing metrics.

### Cache Building Pipeline

The cache building process has 3 stages:

1. **Build Cache**: Extract/preprocess data (CPU-intensive, parallel)
2. **SquashFS Compression**: Convert to compressed read-only filesystem (saves inodes)
3. **Cleanup**: Remove original directory after verifying compression

### Why SquashFS?

HPC filesystems have inode limits. Large caches with millions of files can exhaust your quota. SquashFS solves this:
- Converts directory with 1M files → single `.sqsh` file (1 inode)
- Read-only filesystem, mounted at runtime
- Compression reduces storage (typically 20-40% of original size)
- No performance penalty for training (mounted directly)

### Using the Cache Builder

#### Step 1: Create Your Cache Builder

Extend `BaseCacheBuilder` to implement project-specific cache generation:

```python
from expflow import BaseCacheBuilder, CacheConfig

class MyCacheBuilder(BaseCacheBuilder):
    """Custom cache builder for my project"""

    def _generate_cache_build_script(self, config: CacheConfig) -> str:
        """Generate the cache building SLURM script"""

        cache_type = config.cache_type
        cache_dir = config.cache_output_dir

        if cache_type == "training":
            return self._generate_training_cache_script(config)
        elif cache_type == "validation":
            return self._generate_validation_cache_script(config)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

    def _generate_training_cache_script(self, config: CacheConfig) -> str:
        """Generate training cache building script"""

        # Extract project-specific parameters
        dataset = config.cache_params.get("dataset", "imagenet")
        image_size = config.cache_params.get("image_size", 224)

        script = f'''#!/bin/bash
#SBATCH --job-name=cache_{config.cache_name}
#SBATCH --partition={config.partition}
#SBATCH --cpus-per-task={config.num_cpus}
#SBATCH --mem={config.memory}
#SBATCH --time={config.time_limit}

# Your cache building command
python preprocess_dataset.py \\
    --dataset {dataset} \\
    --output {config.cache_output_dir} \\
    --image-size {image_size} \\
    --workers {config.num_workers}
'''
        return script

    def get_cache_script_command(self, config: CacheConfig) -> str:
        """Get Python command for cache building (for documentation)"""
        dataset = config.cache_params.get("dataset", "imagenet")
        return f"python preprocess_dataset.py --dataset {dataset}"
```

#### Step 2: Create Cache Configuration

```python
builder = MyCacheBuilder()

# Create cache config
builder.create_cache_config(
    cache_name="imagenet_train_224",
    cache_type="training",
    description="ImageNet training set preprocessed to 224x224",
    partition="cpu",  # Usually CPU for caching
    num_cpus=128,
    memory="256G",
    time_limit="24:00:00",
    num_workers=96,
    cache_params={
        "dataset": "imagenet",
        "split": "train",
        "image_size": 224
    }
)
```

#### Step 3: Run Cache Pipeline

```bash
# Option 1: Run full pipeline (recommended)
# Automatically chains: build → squashfs → cleanup
python my_cache_builder.py pipeline imagenet_train_224

# Option 2: Run steps manually
python my_cache_builder.py build imagenet_train_224
python my_cache_builder.py squashfs imagenet_train_224  # Waits for build
python my_cache_builder.py cleanup imagenet_train_224   # Waits for squashfs

# Dry run to preview scripts
python my_cache_builder.py pipeline imagenet_train_224 --dry-run
```

#### Step 4: Use Cache in Training

Mount the SquashFS overlay in your training script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:4

# Both cache directory and overlay are under experiments/cache/
CACHE_DIR="/scratch/USER/experiments/cache/imagenet_train_224"
CACHE_SQSH="/scratch/USER/experiments/cache/overlays/imagenet_train_224.sqsh"

# Run with overlay mounted
apptainer exec \\
    --nv \\
    --bind ${CACHE_SQSH}:${CACHE_DIR}:image-src=/ \\
    container.sif \\
    python train.py --data ${CACHE_DIR}
```

### Complete Example: NAVSIM Cache Builder

See `examples/navsim_cache_builder.py` for a complete implementation:

```python
from expflow import BaseCacheBuilder, CacheConfig

class NavsimCacheBuilder(BaseCacheBuilder):
    """Builds training and metric caches for NAVSIM experiments"""

    def _generate_cache_build_script(self, config: CacheConfig) -> str:
        if config.cache_type == "training":
            return self._generate_training_cache_script(config)
        elif config.cache_type == "metric":
            return self._generate_metric_cache_script(config)

    def _generate_training_cache_script(self, config: CacheConfig) -> str:
        agent = config.cache_params["agent"]
        split = config.cache_params["train_split"]

        return f'''#!/bin/bash
# SLURM directives...

python run_dataset_caching.py \\
    agent={agent} \\
    train_test_split={split} \\
    cache_path={config.cache_output_dir} \\
    worker.threads_per_node={config.num_workers}
'''

    # ... metric cache implementation
```

**Usage:**

```bash
# Create training cache
python navsim_cache_builder.py new training_cache_v4_6cams \\
    --type training \\
    --agent ijepa_planning_agent_v4 \\
    --num-cams 6 \\
    --description "6-camera training cache"

# Run pipeline
python navsim_cache_builder.py pipeline training_cache_v4_6cams

# Create metric cache
python navsim_cache_builder.py new navhard_metric_cache \\
    --type metric \\
    --eval-split navhard_two_stage \\
    --description "Metric cache for navhard eval"

python navsim_cache_builder.py pipeline navhard_metric_cache

# List caches
python navsim_cache_builder.py list
python navsim_cache_builder.py show training_cache_v4_6cams
```

### Cache Management Commands

```bash
# List all caches
python cache_builder.py list

# Filter by type
python cache_builder.py list --type training

# Show cache details
python cache_builder.py show cache_name

# Build only (no compression)
python cache_builder.py build cache_name

# Compress existing cache
python cache_builder.py squashfs cache_name

# Cleanup after compression
python cache_builder.py cleanup cache_name
```

### Cache Configuration Options

```python
CacheConfig(
    cache_name="my_cache",           # Unique identifier
    cache_type="training",           # Project-specific type
    description="Description",       # Human-readable description

    # Paths (auto-detected, usually don't need to override)
    cache_output_dir="/scratch/USER/experiments/cache/my_cache",
    overlay_output_dir="/scratch/USER/experiments/cache/overlays",

    # Resources (for cache building)
    partition="cpu",                 # Usually CPU
    account="my_account",
    num_cpus=128,
    memory="256G",
    time_limit="48:00:00",

    # Cache parameters
    num_workers=96,                  # Parallel workers
    force_rebuild=True,              # Rebuild existing cache
    cache_params={                   # Project-specific params
        "dataset": "imagenet",
        "split": "train",
        "image_size": 224
    },

    # SquashFS options
    squashfs_compression="zstd",     # Compression algorithm
    squashfs_block_size=1048576,     # 1MB blocks
    squashfs_processors=32,          # Compression threads

    # Container (if needed)
    container_image="/path/to/container.sif",
    use_container=False
)
```

### Integrating Cache Building with Experiments

You can integrate cache building into your experiment manager:

```python
from expflow import BaseExperimentManager, BaseCacheBuilder

class MyExperimentManager(BaseExperimentManager):
    def __init__(self, hpc_config):
        super().__init__(hpc_config)
        # Add cache builder
        self.cache_builder = MyCacheBuilder(hpc_config)

    def ensure_cache(self, cache_name, **cache_params):
        """Ensure cache exists, build if needed"""
        if cache_name not in self.cache_builder.cache_metadata:
            self.cache_builder.create_cache_config(
                cache_name=cache_name,
                **cache_params
            )
            self.cache_builder.build_cache_pipeline(cache_name)
            return False  # Cache being built
        return True  # Cache ready
```

### Best Practices

1. **Use CPU partitions** for cache building (no GPU needed)
2. **Test on small subset** before full cache build
3. **Verify SquashFS** works in training before cleanup
4. **Document cache versions** in descriptions
5. **Chain jobs** with `--dependency` to automate pipeline
6. **Monitor inode usage**: `lfs quota -h /scratch`

### Troubleshooting Cache Building

**Problem: Cache build fails**
```bash
# Check logs
tail -f experiments/logs/caching/build_CACHE_NAME_*.err

# Test on single worker
python cache_builder.py new test_cache --num-workers 1
```

**Problem: SquashFS mount fails**
```bash
# Verify file exists
ls -lh /scratch/USER/experiments/cache/overlays/CACHE_NAME.sqsh

# Test mount manually
apptainer exec --bind CACHE.sqsh:/mount/point:image-src=/ container.sif ls /mount/point
```

**Problem: Training can't read cache**
```bash
# Check bind mount syntax
# Correct: --bind overlay.sqsh:/cache/path:image-src=/
# Wrong: --bind overlay.sqsh:/cache/path  (missing :image-src=/)
```

---

## Creating Custom Managers

For project-specific experiment workflows, create a custom manager by subclassing `BaseExperimentManager`.

### Basic Structure

```python
from expflow import BaseExperimentManager

class MyExperimentManager(BaseExperimentManager):
    """Custom manager for my research project"""

    def _generate_train_script(self, config: dict) -> str:
        """Generate SLURM script for training"""
        return f'''#!/bin/bash
#SBATCH --job-name=train_{config['exp_id']}
#SBATCH --partition={config['partition']}
#SBATCH --account={config['account']}
#SBATCH --gres=gpu:{config['num_gpus']}
#SBATCH --cpus-per-task={config['cpus_per_task']}
#SBATCH --time={config['time_limit']}
#SBATCH --output={self.hpc_config.logs_dir}/output/train_{config['exp_id']}_%j.out
#SBATCH --error={self.hpc_config.logs_dir}/error/train_{config['exp_id']}_%j.err

# Load modules
module purge
module load python/3.10

# Activate environment
source ~/.bashrc
conda activate myenv

# Run training
cd {self.hpc_config.project_root}
python train.py \\
    --model {config['model']} \\
    --dataset {config['dataset']} \\
    --batch-size {config['batch_size']} \\
    --lr {config['learning_rate']} \\
    --output {self.hpc_config.checkpoints_dir}/{config['exp_id']}
'''

    def _generate_eval_script(self, config: dict) -> str:
        """Generate SLURM script for evaluation"""
        return f'''#!/bin/bash
#SBATCH --job-name=eval_{config['exp_id']}
#SBATCH --partition={config['partition']}
#SBATCH --account={config['account']}
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Run evaluation
python evaluate.py \\
    --checkpoint {self.hpc_config.checkpoints_dir}/{config['exp_id']} \\
    --output {self.hpc_config.experiments_dir}/results/{config['exp_id']}.json
'''

    def harvest_results(self, exp_id: str) -> dict:
        """Extract results from experiment output"""
        import json
        result_file = f"{self.hpc_config.experiments_dir}/results/{exp_id}.json"

        with open(result_file) as f:
            results = json.load(f)

        return {
            'exp_id': exp_id,
            'accuracy': results['test_accuracy'],
            'loss': results['test_loss'],
            'runtime': results['training_time']
        }
```

### Using Your Custom Manager

```python
# Create CLI for your manager
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="My Experiment Manager")
    parser.add_argument("command", choices=["new", "submit", "status", "harvest", "export"])
    parser.add_argument("--exp-id", help="Experiment ID")
    parser.add_argument("--template", help="Template name")
    parser.add_argument("--description", help="Experiment description")
    parser.add_argument("--output", help="Output file for export")

    args = parser.parse_args()

    manager = MyExperimentManager()

    if args.command == "new":
        manager.create_experiment(
            exp_id=args.exp_id,
            template_name=args.template,
            description=args.description
        )

    elif args.command == "submit":
        manager.submit_experiment(args.exp_id)

    elif args.command == "status":
        manager.status(args.exp_id)

    elif args.command == "harvest":
        results = manager.harvest_results(args.exp_id)
        print(results)

    elif args.command == "export":
        manager.export_results(args.output)

if __name__ == "__main__":
    main()
```

Save as `my_manager.py` and use:

```bash
# Create experiment
python my_manager.py new --exp-id exp001 --template baseline --description "First run"

# Submit to SLURM
python my_manager.py submit --exp-id exp001

# Check status
python my_manager.py status --exp-id exp001

# Harvest results
python my_manager.py harvest --exp-id exp001

# Export all results
python my_manager.py export --output results.csv
```

---

## Advanced Usage

### Environment Variables

ExpFlow respects these environment variables:

- `USER`: Username (auto-detected)
- `HOME`: Home directory (auto-detected)
- `GEMINI_API_KEY`: For AI-powered resource recommendations

### Configuration File

After initialization, your project contains `.hpc_config.yaml`:

```yaml
username: ah7072
user_home: /home/ah7072
scratch_dir: /scratch/ah7072
project_name: my-research
project_root: /scratch/ah7072/my-research
default_account: torch_pr_68_general
default_partition: l40s_public
default_time_limit: '48:00:00'
experiments_dir: /scratch/ah7072/my-research/experiments
logs_dir: /scratch/ah7072/my-research/experiments/logs
cache_dir: /scratch/ah7072/my-research/experiments/cache
checkpoints_dir: /scratch/ah7072/my-research/experiments/checkpoints
cluster_name: greene
available_partitions:
  - l40s_public
  - h200_public
  - rtx8000
```

You can manually edit this file if needed.

### Experiment Metadata

ExpFlow tracks experiments in `experiment_configs/experiments.json`:

```json
{
  "exp001": {
    "config": {
      "exp_id": "exp001",
      "description": "Baseline experiment",
      "template": "baseline",
      "created_at": "2026-01-23T10:30:00",
      "model": "resnet50",
      "batch_size": 256
    },
    "status": "running",
    "train_job_id": "990410",
    "eval_job_id": null,
    "git_commit": "abc123def456",
    "submitted_at": "2026-01-23T10:35:00"
  }
}
```

### Git Integration

ExpFlow automatically tracks git commits for reproducibility:

```python
manager = MyExperimentManager()
manager.create_experiment("exp001", template_name="baseline")
# Automatically records current git commit hash

manager.submit_experiment("exp001")
# Warns if working directory has uncommitted changes
```

### Reproducibility Warnings

ExpFlow warns about changes that may affect reproducibility:

```
WARNING: GPU type changed from L40S to H200
WARNING: Partition changed from l40s_public to h200_public
WARNING: Batch config: 96/GPU × 2 = 192 global (Matches)
WARNING: Consider locking precision mode (bf16)
```

---

## Troubleshooting

### Common Issues

#### 1. No Partitions Detected

**Symptom:**
```
WARNING: No GPU partitions detected
Using fallback partition. You can edit .hpc_config.yaml later.
```

**Solutions:**
- Ensure you're on a SLURM login node (not compute node)
- Check SLURM is available: `sinfo --version`
- Verify your accounts: `sacctmgr show associations user=$USER format=Account -n`
- Try non-interactive mode: `expflow init my-research` (uses fallback defaults)

#### 2. Partition Access Denied

**Symptom:**
```
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```

**Solutions:**
- Check partition-account access: `expflow partitions`
- Use recommended account-partition combinations from interactive init
- Manually test: `sbatch --test-only -p PARTITION -A ACCOUNT -N1 -t 1:00:00 --wrap=hostname`

#### 3. Command Not Found

**Symptom:**
```
expflow: command not found
```

**Solutions:**
- Reinstall: `pip install --force-reinstall git+https://github.com/hurryingauto3/expflow-hpc.git`
- Check installation: `pip show expflow`
- Use full path: `python -m expflow.cli`

#### 4. Project Not Found

**Symptom:**
```
Error: Not in a project directory. Run 'expflow init' first.
```

**Solutions:**
- Navigate to project directory: `cd /scratch/$USER/my-research`
- Verify config exists: `ls .hpc_config.yaml`
- Re-initialize if needed: `expflow init my-research`

#### 5. Experiment Not Found

**Symptom:**
```
Error: Experiment 'exp001' not found
```

**Solutions:**
- List experiments: `expflow list`
- Check metadata: `cat experiment_configs/experiments.json`
- Verify experiment directory exists

### Getting Help

- **GitHub Issues**: [https://github.com/hurryingauto3/expflow-hpc/issues](https://github.com/hurryingauto3/expflow-hpc/issues)
- **Documentation**: See `CHANGELOG.md` for version-specific changes
- **NYU HPC Support**: [HPC Help](https://sites.google.com/nyu.edu/nyu-hpc/)

### Debug Mode

For verbose output:

```bash
# Python logging
export PYTHONVERBOSE=1

# Debug partition detection
python3 -c "
from expflow import PartitionValidator
validator = PartitionValidator()
accounts = validator._get_user_accounts()
print(f'Accounts: {accounts}')
partitions = validator._get_partitions()
print(f'Partitions: {partitions}')
"
```

---

## What's New in v0.3.4

**Experiment Monitoring Commands:**
- `expflow status` - Show all experiments and SLURM jobs
- `expflow list` - List experiments with filtering
- `expflow logs <exp_id>` - View experiment logs
- `expflow tail <exp_id>` - Follow logs in real-time
- `expflow cancel <exp_id>` - Cancel running jobs

**No More Custom Manager Scripts Needed:**
Basic experiment monitoring is now built into the CLI. Create custom managers only for advanced workflows.

See `CHANGELOG.md` for complete version history.

---

## Best Practices

### 1. Use Interactive Init

Always use `expflow init -i` for new projects:
- Gets intelligent recommendations
- Validates partition access
- Sets up optimal configuration

### 2. Version Control Your Templates

Keep experiment templates in git:
```bash
cd /scratch/$USER/my-research
git init
git add experiment_templates/
git commit -m "Add baseline template"
```

### 3. Consistent Naming

Use consistent experiment IDs:
```
exp001_baseline
exp002_lr_sweep
exp003_ablation_dropout
```

### 4. Monitor Resources

Check GPU availability before submitting large batches:
```bash
expflow resources --status
```

### 5. Use Templates

Create templates for common configurations:
```bash
expflow template baseline
expflow template ablation
expflow template sweep
```

### 6. Check Status Regularly

Monitor running experiments:
```bash
expflow status
```

### 7. Document Changes

Use descriptive experiment descriptions:
```python
manager.create_experiment(
    exp_id="exp005",
    template_name="baseline",
    description="Testing new learning rate schedule with warmup"
)
```

---

## Next Steps

1. Read `CHANGELOG.md` for latest features
2. Explore `examples/` for complete implementations
3. Create your custom manager for your research
4. Share your templates with the community!

**Questions?** Open an issue on [GitHub](https://github.com/hurryingauto3/expflow-hpc/issues).

**Stop fighting SLURM. Start doing research.**
