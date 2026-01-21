# Generic HPC Experiment Management Framework

## Overview

This experiment management system is designed to be **framework-agnostic** and can be adapted for any deep learning research project on HPC clusters, not just NavSim.

## What Makes It Generic?

### 1. Modular Architecture
- **experiment_manager.py** - Core experiment lifecycle management
- **resource_advisor.py** - HPC resource analysis and recommendations
- **Templates** - Customizable YAML configs for any project
- **Script generators** - Template-based SLURM generation

### 2. Project-Agnostic Design
The system manages:
- ✅ Experiment configurations (any hyperparameters)
- ✅ SLURM script generation (any training script)
- ✅ Metadata tracking (git, timestamps, resources)
- ✅ Result harvesting (customizable parsers)
- ✅ CSV export (configurable columns)

### 3. Customization Points
To adapt for your project, modify:
1. **Templates** - Define your experiment parameters
2. **Script generator** - Customize SLURM script structure
3. **Result harvester** - Parse your specific output format
4. **CSV schema** - Export your required columns

## Adapting for Other Projects

### Example: Image Classification

```yaml
# Template: imagenet_resnet.yaml
exp_id: resnet50_exp1
description: "ResNet50 on ImageNet"

# Training config
model: resnet50
dataset: imagenet
batch_size: 256
learning_rate: 0.1
epochs: 90
optimizer: sgd
momentum: 0.9
weight_decay: 0.0001

# Augmentation
augmentation:
  - random_crop
  - random_flip
  - color_jitter

# Resources
partition: a100_public
num_gpus: 8
num_nodes: 1
```

### Example: NLP Fine-tuning

```yaml
# Template: llm_finetune.yaml
exp_id: llama_ft_1
description: "Llama-7B fine-tuning on custom dataset"

# Model config
base_model: meta-llama/Llama-2-7b-hf
lora_rank: 8
lora_alpha: 16
target_modules:
  - q_proj
  - v_proj

# Training
batch_size: 4
gradient_accumulation: 8
learning_rate: 0.0002
warmup_steps: 100
max_steps: 1000

# Dataset
dataset_path: /path/to/data
max_seq_length: 2048

# Resources
partition: h200_tandon
num_gpus: 2
precision: bf16
```

### Example: Reinforcement Learning

```yaml
# Template: ppo_atari.yaml
exp_id: ppo_breakout_1
description: "PPO on Atari Breakout"

# Algorithm
algorithm: ppo
env_name: BreakoutNoFrameskip-v4
num_envs: 16
num_steps: 128
num_epochs: 4
clip_range: 0.1
vf_coef: 0.5
ent_coef: 0.01

# Network
network_arch: cnn
hidden_sizes: [512, 512]
activation: relu

# Training
total_timesteps: 10000000
learning_rate: 0.00025

# Resources
partition: l40s_public
num_gpus: 1
```

## Generic Components

### 1. Resource Advisor
**Works for any HPC cluster** - Just update partition configs:

```python
PARTITIONS = {
    "your_partition": {
        "gpu_type": "A100",
        "gpus_per_node": 4,
        "total_nodes": 20,
        "arch": "Ampere",
        "memory_gb": 80,
        "fp8_support": False,
        "priority": "normal"
    }
}
```

### 2. Experiment Manager
**Core functions are project-agnostic:**
- `create_experiment()` - Define any config
- `submit_experiment()` - Generate and submit jobs
- `harvest_results()` - Parse outputs (customize parser)
- `export_results()` - Export to CSV (customize schema)

### 3. Metadata Tracking
**Automatic for all projects:**
- Git commit/branch
- Timestamps
- SLURM job IDs
- Hyperparameters
- Custom tags/notes

## Porting to Your Project

### Step 1: Clone the Framework
```bash
# Copy core files
cp experiment_manager.py your_project/
cp resource_advisor.py your_project/

# Create template directory
mkdir your_project/experiment_templates/
```

### Step 2: Create Templates
```bash
# Define your experiment schema
vim your_project/experiment_templates/my_template.yaml
```

Example template structure:
```yaml
# Required fields
exp_id: null
description: null

# Your project-specific parameters
model: resnet50
dataset: imagenet
# ... add all your hyperparameters

# Resource allocation
partition: a100_public
num_gpus: 4
time_limit: "24:00:00"

# Metadata
tags: []
notes: ""
```

### Step 3: Customize Script Generator

Modify `_generate_train_script()` in `experiment_manager.py`:

```python
def _generate_train_script(self, config: ExperimentConfig) -> str:
    """Generate SLURM training script for YOUR project"""

    script = f'''#!/bin/bash
#SBATCH --partition={config.partition}
#SBATCH --gres=gpu:{config.num_gpus}
#SBATCH --job-name={config.exp_id}
# ... SLURM headers

# YOUR project-specific setup
module load python/3.10
source /path/to/your/venv/bin/activate

# YOUR training command
python train.py \\
    --model {config.model} \\
    --dataset {config.dataset} \\
    --batch-size {config.batch_size} \\
    --lr {config.learning_rate} \\
    # ... your arguments
'''
    return script
```

### Step 4: Customize Result Harvester

Modify `harvest_results()`:

```python
def harvest_results(self, exp_id: str) -> Dict[str, Any]:
    """Harvest results for YOUR project"""

    # Find your output directory
    output_dir = self.experiments_dir / f"exp_{exp_id}"

    # Parse YOUR result format
    # Example: Parse training log
    log_file = output_dir / "train.log"
    with open(log_file) as f:
        for line in f:
            if "Final accuracy:" in line:
                accuracy = float(line.split(":")[-1].strip())

    # Example: Parse validation metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)

    results = {
        "accuracy": accuracy,
        "val_loss": metrics["val_loss"],
        "train_time_hours": metrics["train_time"] / 3600,
        # ... your metrics
    }

    # Update metadata
    self.metadata[exp_id].results = results
    self._save_metadata()

    return results
```

### Step 5: Customize CSV Export

Modify `export_results()`:

```python
def export_results(self, output_file: str = "results.csv"):
    """Export YOUR project results"""

    records = []
    for exp_id, meta in self.metadata.items():
        config = meta.config

        record = {
            "exp_id": exp_id,
            "description": config.description,
            # YOUR project-specific columns
            "model": config.model,
            "dataset": config.dataset,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            # ... your hyperparameters

            # Results
            "accuracy": meta.results.get("accuracy"),
            "val_loss": meta.results.get("val_loss"),
            # ... your metrics

            # Standard metadata
            "status": meta.status,
            "git_commit": config.git_commit,
            "created_at": config.created_at,
        }

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
```

## Universal Workflows

### Workflow 1: Hyperparameter Sweep
```bash
# Works for ANY project
for lr in 0.001 0.0001 0.00001; do
    python experiment_manager.py new \
        --exp-id exp_lr_${lr} \
        --template my_template \
        --description "LR sweep: ${lr}" \
        --learning-rate ${lr} \
        --tags hp-sweep lr

    python experiment_manager.py submit exp_lr_${lr}
done
```

### Workflow 2: Model Comparison
```bash
# Compare different models
for model in resnet50 vit_base efficientnet_b0; do
    python experiment_manager.py new \
        --exp-id ${model}_exp \
        --template imagenet \
        --description "${model} baseline" \
        --model ${model} \
        --tags model-comparison

    python experiment_manager.py submit ${model}_exp
done
```

### Workflow 3: Reproducibility Study
```bash
# Run multiple seeds
for seed in 42 123 456; do
    python experiment_manager.py new \
        --exp-id exp_seed_${seed} \
        --template my_template \
        --description "Seed ${seed}" \
        --seed ${seed} \
        --tags reproducibility

    python experiment_manager.py submit exp_seed_${seed}
done
```

## Benefits for Any Project

### Organized Experiments
- Never lose track of what you ran
- Easy to compare experiments
- Complete audit trail

### Reproducibility
- Every experiment has full config saved
- Git commit automatically tracked
- Easy to recreate experiments months later

### Efficiency
- No manual script editing
- Automatic result extraction
- Batch job submission

### Collaboration
- Share experiment configs easily
- Consistent naming and organization
- Clear documentation of what was run

## Example: Porting to PyTorch Image Classification

Here's a complete example for a standard PyTorch image classification project:

```python
# Modified dataclass for image classification
@dataclass
class ImageClassificationConfig:
    # Required
    exp_id: str
    description: str

    # Model
    model: str  # resnet50, vit_b_16, efficientnet_b0
    pretrained: bool = True
    num_classes: int = 1000

    # Dataset
    dataset: str  # imagenet, cifar10, cifar100
    data_path: str = "/scratch/datasets"
    train_split: str = "train"
    val_split: str = "val"

    # Training
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    epochs: int = 90
    lr_schedule: str = "step"  # step, cosine, linear
    warmup_epochs: int = 5

    # Augmentation
    use_mixup: bool = False
    use_cutmix: bool = False
    use_autoaugment: bool = False

    # Resources
    partition: str = "a100_public"
    num_gpus: int = 4
    num_workers: int = 8

    # Metadata
    created_at: Optional[str] = None
    git_commit: Optional[str] = None
    accuracy: Optional[float] = None
    tags: List[str] = field(default_factory=list)
```

Then use it exactly like the NavSim version:

```bash
# Create experiment
python experiment_manager.py new \
    --exp-id resnet50_exp1 \
    --template imagenet_resnet \
    --description "Baseline ResNet50" \
    --model resnet50

# Submit
python experiment_manager.py submit resnet50_exp1

# Harvest results
python experiment_manager.py harvest resnet50_exp1

# Export all
python experiment_manager.py export imagenet_results.csv
```

## Integration with Existing Tools

### Weights & Biases
Add to script generator:
```python
# In training script
export WANDB_PROJECT="{config.wandb_project}"
export WANDB_RUN_NAME="{config.exp_id}"

python train.py \\
    --wandb-entity your_team \\
    # ... other args
```

### MLflow
```python
# In result harvester
import mlflow

mlflow.log_params(asdict(config))
mlflow.log_metrics(results)
```

### TensorBoard
```python
# In training script
python train.py \\
    --logdir /scratch/tb_logs/{config.exp_id} \\
    # ... other args
```

## Distribution Package

To make this a proper Python package for NYU HPC users:

```
nyu-hpc-experiment-manager/
├── setup.py
├── README.md
├── nyu_hpc_experiments/
│   ├── __init__.py
│   ├── manager.py          # Core ExperimentManager
│   ├── resource_advisor.py # ResourceAdvisor
│   ├── templates/          # Built-in templates
│   └── utils/
├── examples/
│   ├── navsim/
│   ├── image_classification/
│   ├── nlp_finetuning/
│   └── reinforcement_learning/
└── docs/
```

Install with:
```bash
pip install nyu-hpc-experiment-manager
```

Use in any project:
```python
from nyu_hpc_experiments import ExperimentManager

manager = ExperimentManager(project_name="my_research")
manager.create_experiment(...)
```

## Conclusion

This framework provides a **universal experiment management system** for HPC deep learning research. By customizing the templates, script generator, and result parser, it can be adapted to virtually any project while maintaining the same clean workflow and automatic metadata tracking.

The resource advisor is particularly valuable as it works for **any NYU Greene partition** and helps researchers make informed decisions about GPU allocation and reproducibility.
