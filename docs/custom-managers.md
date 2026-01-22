# Creating Custom Experiment Managers

This guide shows how to create a custom experiment manager for your research.

## Quick Start

```python
from expflow import BaseExperimentManager

class MyManager(BaseExperimentManager):
    def _generate_train_script(self, config):
        """Generate SLURM training script"""
        return f'''#!/bin/bash
#SBATCH --partition={config['partition']}
#SBATCH --gres=gpu:{config['num_gpus']}
python train.py --model {config['model']}
'''

    def _generate_eval_script(self, config):
        """Generate evaluation script"""
        return "#!/bin/bash\npython evaluate.py ..."

    def harvest_results(self, exp_id):
        """Parse your results"""
        # Parse your result files
        return {"accuracy": 0.95}
```

## Step-by-Step Guide

### 1. Initialize Project

```bash
expflow init my-research
cd /scratch/YOUR_ID/my-research
```

### 2. Create Manager File

Create `my_manager.py`:

```python
#!/usr/bin/env python3
from expflow import BaseExperimentManager, load_project_config
import argparse

class MyManager(BaseExperimentManager):
    # ... implementation ...

# CLI
if __name__ == "__main__":
    config = load_project_config()
    manager = MyManager(config)

    # Add commands: new, submit, harvest, list, export, show
    # See examples/simple_image_classification.py for full CLI
```

### 3. Implement Required Methods

#### `_generate_train_script(config)`

Generates the SLURM training script:

```python
def _generate_train_script(self, config):
    scratch = self.hpc_config.scratch_dir  # Auto-detected!
    username = self.hpc_config.username    # Auto-detected!

    return f'''#!/bin/bash
#SBATCH --partition={config['partition']}
#SBATCH --gres=gpu:{config['num_gpus']}
#SBATCH --account={config['account']}
#SBATCH --output={self.logs_dir}/train_{config['exp_id']}_%j.out

# Your training command
python train.py \\
    --data {scratch}/data \\
    --model {config['model']} \\
    --batch-size {config['batch_size']}
'''
```

#### `_generate_eval_script(config)`

Similar to training, but for evaluation.

#### `harvest_results(exp_id)`

Parse your result files:

```python
def harvest_results(self, exp_id):
    import json

    results_file = self.experiments_dir / f"results_{exp_id}.json"

    with open(results_file) as f:
        results = json.load(f)

    # Update metadata
    self.metadata[exp_id]["results"] = results
    self.metadata[exp_id]["status"] = "completed"
    self._save_metadata()

    return results
```

## Complete Examples

See [`examples/`](../examples/) for full implementations:

- **Image Classification**: `simple_image_classification.py`
- **LLM Fine-tuning**: Coming soon
- **Reinforcement Learning**: Coming soon

## Tips

### Use Auto-Detected Paths

```python
# Good
scratch = self.hpc_config.scratch_dir
data_path = f"{scratch}/data"

# Bad
data_path = "/scratch/ah7072/data"  # Hardcoded!
```

### Access Configuration

```python
# In your methods
partition = config['partition']
num_gpus = config['num_gpus']
account = config.get('account', 'default')
```

### Custom Configuration

Extend `BaseExperimentConfig`:

```python
from expflow import BaseExperimentConfig
from dataclasses import dataclass

@dataclass
class MyConfig(BaseExperimentConfig):
    model: str = "resnet50"
    dataset: str = "imagenet"
    # ... your fields
```

## Next Steps

- See [User Guide](user-guide.md) for full API
- Check [Examples](../examples/) for complete code
- Read [API Reference](api-reference.md) for details
