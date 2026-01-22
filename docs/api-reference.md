# API Reference

Python API documentation for ExpFlow.

## Core Classes

### `BaseExperimentManager`

Base class for custom experiment managers.

```python
from expflow import BaseExperimentManager

class MyManager(BaseExperimentManager):
    def __init__(self, hpc_config):
        super().__init__(hpc_config)
```

**Methods to Implement:**

- `_generate_train_script(config: dict) -> str` - Generate training SLURM script
- `_generate_eval_script(config: dict) -> str` - Generate evaluation SLURM script
- `harvest_results(exp_id: str) -> dict` - Parse and return results

**Provided Methods:**

- `create_experiment(exp_id, template=None, description="", **kwargs)` - Create experiment
- `submit_experiment(exp_id, train_only=False, eval_only=False, dry_run=False)` - Submit to SLURM
- `list_experiments(status=None, tags=None)` - List experiments
- `show_experiment(exp_id)` - Show experiment details
- `export_results(output_file="results.csv")` - Export to CSV

**Attributes:**

- `hpc_config: HPCConfig` - HPC configuration
- `metadata: dict` - Experiment metadata database
- `experiments_dir: Path` - Experiments output directory
- `logs_dir: Path` - SLURM logs directory
- `checkpoints_dir: Path` - Model checkpoints directory

### `HPCConfig`

HPC environment configuration (auto-detected).

```python
from expflow import HPCConfig, load_project_config

config = load_project_config()
```

**Attributes:**

- `username: str` - Current username (auto-detected)
- `scratch_dir: str` - Scratch directory path
- `project_name: str` - Project name
- `project_root: str` - Project root directory
- `default_account: str` - Default SLURM account
- `default_partition: str` - Default SLURM partition
- `available_partitions: List[str]` - Available partitions

### `ResourceAdvisor`

GPU resource analysis and recommendations.

```python
from expflow import ResourceAdvisor

advisor = ResourceAdvisor()
status = advisor.get_queue_status()
recommendations = advisor.get_recommendations()
```

**Methods:**

- `get_queue_status() -> Dict[str, PartitionInfo]` - Query SLURM for partition status
- `get_recommendations(exp_config=None, target_global_batch=192) -> List[ResourceRecommendation]`
- `print_status()` - Print current resource status
- `print_recommendations(recommendations, gemini_suggestion=None)` - Print recommendations

## Functions

### `initialize_project(project_name: str) -> HPCConfig`

Initialize a new HPC experiment project.

```python
from expflow import initialize_project

config = initialize_project("my-research")
```

### `load_project_config(project_root=None) -> HPCConfig`

Load project configuration.

```python
from expflow import load_project_config

config = load_project_config()  # Auto-finds .hpc_config.yaml
```

## Data Classes

### `BaseExperimentConfig`

Base configuration for experiments. Extend for your use case.

```python
from expflow import BaseExperimentConfig
from dataclasses import dataclass

@dataclass
class MyConfig(BaseExperimentConfig):
    model: str = "resnet50"
    dataset: str = "imagenet"
```

**Base Fields:**

- `exp_id: str` - Experiment ID
- `description: str` - Description
- `partition: str` - SLURM partition
- `num_gpus: int` - Number of GPUs
- `num_nodes: int` - Number of nodes
- `created_at: Optional[str]` - Creation timestamp
- `git_commit: Optional[str]` - Git commit hash
- `tags: List[str]` - Organization tags

### `ExperimentMetadata`

Runtime metadata for tracking experiments.

**Fields:**

- `exp_id: str` - Experiment ID
- `config: Any` - Experiment configuration
- `status: str` - Status (created, submitted, training, completed, failed)
- `train_job_id: Optional[str]` - Training SLURM job ID
- `results: Dict[str, Any]` - Harvested results

## Environment Detection

### `HPCEnvironment`

Static methods for environment detection.

```python
from expflow import HPCEnvironment

username = HPCEnvironment.get_username()
scratch = HPCEnvironment.get_scratch_dir()
cluster = HPCEnvironment.detect_cluster()
accounts = HPCEnvironment.get_slurm_accounts()
```

**Methods:**

- `get_username() -> str` - Get current username
- `get_home_dir() -> str` - Get home directory
- `get_scratch_dir() -> str` - Auto-detect scratch directory
- `detect_cluster() -> str` - Detect cluster name
- `get_slurm_accounts() -> List[str]` - Get SLURM accounts
- `get_available_partitions() -> List[str]` - Get available partitions

## Examples

### Complete Custom Manager

```python
from expflow import BaseExperimentManager, load_project_config
import argparse

class ImageClassificationManager(BaseExperimentManager):
    def _generate_train_script(self, config):
        return f'''#!/bin/bash
#SBATCH --partition={config['partition']}
#SBATCH --gres=gpu:{config['num_gpus']}
python train.py --model {config['model']}
'''

    def _generate_eval_script(self, config):
        return "#!/bin/bash\npython evaluate.py ..."

    def harvest_results(self, exp_id):
        import json
        with open(f"results_{exp_id}.json") as f:
            return json.load(f)

if __name__ == "__main__":
    config = load_project_config()
    manager = ImageClassificationManager(config)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Add commands: new, submit, harvest, list, export
    # ...

    args = parser.parse_args()
    # Handle commands
```

See [`examples/`](../examples/) for full implementations.
