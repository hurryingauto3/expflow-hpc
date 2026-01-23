# Copilot Instructions - ExpFlow HPC

## Project Overview
ExpFlow is a lightweight experiment tracking framework for SLURM-based HPC clusters (primarily NYU Greene). Core principle: **auto-detect everything** - no hardcoded usernames, paths, or SLURM accounts.

## Architecture

### Module Responsibilities
- [hpc_config.py](../src/expflow/hpc_config.py) - Auto-detection foundation: username via `pwd.getpwuid()`, scratch dirs, SLURM accounts via `sacctmgr`
- [hpcexp_core.py](../src/expflow/hpcexp_core.py) - `BaseExperimentManager` ABC users extend with 3 methods: `_generate_train_script()`, `_generate_eval_script()`, `harvest_results()`
- [cli.py](../src/expflow/cli.py) - Two-tier CLI: `expflow` for framework, custom managers for experiments
- [resource_advisor.py](../src/expflow/resource_advisor.py) - Real-time SLURM queue analysis via `squeue`

### Data Flow
1. `HPCEnvironment` detects user context → `HPCConfig` dataclass
2. User subclasses `BaseExperimentManager` with domain-specific logic
3. Framework generates SLURM scripts, handles lifecycle, stores metadata in JSON

### Key Design Decisions
- **`src/` layout**: Modern packaging, prevents accidental local imports
- **JSON metadata**: No dependencies, human-readable, git-friendly diffs
- **Filesystem-based tracking**: `/scratch/{user}/{project}/experiments/` - no database setup
- **Abstract base class**: Flexibility for varied ML workflows (classification, LLM, RL)

## Code Conventions

### Critical: No Emojis
Use text indicators only:
```python
# Correct
print("[OK] Environment detected")
print("WARNING: Partition not accessible")

# Wrong - no emojis
print("✓ Environment detected")
print("⚠ Warning")
```

### No Hardcoded Paths
```python
# Correct
data_path = f"{self.hpc_config.scratch_dir}/data"

# Wrong - forbidden
data_path = "/scratch/ah7072/data"
```

### Dependencies
Keep minimal: only `pyyaml` and `pandas` required. Avoid adding heavy dependencies.

## Development Commands

```bash
# Install editable
pip install -e .

# Test auto-detection
expflow info

# Test project init
expflow init -i test-project

# Test custom manager (dry run)
python examples/simple_image_classification.py submit test001 --dry-run
```

## SLURM Integration Testing
```bash
# Key commands for debugging
sinfo -h -o "%R"                                             # List partitions
sacctmgr show associations user=$USER format=Account%40 -n   # User accounts (correct format)
squeue -u $USER -o "%.18i %.9P"                              # User jobs
```

## Common Patterns

### Extending BaseExperimentManager
See [simple_image_classification.py](../examples/simple_image_classification.py):
```python
@dataclass
class MyConfig(BaseExperimentConfig):
    model: str = "resnet50"
    # Add domain-specific fields

class MyManager(BaseExperimentManager):
    def _generate_train_script(self, config: dict) -> str:
        # Return SLURM script content
```

### Auto-Detection Pattern
```python
# Test detection locally
from expflow import HPCEnvironment
print(HPCEnvironment.get_scratch_dir())
print(HPCEnvironment.get_slurm_accounts())
```

## File Structure
- User configs: `experiment_configs/experiments.json`
- Templates: `experiment_templates/*.yaml`
- Generated scripts: `generated_scripts/`
- Logs: `experiments/logs/{output,error}/`

## Testing Notes
No automated tests yet. When adding:
- Mock `subprocess` calls for SLURM commands
- Mock filesystem for path detection
- Test `BaseExperimentManager` with concrete implementations
