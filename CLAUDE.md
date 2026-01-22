# Claude Development Guide - ExpFlow HPC

This document provides context for Claude Code instances working in this repository.

## Project Overview

**ExpFlow** is a lightweight experiment tracking framework for HPC clusters, specifically designed for NYU Greene HPC but adaptable to other SLURM-based systems. It replaces heavyweight frameworks like MLflow with a minimal, filesystem-based approach.

**Core Purpose**: Allow researchers to easily track ML experiments on HPC without manually editing SLURM scripts or hardcoding paths.

**Key Design Principle**: Auto-detect everything. No hardcoded usernames, paths, or SLURM accounts.

## Architecture

### 1. Auto-Detection System (`src/expflow/hpc_config.py`)

The foundation of the framework. Uses system calls to detect the user's environment:

```python
# Username detection
pwd.getpwuid(os.getuid()).pw_name

# Scratch directory discovery (tries multiple patterns)
/scratch/{username}
/scratch/users/{username}
/global/scratch/{username}

# SLURM configuration
sacctmgr show user {username} -P  # Get accounts
sinfo -h -o "%R"                   # Get partitions
```

**Why this matters**: The original problem was scripts with hardcoded `/scratch/ah7072`. This system makes the framework work for ANY NYU HPC user without modification.

### 2. Base Manager Pattern (`src/expflow/hpcexp_core.py`)

Users extend `BaseExperimentManager` and implement 3 methods:

1. `_generate_train_script(config)` - Generate SLURM training script
2. `_generate_eval_script(config)` - Generate SLURM evaluation script
3. `harvest_results(exp_id)` - Parse experiment results

The base class handles:
- Experiment lifecycle (create, submit, track)
- Metadata database (JSON-based)
- File organization (configs, logs, checkpoints)
- Git integration for reproducibility

**Why this matters**: Provides structure without forcing opinions. Users write domain-specific logic, framework handles infrastructure.

### 3. CLI Layer (`src/expflow/cli.py`)

Two-tier command system:
- `expflow` commands: Project initialization, resource checking, templates
- Custom manager commands: Users implement `new`, `submit`, `list`, `export` in their manager scripts

**Why this matters**: Separation of concerns. Framework handles environment, users handle experiments.

### 4. Resource Advisor (`src/expflow/resource_advisor.py`)

Real-time SLURM queue analysis:
- Queries `squeue` for current GPU usage
- Calculates available resources per partition
- Warns about reproducibility when switching GPU types (L40s vs H200)

**Why this matters**: NYU HPC has multiple GPU types. Users need to know what's available and avoid accidental GPU changes that break reproducibility.

## Development Commands

### Setup for Development
```bash
# Install in editable mode
pip install -e .

# Install with dev dependencies (if added later)
pip install -e .[dev]
```

### Testing the CLI
```bash
# Test detection system
expflow info

# Test resource checking
expflow resources --status

# Test project initialization
expflow init test-project
```

### Testing Custom Manager
```bash
cd /scratch/YOUR_ID/test-project
python -m expflow.examples.simple new --exp-id test001
python -m expflow.examples.simple submit test001 --dry-run
```

## Important Files

### Configuration Files
- `.hpc_config.yaml` - Generated per-project, contains auto-detected settings
- `metadata.json` - Experiment database, tracks all experiments
- `experiment_templates/` - User-defined config templates

### Package Structure
```
src/expflow/
├── __init__.py          # Exports public API
├── cli.py               # expflow command implementation
├── hpc_config.py        # Auto-detection + HPCConfig dataclass
├── hpcexp_core.py       # BaseExperimentManager abstract class
└── resource_advisor.py  # SLURM queue analysis
```

### Documentation
All docs in `docs/` directory. `README.md` is intentionally minimal - detailed docs are in:
- `docs/getting-started.md` - Tutorial
- `docs/user-guide.md` - Full features
- `docs/api-reference.md` - Python API
- `docs/custom-managers.md` - Extension guide

## Key Architectural Decisions

### 1. Why `src/` layout?
Modern Python packaging best practice. Prevents accidental imports from local directory during development.

### 2. Why JSON for metadata database?
- No dependencies (no SQLite, no pandas)
- Human-readable for debugging
- Git-friendly (can see diffs)
- Fast enough for typical use (hundreds of experiments)

### 3. Why abstract base class pattern?
Flexibility. Users might be doing:
- Image classification (ResNet, ViT)
- LLM fine-tuning (different resource needs)
- Reinforcement learning (multi-stage pipelines)

Framework can't predict all use cases. ABC provides structure without constraints.

### 4. Why filesystem-based tracking?
HPC filesystem IS the database:
- Experiments in `/scratch/{user}/{project}/experiments/{exp_id}/`
- Logs in `/scratch/{user}/{project}/logs/`
- Checkpoints in `/scratch/{user}/{project}/checkpoints/`

No network calls, no database setup, works offline.

## NYU HPC Specifics

### Greene Cluster Partitions
```
rtx8000       # RTX 8000 GPUs (older)
l40s_public   # L40s GPUs (newer, popular)
h200_tandon   # H200 GPUs (newer, limited access)
```

### Scratch Directory
- Primary workspace: `/scratch/{netid}`
- NOT backed up, purged after 60 days of inactivity
- High-performance parallel filesystem (GPFS)

### SLURM Accounts
Users typically have multiple accounts (e.g., `torch_pr_68_tandon_advanced`). Auto-detected via `sacctmgr`.

### Common Issues
1. **Partition access**: Not all users can access all partitions
2. **Account limits**: Different accounts have different priority/limits
3. **GPU quotas**: Limited number of GPUs per user

## Code Style Requirements

**CRITICAL: NO EMOJIS**

This was explicitly requested. The project is production-ready and professional. Previous versions had emojis removed via:
```python
# Unicode emoji ranges removed
\U0001F300-\U0001F9FF  # Misc symbols & pictographs
\U0001F600-\U0001F64F  # Emoticons
# etc.
```

Use text instead:
- `[OK]` not ✓
- `WARNING:` not ⚠
- `[READY]` not 🟢
- `[BUSY]` not 🔴

## Common Tasks for Claude

### Adding a New Feature
1. Read existing code in `src/expflow/` first
2. Check if it fits in existing classes or needs new module
3. Update relevant docs in `docs/`
4. Add example to `examples/` if user-facing

### Fixing Auto-Detection Issues
1. Check `HPCEnvironment` static methods in `hpc_config.py`
2. Test with: `python -c "from expflow import HPCEnvironment; print(HPCEnvironment.get_scratch_dir())"`
3. Remember: Must work on Greene HPC (can't test locally)

### Updating Documentation
- `README.md` - Keep minimal (installation + quick start only)
- Detailed docs go in `docs/`
- Code examples must have no hardcoded paths

### Debugging SLURM Integration
Key commands for testing:
```bash
sinfo -h -o "%R"                    # List partitions
squeue -u $USER -o "%.18i %.9P"     # User's jobs
sacctmgr show user $USER -P         # User's accounts
```

## What NOT to Do

1. **Don't hardcode paths**: `/scratch/ah7072` is forbidden. Use `hpc_config.scratch_dir`
2. **Don't add emojis**: Explicitly removed for professional presentation
3. **Don't create unnecessary docs**: User wanted clean repo. New docs must be essential.
4. **Don't use heavy dependencies**: Keep it lightweight. Current deps: PyYAML only.
5. **Don't modify `.archive/`**: Old files kept for reference, excluded from git

## Testing Strategy

Currently no automated tests. If adding tests:
- Mock SLURM commands (subprocess calls)
- Mock filesystem operations
- Test auto-detection with various configurations
- Test abstract base class with concrete implementation

Suggested structure:
```
tests/
├── test_hpc_config.py       # Auto-detection
├── test_experiment_manager.py  # Base class
├── test_resource_advisor.py    # SLURM queries
└── fixtures/                   # Mock data
```

## Git Workflow

Repository is version controlled. Key files:

`.gitignore` excludes:
- `__pycache__/`, `*.pyc`
- `experiment_configs/`, `generated_scripts/` (user-specific)
- `.hpc_config.yaml` (per-user configuration)
- `.archive/` (old development files)
- Virtual environments

Commit messages should be clear and reference specific changes.

## Release Process (Future)

Not yet published. When ready:
1. Choose version number (semantic versioning)
2. Update `setup.py` version
3. Create git tag
4. Push to GitHub
5. Users install via: `pip install git+https://github.com/hurryingauto3/expflow-hpc.git`

Optional: Publish to PyPI for `pip install expflow-hpc`

## Contact and Support

- User: ah7072 (NYU HPC NetID)
- Target users: NYU HPC researchers
- Primary use case: ML experiment tracking (image classification, LLMs, RL)

## Quick Reference

### File Locations
- Source code: `src/expflow/`
- Documentation: `docs/`
- Examples: `examples/`
- Old files: `.archive/` (not in git)

### Key Classes
- `HPCConfig` - dataclass, auto-detected configuration
- `BaseExperimentManager` - abstract base for custom managers
- `ResourceAdvisor` - SLURM queue analysis
- `HPCEnvironment` - static methods for detection

### Key Functions
- `initialize_project(name)` - Create new project
- `load_project_config()` - Load .hpc_config.yaml

### Entry Points
- `expflow` command - installed via setup.py console_scripts
- Custom manager - users run as `python my_manager.py [command]`

## Future Improvements (Potential)

Ideas for future development:
- Automatic experiment comparison (diff configs)
- Result visualization (plots, tables)
- Dependency tracking (data versions, code versions)
- Multi-cluster support (non-NYU HPC)
- Web dashboard (optional, keep CLI-first)

But remember: **Keep it lightweight**. That's the core value proposition.
