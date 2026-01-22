# ExpFlow Documentation

Lightweight experiment tracking for HPC clusters.

## Documentation Index

### Getting Started
- **[Getting Started Guide](getting-started.md)** - Complete setup tutorial
- **[Quick Reference](quick-reference.md)** - Command cheat sheet

### User Guides
- **[User Guide](user-guide.md)** - Full feature documentation
- **[Creating Custom Managers](custom-managers.md)** - Adapt for your research
- **[Migration Guide](migration-guide.md)** - Converting existing code
- **[Partition Access Guide](partition-access-guide.md)** - NYU Greene partition/account setup

### Reference
- **[API Reference](api-reference.md)** - Python API documentation
- **[CLI Reference](cli-reference.md)** - Command-line interface

## Quick Links

### Installation
```bash
pip install git+https://github.com/YOUR_USERNAME/expflow-hpc.git
```

### Basic Usage
```bash
# Initialize
expflow init my-research

# Check resources
expflow resources --status

# Create experiment
cd /scratch/YOUR_ID/my-research
python -m expflow.examples.simple new --exp-id exp001 --template baseline

# Submit
python -m expflow.examples.simple submit exp001
```

### Examples
- [Image Classification](../examples/simple_image_classification.py)

## Common Tasks

| Task | Command |
|------|---------|
| Check GPU availability | `expflow resources --status` |
| Show environment info | `expflow info` |
| Create template | `expflow template <name>` |
| List experiments | `python my_manager.py list` |
| Export results | `python my_manager.py export results.csv` |

## Support

- **Issues**: [GitHub Issues](https://github.com/hurryingauto3/expflow-hpc/issues)
- **Docs**: Full documentation in this directory
- **NYU HPC**: [NYU HPC Wiki](https://sites.google.com/nyu.edu/nyu-hpc/)
