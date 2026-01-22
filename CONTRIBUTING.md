# Contributing to ExpFlow

Thank you for your interest in contributing! 

## Ways to Contribute

### 1. Share Templates

Share experiment templates for common use cases:

```bash
examples/templates/
 bert_finetuning.yaml
 stable_diffusion.yaml
 your_template.yaml
```

Submit via pull request!

### 2. Report Bugs

Open an [issue](https://github.com/hurryingauto3/expflow-hpc/issues) with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment (cluster, Python version)

### 3. Add Examples

Create example implementations for new use cases:

```bash
examples/
 llm_finetuning.py          # Wanted!
 reinforcement_learning.py  # Wanted!
 your_example.py            # Submit it!
```

### 4. Improve Documentation

Help make docs clearer:
- Fix typos
- Add missing sections
- Create tutorials
- Add screenshots/GIFs

### 5. Add Features

Larger contributions:
- Support for other HPC clusters
- Integration with W&B, MLflow
- Web UI
- Jupyter notebook integration

## Development Setup

```bash
# Clone
git clone https://github.com/hurryingauto3/expflow-hpc.git
cd expflow-hpc

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## Coding Standards

- **Python 3.8+** compatibility
- **Type hints** where possible
- **Docstrings** for public APIs
- **Black** for formatting
- **Tests** for new features

## Pull Request Process

1. **Fork** the repository
2. **Create branch** (`git checkout -b feature/amazing-feature`)
3. **Make changes** and commit
4. **Add tests** if applicable
5. **Update docs** if needed
6. **Push** to your fork
7. **Submit PR** with clear description

## Testing on HPC

Before submitting, test on actual HPC cluster:

```bash
# Upload to HPC
scp -r expflow-hpc YOUR_ID@greene.hpc.nyu.edu:~/

# SSH and test
ssh YOUR_ID@greene.hpc.nyu.edu
cd expflow-hpc
pip install -e .

# Test commands
expflow info
expflow init test-project
# ... test functionality
```

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers
- Give credit where due

## Questions?

- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/expflow-hpc/discussions)
- **Chat**: (Add Slack/Discord if you create one)

## Recognition

Contributors will be:
- Listed in README
- Mentioned in release notes
- Credited in citations

Thank you for making ExpFlow better! 
