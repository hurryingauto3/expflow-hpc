# Project Structure

```
expflow-hpc/
├── README.md                   # Main documentation
├── CONTRIBUTING.md             # Contribution guide
├── LICENSE                     # MIT License
├── setup.py                    # Pip installation
├── requirements.txt            # Dependencies
├── .gitignore                  # Git ignore patterns
│
├── src/expflow/                # Main package
│   ├── __init__.py
│   ├── cli.py
│   ├── hpc_config.py
│   ├── hpcexp_core.py
│   └── resource_advisor.py
│
├── docs/                       # Documentation
│   ├── README.md              # Docs index
│   ├── getting-started.md     # Setup tutorial
│   ├── user-guide.md          # Full guide
│   ├── quick-reference.md     # Command cheat sheet
│   ├── custom-managers.md     # Extending the framework
│   ├── migration-guide.md     # Migration guide
│   ├── api-reference.md       # Python API
│   └── cli-reference.md       # CLI reference
│
└── examples/                   # Example implementations
    └── simple_image_classification.py
```

## Installation

```bash
pip install git+https://github.com/YOUR_USERNAME/expflow-hpc.git
```

## Key Files

- **`setup.py`** - Package configuration for pip
- **`src/expflow/`** - Main package source code
- **`docs/`** - All user documentation
- **`examples/`** - Example implementations
- **`.archive/`** - Archived development files (not in git)
