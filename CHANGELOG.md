# Changelog

All notable changes to ExpFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-22

### Added
- **PartitionValidator**: Automatic partition-account validation and selection
  - Auto-detects which accounts can access which partitions
  - Uses `sbatch --test-only` to validate without submitting jobs
  - Intelligently selects best account for each partition
  - Handles GPU-only partition requirements (H200, L40s)
  - Provides partition access map visualization
- **CLI command**: `expflow partitions` to show partition-account access map
- **Example**: `partition_aware_manager.py` demonstrating automatic selection
- **Documentation**:
  - `PARTITION_VALIDATOR_USAGE.md` - Quick start guide
  - `docs/partition-access-guide.md` - Comprehensive partition guide
- **Templates**: Now include `account` field automatically

### Fixed
- SLURM account detection now uses correct `sacctmgr` format
  - Changed from `show user -P` to `show associations format=Account -n`
  - Filters out default 'users' account automatically
  - Returns clean list of accessible accounts
- Import errors: Changed absolute imports to relative imports in `hpcexp_core.py`
- Partition detection now properly deduplicates results

### Changed
- Account detection prefers non-general accounts for specific partitions
  - E.g., prefers `torch_pr_68_tandon_advanced` for `h200_tandon`
- Templates now include both `partition` and `account` fields

### Documentation
- Added conda installation instructions
- Added local testing guide (TESTING.md)
- Added CLAUDE.md for future AI assistance
- Updated all docs with GitHub repository links

## [0.1.0] - 2026-01-21

### Added
- Initial release
- **Core Features**:
  - Auto-detection of HPC environment (username, scratch paths, SLURM accounts)
  - Base experiment manager framework
  - YAML-based configuration
  - Git integration for reproducibility
  - Resource advisor for GPU availability
  - Metadata tracking with JSON database

- **CLI Commands**:
  - `expflow init` - Initialize project
  - `expflow info` - Show environment
  - `expflow resources` - Check GPU availability
  - `expflow template` - Create experiment templates
  - `expflow config` - Show configuration

- **Documentation**:
  - Complete user guide
  - API reference
  - Getting started tutorial
  - Custom manager guide
  - Migration guide

- **Examples**:
  - Simple image classification manager
  - Template configurations

### Infrastructure
- Pip-installable package structure
- Console script entry points
- Professional documentation (no emojis)
- Clean repository organization

## [Unreleased]

### Planned
- Web UI for experiment tracking
- Integration with W&B, MLflow
- Jupyter notebook support
- Multi-cluster support
- Automated testing suite

---

## Version Numbering

- **Major (X.0.0)**: Breaking changes, major rewrites
- **Minor (0.X.0)**: New features, non-breaking changes
- **Patch (0.0.X)**: Bug fixes, documentation updates

## Links

- [GitHub Repository](https://github.com/hurryingauto3/expflow-hpc)
- [Issues](https://github.com/hurryingauto3/expflow-hpc/issues)
- [Documentation](https://github.com/hurryingauto3/expflow-hpc/docs)
