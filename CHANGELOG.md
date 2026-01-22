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

## [0.3.0] - 2026-01-22

### Added
- **Interactive Initialization**: Professional CLI menu-based setup experience
  - `expflow init -i <project>` for interactive mode with guided menus
  - `expflow init -q <project>` for quick mode with smart defaults
  - Account selection with intelligent recommendations (prefers "general" accounts)
  - GPU/Partition selection with categorization (H200, L40s, A100, etc.)
  - Real-time partition access validation during setup
  - Time limit preferences (6h, 12h, 24h, 48h, 72h, custom)
  - Configuration summary with confirmation before proceeding

### Changed
- **Partition validation integrated into core** (removed from examples/)
  - PartitionValidator now used during interactive init
  - Moved `partition_aware_manager.py` to `.archive/`
  - Interactive init is now the recommended setup method

### Improved
- Account recommendations based on naming patterns (general > public > specific)
- Partition recommendations based on accessibility (public > specific)
- GPU categorization for easier selection (H200/L40s/A100/RTX8000)
- Better user experience with clear markers: [RECOMMENDED], [Public access]

## [0.2.2] - 2026-01-22

### Fixed
- **Default partition selection**: Now uses preference-based selection without slow validation
  - Prioritizes broadly accessible partitions (l40s_public, h200_public)
  - Removed slow sbatch --test-only during initialization
  - Always picks l40s_public if available (most accessible)
  - Falls through preference list: l40s_public → h200_public → rtx8000 → etc.

### Changed
- Removed `_quick_test_partition()` from initialization path (too slow)
- Partition preferences now NYU Greene specific
- Init is now instant (no validation delays)

## [0.2.1] - 2026-01-22

### Fixed
- **Account truncation**: Changed `format=Account` to `format=Account%40` to prevent truncation
  - Was showing `torch_pr_+` instead of `torch_pr_68_general`
- **Default partition selection**: Now intelligently selects accessible partition
  - Tests partition-account combinations during initialization
  - Prefers public partitions (l40s_public, h200_public)
  - Falls back to first accessible partition if preferred ones unavailable
  - Prevents selecting inaccessible partitions like h200_bpeher

### Changed
- `create_default_config()` now validates partition access before setting default
- Added `_quick_test_partition()` helper for fast partition validation

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
