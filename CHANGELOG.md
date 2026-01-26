# Changelog

All notable changes to ExpFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-26

### Added
- **Cache Building Framework**: Generic infrastructure for building and managing HPC caches
  - `BaseCacheBuilder`: Abstract base class for project-specific cache builders
  - `CacheConfig`: Dataclass for cache configuration
  - 3-stage pipeline: Build → SquashFS → Cleanup
  - Automatic job dependency chaining with SLURM
  - SquashFS compression for inode optimization (millions of files → 1 file)
  - Support for multiple cache types (training, metric, validation, etc.)
  - Container (Apptainer/Singularity) support for cache operations

- **NAVSIM Cache Builder Example**: Complete implementation for NAVSIM experiments
  - Training cache: Dataset feature extraction with I-JEPA encoder
  - Metric cache: PDM score metric precomputation
  - Support for multi-camera configurations (3-cam, 6-cam)
  - CPU-optimized parallel processing with Ray workers
  - Example CLI: `navsim_cache_builder.py`

- **Cache Management Commands**:
  - `create_cache_config()`: Define new cache configuration
  - `build_cache()`: Submit cache building job
  - `squashfs_cache()`: Compress cache to SquashFS
  - `cleanup_cache()`: Remove original directory after compression
  - `build_cache_pipeline()`: Run full pipeline with dependencies
  - `list_caches()`: List all caches with status
  - `show_cache()`: Show detailed cache information

- **Documentation**:
  - Cache building section in USER_GUIDE.md
  - `examples/CACHE_BUILDER_README.md`: Complete cache building guide
  - Example implementation with NAVSIM use cases
  - Integration patterns with experiment managers

### Features
- Automatic SquashFS overlay generation for read-only cache mounting
- Support for custom cache parameters via `cache_params` dict
- Configurable SquashFS compression (zstd, gzip, etc.)
- Dry-run mode for all cache operations
- Job dependency management (wait_for parameter)
- Container-based cache operations with Apptainer

### Use Cases
- Training dataset preprocessing and caching
- Metric precomputation for faster evaluation
- Feature extraction pipelines
- Large-scale data transformations
- Inode quota optimization on HPC

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

## [0.3.4] - 2026-01-23

### Added
- **Experiment monitoring commands in CLI**: No longer need custom manager scripts for basic operations
  - `expflow status` - Show all experiments with SLURM job status (running, pending, completed)
  - `expflow list` - List experiments with optional status filtering
  - `expflow logs <exp_id>` - View experiment output logs (last N lines)
  - `expflow tail <exp_id>` - Follow logs in real-time (like `tail -f`)
  - `expflow cancel <exp_id>` - Cancel running/pending SLURM jobs
- **Log viewing options**:
  - `--type train|eval` - Choose between training and evaluation logs
  - `-n, --lines` - Number of lines to display (default: 50)
  - `-e, --errors` - View error logs instead of stdout
- **BaseExperimentManager methods**: Added `status()`, `logs()`, `tail_logs()`, `cancel()` to base class

### Fixed
- `expflow template` now creates `experiment_templates/` directory if missing

### Changed
- Experiment monitoring is now a core CLI feature, not just available through custom managers
- Users can track experiments without writing any Python code


## [0.3.3] - 2026-01-22

### Fixed
- **Interactive/Quick init not creating directories**: Fixed missing directory creation
  - `interactive_init()` and `quick_init()` now properly create project directories
  - Config file is now saved to `.hpc_config.yaml`
  - Added `_finalize_project_setup()` helper function in CLI

## [0.3.2] - 2026-01-22

### Fixed
- **CRITICAL: Partition detection bug**: Fixed stdout/stderr check in partition validation
  - `sbatch --test-only` writes success message to stderr, not stdout
  - Was checking `result.stdout` causing all partition tests to fail
  - Now correctly checks `result.stderr` for "to start at" message
  - This fixes the "No GPU partitions detected" issue during interactive init

## [0.3.1] - 2026-01-22

### Fixed
- **Partition detection timeout**: Optimized partition validation to be much faster
  - Now filters to only known GPU partitions (h200, l40s, a100, rtx8000, v100)
  - Reduced test timeout from 5s to 2s per partition
  - Reduced total tests from 36 (2 accts × 18 partitions) to ~10-12 tests
  - Detection now completes in ~5-10 seconds instead of timing out
  - Added better error handling for timeouts

### Added
- **More GPU partitions recognized**:
  - H200: h200_public, h200_tandon, h200_bpeher, h200_courant, h200_cds, h200
  - L40s: l40s_public, l40s
  - A100: a100_public, a100
  - RTX8000: rtx8000
  - V100: v100

### Changed
- `detect_partition_access()` now has `filter_known_gpus=True` by default
- Interactive init uses optimized detection (only GPU partitions)
- Better warning messages when no partitions detected

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
