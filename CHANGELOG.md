# Changelog

All notable changes to ExpFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-02-04

### Added
- **Container Integration System**: Generic container support with configurable images and bind mounts
  - `HPCConfig.container_image`: Default container image path (auto-detected on Greene)
  - `HPCConfig.container_bind_mounts`: List of custom bind mounts
  - `BaseExperimentManager._generate_container_exec()`: Wrap scripts in apptainer execution
  - `BaseExperimentManager._prepare_bind_mounts()`: Auto-prepare bind mounts with scratch/tmp
  - Automatic detection of NYU Greene container images

- **Conda/Environment Management**: Auto-detection and configuration
  - `HPCConfig.conda_root`: Auto-detected conda installation path
  - `HPCConfig.conda_env`: Environment name to activate
  - `HPCConfig.module_loads`: List of modules to load (e.g., anaconda3/2025.06)
  - `BaseExperimentManager._generate_conda_activation()`: Generate activation commands
  - Supports both module-based conda and direct installations
  - Auto-detects conda from CONDA_PREFIX, scratch directory, or home directory

- **SquashFS Overlay System**: Generic overlay mounting helpers
  - `HPCConfig.overlay_cache_dir`: Directory for .sqsh overlays (auto: cache/overlays/)
  - `BaseExperimentManager._generate_overlay_mount()`: Generate apptainer overlay bind
  - `BaseExperimentManager._check_overlay_availability()`: Check if overlay exists
  - Automatic fallback suggestions when overlays are missing

- **Checkpoint Registry**: Structured checkpoint tracking
  - `BaseExperimentManager.register_checkpoint()`: Register checkpoints with metadata
  - `BaseExperimentManager.get_registered_checkpoint()`: Retrieve best/latest checkpoint
  - JSON-based registry at `checkpoints/checkpoint_registry.json`
  - Tracks epoch, metrics (val_loss), and registration time
  - Smart selection: prefer best (lowest val_loss) or most recent

- **GPU Monitoring Helper**: Configurable nvidia-smi monitoring
  - `HPCConfig.enable_gpu_monitoring`: Enable/disable GPU monitoring
  - `HPCConfig.gpu_monitor_interval`: Monitoring interval in seconds (default: 60)
  - `BaseExperimentManager._generate_gpu_monitoring()`: Generate monitoring commands
  - Automatic cleanup with trap handlers
  - Logs to: logs/output/{exp_id}_gpu_{job_id}.csv

- **NCCL Optimization Presets**: GPU-specific NCCL tuning
  - `HPCConfig.nccl_preset`: Preset name ('h200', 'a100', 'l40s', 'rtx8000')
  - `HPCConfig.nccl_env_vars`: Custom NCCL environment variables
  - `BaseExperimentManager._get_nccl_env_vars()`: Get optimized NCCL settings
  - Auto-detection from partition name if preset not specified
  - Presets for H200 (NVL P2P), A100, L40s, RTX8000

- **Environment Variable Templates**: Variable substitution system
  - `BaseExperimentManager._substitute_env_vars()`: Substitute ${var} placeholders
  - Supports: ${scratch_dir}, ${project_root}, ${username}, ${experiments_dir}, etc.
  - Works with any config field for dynamic path generation

### Changed
- `HPCConfig` extended with 10+ new configuration fields
- `HPCEnvironment.detect_conda_root()`: New static method for conda detection
- `HPCEnvironment.detect_container_image()`: New static method for container detection
- `HPCEnvironment.create_default_config()`: Now auto-detects container and conda
- Package version bumped to 0.7.0

### Features
- **Generic Container Support**: Works with any apptainer/singularity image
- **Flexible Environment Management**: Module-based or direct conda installations
- **Smart Auto-Detection**: Container images, conda paths, NCCL settings
- **Structured Checkpoint Tracking**: Beyond simple file discovery
- **Production Monitoring**: Built-in GPU utilization tracking
- **Performance Optimization**: Partition-specific NCCL tuning

### Use Cases
- **Container-based workflows**: Run training in consistent environments
- **Multi-user projects**: Auto-detect each user's conda/container setup
- **Performance tuning**: Apply GPU-specific NCCL optimizations
- **Checkpoint management**: Track and retrieve best/latest checkpoints systematically
- **GPU utilization analysis**: Monitor GPU usage across training runs
- **Dynamic path configuration**: Use templates for portable configs

### Migration Notes
Users with existing navsim_manager.py or similar can now:
1. Remove hardcoded container paths → Use `config['container_image']`
2. Remove hardcoded conda paths → Use `_generate_conda_activation(config)`
3. Remove overlay mounting logic → Use `_generate_overlay_mount(cache_name, cache_path)`
4. Remove GPU monitoring duplicates → Use `_generate_gpu_monitoring(exp_id)`
5. Remove NCCL env var blocks → Use `_get_nccl_env_vars(partition)`
6. Simplify checkpoint discovery → Use checkpoint registry

See MIGRATION_v0.7.md for detailed migration guide.

## [0.6.0] - 2026-01-26

### Added
- **Experiment Resume Support**: Framework-level checkpoint resumption capability
  - `BaseExperimentManager.resume_experiment()`: Create new experiment that resumes from checkpoint
  - `_find_latest_checkpoint()`: Automatic checkpoint detection with pattern matching
  - `_extract_epoch_from_checkpoint()`: Smart epoch extraction from checkpoint filenames
  - Support for multiple checkpoint formats: PyTorch (.pth, .pt), PyTorch Lightning (.ckpt)
  - Automatic "best" vs "latest" checkpoint detection
  - Resume tracking in metadata: `resume_from_exp_id`, `resume_checkpoint_path`, `resume_epoch`
  - Auto-generated resume experiment IDs: `{original_exp_id}_resume{N}`
  - Resume count tracking to prevent ID collisions

- **Resume Metadata Fields**:
  - `ExperimentMetadata.resume_from_exp_id`: Source experiment being resumed
  - `ExperimentMetadata.resume_checkpoint_path`: Path to checkpoint file
  - `ExperimentMetadata.resume_epoch`: Epoch number being resumed from
  - `ExperimentMetadata.resume_count`: Number of times experiment has been resumed

- **Checkpoint Detection Patterns**:
  - Best checkpoints: `checkpoint_best.pth`, `best_checkpoint.pth`, `model_best.pth`
  - Latest checkpoints: `checkpoint_latest.pth`, `latest_checkpoint.pth`
  - Epoch-specific: `checkpoint_epoch_*.pth`, `epoch_*.pth`
  - PyTorch Lightning: `*.ckpt`
  - Custom checkpoints: User can specify exact path

### Features
- **Automatic checkpoint discovery**: Searches `checkpoints/{exp_id}/` directory
- **Smart checkpoint selection**: Prefers "best" checkpoints over "latest"
- **Epoch tracking**: Extracts epoch number from filenames for accurate resumption
- **Config inheritance**: Resumed experiments inherit source config with overrides
- **Git tracking**: Captures git state at resume time for reproducibility
- **Resume chain tracking**: Handles experiments resumed from resumed experiments
- **Collision prevention**: Auto-increments resume count if ID exists
- **User overrides**: Allow config modifications during resume (learning rate, batch size, etc.)

### Use Cases
- Resume training from failed/interrupted SLURM jobs
- Continue experiments with different hyperparameters
- Extend training beyond original epoch count
- Test different evaluation strategies on same checkpoint
- Recover from node failures or time limit exceeded errors
- Fine-tune from intermediate checkpoints

### Changed
- `ExperimentMetadata` dataclass extended with resume fields
- `BaseExperimentManager` now includes checkpoint discovery methods
- Package version bumped to 0.6.0 in setup.py and __init__.py

### Documentation
- USER_GUIDE.md updated with "Resuming Experiments" section
- Example implementation in navsim_manager.py
- API documentation for resume methods

## [0.5.0] - 2026-01-26

### Added
- **Experiment Pruning System**: Comprehensive cleanup functionality for duplicate and invalid experiments
  - `ExperimentPruner`: Core pruning engine with duplicate detection and validation
  - `expflow prune` CLI command: Clean up experiments with multiple modes
  - `BaseExperimentManager.prune_experiments()`: Programmatic pruning API
  - Three pruning modes: `all`, `duplicates`, `invalid`
  - Smart duplicate detection by base experiment name (ignores timestamps)
  - Checkpoint validation (`.pth`, `.pt`, `.ckpt` files)
  - Evaluation results validation (`results.json`, `metrics.json`, etc.)
  - Epoch-based checkpoint filtering (`--required-epochs`)
  - Safe archival to `.archive/experiments/YYYYMMDD/` instead of permanent deletion
  - Dry-run mode for previewing changes without deletion
  - Space tracking (reports MB/GB freed)
  - `PruneStats` dataclass for operation results

- **Pruning CLI Options**:
  - `--mode {all,duplicates,invalid}`: Select pruning strategy
  - `--keep N`: Keep N most recent runs per experiment (default: 1)
  - `--dry-run`: Preview without deleting
  - `--no-checkpoint-check`: Skip checkpoint validation
  - `--no-eval-check`: Skip eval results validation
  - `--required-epochs N`: Require checkpoints with >= N epochs

- **Documentation**:
  - USER_GUIDE.md: Added comprehensive "Experiment Pruning" section
  - examples/test_pruner.py: Verified test script

### Features
- Automatic timestamp extraction from experiment directories (`YYYYMMDD_HHMMSS`)
- Groups experiments by base name for intelligent duplicate detection
- Configurable checkpoint and evaluation validation
- Support for custom archive directories
- Collision handling in archive (appends timestamp if needed)
- Works with both flat and nested experiment directory structures
- Integration with training and evaluation subdirectories

### Use Cases
- Clean up duplicate experiment runs (keep only most recent)
- Remove failed experiments without valid outputs
- Free up disk space on HPC scratch storage (10-25 GB typical savings)
- Maintain organized experiment directories
- Prepare for storage quota limits
- Archive old experiments safely

### Changed
- Package exports now include `ExperimentPruner` and `PruneStats`
- README updated with pruning feature in key features and commands
- CLI help text includes pruning examples

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

#### Results Harvesting Framework
- **Results Harvesting Framework**: Infrastructure for extracting and analyzing experiment results
  - `BaseResultsHarvester`: Abstract base class for project-specific result parsing
  - `TrainingMetrics` / `EvaluationMetrics`: Type-safe result containers
  - TensorBoard log parsing (train/val loss, learning rate, custom metrics)
  - Evaluation result extraction from CSVs/JSONs
  - Automatic visualization generation (training curves, comparison plots)
  - CSV/JSON export for analysis

- **NAVSIM Results Harvester Example**: Complete implementation for NAVSIM experiments
  - PyTorch Lightning TensorBoard log parsing
  - PDM Score evaluation result extraction
  - Multi-stage evaluation support (one-stage, two-stage)
  - Comparison plots by backbone/agent
  - Example CLI: `navsim_results_harvester.py`

- **Results Management**:
  - `harvest_experiment()`: Harvest single experiment
  - `harvest_all_experiments()`: Batch harvesting
  - `export_to_csv()` / `export_to_json()`: Result export
  - `plot_training_curves()`: Visualization generation
  - Structured results directory: `experiments/results/{plots,csvs,analysis}`

- **Documentation**:
  - Results harvesting section in USER_GUIDE.md
  - Integration examples with experiment managers
  - Comparison with manual result extraction scripts

#### Directory Structure Improvements
- Cache overlays moved to `experiments/cache/overlays/` (better organization)
- Results harvested to `experiments/results/` with subdirectories
- All experiment-related files now under `experiments/` hierarchy

### Changed
- **Cache directory structure**: Overlays now in `experiments/cache/overlays/` instead of `/scratch/USER/overlays/`
- **Results directory**: Added `experiments/results/` for harvested results
- **BaseExperimentManager**: Auto-creates results subdirectories

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
