#!/usr/bin/env python3
"""
Generic HPC Experiment Manager - Core Framework

This is the base framework that can be customized for any deep learning project.
Users subclass BaseExperimentManager and implement project-specific methods.
"""

import argparse
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from .hpc_config import HPCConfig, load_project_config


# =============================================================================
# Base Configuration Classes
# =============================================================================

@dataclass
class BaseExperimentConfig:
    """Base experiment configuration - extend this for your project"""

    # Core fields (always required)
    exp_id: str
    description: str

    # Resource configuration
    partition: str = "gpu"
    num_gpus: int = 4
    num_nodes: int = 1
    cpus_per_task: int = 16
    time_limit: str = "48:00:00"
    account: str = "default"

    # Metadata (auto-populated)
    created_at: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    train_job_id: Optional[str] = None
    eval_job_id: Optional[str] = None

    # Git tracking
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None

    # Organization
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentMetadata:
    """Runtime metadata for tracking experiments"""
    exp_id: str
    config: Any  # Will be your custom config class
    status: str  # created, submitted, training, evaluating, completed, failed
    train_script_path: Optional[str] = None
    eval_script_path: Optional[str] = None
    run_id: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    # Resume tracking
    resume_from_exp_id: Optional[str] = None
    resume_checkpoint_path: Optional[str] = None
    resume_epoch: Optional[int] = None
    resume_count: int = 0  # Number of times this experiment has been resumed


@dataclass
class ConsistencyReport:
    """Result of validate_consistency() — describes YAML/JSON/filesystem alignment"""
    missing_yaml: List[str]          # In metadata but no YAML config file
    orphan_yaml: List[str]           # YAML file exists but not registered in metadata
    stale_config_copy: List[str]     # Metadata entries with embedded "config" key (old format)
    broken_script_paths: List[str]   # Script paths in metadata pointing to missing files
    ok: bool                         # True when all lists are empty

    def __str__(self) -> str:
        if self.ok:
            return "Consistency: [OK] all checks passed"
        lines = ["Consistency: issues found"]
        if self.missing_yaml:
            lines.append(f"  missing_yaml ({len(self.missing_yaml)}): {', '.join(self.missing_yaml[:5])}"
                         + ("..." if len(self.missing_yaml) > 5 else ""))
        if self.orphan_yaml:
            lines.append(f"  orphan_yaml ({len(self.orphan_yaml)}): {', '.join(self.orphan_yaml[:5])}"
                         + ("..." if len(self.orphan_yaml) > 5 else ""))
        if self.stale_config_copy:
            lines.append(f"  stale_config_copy ({len(self.stale_config_copy)}): "
                         + f"{', '.join(self.stale_config_copy[:5])}"
                         + ("..." if len(self.stale_config_copy) > 5 else ""))
        if self.broken_script_paths:
            lines.append(f"  broken_script_paths ({len(self.broken_script_paths)}): "
                         + f"{', '.join(self.broken_script_paths[:5])}"
                         + ("..." if len(self.broken_script_paths) > 5 else ""))
        return "\n".join(lines)


@dataclass
class BatchPreview:
    """Preview result from preview_batch() before submitting"""
    total_experiments: int
    total_gpus: int
    estimated_gpu_hours: float
    partition_summary: Dict[str, int]   # partition -> experiment count
    ready: List[str]                    # exp_ids with YAML config present
    not_ready: List[str]                # exp_ids missing config or not registered
    warnings: List[str]


# =============================================================================
# Base Experiment Manager
# =============================================================================

class BaseExperimentManager(ABC):
    """
    Base class for HPC experiment management.

    To use:
    1. Subclass this and implement abstract methods
    2. Define your own ExperimentConfig (subclass BaseExperimentConfig)
    3. Implement _generate_train_script() and _generate_eval_script()
    4. Implement harvest_results() for your output format
    """

    def __init__(self, hpc_config: HPCConfig):
        """
        Initialize experiment manager

        Args:
            hpc_config: HPC configuration object
        """
        self.hpc_config = hpc_config

        # Setup paths
        self.project_root = Path(hpc_config.project_root)
        self.experiments_dir = Path(hpc_config.experiments_dir)
        self.logs_dir = Path(hpc_config.logs_dir)
        self.cache_dir = Path(hpc_config.cache_dir)
        self.checkpoints_dir = Path(hpc_config.checkpoints_dir)
        self.results_dir = self.experiments_dir / "results"  # Results harvesting directory

        # Framework directories
        self.configs_dir = self.project_root / "experiment_configs"
        self.templates_dir = self.project_root / "experiment_templates"
        self.generated_dir = self.project_root / "generated_scripts"
        self.metadata_db = self.configs_dir / "experiments.json"

        # Create necessary directories
        for directory in [
            self.configs_dir,
            self.templates_dir,
            self.generated_dir,
            self.experiments_dir,
            self.logs_dir / "output",
            self.logs_dir / "error",
            self.checkpoints_dir,
            self.results_dir,
            self.results_dir / "plots",
            self.results_dir / "csvs",
            self.results_dir / "analysis"
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load experiment metadata database"""
        if self.metadata_db.exists():
            with open(self.metadata_db, 'r') as f:
                data = json.load(f)
                self.metadata = data
            # Warn once if old-format entries (with embedded "config" key) are detected
            stale = [eid for eid, entry in self.metadata.items() if "config" in entry]
            if stale:
                print(f"WARNING: {len(stale)} metadata entries are in old format "
                      f"(embedded 'config' copy). Run manager.sync_metadata() to migrate.")
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save experiment metadata database"""
        with open(self.metadata_db, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _get_git_info(self) -> Dict[str, Any]:
        """Get current git commit info"""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            return {
                "git_commit": commit,
                "git_branch": branch,
                "git_dirty": len(status) > 0
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                "git_commit": None,
                "git_branch": None,
                "git_dirty": None
            }

    # =========================================================================
    # Config / Metadata Unified Access
    # =========================================================================

    def _load_config(self, exp_id: str) -> Dict[str, Any]:
        """
        Load fresh experiment config from its YAML file.

        This is the authoritative way to read experiment parameters.
        Raises FileNotFoundError if the YAML does not exist.
        """
        config_path = self.configs_dir / f"{exp_id}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config YAML not found for {exp_id}: {config_path}"
            )
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def get_experiment_record(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a complete experiment record by merging YAML config and JSON runtime state.

        YAML config is the source of truth for experiment parameters.
        JSON metadata holds runtime state (status, job IDs, results, etc.).
        JSON state keys take precedence when both have the same key.

        Returns None if the experiment is not registered in metadata.
        """
        if exp_id not in self.metadata:
            return None
        state = self.metadata[exp_id]
        try:
            config = self._load_config(exp_id)
        except FileNotFoundError:
            # Fall back to embedded config copy if YAML is missing (old-format support)
            config = state.get("config", {})
        return {**config, **{k: v for k, v in state.items() if k != "config"}}

    def sync_metadata(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate metadata database from old format (embedded "config" copy) to new format.

        New format: JSON stores runtime state only; YAML is the config source of truth.

        For each experiment with an embedded "config" key:
        - If the YAML file already exists: drop the embedded copy from JSON.
        - If no YAML exists: write one from the embedded copy, then drop the copy.

        This method is idempotent — safe to run multiple times.

        Args:
            dry_run: If True, show what would change without modifying files.

        Returns:
            {"migrated": [...], "skipped": [...], "warnings": [...]}
        """
        migrated = []
        skipped = []
        warnings = []

        for exp_id, entry in self.metadata.items():
            if "config" not in entry:
                skipped.append(exp_id)
                continue

            embedded_config = entry["config"]
            config_path = self.configs_dir / f"{exp_id}.yaml"

            if not config_path.exists():
                if dry_run:
                    print(f"  [DRY RUN] Would write YAML for {exp_id} (missing) and strip config copy")
                else:
                    with open(config_path, 'w') as f:
                        yaml.dump(embedded_config, f, default_flow_style=False, sort_keys=False)
                warnings.append(exp_id)
            else:
                if dry_run:
                    print(f"  [DRY RUN] Would strip config copy from metadata for {exp_id}")

            if not dry_run:
                del entry["config"]
            migrated.append(exp_id)

        if not dry_run and migrated:
            self._save_metadata()

        print(f"sync_metadata: migrated={len(migrated)}, skipped={len(skipped)}, "
              f"yaml_written={len(warnings)}")
        return {"migrated": migrated, "skipped": skipped, "warnings": warnings}

    @abstractmethod
    def _generate_train_script(self, config: Any) -> str:
        """
        Generate SLURM training script from config

        Args:
            config: Your custom experiment config

        Returns:
            SLURM script as string
        """
        pass

    @abstractmethod
    def _generate_eval_script(self, config: Any) -> str:
        """
        Generate SLURM evaluation script from config

        Args:
            config: Your custom experiment config

        Returns:
            SLURM script as string
        """
        pass

    @abstractmethod
    def harvest_results(self, exp_id: str) -> Dict[str, Any]:
        """
        Harvest evaluation results for an experiment

        Args:
            exp_id: Experiment ID

        Returns:
            Dictionary of results
        """
        pass

    # =========================================================================
    # Helper Methods for Script Generation (v0.7.0+)
    # =========================================================================

    def _generate_conda_activation(self, config: Dict[str, Any]) -> str:
        """
        Generate conda/environment activation commands with fallback support

        This enhanced version includes fallback logic for HPC environments
        where conda may be provided via modules instead of a direct path.

        Args:
            config: Experiment configuration

        Returns:
            Shell commands for environment activation

        Example:
            # With conda_root specified
            script = manager._generate_conda_activation(config)

            # Falls back to module-provided conda if path doesn't exist
        """
        conda_root = config.get('conda_root', self.hpc_config.conda_root)
        conda_env = config.get('conda_env', self.hpc_config.conda_env)
        module_loads = config.get('module_loads', self.hpc_config.module_loads)

        script_lines = []

        # Load modules first
        if module_loads:
            for module in module_loads:
                script_lines.append(f"module load {module}")

        # Conda activation with robust fallback logic
        if conda_root and conda_env:
            script_lines.extend([
                f'CONDA_ROOT="{conda_root}"',
                'if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then',
                '    # Use specified conda installation',
                '    source "${CONDA_ROOT}/etc/profile.d/conda.sh"',
                f'    conda activate {conda_env}',
                'else',
                '    # Fallback: try module-provided conda',
                '    echo "Warning: Conda not found at ${CONDA_ROOT}, trying module fallback..."',
                '    module purge || true',
                '    module load anaconda3/2025.06 || module load anaconda3 || true',
                '    source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true',
                f'    conda activate {conda_env} || echo "ERROR: Could not activate {conda_env}"',
                'fi'
            ])
        elif module_loads and conda_env:
            # Assume module provides conda
            script_lines.extend([
                f"source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {conda_env}"
            ])

        return "\n".join(script_lines)

    def _generate_container_exec(
        self,
        config: Dict[str, Any],
        script_content: str,
        working_dir: Optional[str] = None
    ) -> str:
        """
        Generate apptainer/singularity execution wrapper

        Args:
            config: Experiment configuration
            script_content: The actual script to run inside container
            working_dir: Working directory inside container

        Returns:
            Shell commands that wrap script_content in container exec
        """
        container = config.get('container_image', self.hpc_config.container_image)

        if not container:
            # No container - return script as-is
            return script_content

        # Build bind mounts
        bind_mounts = self._prepare_bind_mounts(config)
        bind_args = " \\\n        ".join([f'--bind "{b}"' for b in bind_mounts])

        # Get environment variables to pass
        env_vars = config.get('environment_variables', {})
        env_args = " \\\n        ".join([f'--env "{k}={v}"' for k, v in env_vars.items()])

        # Create temporary script
        exp_id = config.get('exp_id', 'unknown')
        script_lines = [
            f"# Create temporary script for container execution",
            f"TEMP_SCRIPT=$(mktemp /tmp/{exp_id}_XXXXXX.sh)",
            f"cat > \"${{TEMP_SCRIPT}}\" << 'CONTAINER_SCRIPT_EOF'",
            f"#!/bin/bash",
            script_content,
            f"CONTAINER_SCRIPT_EOF",
            f"chmod +x \"${{TEMP_SCRIPT}}\"",
            f"",
            f"# Execute in container",
            f"apptainer exec \\",
            f"    --nv \\",
        ]

        if bind_args:
            script_lines.append(f"    {bind_args} \\")

        if working_dir:
            script_lines.append(f"    --pwd \"{working_dir}\" \\")

        if env_args:
            script_lines.append(f"    {env_args} \\")

        script_lines.extend([
            f"    \"{container}\" \\",
            f"    bash \"${{TEMP_SCRIPT}}\"",
            f"",
            f"# Cleanup",
            f"rm -f \"${{TEMP_SCRIPT}}\""
        ])

        return "\n".join(script_lines)

    def _prepare_bind_mounts(self, config: Dict[str, Any]) -> List[str]:
        """
        Prepare bind mount list for container

        Args:
            config: Experiment configuration

        Returns:
            List of bind mount strings (e.g., "/scratch/user:/scratch/user")
        """
        bind_mounts = []

        # Default binds from config
        bind_mounts.extend(self.hpc_config.container_bind_mounts)

        # Auto-bind scratch directory
        bind_mounts.append(f"{self.hpc_config.scratch_dir}:{self.hpc_config.scratch_dir}")

        # Auto-bind /tmp
        bind_mounts.append("/tmp:/tmp")

        # Add any experiment-specific binds
        if 'container_bind_mounts' in config:
            bind_mounts.extend(config['container_bind_mounts'])

        # Remove duplicates while preserving order
        seen = set()
        unique_binds = []
        for bind in bind_mounts:
            if bind not in seen:
                seen.add(bind)
                unique_binds.append(bind)

        return unique_binds

    def _generate_overlay_mount(
        self,
        cache_name: str,
        cache_path: str,
        overlay_path: Optional[str] = None
    ) -> str:
        """
        Generate SquashFS overlay mount for apptainer

        Args:
            cache_name: Name of the cache
            cache_path: Path where cache should be mounted
            overlay_path: Path to .sqsh file (auto-detected if None)

        Returns:
            Bind mount string for apptainer (e.g., "--bind overlay.sqsh:/cache:image-src=/")
        """
        if overlay_path is None:
            overlay_path = f"{self.hpc_config.overlay_cache_dir}/{cache_name}.sqsh"

        return f'--bind "{overlay_path}:{cache_path}:image-src=/"'

    def _check_overlay_availability(
        self,
        cache_name: str,
        overlay_path: Optional[str] = None
    ) -> bool:
        """
        Check if SquashFS overlay exists

        Args:
            cache_name: Name of the cache
            overlay_path: Path to .sqsh file (auto-detected if None)

        Returns:
            True if overlay exists, False otherwise
        """
        if overlay_path is None:
            overlay_path = f"{self.hpc_config.overlay_cache_dir}/{cache_name}.sqsh"

        return Path(overlay_path).exists()

    def _generate_gpu_monitoring(
        self,
        exp_id: str,
        interval: Optional[int] = None
    ) -> str:
        """
        Generate GPU monitoring command for SLURM script

        Args:
            exp_id: Experiment ID
            interval: Monitoring interval in seconds (uses config default if None)

        Returns:
            Shell commands for GPU monitoring
        """
        if not self.hpc_config.enable_gpu_monitoring:
            return "# GPU monitoring disabled"

        if interval is None:
            interval = self.hpc_config.gpu_monitor_interval

        log_file = self.logs_dir / "output" / f"{exp_id}_gpu_${{SLURM_JOB_ID}}.csv"

        script_lines = [
            f"# Start GPU monitoring",
            f"nvidia-smi \\",
            f"    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total \\",
            f"    --format=csv \\",
            f"    -l {interval} \\",
            f"    > {log_file} &",
            f"GPU_MONITOR_PID=$!",
            f"",
            f"# Cleanup function",
            f"cleanup_gpu_monitor() {{",
            f"    if [ ! -z \"${{GPU_MONITOR_PID}}\" ]; then",
            f"        kill ${{GPU_MONITOR_PID}} 2>/dev/null || true",
            f"    fi",
            f"}}",
            f"trap cleanup_gpu_monitor EXIT"
        ]

        return "\n".join(script_lines)

    def _get_nccl_env_vars(
        self,
        partition: Optional[str] = None,
        preset: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get NCCL optimization environment variables

        Args:
            partition: SLURM partition (auto-detects GPU type if preset not given)
            preset: Preset name ('h200', 'a100', 'l40s', 'rtx8000') or None

        Returns:
            Dictionary of NCCL environment variables
        """
        # Start with custom env vars from config
        nccl_vars = dict(self.hpc_config.nccl_env_vars)

        # Determine preset
        if preset is None:
            preset = self.hpc_config.nccl_preset

        # Auto-detect from partition if still None
        if preset is None and partition:
            if 'h200' in partition.lower():
                preset = 'h200'
            elif 'a100' in partition.lower():
                preset = 'a100'
            elif 'l40s' in partition.lower():
                preset = 'l40s'
            elif 'rtx8000' in partition.lower():
                preset = 'rtx8000'

        # Apply preset
        presets = {
            'h200': {
                'NCCL_IB_DISABLE': '0',
                'NCCL_P2P_LEVEL': 'NVL',
                'NCCL_NET_GDR_LEVEL': '2',
                'CUDA_LAUNCH_BLOCKING': '0',
                'TORCH_CUDNN_V8_API_ENABLED': '1'
            },
            'a100': {
                'NCCL_IB_DISABLE': '0',
                'NCCL_P2P_LEVEL': 'NVL',
                'NCCL_NET_GDR_LEVEL': '1',
                'CUDA_LAUNCH_BLOCKING': '0',
                'TORCH_CUDNN_V8_API_ENABLED': '1'
            },
            'l40s': {
                'NCCL_IB_DISABLE': '0',
                'NCCL_P2P_LEVEL': 'SYS',
                'NCCL_NET_GDR_LEVEL': '1',
                'CUDA_LAUNCH_BLOCKING': '0'
            },
            'rtx8000': {
                'NCCL_IB_DISABLE': '0',
                'NCCL_P2P_LEVEL': 'SYS',
                'NCCL_NET_GDR_LEVEL': '0',
                'CUDA_LAUNCH_BLOCKING': '0'
            }
        }

        if preset and preset in presets:
            # Merge preset with custom vars (custom vars take precedence)
            preset_vars = presets[preset]
            for key, value in preset_vars.items():
                if key not in nccl_vars:
                    nccl_vars[key] = value

        return nccl_vars

    def _generate_nccl_exports(
        self,
        partition: Optional[str] = None,
        preset: Optional[str] = None
    ) -> str:
        """
        Generate NCCL environment variable export statements

        Utility method that generates shell export statements from NCCL configuration.
        Useful for including in SLURM scripts.

        Args:
            partition: SLURM partition (for auto-detection)
            preset: NCCL preset name ('h200', 'a100', 'l40s', 'rtx8000')

        Returns:
            Shell export statements as string

        Example:
            # In script generation
            nccl_exports = manager._generate_nccl_exports(partition='l40s_public')
            # Returns:
            # export NCCL_IB_DISABLE=0
            # export NCCL_P2P_LEVEL=SYS
            # export NCCL_NET_GDR_LEVEL=1
        """
        nccl_vars = self._get_nccl_env_vars(partition=partition, preset=preset)

        if not nccl_vars:
            return "# No NCCL optimizations configured"

        lines = ["# NCCL optimizations (auto-detected from partition)"]
        for key, value in nccl_vars.items():
            lines.append(f"export {key}={value}")

        return "\n".join(lines)

    def _substitute_env_vars(
        self,
        template: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Substitute environment variable templates

        Supports: ${scratch_dir}, ${project_root}, ${experiments_dir}, ${username}, etc.

        Args:
            template: Template string with ${var} placeholders
            config: Experiment configuration

        Returns:
            String with variables substituted
        """
        substitutions = {
            'scratch_dir': self.hpc_config.scratch_dir,
            'project_root': self.hpc_config.project_root,
            'experiments_dir': self.hpc_config.experiments_dir,
            'logs_dir': self.logs_dir,
            'cache_dir': self.cache_dir,
            'checkpoints_dir': self.checkpoints_dir,
            'username': self.hpc_config.username,
            'user_home': self.hpc_config.user_home,
        }

        # Add config-specific substitutions
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                substitutions[key] = str(value)

        # Perform substitutions
        result = template
        for key, value in substitutions.items():
            result = result.replace(f"${{{key}}}", str(value))

        return result

    # =========================================================================
    # HPC Configuration Helpers
    # =========================================================================

    def _get_overlay_path(self, cache_name: str) -> str:
        """
        Get the path to a SquashFS overlay file

        Args:
            cache_name: Name of the cache (without .sqsh extension)

        Returns:
            Full path to overlay file

        Example:
            overlay_path = manager._get_overlay_path('training_cache_v2')
            # Returns: /scratch/user/cache/overlays/training_cache_v2.sqsh
        """
        overlay_dir = getattr(
            self.hpc_config,
            'overlay_cache_dir',
            f"{self.hpc_config.cache_dir}/overlays"
        )
        return f"{overlay_dir}/{cache_name}.sqsh"

    def _get_container_image(self) -> Optional[str]:
        """
        Get the configured container image path

        Returns:
            Path to Apptainer/Singularity image or None if not configured

        Example:
            image = manager._get_container_image()
            if image:
                # Use container
        """
        return getattr(self.hpc_config, 'container_image', None)

    # =========================================================================
    # Checkpoint Registry (v0.7.0+)
    # =========================================================================

    def register_checkpoint(
        self,
        exp_id: str,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Register a checkpoint for an experiment

        Args:
            exp_id: Experiment ID
            checkpoint_path: Path to checkpoint file
            epoch: Epoch number (optional)
            metrics: Checkpoint metrics like val_loss (optional)
        """
        registry_file = self.checkpoints_dir / "checkpoint_registry.json"

        # Load existing registry
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}

        # Update entry
        if exp_id not in registry:
            registry[exp_id] = []

        checkpoint_info = {
            'path': str(checkpoint_path),
            'registered_at': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics or {}
        }

        registry[exp_id].append(checkpoint_info)

        # Save registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)

        print(f"Registered checkpoint for {exp_id}: {Path(checkpoint_path).name}")

    def get_registered_checkpoint(
        self,
        exp_id: str,
        prefer_best: bool = True
    ) -> Optional[str]:
        """
        Retrieve registered checkpoint for experiment

        Args:
            exp_id: Experiment ID
            prefer_best: If True, returns checkpoint with lowest val_loss

        Returns:
            Checkpoint path or None
        """
        registry_file = self.checkpoints_dir / "checkpoint_registry.json"

        if not registry_file.exists():
            return None

        with open(registry_file, 'r') as f:
            registry = json.load(f)

        if exp_id not in registry or not registry[exp_id]:
            return None

        checkpoints = registry[exp_id]

        if prefer_best and len(checkpoints) > 1:
            # Find checkpoint with lowest val_loss
            best_ckpt = None
            best_loss = float('inf')

            for ckpt in checkpoints:
                metrics = ckpt.get('metrics', {})
                val_loss = metrics.get('val_loss')

                if val_loss is not None and val_loss < best_loss:
                    best_loss = val_loss
                    best_ckpt = ckpt

            if best_ckpt:
                return best_ckpt['path']

        # Return most recent
        return checkpoints[-1]['path']

    def create_experiment(
        self,
        exp_id: str,
        template: Optional[str] = None,
        description: str = "",
        **kwargs
    ):
        """
        Create a new experiment configuration

        Args:
            exp_id: Unique experiment identifier
            template: Template name to use (optional)
            description: Experiment description
            **kwargs: Additional config parameters
        """

        # Check if exists
        if exp_id in self.metadata:
            print(f"Warning: Experiment {exp_id} already exists. Overwriting.")

        # Load template if provided
        template_config = {}
        if template:
            template_path = self.templates_dir / f"{template}.yaml"
            if template_path.exists():
                with open(template_path, 'r') as f:
                    template_config = yaml.safe_load(f)

        # Merge configurations
        config_dict = {
            "exp_id": exp_id,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "account": self.hpc_config.default_account,
            "partition": self.hpc_config.default_partition,
            **self._get_git_info(),
            **template_config,
            **kwargs
        }

        # Save config as YAML
        config_path = self.configs_dir / f"{exp_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Create metadata entry — config lives in YAML only; JSON holds runtime state
        self.metadata[exp_id] = {
            "exp_id": exp_id,
            "status": "created",
            "train_script_path": None,
            "eval_script_path": None,
            "run_id": None,
            "results": {}
        }
        self._save_metadata()

        print(f" Created experiment: {exp_id}")
        print(f"  Config: {config_path}")

        return config_dict

    def submit_experiment(
        self,
        exp_id: str,
        train_only: bool = False,
        eval_only: bool = False,
        dry_run: bool = False
    ) -> Dict[str, str]:
        """Submit experiment to SLURM"""

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found.")
            print("Run 'hpcexp new <exp_id>' to create it first.")
            sys.exit(1)

        meta = self.metadata[exp_id]

        # Load fresh config from YAML (single source of truth)
        full_config = self._load_config(exp_id)

        # Generate scripts
        train_script_path = self.generated_dir / f"train_{exp_id}.slurm"
        eval_script_path = self.generated_dir / f"eval_{exp_id}.slurm"

        if not eval_only:
            train_script = self._generate_train_script(full_config)
            with open(train_script_path, 'w') as f:
                f.write(train_script)
            meta["train_script_path"] = str(train_script_path)
            print(f" Generated training script: {train_script_path}")

        if not train_only:
            eval_script = self._generate_eval_script(full_config)
            with open(eval_script_path, 'w') as f:
                f.write(eval_script)
            meta["eval_script_path"] = str(eval_script_path)
            print(f" Generated evaluation script: {eval_script_path}")

        if dry_run:
            print("\n[DRY RUN] Would submit the following jobs:")
            if not eval_only:
                print(f"  Training: sbatch {train_script_path}")
            if not train_only:
                print(f"  Evaluation: sbatch --dependency=afterok:TRAIN_JOB_ID {eval_script_path}")
            return {}

        # Submit to SLURM
        job_ids = {}

        try:
            if not eval_only:
                result = subprocess.run(
                    ["sbatch", str(train_script_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Parse job ID from "Submitted batch job 12345"
                train_job_id = result.stdout.strip().split()[-1]
                job_ids["train_job_id"] = train_job_id
                print(f" Submitted training job: {train_job_id}")

            if not train_only:
                cmd = ["sbatch"]
                if "train_job_id" in job_ids:
                    cmd.extend(["--dependency", f"afterok:{job_ids['train_job_id']}"])
                cmd.append(str(eval_script_path))

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                eval_job_id = result.stdout.strip().split()[-1]
                job_ids["eval_job_id"] = eval_job_id
                print(f" Submitted evaluation job: {eval_job_id}")

            # Update metadata
            meta["status"] = "submitted"
            meta["submitted_at"] = datetime.now().isoformat()
            meta.update(job_ids)
            self._save_metadata()

            return job_ids

        except subprocess.CalledProcessError as e:
            print(f"Error submitting jobs:")
            print(e.stderr)
            sys.exit(1)

    def list_experiments(self, status: Optional[str] = None, tags: Optional[List[str]] = None):
        """List all experiments with optional filtering"""

        filtered = []
        for exp_id, meta in self.metadata.items():
            if status and meta.get("status") != status:
                continue
            record = self.get_experiment_record(exp_id) or {}
            if tags and not any(tag in record.get("tags", []) for tag in tags):
                continue
            filtered.append((exp_id, meta, record))

        if not filtered:
            print("No experiments found")
            return

        print(f"\nFound {len(filtered)} experiments:")
        print(f"{'ID':<15} {'Status':<12} {'Description':<50}")
        print("-" * 80)

        for exp_id, meta, record in sorted(filtered, key=lambda x: x[0]):
            raw_desc = record.get("description", "")
            desc = raw_desc[:47] + "..." if len(raw_desc) > 50 else raw_desc
            status_str = meta.get("status", "unknown")
            print(f"{exp_id:<15} {status_str:<12} {desc:<50}")

    def show_experiment(self, exp_id: str):
        """Show detailed information about an experiment"""

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            sys.exit(1)

        meta = self.metadata[exp_id]
        record = self.get_experiment_record(exp_id) or {}

        print(f"\n{'='*70}")
        print(f"Experiment: {exp_id}")
        print(f"{'='*70}")
        print(f"\nDescription: {record.get('description', 'N/A')}")
        print(f"Status: {meta.get('status', 'unknown')}")

        state_keys = {
            "exp_id", "description", "created_at", "submitted_at", "completed_at",
            "status", "train_script_path", "eval_script_path", "run_id",
            "results", "resume_from_exp_id", "resume_checkpoint_path",
            "resume_epoch", "resume_count", "train_job_id", "eval_job_id",
            "eval_job_ids", "cancelled_at"
        }
        print(f"\nConfiguration:")
        for key, value in record.items():
            if key not in state_keys:
                print(f"  {key}: {value}")

        if meta.get("results"):
            print(f"\nResults:")
            for key, value in meta["results"].items():
                print(f"  {key}: {value}")

        print(f"\nTimeline:")
        print(f"  Created: {record.get('created_at', 'N/A')}")
        if meta.get("submitted_at"):
            print(f"  Submitted: {meta['submitted_at']}")
        if meta.get("completed_at"):
            print(f"  Completed: {meta['completed_at']}")

        if record.get("git_commit"):
            print(f"\nGit:")
            print(f"  Commit: {record['git_commit'][:8]}")
            print(f"  Branch: {record.get('git_branch', 'N/A')}")

        print(f"\n{'='*70}\n")

    def export_results(self, output_file: str = "results.csv"):
        """
        Export all experiment results to CSV

        Override this in subclass to customize columns
        """
        import pandas as pd

        records = []
        for exp_id, meta in self.metadata.items():
            r = self.get_experiment_record(exp_id) or {}
            record = {
                "exp_id": exp_id,
                "description": r.get("description", ""),
                "status": meta.get("status", "unknown"),
                "partition": r.get("partition", ""),
                "num_gpus": r.get("num_gpus", 0),
                "created_at": r.get("created_at", ""),
                "submitted_at": meta.get("submitted_at", ""),
                "completed_at": meta.get("completed_at", ""),
                **meta.get("results", {})
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        print(f" Exported {len(records)} experiments to {output_file}")

    def _get_slurm_jobs(self) -> Dict[str, Dict[str, str]]:
        """Get current SLURM jobs for this user"""
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h",
                 "-o", "%.18i %.9P %.50j %.8T %.10M %.6D %R"],
                capture_output=True, text=True
            )
            jobs = {}
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    job_id = parts[0].strip()
                    jobs[job_id] = {
                        "job_id": job_id,
                        "partition": parts[1].strip(),
                        "name": parts[2].strip(),
                        "state": parts[3].strip(),
                        "time": parts[4].strip() if len(parts) > 4 else "",
                        "nodes": parts[5].strip() if len(parts) > 5 else "",
                        "nodelist": parts[6].strip() if len(parts) > 6 else ""
                    }
            return jobs
        except Exception:
            return {}

    def _find_log_file(self, exp_id: str, log_type: str = "train") -> Optional[Path]:
        """Find the most recent log file for an experiment"""
        if log_type == "train":
            pattern = f"train_{exp_id}_*.out"
        else:
            pattern = f"eval_{exp_id}_*.out"

        log_dir = self.logs_dir / "output"
        matches = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None

    def _find_error_file(self, exp_id: str, log_type: str = "train") -> Optional[Path]:
        """Find the most recent error file for an experiment"""
        if log_type == "train":
            pattern = f"train_{exp_id}_*.err"
        else:
            pattern = f"eval_{exp_id}_*.err"

        log_dir = self.logs_dir / "error"
        matches = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None

    def _find_latest_checkpoint(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Find the latest checkpoint for an experiment

        Returns:
            Dictionary with checkpoint info: {path, epoch, type} or None if not found
        """
        checkpoint_dir = self.checkpoints_dir / exp_id

        if not checkpoint_dir.exists():
            return None

        # Look for common checkpoint patterns
        checkpoint_patterns = [
            "checkpoint_best.pth",
            "best_checkpoint.pth",
            "model_best.pth",
            "checkpoint_latest.pth",
            "latest_checkpoint.pth",
            "checkpoint_epoch_*.pth",
            "checkpoint_*.pth",
            "epoch_*.pth",
            "*.ckpt",  # PyTorch Lightning format
        ]

        best_checkpoint = None
        latest_checkpoint = None

        for pattern in checkpoint_patterns:
            matches = list(checkpoint_dir.glob(pattern))

            if not matches:
                continue

            # Check for "best" checkpoints first
            if "best" in pattern.lower():
                if matches:
                    best_checkpoint = max(matches, key=lambda p: p.stat().st_mtime)
                    return {
                        "path": str(best_checkpoint),
                        "epoch": self._extract_epoch_from_checkpoint(best_checkpoint),
                        "type": "best"
                    }

            # Otherwise track the latest checkpoint by modification time
            for match in matches:
                if latest_checkpoint is None or match.stat().st_mtime > latest_checkpoint.stat().st_mtime:
                    latest_checkpoint = match

        if latest_checkpoint:
            return {
                "path": str(latest_checkpoint),
                "epoch": self._extract_epoch_from_checkpoint(latest_checkpoint),
                "type": "latest"
            }

        return None

    def _extract_epoch_from_checkpoint(self, checkpoint_path: Path) -> Optional[int]:
        """Extract epoch number from checkpoint filename"""
        import re

        # Try to extract epoch from filename
        patterns = [
            r"epoch_(\d+)",
            r"epoch(\d+)",
            r"checkpoint_(\d+)",
            r"step_(\d+)",
        ]

        filename = checkpoint_path.stem
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def resume_experiment(
        self,
        source_exp_id: str,
        new_exp_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create a new experiment that resumes from a previous experiment's checkpoint

        Args:
            source_exp_id: Experiment ID to resume from
            new_exp_id: ID for the new resumed experiment (auto-generated if None)
            checkpoint_path: Specific checkpoint path (auto-detects latest if None)
            **kwargs: Additional config overrides

        Returns:
            The new experiment ID
        """
        # Validate source experiment exists
        if source_exp_id not in self.metadata:
            print(f"Error: Source experiment {source_exp_id} not found")
            sys.exit(1)

        source_meta = self.metadata[source_exp_id]
        source_config = self._load_config(source_exp_id) if (self.configs_dir / f"{source_exp_id}.yaml").exists() else source_meta.get("config", {})

        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_info = self._find_latest_checkpoint(source_exp_id)
            if checkpoint_info is None:
                print(f"Error: No checkpoint found for experiment {source_exp_id}")
                print(f"  Looked in: {self.checkpoints_dir / source_exp_id}")
                sys.exit(1)
            checkpoint_path = checkpoint_info["path"]
            resume_epoch = checkpoint_info["epoch"]
            checkpoint_type = checkpoint_info["type"]
            print(f"Found {checkpoint_type} checkpoint: {Path(checkpoint_path).name}")
            if resume_epoch is not None:
                print(f"  Resuming from epoch: {resume_epoch}")
        else:
            # Validate provided checkpoint exists
            if not Path(checkpoint_path).exists():
                print(f"Error: Checkpoint not found: {checkpoint_path}")
                sys.exit(1)
            resume_epoch = self._extract_epoch_from_checkpoint(Path(checkpoint_path))
            checkpoint_type = "custom"

        # Generate new experiment ID if not provided
        if new_exp_id is None:
            # Get resume count for source experiment
            resume_count = source_meta.get("resume_count", 0) + 1
            new_exp_id = f"{source_exp_id}_resume{resume_count}"

            # Handle case where resumed experiment was itself resumed
            if source_meta.get("resume_from_exp_id"):
                original_exp_id = source_meta["resume_from_exp_id"]
                new_exp_id = f"{original_exp_id}_resume{resume_count}"

        # Check if new experiment ID already exists
        if new_exp_id in self.metadata:
            print(f"Error: Experiment {new_exp_id} already exists")
            print(f"  Specify a different new_exp_id or delete the existing experiment")
            sys.exit(1)

        # Create new config based on source config
        new_config = {
            **source_config,
            "exp_id": new_exp_id,
            "description": f"Resume from {source_exp_id}: {source_config.get('description', '')}",
            "created_at": datetime.now().isoformat(),
            "resume_from_exp_id": source_exp_id,
            "resume_checkpoint_path": checkpoint_path,
            "resume_epoch": resume_epoch,
            **self._get_git_info(),
            **kwargs  # Allow user to override any config values
        }

        # Save new config
        config_path = self.configs_dir / f"{new_exp_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

        # Create metadata entry — config lives in YAML; resume fields stay in JSON state
        self.metadata[new_exp_id] = {
            "exp_id": new_exp_id,
            "status": "created",
            "train_script_path": None,
            "eval_script_path": None,
            "run_id": None,
            "results": {},
            "resume_from_exp_id": source_exp_id,
            "resume_checkpoint_path": checkpoint_path,
            "resume_epoch": resume_epoch,
            "resume_count": 0
        }

        # Update source experiment's resume count
        source_meta["resume_count"] = source_meta.get("resume_count", 0) + 1

        self._save_metadata()

        print(f"\n Created resume experiment: {new_exp_id}")
        print(f"  Config: {config_path}")
        print(f"  Resuming from: {source_exp_id}")
        print(f"  Checkpoint: {Path(checkpoint_path).name}")
        if resume_epoch is not None:
            print(f"  Starting epoch: {resume_epoch + 1}")

        return new_exp_id

    def status(self):
        """Show status of all experiments with SLURM job info"""
        slurm_jobs = self._get_slurm_jobs()

        # Build job_id to experiment mapping
        job_to_exp = {}
        for exp_id, meta in self.metadata.items():
            if meta.get("train_job_id"):
                job_to_exp[meta["train_job_id"]] = (exp_id, "train")
            if meta.get("eval_job_id"):
                job_to_exp[meta["eval_job_id"]] = (exp_id, "eval")

        print(f"\n{'='*80}")
        print("Experiment Status")
        print(f"{'='*80}")

        # Show running/pending jobs
        active_jobs = []
        for job_id, job_info in slurm_jobs.items():
            if job_id in job_to_exp:
                exp_id, job_type = job_to_exp[job_id]
                active_jobs.append({
                    "exp_id": exp_id,
                    "job_type": job_type,
                    "job_id": job_id,
                    **job_info
                })

        if active_jobs:
            print(f"\nActive Jobs ({len(active_jobs)}):")
            print(f"{'Experiment':<15} {'Type':<6} {'JobID':<10} {'State':<10} {'Time':<12} {'Node'}")
            print("-" * 80)
            for job in active_jobs:
                print(f"{job['exp_id']:<15} {job['job_type']:<6} {job['job_id']:<10} "
                      f"{job['state']:<10} {job['time']:<12} {job.get('nodelist', '')}")
        else:
            print("\nNo active jobs")

        # Show recent experiments
        print(f"\nRecent Experiments:")
        print(f"{'ID':<15} {'Status':<12} {'Train Job':<12} {'Eval Job':<12} {'Description'}")
        print("-" * 80)

        sorted_exps = sorted(
            self.metadata.items(),
            key=lambda x: (self.get_experiment_record(x[0]) or {}).get("created_at", ""),
            reverse=True
        )[:10]

        for exp_id, meta in sorted_exps:
            record = self.get_experiment_record(exp_id) or {}
            desc = record.get("description", "")[:30]
            train_job = meta.get("train_job_id", "-")
            eval_job = meta.get("eval_job_id", "-")
            status = meta.get("status", "unknown")
            print(f"{exp_id:<15} {status:<12} {train_job:<12} {eval_job:<12} {desc}")

        print(f"{'='*80}\n")

    def logs(self, exp_id: str, log_type: str = "train", lines: int = 50, errors: bool = False):
        """View logs for an experiment"""
        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            return

        if errors:
            log_file = self._find_error_file(exp_id, log_type)
            file_type = "error"
        else:
            log_file = self._find_log_file(exp_id, log_type)
            file_type = "output"

        if not log_file:
            print(f"No {file_type} log found for {exp_id} ({log_type})")
            print(f"  Looked in: {self.logs_dir}/{file_type}/")
            return

        print(f"\n{'='*70}")
        print(f"Log: {log_file.name}")
        print(f"{'='*70}\n")

        try:
            with open(log_file, 'r') as f:
                content = f.readlines()
                if len(content) > lines:
                    print(f"[Showing last {lines} lines of {len(content)} total]\n")
                    content = content[-lines:]
                for line in content:
                    print(line, end='')
        except Exception as e:
            print(f"Error reading log: {e}")

        print(f"\n{'='*70}")
        print(f"Full log: {log_file}")
        print(f"{'='*70}\n")

    def tail_logs(self, exp_id: str, log_type: str = "train", errors: bool = False):
        """Tail logs for an experiment (live follow)"""
        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            return

        if errors:
            log_file = self._find_error_file(exp_id, log_type)
        else:
            log_file = self._find_log_file(exp_id, log_type)

        if not log_file:
            print(f"No log found for {exp_id} ({log_type})")
            return

        print(f"Tailing: {log_file}")
        print("Press Ctrl+C to stop\n")
        print("-" * 70)

        try:
            subprocess.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            print("\n[Stopped]")

    def cancel(
        self,
        exp_id: str,
        job_type: Optional[str] = None,
        include_matching_name: bool = True,
        dry_run: bool = False
    ) -> List[str]:
        """
        Cancel all SLURM jobs related to an experiment

        Enhanced version that finds jobs from metadata AND live SLURM queue.
        This catches jobs even if metadata is incomplete or outdated.

        Args:
            exp_id: Experiment ID
            job_type: Deprecated - kept for backward compatibility (ignored if include_matching_name=True)
            include_matching_name: If True, also cancel live SLURM jobs matching exp_id
            dry_run: If True, show what would be canceled without actually canceling

        Returns:
            List of canceled job IDs

        Example:
            # Cancel all jobs for experiment (from metadata + live queue)
            manager.cancel('exp_b20')

            # Preview what would be canceled
            manager.cancel('exp_b20', dry_run=True)

            # Only cancel from metadata (old behavior)
            manager.cancel('exp_b20', include_matching_name=False)
        """
        if exp_id not in self.metadata:
            print(f"Warning: Experiment {exp_id} not found in metadata")
            # Continue anyway - might find jobs in live queue

        job_ids = set()
        meta = self.metadata.get(exp_id, {})

        # Collect job IDs from metadata
        train_job = meta.get("train_job_id")
        eval_job = meta.get("eval_job_id")
        eval_jobs = meta.get("eval_job_ids", [])  # Support multiple eval jobs

        for jid in [train_job, eval_job]:
            if jid:
                job_ids.add(str(jid))

        for jid in eval_jobs or []:
            if jid:
                job_ids.add(str(jid))

        # Smart job name matching helper
        def _is_related_job(name: str) -> bool:
            """Check if job name is related to exp_id"""
            if name == exp_id:
                return True
            if not name.startswith(exp_id):
                return False
            if len(name) == len(exp_id):
                return True
            # Allow common separators: exp_b20_eval, exp_b20-train, exp_b20.1
            return name[len(exp_id)] in {"_", "-", "."}

        # Query live SLURM queue for matching job names
        if include_matching_name:
            try:
                result = subprocess.run(
                    ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    jid, name = parts[0].strip(), parts[1].strip()
                    if _is_related_job(name):
                        job_ids.add(jid)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not query squeue: {e.stderr}")

        if not job_ids:
            print(f"No related SLURM jobs found for {exp_id}")
            return []

        job_ids_list = sorted(job_ids)

        # Dry run mode
        if dry_run:
            print(f"[DRY RUN] Would cancel {len(job_ids_list)} jobs for {exp_id}:")
            for jid in job_ids_list:
                print(f"  {jid}")
            return job_ids_list

        # Cancel jobs
        try:
            subprocess.run(["scancel", *job_ids_list], check=True)
            print(f"[OK] Canceled {len(job_ids_list)} jobs for {exp_id}")
            if len(job_ids_list) <= 10:
                print(f"     Job IDs: {' '.join(job_ids_list)}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to cancel jobs: {e.stderr}")
            return []

        # Update metadata
        if exp_id in self.metadata:
            meta["status"] = "cancelled"
            meta["cancelled_at"] = datetime.now().isoformat()
            self._save_metadata()

        return job_ids_list

    # =========================================================================
    # Results Storage Integration (v0.8.0+)
    # =========================================================================

    @property
    def results_storage(self):
        """
        Lazily initialize results storage with multi-backend support

        Supports:
        - SQLite (default): Local file-based storage
        - MongoDB: Remote cloud database (requires pymongo)
        - PostgreSQL: Remote SQL database (requires psycopg2-binary)

        Backend is selected via environment variables:
        - EXPFLOW_BACKEND: 'sqlite', 'mongodb', or 'postgresql' (default: 'sqlite')
        - EXPFLOW_CONNECTION_STRING: Connection string for remote databases
        - EXPFLOW_MONGODB_TLS_INSECURE: Set to '1' to bypass TLS cert validation (HPC environments)

        Returns:
            ResultsStorage instance connected to experiments database

        Example:
            # SQLite (default)
            manager.results_storage

            # MongoDB
            os.environ['EXPFLOW_BACKEND'] = 'mongodb'
            os.environ['EXPFLOW_CONNECTION_STRING'] = 'mongodb+srv://user:pass@cluster.mongodb.net/'
            manager.results_storage

            # MongoDB with TLS bypass (for HPC)
            os.environ['EXPFLOW_MONGODB_TLS_INSECURE'] = '1'
            manager.results_storage
        """
        if not hasattr(self, '_results_storage'):
            from .results_storage import ResultsStorage
            import os

            # Get backend configuration from environment
            backend = os.getenv('EXPFLOW_BACKEND', 'sqlite')
            connection_string = os.getenv('EXPFLOW_CONNECTION_STRING')

            if backend == 'mongodb':
                # MongoDB backend (remote)
                if not connection_string:
                    raise ValueError(
                        "EXPFLOW_CONNECTION_STRING required for MongoDB backend. "
                        "Get free tier at: https://www.mongodb.com/cloud/atlas"
                    )

                # Optional: allow TLS validation bypass for restricted HPC environments
                # This is useful when compute nodes have certificate trust issues
                if os.getenv('EXPFLOW_MONGODB_TLS_INSECURE', '').lower() in ('1', 'true', 'yes'):
                    if 'tlsAllowInvalidCertificates=true' not in connection_string:
                        sep = '&' if '?' in connection_string else '?'
                        connection_string = f"{connection_string}{sep}tlsAllowInvalidCertificates=true"
                    print("[WARN] MongoDB TLS cert validation disabled via EXPFLOW_MONGODB_TLS_INSECURE")

                self._results_storage = ResultsStorage(
                    backend='mongodb',
                    connection_string=connection_string,
                    database=os.getenv('EXPFLOW_MONGODB_DATABASE', 'experiments')
                )
                print(f"[INFO] Using MongoDB backend")

            elif backend == 'postgresql':
                # PostgreSQL backend (remote)
                if not connection_string:
                    raise ValueError(
                        "EXPFLOW_CONNECTION_STRING required for PostgreSQL backend"
                    )

                self._results_storage = ResultsStorage(
                    backend='postgresql',
                    connection_string=connection_string,
                    table_name=os.getenv('EXPFLOW_POSTGRES_TABLE', 'experiments')
                )
                print(f"[INFO] Using PostgreSQL backend")

            else:
                # SQLite backend (local, default)
                db_path = self.project_root / "experiments_results.db"
                self._results_storage = ResultsStorage(
                    backend='sqlite',
                    path=str(db_path)
                )
                print(f"[INFO] Using SQLite backend: {db_path}")

        return self._results_storage

    def store_experiment_results(
        self,
        exp_id: str,
        results: Dict[str, Any] = None,
        auto_harvest: bool = True
    ) -> bool:
        """
        Store experiment results in database

        Args:
            exp_id: Experiment identifier
            results: Results dictionary (if None, will call harvest_results)
            auto_harvest: If True and results is None, automatically harvest results

        Returns:
            True if successful, False otherwise

        Example:
            # Manual storage
            results = {'accuracy': 0.95, 'loss': 0.05}
            manager.store_experiment_results('exp_001', results)

            # Auto-harvest and store
            manager.store_experiment_results('exp_001', auto_harvest=True)
        """
        if exp_id not in self.metadata:
            print(f"WARNING: Experiment {exp_id} not found in metadata")
            return False

        # Get experiment metadata
        exp_meta = self.metadata[exp_id]

        # Harvest results if not provided
        if results is None and auto_harvest:
            try:
                results = self.harvest_results(exp_id)
            except Exception as e:
                print(f"WARNING: Could not harvest results for {exp_id}: {e}")
                results = {}

        # Load fresh config from YAML (single source of truth)
        record = self.get_experiment_record(exp_id) or {}

        # Build complete experiment data
        exp_data = {
            'exp_id': exp_id,
            'status': exp_meta.get('status', 'unknown'),
            'created_at': record.get('created_at'),
            'submitted_at': exp_meta.get('submitted_at'),
            'completed_at': exp_meta.get('completed_at'),
            'config': record,
            'slurm': {
                'partition': record.get('partition'),
                'num_gpus': record.get('num_gpus'),
                'train_job_id': exp_meta.get('train_job_id'),
                'eval_job_id': exp_meta.get('eval_job_id')
            },
            'git': {
                'commit': record.get('git_commit'),
                'branch': record.get('git_branch'),
                'dirty': record.get('git_dirty')
            },
            'results': results or {},
            'stored_at': datetime.now().isoformat()
        }

        # Store in database
        with self.results_storage as storage:
            success = storage.store(exp_id, exp_data)

        if success:
            print(f"[OK] Stored results for {exp_id}")
        else:
            print(f"[ERROR] Failed to store results for {exp_id}")

        return success

    def collect_all_results(
        self,
        status_filter: str = 'completed',
        force_reharvest: bool = False,
        verbose: bool = True,
        submit_after_current: bool = False,
        slurm_account: Optional[str] = None,
        manager_script_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Unified results collection - replaces harvest/harvest-all commands

        Automatically harvests results from all experiments and stores in database

        Args:
            status_filter: Only process experiments with this status ('completed', 'all', etc)
            force_reharvest: If True, re-harvest even if already in database
            verbose: If True, print progress
            submit_after_current: If True, submit as SLURM job after current running jobs
            slurm_account: SLURM account for job submission (required if submit_after_current=True)
            manager_script_path: Path to manager script for SLURM submission

        Returns:
            Dictionary mapping exp_id to results

        Example:
            # Collect all completed experiments
            all_results = manager.collect_all_results()

            # Force re-harvest everything
            all_results = manager.collect_all_results(
                status_filter='all',
                force_reharvest=True
            )

            # Fire-and-forget: submit harvest job after current experiments finish
            manager.collect_all_results(
                status_filter='completed',
                submit_after_current=True,
                slurm_account='my_account'
            )
        """
        # Handle fire-and-forget SLURM job submission
        if submit_after_current:
            return self._submit_collection_job(
                status_filter=status_filter,
                force_reharvest=force_reharvest,
                slurm_account=slurm_account,
                manager_script_path=manager_script_path,
                verbose=verbose
            )

        # Perform actual collection
        return self._collect_results_internal(status_filter, force_reharvest, verbose)

    def _submit_collection_job(
        self,
        status_filter: str,
        force_reharvest: bool,
        slurm_account: Optional[str],
        manager_script_path: Optional[str],
        verbose: bool
    ) -> Dict[str, Dict[str, Any]]:
        """
        Submit results collection as SLURM job with auto-dependency on current jobs

        This enables fire-and-forget workflow: submit experiments, then submit
        a collection job that runs automatically after all experiments finish.

        Args:
            status_filter: Status filter for collection
            force_reharvest: Force reharvest flag
            slurm_account: SLURM account to use
            manager_script_path: Path to manager script
            verbose: Verbose output

        Returns:
            Empty dict (actual collection happens in SLURM job)
        """
        # Query current SLURM jobs
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
                capture_output=True,
                text=True,
                check=True
            )
            job_ids = [j.strip() for j in result.stdout.strip().split() if j.strip()]
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to query SLURM jobs: {e.stderr}")
            return {}

        if not job_ids:
            if verbose:
                print("No active jobs found. Collecting results now instead of submitting job.")
            # Fall through to normal collection
            return self._collect_results_internal(status_filter, force_reharvest, verbose)

        # Build dependency string
        dependency = "afterok:" + ":".join(job_ids)

        # Determine SLURM account
        if not slurm_account:
            slurm_account = self.hpc_config.default_account if self.hpc_config else None
        if not slurm_account:
            print("[ERROR] SLURM account required for job submission")
            print("  Set via slurm_account parameter or EXPFLOW_DEFAULT_ACCOUNT env var")
            return {}

        # Determine manager script path
        if not manager_script_path:
            # Try to infer from calling script
            manager_script_path = sys.argv[0]
            if not Path(manager_script_path).exists():
                print("[ERROR] Could not determine manager script path")
                print("  Specify via manager_script_path parameter")
                return {}

        # Build collection command
        conda_activation = ""
        if self.hpc_config and self.hpc_config.conda_env:
            conda_root = self.hpc_config.conda_root or os.environ.get('CONDA_PREFIX', '').rsplit('/', 2)[0]
            if conda_root:
                conda_activation = (
                    f"source {conda_root}/etc/profile.d/conda.sh && "
                    f"conda activate {self.hpc_config.conda_env} && "
                )

        collection_cmd = (
            f"{conda_activation}"
            f"cd {self.project_root} && "
            f"python {manager_script_path} collect-results --status={status_filter}"
        )
        if force_reharvest:
            collection_cmd += " --force"

        # Export environment variables for remote database access
        export_vars = [
            "ALL",
            "EXPFLOW_BACKEND",
            "EXPFLOW_CONNECTION_STRING",
            "EXPFLOW_MONGODB_TLS_INSECURE",
            "EXPFLOW_MONGODB_DATABASE",
            "EXPFLOW_POSTGRES_TABLE"
        ]
        export_str = ",".join(export_vars)

        # Build sbatch command
        output_log = self.logs_dir / "output" / "collect_results_%j.out"
        error_log = self.logs_dir / "error" / "collect_results_%j.err"

        sbatch_cmd = [
            "sbatch",
            f"--job-name=expflow_collect_results",
            f"--account={slurm_account}",
            f"--dependency={dependency}",
            f"--export={export_str}",
            f"--output={output_log}",
            f"--error={error_log}",
            "--wrap",
            collection_cmd
        ]

        # Submit job
        try:
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            job_id = result.stdout.strip().split()[-1]
            if verbose:
                print(f"[OK] Submitted results collection job: {job_id}")
                print(f"     Dependencies: {len(job_ids)} jobs ({', '.join(job_ids[:5])}{'...' if len(job_ids) > 5 else ''})")
                print(f"     Status filter: {status_filter}")
                print(f"     Collection will start after all current jobs finish")
            return {}
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to submit collection job: {e.stderr}")
            return {}

    def _collect_results_internal(
        self,
        status_filter: str,
        force_reharvest: bool,
        verbose: bool
    ) -> Dict[str, Dict[str, Any]]:
        """
        Internal method for actual results collection logic.
        Extracted from collect_all_results for reuse.
        """
        results_collected = {}

        # Filter experiments by status
        experiments = []
        for exp_id, meta in self.metadata.items():
            exp_status = meta.get('status', 'unknown')

            if status_filter == 'all' or exp_status == status_filter:
                experiments.append(exp_id)

        if not experiments:
            if verbose:
                print(f"No experiments found with status '{status_filter}'")
            return {}

        if verbose:
            print(f"Collecting results from {len(experiments)} experiments...")

        # Check which are already in database
        with self.results_storage as storage:
            for exp_id in experiments:
                # Check if already stored
                if not force_reharvest:
                    existing = storage.get(exp_id)
                    if existing:
                        if verbose:
                            print(f"  [SKIP] {exp_id} (already in database)")
                        results_collected[exp_id] = existing.get('results', {})
                        continue

                # Harvest and store
                if verbose:
                    print(f"  [HARVEST] {exp_id}...")

                try:
                    results = self.harvest_results(exp_id)
                    results_collected[exp_id] = results

                    # Store in database
                    success = self.store_experiment_results(
                        exp_id,
                        results=results,
                        auto_harvest=False
                    )

                    if not success and verbose:
                        print(f"    WARNING: Failed to store in database")

                except Exception as e:
                    if verbose:
                        print(f"    ERROR: {e}")
                    results_collected[exp_id] = {}

        if verbose:
            print(f"\n[OK] Collected results from {len(results_collected)} experiments")
            backend_name = os.getenv('EXPFLOW_BACKEND', 'sqlite')
            if backend_name == 'sqlite':
                print(f"     Database: {self.project_root / 'experiments_results.db'}")
            else:
                print(f"     Database: {backend_name}")

        return results_collected

    def export_results_for_web(
        self,
        output_format: str = 'json',
        output_path: str = None,
        fields: List[str] = None,
        filters: Dict[str, Any] = None
    ):
        """
        Export results for web visualization

        Args:
            output_format: Export format ('json' or 'csv')
            output_path: Output file path (auto-generated if None)
            fields: List of fields to export (CSV only)
            filters: Filters to apply (e.g., {'status': 'completed'})

        Example:
            # Export all results to JSON for static website
            manager.export_results_for_web(
                output_format='json',
                output_path='public/experiments.json'
            )

            # Export specific fields to CSV
            manager.export_results_for_web(
                output_format='csv',
                fields=['exp_id', 'status', 'results.pdm_score'],
                filters={'status': 'completed'}
            )
        """
        from .results_storage import export_to_json, export_to_csv

        # Auto-generate output path if not provided
        if output_path is None:
            if output_format == 'json':
                output_path = self.results_dir / "experiments_export.json"
            else:
                output_path = self.results_dir / "experiments_export.csv"

        with self.results_storage as storage:
            if output_format == 'json':
                export_to_json(storage, str(output_path), filters)
            elif output_format == 'csv':
                export_to_csv(storage, str(output_path), fields, filters)
            else:
                raise ValueError(f"Unsupported format: {output_format}")

        print(f"[OK] Exported results to {output_path}")

    def query_results(
        self,
        metric: str = None,
        n: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query experiment results from database

        Args:
            metric: Metric to sort by (e.g., 'results.pdm_score')
            n: Number of results to return
            filters: Filters to apply

        Returns:
            List of experiment data dictionaries

        Example:
            # Get top 10 experiments by PDM score
            best = manager.query_results(
                metric='results.pdm_score',
                n=10
            )

            # Get experiments on L40s partition
            l40s_exps = manager.query_results(
                filters={'slurm.partition': 'l40s_public'}
            )
        """
        from .results_storage import ResultsQueryAPI

        with self.results_storage as storage:
            api = ResultsQueryAPI(storage)

            if metric:
                # Sort by metric
                return api.best_experiments(metric, n=n, ascending=False)
            else:
                # Simple query
                experiments = storage.query(**(filters or {}))
                return experiments[:n]

    def prune_experiments(
        self,
        mode: str = "all",
        keep_n: int = 1,
        require_checkpoint: bool = True,
        require_eval: bool = True,
        required_epochs: Optional[int] = None,
        dry_run: bool = False,
        verbose: bool = True
    ):
        """
        Prune duplicate and invalid experiments

        Args:
            mode: Pruning mode - "all", "duplicates", or "invalid"
            keep_n: Number of most recent runs to keep per experiment
            require_checkpoint: If True, prune experiments without checkpoints
            require_eval: If True, prune experiments without eval results
            required_epochs: If specified, require checkpoint with this many epochs
            dry_run: If True, preview without actually deleting
            verbose: If True, print detailed information

        Returns:
            PruneStats with operation results
        """
        from .pruner import ExperimentPruner

        # Find evaluation directory if it exists
        eval_dir = None
        if self.experiments_dir.parent.exists():
            possible_eval = self.experiments_dir.parent / "evaluations"
            if possible_eval.exists():
                eval_dir = possible_eval

        # Initialize pruner
        pruner = ExperimentPruner(
            experiments_dir=self.experiments_dir,
            evaluations_dir=eval_dir,
            archive_dir=self.experiments_dir.parent / ".archive" / "experiments"
        )

        # Perform pruning
        if mode == "duplicates":
            return pruner.prune_duplicates(
                keep_n=keep_n,
                dry_run=dry_run,
                verbose=verbose
            )
        elif mode == "invalid":
            return pruner.prune_invalid(
                require_checkpoint=require_checkpoint,
                require_eval=require_eval,
                required_epochs=required_epochs,
                dry_run=dry_run,
                verbose=verbose
            )
        else:  # all
            return pruner.prune_all(
                keep_n=keep_n,
                require_checkpoint=require_checkpoint,
                require_eval=require_eval,
                required_epochs=required_epochs,
                dry_run=dry_run,
                verbose=verbose
            )

    # =========================================================================
    # Metadata Bulk Operations (v0.9.0+)
    # =========================================================================

    def backup_metadata(self, label: Optional[str] = None) -> Path:
        """
        Create a timestamped backup of experiments.json.

        Args:
            label: Optional label appended to filename (e.g. "before_rename").

        Returns:
            Path to the backup file.
        """
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{label}" if label else ""
        backup_path = self.metadata_db.parent / f"experiments.json.{timestamp}{suffix}.bak"
        shutil.copy2(self.metadata_db, backup_path)
        print(f"Backup created: {backup_path}")
        return backup_path

    def rename_experiment(
        self,
        old_id: str,
        new_id: str,
        dry_run: bool = False
    ) -> bool:
        """
        Rename a single experiment with cascading updates across all locations.

        Cascade order:
        1. experiments.json — rename key, update exp_id field, update path strings
        2. {old_id}.yaml   — rename file, update exp_id: line inside
        3. Training/eval directories — rename by replacing old_id in name

        Args:
            old_id: Existing experiment ID.
            new_id: Target experiment ID.
            dry_run: If True, print planned changes without modifying anything.

        Returns:
            True if successful (or dry_run). False on validation failure.
        """
        import re as _re

        if old_id not in self.metadata:
            print(f"ERROR: Experiment {old_id} not found in metadata")
            return False
        if new_id in self.metadata and new_id != old_id:
            print(f"ERROR: Experiment {new_id} already exists")
            return False

        changes = []

        # 1. Metadata JSON
        old_yaml = self.configs_dir / f"{old_id}.yaml"
        new_yaml = self.configs_dir / f"{new_id}.yaml"
        changes.append(f"  metadata key: {old_id} -> {new_id}")
        changes.append(f"  YAML rename: {old_yaml.name} -> {new_yaml.name}")

        # Check directories
        def _dirs_matching(root: Path, pattern_fn) -> List[Path]:
            if not root.exists():
                return []
            return [d for d in root.iterdir() if d.is_dir() and pattern_fn(d.name)]

        def _is_related(name: str) -> bool:
            return (name == old_id
                    or name.startswith(f"{old_id}_")
                    or name.startswith(f"{old_id}-")
                    or f"_{old_id}_" in name
                    or name.endswith(f"_{old_id}"))

        related_dirs = _dirs_matching(self.experiments_dir, _is_related)
        for d in related_dirs:
            new_name = d.name.replace(old_id, new_id)
            changes.append(f"  dir rename: {d.name} -> {new_name}")

        if dry_run:
            print(f"[DRY RUN] rename_experiment({old_id} -> {new_id}):")
            for c in changes:
                print(c)
            return True

        # Apply metadata update
        entry = self.metadata.pop(old_id)
        entry["exp_id"] = new_id
        # Update string-valued path fields
        for field_name in ("train_script_path", "eval_script_path"):
            if entry.get(field_name):
                entry[field_name] = entry[field_name].replace(old_id, new_id)
        if isinstance(entry.get("results"), dict):
            csv_path = entry["results"].get("csv_path")
            if csv_path:
                entry["results"]["csv_path"] = csv_path.replace(old_id, new_id)
        self.metadata[new_id] = entry
        self._save_metadata()

        # Rename YAML
        if old_yaml.exists():
            content = old_yaml.read_text()
            content = _re.sub(
                r'^(exp_id:\s+)' + _re.escape(old_id) + r'\s*$',
                f'\\g<1>{new_id}',
                content,
                flags=_re.MULTILINE
            )
            new_yaml.write_text(content)
            old_yaml.unlink()

        # Rename directories
        for d in related_dirs:
            new_name = d.name.replace(old_id, new_id)
            d.rename(d.parent / new_name)

        print(f"[OK] Renamed {old_id} -> {new_id}")
        return True

    def bulk_rename(
        self,
        rename_map: Dict[str, str],
        dry_run: bool = False,
        auto_backup: bool = True
    ) -> Dict[str, bool]:
        """
        Rename multiple experiments atomically.

        Args:
            rename_map: Dict mapping old_id -> new_id.
            dry_run: Preview changes without applying.
            auto_backup: Create timestamped backup before applying any changes.

        Returns:
            Dict mapping old_id -> success bool.
        """
        # Pre-validate
        errors = []
        for old_id, new_id in rename_map.items():
            if old_id not in self.metadata:
                errors.append(f"  {old_id}: not found")
            if new_id in self.metadata and new_id not in rename_map.values():
                errors.append(f"  {new_id}: already exists as target")
        if errors:
            print("Validation errors:")
            for e in errors:
                print(e)
            return {old_id: False for old_id in rename_map}

        if not dry_run and auto_backup:
            self.backup_metadata("before_bulk_rename")

        results = {}
        for old_id, new_id in rename_map.items():
            results[old_id] = self.rename_experiment(old_id, new_id, dry_run=dry_run)

        ok = sum(v for v in results.values())
        print(f"bulk_rename: {ok}/{len(rename_map)} succeeded")
        return results

    def validate_consistency(self) -> "ConsistencyReport":
        """
        Check for drift between metadata JSON, YAML configs, and filesystem.

        Returns:
            ConsistencyReport describing issues found.
        """
        missing_yaml = []
        orphan_yaml = []
        stale_config_copy = []
        broken_script_paths = []

        # Check each metadata entry
        for exp_id, entry in self.metadata.items():
            config_path = self.configs_dir / f"{exp_id}.yaml"
            if not config_path.exists():
                missing_yaml.append(exp_id)
            if "config" in entry:
                stale_config_copy.append(exp_id)
            for field_name in ("train_script_path", "eval_script_path"):
                path_val = entry.get(field_name)
                if path_val and not Path(path_val).exists():
                    broken_script_paths.append(exp_id)
                    break

        # Check for orphan YAML files
        registered = set(self.metadata.keys())
        for yaml_path in self.configs_dir.glob("*.yaml"):
            exp_id = yaml_path.stem
            if exp_id not in registered and yaml_path.name != "experiments.yaml":
                orphan_yaml.append(exp_id)

        ok = not any([missing_yaml, orphan_yaml, stale_config_copy, broken_script_paths])
        report = ConsistencyReport(
            missing_yaml=missing_yaml,
            orphan_yaml=orphan_yaml,
            stale_config_copy=stale_config_copy,
            broken_script_paths=broken_script_paths,
            ok=ok
        )
        print(str(report))
        return report

    def repair_consistency(self, dry_run: bool = False) -> "ConsistencyReport":
        """
        Fix detected consistency issues automatically.

        Actions taken:
        - Migrate stale config copies (sync_metadata)
        - Register orphan YAMLs with status="unknown"
        - Clear broken script paths (set to None)

        Does NOT delete any files or experiments.

        Returns:
            ConsistencyReport after repairs.
        """
        report = self.validate_consistency()
        if report.ok:
            return report

        if report.stale_config_copy:
            print(f"Migrating {len(report.stale_config_copy)} stale config copies...")
            self.sync_metadata(dry_run=dry_run)

        for exp_id in report.orphan_yaml:
            print(f"Registering orphan: {exp_id}")
            if not dry_run:
                self.metadata[exp_id] = {"exp_id": exp_id, "status": "unknown", "results": {}}

        for exp_id in report.broken_script_paths:
            entry = self.metadata.get(exp_id, {})
            for field_name in ("train_script_path", "eval_script_path"):
                if entry.get(field_name) and not Path(entry[field_name]).exists():
                    print(f"Clearing broken {field_name} for {exp_id}")
                    if not dry_run:
                        entry[field_name] = None

        if not dry_run:
            self._save_metadata()

        return self.validate_consistency()

    # =========================================================================
    # Run History Tracking (v0.9.0+)
    # =========================================================================

    @property
    def run_results_storage(self):
        """
        Lazily initialize a separate storage backend for per-run history.

        Uses the same backend configuration as results_storage but stores
        in a separate table/collection ('experiment_runs') so run history
        is additive and never overwrites aggregate results.
        """
        if not hasattr(self, '_run_results_storage'):
            from .results_storage import ResultsStorage
            backend = os.getenv('EXPFLOW_BACKEND', 'sqlite')
            connection_string = os.getenv('EXPFLOW_CONNECTION_STRING')

            if backend == 'mongodb':
                self._run_results_storage = ResultsStorage(
                    backend='mongodb',
                    connection_string=connection_string,
                    database=os.getenv('EXPFLOW_MONGODB_DATABASE', 'experiments'),
                    collection='experiment_runs'
                )
            elif backend == 'postgresql':
                self._run_results_storage = ResultsStorage(
                    backend='postgresql',
                    connection_string=connection_string,
                    table_name='experiment_runs'
                )
            else:
                db_path = self.project_root / "experiment_runs.db"
                self._run_results_storage = ResultsStorage(
                    backend='sqlite', path=str(db_path)
                )
        return self._run_results_storage

    def _parse_eval_dir_metadata(self, dir_name: str) -> Dict[str, Any]:
        """
        Extract structured metadata from an evaluation directory name.

        Supports naming patterns:
          - {exp_id}_eval_{split}_{city}_{date}_{time}_{job_id}
          - {exp_id}_eval_{split}_{date}_{time}_{job_id}
          - {exp_id}_eval_{split}_{city}_{date}_{time}
          - {exp_id}_eval_{split}_{city}_{date}

        Returns:
            Dict with keys: exp_id, eval_split, city, timestamp, slurm_job_id
        """
        known_cities = {"boston", "vegas", "pittsburgh", "singapore", "all"}
        parts = dir_name.split("_eval_")

        if len(parts) != 2:
            return {"exp_id": dir_name, "eval_split": None, "city": None,
                    "timestamp": None, "slurm_job_id": None}

        exp_id = parts[0]
        tokens = parts[1].split("_")
        eval_split = tokens[0] if tokens else None
        city = None
        date_start = 1

        if len(tokens) > 1 and tokens[1] in known_cities:
            city = tokens[1]
            date_start = 2

        timestamp = None
        slurm_job_id = None
        date_tokens = tokens[date_start:]
        if len(date_tokens) >= 2:
            timestamp = f"{date_tokens[0]}_{date_tokens[1]}"
            if len(date_tokens) >= 3:
                slurm_job_id = date_tokens[2]

        return {"exp_id": exp_id, "eval_split": eval_split, "city": city,
                "timestamp": timestamp, "slurm_job_id": slurm_job_id}

    def store_run_results(self, exp_id: str, force: bool = False) -> int:
        """
        Store per-run evaluation results for an experiment in run_results_storage.

        Discovers every evaluation directory for exp_id, groups runs by timestamp
        so multi-city evaluations become one record, and stores each run separately.
        Unlike collect_all_results(), this is additive and never overwrites.

        Args:
            exp_id: Experiment ID.
            force: If True, overwrite existing run records.

        Returns:
            Number of new run records stored.
        """
        if exp_id not in self.metadata:
            return 0

        # Find all evaluation directories matching exp_id
        eval_root = self.experiments_dir.parent / "evaluations"
        if not eval_root.exists():
            eval_root = self.experiments_dir / "evaluations"
        if not eval_root.exists():
            return 0

        eval_dirs = []
        for d in eval_root.glob(f"*{exp_id}*"):
            if not d.is_dir():
                continue
            parsed = self._parse_eval_dir_metadata(d.name)
            if parsed["exp_id"] == exp_id:
                eval_dirs.append(d)

        if not eval_dirs:
            return 0

        # Group by timestamp
        runs_by_timestamp: Dict[str, Dict] = {}
        for d in eval_dirs:
            meta = self._parse_eval_dir_metadata(d.name)
            ts = meta.get("timestamp") or d.name
            if ts not in runs_by_timestamp:
                runs_by_timestamp[ts] = {"dirs": [], "meta": meta, "cities": {}}
            city = meta.get("city") or "all"
            runs_by_timestamp[ts]["dirs"].append(d)
            runs_by_timestamp[ts]["cities"][city] = d

        stored = 0
        record = self.get_experiment_record(exp_id) or {}

        with self.run_results_storage as storage:
            for ts, run_group in sorted(runs_by_timestamp.items()):
                first_meta = run_group["meta"]
                job_id = first_meta.get("slurm_job_id", "")
                run_id = f"{exp_id}__{ts}"
                if job_id:
                    run_id += f"__{job_id}"

                if not force and storage.get(run_id):
                    continue

                # Parse CSVs for each city
                city_results = {}
                for city, eval_dir in run_group["cities"].items():
                    csv_files = sorted(
                        eval_dir.glob("*.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True
                    )
                    if not csv_files:
                        continue
                    try:
                        import csv as _csv
                        rows = []
                        with open(csv_files[0], newline='') as fh:
                            reader = _csv.DictReader(fh)
                            rows = [r for r in reader if r.get("token") != "average_all_frames"]
                        if not rows:
                            continue
                        # Map common column names to short keys
                        col_map = {
                            "NC": ["no_at_fault_collisions", "no_collision", "NC"],
                            "DAC": ["drivable_area_compliance", "DAC"],
                            "EP": ["ego_progress", "EP"],
                            "TTC": ["time_to_collision_within_bound", "time_to_collision", "TTC"],
                            "C": ["history_comfort", "comfort", "C"],
                            "PDMS": ["score", "pdm_score", "PDMS"],
                        }
                        metrics = {}
                        for short, candidates in col_map.items():
                            for col in candidates:
                                vals = [float(r[col]) for r in rows if col in r and r[col]]
                                if vals:
                                    metrics[short] = sum(vals) / len(vals)
                                    break
                        city_results[city] = {
                            "pdms": metrics.get("PDMS", 0.0),
                            "scenarios": len(rows),
                            **{k: v for k, v in metrics.items() if k != "PDMS"},
                            "csv_file": csv_files[0].name,
                            "eval_dir": eval_dir.name,
                        }
                    except Exception as e:
                        city_results[city] = {"pdms": 0.0, "error": str(e), "eval_dir": eval_dir.name}

                if not city_results:
                    continue

                per_city = {k: v for k, v in city_results.items() if k != "all"}
                source = per_city if per_city else city_results
                valid = [v["pdms"] for v in source.values() if v.get("pdms", 0) > 0]
                avg_pdms = sum(valid) / len(valid) if valid else 0.0

                state = self.metadata[exp_id]
                run_record = {
                    "run_id": run_id,
                    "exp_id": exp_id,
                    "eval_timestamp": ts,
                    "harvested_at": datetime.now().isoformat(),
                    "git_commit": record.get("git_commit"),
                    "git_branch": record.get("git_branch"),
                    "git_dirty": record.get("git_dirty"),
                    "slurm_train_job_id": state.get("train_job_id"),
                    "slurm_eval_job_id": job_id or None,
                    "eval_job_ids": state.get("eval_job_ids", []),
                    "config": {k: record.get(k) for k in (
                        "agent", "backbone", "epochs", "batch_size",
                        "learning_rate", "partition", "num_gpus"
                    )},
                    "cities": city_results,
                    "avg_pdms": avg_pdms,
                    "total_scenarios": sum(v.get("scenarios", 0) for v in city_results.values()),
                    "status": state.get("status", "unknown"),
                    "description": record.get("description", ""),
                }
                if storage.store(run_id, run_record):
                    stored += 1

        return stored

    def collect_all_run_results(
        self,
        exp_ids: Optional[List[str]] = None,
        force: bool = False,
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Collect per-run results for all (or specified) experiments.

        Args:
            exp_ids: Experiments to process. Defaults to all in metadata.
            force: Re-harvest existing run records.
            verbose: Print progress.

        Returns:
            Dict mapping exp_id to number of new runs stored.
        """
        if exp_ids is None:
            exp_ids = list(self.metadata.keys())
        if verbose:
            print(f"Collecting run history for {len(exp_ids)} experiments...")
        totals: Dict[str, int] = {}
        total_new = 0
        for exp_id in sorted(exp_ids):
            count = self.store_run_results(exp_id, force=force)
            totals[exp_id] = count
            total_new += count
            if verbose and count > 0:
                print(f"  [OK] {exp_id}: {count} new runs stored")
        if verbose:
            print(f"\n[OK] Stored {total_new} new runs across {len(totals)} experiments")
        return totals

    def show_run_history(self, exp_id: str, output_json: bool = False) -> List[Dict]:
        """
        Show all stored evaluation runs for an experiment.

        Args:
            exp_id: Experiment ID.
            output_json: If True, print JSON instead of a table.

        Returns:
            List of run records.
        """
        with self.run_results_storage as storage:
            all_runs = storage.query()

        runs = [r for r in all_runs if r.get("exp_id") == exp_id]
        runs.sort(key=lambda r: r.get("eval_timestamp", ""))

        if output_json:
            import json as _json
            print(_json.dumps(runs, indent=2))
            return runs

        if not runs:
            print(f"No run history found for {exp_id}")
            return []

        print(f"\n{'='*90}")
        print(f"  Run History for {exp_id}  ({len(runs)} runs)")
        print(f"{'='*90}")
        print(f"  {'Run ID':<40} {'Checkpoint':<20} {'Git':<10} {'PDMS':>8} {'Date'}")
        print(f"  {'-'*40} {'-'*20} {'-'*10} {'-'*8} {'-'*16}")

        for run in runs:
            run_id = run.get("run_id", "?")
            display_id = run_id.replace(f"{exp_id}__", "")
            if len(display_id) > 38:
                display_id = display_id[:35] + "..."
            ckpt = run.get("checkpoint_path", "")
            ckpt_name = (Path(ckpt).name[:18] + "..." if len(Path(ckpt).name) > 20 else Path(ckpt).name) if ckpt else "-"
            git = (run.get("git_commit") or "-")[:8]
            pdms = run.get("avg_pdms", 0.0)
            pdms_str = f"{pdms:.4f}" if pdms else "-"
            ts = run.get("eval_timestamp", "")
            if ts and len(ts) >= 15:
                date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"
            else:
                date_str = ts or "-"
            print(f"  {display_id:<40} {ckpt_name:<20} {git:<10} {pdms_str:>8} {date_str}")

        if runs:
            latest = runs[-1]
            cities = latest.get("cities", {})
            if cities:
                print(f"\n  Latest run per-city PDMS:")
                for city, data in sorted(cities.items()):
                    city_pdms = data.get("pdms", 0.0)
                    scenarios = data.get("scenarios", 0)
                    print(f"    {city:<15} {city_pdms:.4f}  ({scenarios} scenarios)")

        print(f"{'='*90}")
        return runs

    # =========================================================================
    # Batch Job Orchestrator (v0.9.0+)
    # =========================================================================

    def _parse_time_limit(self, time_str: str) -> float:
        """Parse SLURM time limit string to fractional hours. Returns 0.0 on failure."""
        if not time_str:
            return 0.0
        try:
            # Handle D-HH:MM:SS
            if "-" in time_str:
                days_str, rest = time_str.split("-", 1)
                days = int(days_str)
                parts = rest.split(":")
            else:
                days = 0
                parts = time_str.split(":")

            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                h, m, s = 0, int(parts[0]), int(parts[1])
            else:
                return float(parts[0]) / 60.0  # minutes
            return days * 24 + h + m / 60.0 + s / 3600.0
        except (ValueError, IndexError):
            return 0.0

    def preview_batch(self, exp_ids: List[str]) -> "BatchPreview":
        """
        Preview resource requirements for a list of experiments.

        Loads each experiment's YAML config and sums GPU hours, groups by partition.

        Args:
            exp_ids: Experiment IDs to preview.

        Returns:
            BatchPreview dataclass with resource summary.
        """
        ready = []
        not_ready = []
        warnings_list = []
        total_gpus = 0
        estimated_gpu_hours = 0.0
        partition_summary: Dict[str, int] = {}

        for exp_id in exp_ids:
            if exp_id not in self.metadata:
                not_ready.append(exp_id)
                warnings_list.append(f"{exp_id}: not registered in metadata")
                continue
            try:
                config = self._load_config(exp_id)
                ready.append(exp_id)
                gpus = int(config.get("num_gpus", 1))
                nodes = int(config.get("num_nodes", 1))
                time_h = self._parse_time_limit(config.get("time_limit", "0"))
                gpu_hours = gpus * nodes * time_h
                total_gpus += gpus * nodes
                estimated_gpu_hours += gpu_hours
                partition = config.get("partition", "unknown")
                partition_summary[partition] = partition_summary.get(partition, 0) + 1
            except FileNotFoundError:
                not_ready.append(exp_id)
                warnings_list.append(f"{exp_id}: YAML config missing")

        preview = BatchPreview(
            total_experiments=len(exp_ids),
            total_gpus=total_gpus,
            estimated_gpu_hours=round(estimated_gpu_hours, 2),
            partition_summary=partition_summary,
            ready=ready,
            not_ready=not_ready,
            warnings=warnings_list
        )

        print(f"Batch preview: {len(ready)}/{len(exp_ids)} ready")
        print(f"  Total GPUs requested: {total_gpus}")
        print(f"  Estimated GPU-hours:  {preview.estimated_gpu_hours}")
        if partition_summary:
            for p, count in partition_summary.items():
                print(f"  Partition {p}: {count} experiments")
        if not_ready:
            print(f"  WARNING: {len(not_ready)} experiments not ready: {not_ready}")

        return preview

    def submit_batch(
        self,
        exp_ids: List[str],
        dry_run: bool = False,
        train_only: bool = False,
        eval_only: bool = False
    ) -> Dict[str, Optional[str]]:
        """
        Submit multiple experiments sequentially.

        Individual failures are logged but do not stop the batch.

        Args:
            exp_ids: Ordered list of experiment IDs.
            dry_run: Pass through to submit_experiment().
            train_only: Submit only training jobs.
            eval_only: Submit only evaluation jobs.

        Returns:
            Dict mapping exp_id -> primary job ID (or None on failure / dry_run).
        """
        results: Dict[str, Optional[str]] = {}
        failed = []

        for exp_id in exp_ids:
            print(f"=== Submitting {exp_id} ===")
            try:
                job_ids = self.submit_experiment(
                    exp_id,
                    train_only=train_only,
                    eval_only=eval_only,
                    dry_run=dry_run
                )
                primary = job_ids.get("train_job_id") or job_ids.get("eval_job_id")
                results[exp_id] = primary
            except SystemExit:
                print(f"  FAILED: {exp_id}")
                results[exp_id] = None
                failed.append(exp_id)
            print()

        ok = len(exp_ids) - len(failed)
        print(f"submit_batch: {ok}/{len(exp_ids)} submitted, {len(failed)} failed")
        if failed:
            print(f"  Failed: {failed}")
        return results

    def generate_batch_script(
        self,
        exp_ids: List[str],
        slurm_account: str,
        partition: str,
        job_name: str = "expflow_batch",
        time_limit: str = "04:00:00",
        submit_flags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a SLURM wrapper script that submits a batch of experiments.

        Mirrors the pattern in scripts/trigger-train.slurm. Returns script as
        string; caller writes to disk and submits with sbatch.

        Args:
            exp_ids: Experiments to include.
            slurm_account: SBATCH --account value for the wrapper job.
            partition: SBATCH --partition for the wrapper job.
            job_name: SBATCH --job-name.
            time_limit: SBATCH --time for the wrapper (not individual jobs).
            submit_flags: Extra flags per submit call e.g. {"--sweep-cities": ""}.

        Returns:
            SLURM script as string.
        """
        manager_script = sys.argv[0] if sys.argv else "manager.py"
        output_log = self.logs_dir / "output" / f"{job_name}_%j.out"
        error_log = self.logs_dir / "error" / f"{job_name}_%j.err"

        conda_block = self._generate_conda_activation({})
        flags_str = ""
        if submit_flags:
            flags_str = " " + " ".join(
                f"{k} {v}" if v else k for k, v in submit_flags.items()
            )

        exp_list = " ".join(exp_ids)
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --account={slurm_account}",
            f"#SBATCH --partition={partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --mem=4G",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={output_log}",
            f"#SBATCH --error={error_log}",
            "",
            conda_block,
            "",
            f"cd {self.project_root}",
            "",
            f"for exp in {exp_list}; do",
            f'    echo "=== Submitting $exp ==="',
            f'    python {manager_script} submit "$exp"{flags_str} 2>&1',
            '    echo ""',
            "done",
            "",
            f"echo 'Batch complete: {len(exp_ids)} experiments submitted'",
        ]
        return "\n".join(lines)
