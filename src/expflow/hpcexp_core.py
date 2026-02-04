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
                # Note: Subclass should handle config reconstruction
                self.metadata = data
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
        Generate conda/environment activation commands

        Args:
            config: Experiment configuration

        Returns:
            Shell commands for environment activation
        """
        conda_root = config.get('conda_root', self.hpc_config.conda_root)
        conda_env = config.get('conda_env', self.hpc_config.conda_env)
        module_loads = config.get('module_loads', self.hpc_config.module_loads)

        script_lines = []

        # Load modules first
        if module_loads:
            for module in module_loads:
                script_lines.append(f"module load {module}")

        # Conda activation
        if conda_root and conda_env:
            conda_sh = f"{conda_root}/etc/profile.d/conda.sh"
            script_lines.extend([
                f"# Activate conda environment",
                f"if [ -f \"{conda_sh}\" ]; then",
                f"    source \"{conda_sh}\"",
                f"    conda activate {conda_env}",
                f"else",
                f"    echo \"Warning: Conda not found at {conda_root}\"",
                f"fi"
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

        # Create metadata entry
        self.metadata[exp_id] = {
            "exp_id": exp_id,
            "config": config_dict,
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
        config = meta["config"]

        # Load full config for script generation
        config_path = self.configs_dir / f"{exp_id}.yaml"
        with open(config_path) as f:
            full_config = yaml.safe_load(f)

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
            config = meta.get("config", {})
            if tags and not any(tag in config.get("tags", []) for tag in tags):
                continue
            filtered.append((exp_id, meta))

        if not filtered:
            print("No experiments found")
            return

        print(f"\nFound {len(filtered)} experiments:")
        print(f"{'ID':<15} {'Status':<12} {'Description':<50}")
        print("-" * 80)

        for exp_id, meta in sorted(filtered, key=lambda x: x[0]):
            config = meta.get("config", {})
            desc = config.get("description", "")[:47] + "..." if len(config.get("description", "")) > 50 else config.get("description", "")
            status_str = meta.get("status", "unknown")
            print(f"{exp_id:<15} {status_str:<12} {desc:<50}")

    def show_experiment(self, exp_id: str):
        """Show detailed information about an experiment"""

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            sys.exit(1)

        meta = self.metadata[exp_id]
        config = meta.get("config", {})

        print(f"\n{'='*70}")
        print(f"Experiment: {exp_id}")
        print(f"{'='*70}")
        print(f"\nDescription: {config.get('description', 'N/A')}")
        print(f"Status: {meta.get('status', 'unknown')}")

        print(f"\nConfiguration:")
        for key, value in config.items():
            if key not in ["exp_id", "description", "created_at", "submitted_at", "completed_at"]:
                print(f"  {key}: {value}")

        if meta.get("results"):
            print(f"\nResults:")
            for key, value in meta["results"].items():
                print(f"  {key}: {value}")

        print(f"\nTimeline:")
        print(f"  Created: {config.get('created_at', 'N/A')}")
        if meta.get("submitted_at"):
            print(f"  Submitted: {meta['submitted_at']}")
        if meta.get("completed_at"):
            print(f"  Completed: {meta['completed_at']}")

        if config.get("git_commit"):
            print(f"\nGit:")
            print(f"  Commit: {config['git_commit'][:8]}")
            print(f"  Branch: {config.get('git_branch', 'N/A')}")

        print(f"\n{'='*70}\n")

    def export_results(self, output_file: str = "results.csv"):
        """
        Export all experiment results to CSV

        Override this in subclass to customize columns
        """
        import pandas as pd

        records = []
        for exp_id, meta in self.metadata.items():
            config = meta.get("config", {})
            record = {
                "exp_id": exp_id,
                "description": config.get("description", ""),
                "status": meta.get("status", "unknown"),
                "partition": config.get("partition", ""),
                "num_gpus": config.get("num_gpus", 0),
                "created_at": config.get("created_at", ""),
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
        source_config = source_meta.get("config", {})

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

        # Create metadata entry
        self.metadata[new_exp_id] = {
            "exp_id": new_exp_id,
            "config": new_config,
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
            key=lambda x: x[1].get("config", {}).get("created_at", ""),
            reverse=True
        )[:10]

        for exp_id, meta in sorted_exps:
            config = meta.get("config", {})
            desc = config.get("description", "")[:30]
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

    def cancel(self, exp_id: str, job_type: Optional[str] = None):
        """Cancel running jobs for an experiment"""
        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            return

        meta = self.metadata[exp_id]
        cancelled = []

        jobs_to_cancel = []
        if job_type in (None, "train") and meta.get("train_job_id"):
            jobs_to_cancel.append(("train", meta["train_job_id"]))
        if job_type in (None, "eval") and meta.get("eval_job_id"):
            jobs_to_cancel.append(("eval", meta["eval_job_id"]))

        if not jobs_to_cancel:
            print(f"No jobs to cancel for {exp_id}")
            return

        for jtype, job_id in jobs_to_cancel:
            try:
                result = subprocess.run(
                    ["scancel", job_id],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    cancelled.append(f"{jtype} ({job_id})")
                else:
                    print(f"  Warning: Could not cancel {jtype} job {job_id}")
            except Exception as e:
                print(f"  Error cancelling {jtype} job: {e}")

        if cancelled:
            print(f"Cancelled jobs for {exp_id}: {', '.join(cancelled)}")
            meta["status"] = "cancelled"
            self._save_metadata()

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
