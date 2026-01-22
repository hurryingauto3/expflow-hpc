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
            self.checkpoints_dir
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
