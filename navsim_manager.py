#!/usr/bin/env python3
"""
NAVSIM Experiment Manager - Built on ExpFlow

Minimal-change integration: wraps existing SLURM scripts,
adds YAML-based configs and experiment tracking.

Usage:
    python navsim_manager.py new exp_b15 --template ijepa_mlp_v4
    python navsim_manager.py submit exp_b15
    python navsim_manager.py submit exp_b15 --dry-run
    python navsim_manager.py submit exp_b15 --eval-type one_stage
    python navsim_manager.py list
    python navsim_manager.py show exp_b15
    python navsim_manager.py harvest exp_b15
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Try to use expflow for additional features
try:
    from expflow import BaseExperimentManager, load_project_config
    from expflow.hpc_config import HPCConfig
    from expflow.cache_builder import BaseCacheBuilder, CacheConfig
    from expflow.results_harvester import (
        BaseResultsHarvester,
        TrainingMetrics,
        EvaluationMetrics,
    )
    from expflow.pruner import ExperimentPruner, PruneStats

    EXPFLOW_AVAILABLE = True
except ImportError:
    EXPFLOW_AVAILABLE = False
    BaseCacheBuilder = None  # Will define fallback below
    CacheConfig = None
    BaseResultsHarvester = None
    TrainingMetrics = None
    EvaluationMetrics = None
    ExperimentPruner = None
    PruneStats = None


# =============================================================================
# NAVSIM Cache Builder (inherits from ExpFlow)
# =============================================================================

if EXPFLOW_AVAILABLE and BaseCacheBuilder is not None:

    class NavsimCacheBuilder(BaseCacheBuilder):
        """
        NAVSIM-specific cache builder inheriting from ExpFlow's BaseCacheBuilder.

        Only implements the abstract methods - squashfs/cleanup scripts are inherited.
        """

        def __init__(self, hpc_config: HPCConfig = None, project_root: Path = None):
            # Load from .hpc_config.yaml if not provided
            if hpc_config is None:
                if project_root is None:
                    username = os.environ.get("USER", "ah7072")
                    project_root = Path(
                        f"/scratch/{username}/navsim-ssl-city-generalization"
                    )
                hpc_config = load_project_config(project_root)
            super().__init__(hpc_config)
            self.username = hpc_config.username

        def _generate_cache_build_script(self, config: CacheConfig) -> str:
            """Generate NAVSIM-specific cache building script"""
            cache_type = config.cache_type
            if cache_type == "training":
                return self._generate_training_cache_script(config)
            elif cache_type == "metric":
                return self._generate_metric_cache_script(config)
            else:
                raise ValueError(f"Unknown cache_type: {cache_type}")

        def get_cache_script_command(self, config: CacheConfig) -> str:
            """Get the Python command for cache building"""
            if config.cache_type == "training":
                params = config.cache_params
                agent = params.get("agent", "transfuser_agent")

                # TransFuser has its own sensor config, doesn't need vision_views override
                if "transfuser" in agent:
                    return (
                        f"python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_dataset_caching.py "
                        f"agent={agent} "
                        f"train_test_split={params.get('train_split', 'navtrain')} "
                        f'cache_path="{config.cache_output_dir}" '
                        f"force_cache_computation={str(config.force_rebuild).lower()} "
                        f'experiment_name="cache_build_{config.cache_name}" '
                        f"worker=ray_distributed_no_torch "
                        f"worker.threads_per_node={config.num_workers}"
                    )
                else:
                    vision_views = params.get(
                        "vision_views", ["cam_l0", "cam_f0", "cam_r0"]
                    )
                    vision_views_str = (
                        str(vision_views).replace("'", "").replace(" ", "")
                    )
                    return (
                        f"python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_dataset_caching.py "
                        f"agent={agent} "
                        f"agent.use_multi_camera=true "
                        f"agent.vision_views={vision_views_str} "
                        f"train_test_split={params.get('train_split', 'navtrain')} "
                        f'cache_path="{config.cache_output_dir}" '
                        f"force_cache_computation={str(config.force_rebuild).lower()} "
                        f'experiment_name="cache_build_{config.cache_name}" '
                        f"worker=ray_distributed_no_torch "
                        f"worker.threads_per_node={config.num_workers}"
                    )
            else:  # metric
                params = config.cache_params
                return (
                    f"python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_metric_caching.py "
                    f"train_test_split=\"{params.get('eval_split', 'navtest')}\" "
                    f'metric_cache_path="{config.cache_output_dir}" '
                    f"worker=ray_distributed_no_torch "
                    f"max_number_of_workers={config.num_workers}"
                )

        def _generate_training_cache_script(self, config: CacheConfig) -> str:
            """Generate training cache SLURM script"""
            params = config.cache_params
            num_cams = params.get("num_cams", 6)
            vision_views = params.get("vision_views")
            if vision_views is None:
                vision_views = (
                    ["cam_l0", "cam_f0", "cam_r0", "cam_l1", "cam_r1", "cam_b0"]
                    if num_cams == 6
                    else ["cam_l0", "cam_f0", "cam_r0"]
                )
            vision_views_str = str(vision_views).replace("'", "").replace(" ", "")
            agent = params.get("agent", "ijepa_planning_agent_v4")
            train_split = params.get("train_split", "navtrain")

            return f"""#!/bin/bash
# =============================================================================
# Training Cache Builder - Auto-generated via ExpFlow inheritance
# Cache: {config.cache_name}
# Type: training (dataset features)
# Cameras: {num_cams}
# Generated: {datetime.now().isoformat()}
# =============================================================================

#SBATCH --job-name=tcache_{config.cache_name[:12]}
#SBATCH --partition={config.partition}
#SBATCH --account={config.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.num_cpus}
#SBATCH --mem={config.memory}
#SBATCH --time={config.time_limit}
#SBATCH --output={self.logs_dir}/caching/build_{config.cache_name}_%j.out
#SBATCH --error={self.logs_dir}/caching/build_{config.cache_name}_%j.err

set -e

echo "=============================================="
echo "Building NAVSIM Training Cache"
echo "Cache: {config.cache_name}"
echo "Cameras: {num_cams}"
echo "Agent: {agent}"
echo "Split: {train_split}"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=============================================="

export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/{self.username}/navsim-ssl-city-generalization/navsim"
export OPENSCENE_DATA_ROOT="/scratch/{self.username}/data"
export NUPLAN_MAPS_ROOT="/scratch/{self.username}/data/maps"
export NAVSIM_EXP_ROOT="/scratch/{self.username}/experiments"

mkdir -p "{config.cache_output_dir}"
cd "${{NAVSIM_DEVKIT_ROOT}}"

source /scratch/{self.username}/miniconda3/etc/profile.d/conda.sh
conda activate navsim
export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

{self.get_cache_script_command(config)}

echo ""
echo "=============================================="
echo "Training cache completed: $(date)"
echo "Directory: {config.cache_output_dir}"
du -sh "{config.cache_output_dir}"
echo "=============================================="
"""

        def _generate_metric_cache_script(self, config: CacheConfig) -> str:
            """Generate metric cache SLURM script"""
            params = config.cache_params
            eval_split = params.get("eval_split", "navtest")

            return f"""#!/bin/bash
# =============================================================================
# Metric Cache Builder - Auto-generated via ExpFlow inheritance
# Cache: {config.cache_name}
# Type: metric (evaluation precomputation)
# Split: {eval_split}
# Generated: {datetime.now().isoformat()}
# =============================================================================

#SBATCH --job-name=mcache_{config.cache_name[:12]}
#SBATCH --partition={config.partition}
#SBATCH --account={config.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.num_cpus}
#SBATCH --mem={config.memory}
#SBATCH --time=08:00:00
#SBATCH --output={self.logs_dir}/caching/build_{config.cache_name}_%j.out
#SBATCH --error={self.logs_dir}/caching/build_{config.cache_name}_%j.err

set -e

echo "=============================================="
echo "Building NAVSIM Metric Cache"
echo "Cache: {config.cache_name}"
echo "Split: {eval_split}"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=============================================="

export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/{self.username}/navsim-ssl-city-generalization/navsim"
export OPENSCENE_DATA_ROOT="/scratch/{self.username}/data"
export NUPLAN_MAPS_ROOT="/scratch/{self.username}/data/maps"
export NAVSIM_EXP_ROOT="/scratch/{self.username}/experiments"

mkdir -p "{config.cache_output_dir}"
cd "${{NAVSIM_DEVKIT_ROOT}}"

source /scratch/{self.username}/miniconda3/etc/profile.d/conda.sh
conda activate navsim
export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

{self.get_cache_script_command(config)}

echo ""
echo "=============================================="
echo "Metric cache completed: $(date)"
echo "Directory: {config.cache_output_dir}"
du -sh "{config.cache_output_dir}"
echo "=============================================="
"""

else:
    # Fallback: ExpFlow not available
    NavsimCacheBuilder = None


# =============================================================================
# NAVSIM Results Harvester (inherits from ExpFlow)
# =============================================================================

if EXPFLOW_AVAILABLE and BaseResultsHarvester is not None:

    class NavsimResultsHarvester(BaseResultsHarvester):
        """
        NAVSIM-specific results harvester inheriting from ExpFlow's BaseResultsHarvester.

        Handles:
        - PDM Score evaluation results from CSVs
        - Lightning training logs with TensorBoard events
        - NAVSIM-specific metrics (NC, DAC, EP, TTC, Comfort)
        """

        def __init__(self, experiments_dir: Path = None, results_dir: Path = None):
            username = os.environ.get("USER", "ah7072")
            if experiments_dir is None:
                experiments_dir = Path(f"/scratch/{username}/experiments")
            if results_dir is None:
                results_dir = experiments_dir / "analysis"

            super().__init__(experiments_dir, results_dir)
            self.username = username

            # NAVSIM-specific paths
            self.training_dir = self.experiments_dir / "training"
            self.evaluations_dir = self.experiments_dir / "evaluations"

        def find_tensorboard_logs(self, exp_id: str) -> List[Path]:
            """Find TensorBoard event files for a NAVSIM experiment."""
            import glob

            # NAVSIM training logs are in: training/{exp_id}_*/*/lightning_logs/*/
            patterns = [
                self.training_dir / f"{exp_id}_*" / "**" / "events.out.tfevents*",
                self.training_dir / f"*{exp_id}*" / "**" / "events.out.tfevents*",
            ]

            event_files = []
            for pattern in patterns:
                event_files.extend(glob.glob(str(pattern), recursive=True))

            return [Path(f) for f in event_files]

        def find_evaluation_results(self, exp_id: str) -> List[Path]:
            """Find evaluation result files for a NAVSIM experiment."""
            import glob

            # NAVSIM eval results are in: evaluations/eval_*{exp_id}*/*.csv
            patterns = [
                self.evaluations_dir / f"*{exp_id}*" / "*.csv",
                self.evaluations_dir / f"eval_*{exp_id}*" / "*.csv",
            ]

            result_files = []
            for pattern in patterns:
                result_files.extend(glob.glob(str(pattern), recursive=True))

            return [Path(f) for f in result_files]

        def parse_evaluation_results(self, result_file: Path) -> EvaluationMetrics:
            """
            Parse NAVSIM PDM Score evaluation results from CSV.

            Expected columns: pdm_score, no_collision, drivable_area_compliance,
                            ego_progress, time_to_collision, comfort
            """
            import pandas as pd

            # Extract exp_id and split from path
            # Path format: evaluations/eval_{split}_{exp_id}_{timestamp}/results.csv
            parent_name = result_file.parent.name
            parts = parent_name.split("_")

            # Try to extract exp_id and split
            exp_id = "unknown"
            eval_split = "unknown"

            if parent_name.startswith("eval_"):
                # Format: eval_navtest_B21_20260126_014308
                if len(parts) >= 3:
                    eval_split = parts[1]  # navtest
                    exp_id = parts[2]  # B21

            try:
                df = pd.read_csv(result_file)

                # Calculate mean scores
                metrics = {}
                score = None

                # PDM Score
                for col in ["pdm_score", "PDMS", "score"]:
                    if col in df.columns:
                        score = float(df[col].mean())
                        metrics["pdm_score"] = score
                        break

                # Component metrics with various column name formats
                metric_mapping = {
                    "no_collision": ["no_collision", "NC", "nc"],
                    "drivable_area_compliance": [
                        "drivable_area_compliance",
                        "DAC",
                        "dac",
                    ],
                    "ego_progress": ["ego_progress", "EP", "ep"],
                    "time_to_collision": ["time_to_collision", "TTC", "ttc"],
                    "comfort": ["comfort", "Comfort", "C"],
                }

                for metric_name, possible_cols in metric_mapping.items():
                    for col in possible_cols:
                        if col in df.columns:
                            metrics[metric_name] = float(df[col].mean())
                            break

                # Add scenario count
                metrics["num_scenarios"] = len(df)

                return EvaluationMetrics(
                    exp_id=exp_id,
                    eval_split=eval_split,
                    score=score,
                    metrics=metrics,
                    csv_path=str(result_file),
                )

            except Exception as e:
                print(f"Warning: Failed to parse {result_file}: {e}")
                return EvaluationMetrics(
                    exp_id=exp_id,
                    eval_split=eval_split,
                    score=None,
                    metrics={},
                    csv_path=str(result_file),
                )

        def harvest_training_metrics(self, exp_id: str) -> Optional[TrainingMetrics]:
            """
            Harvest training metrics for a NAVSIM experiment.

            Looks for Lightning TensorBoard logs and extracts loss curves.
            """
            # Find TensorBoard logs
            event_files = self.find_tensorboard_logs(exp_id)
            if not event_files:
                # Try to find metrics from log files
                return self._harvest_from_log_files(exp_id)

            # Use most recent event file
            event_dir = event_files[0].parent
            scalars = self.extract_tensorboard_scalars(event_dir)

            if not scalars:
                return self._harvest_from_log_files(exp_id)

            metrics = TrainingMetrics(exp_id=exp_id)

            # NAVSIM Lightning loss keys
            train_keys = ["train_loss", "train/loss", "loss", "train_loss_epoch"]
            val_keys = ["val_loss", "val/loss", "validation_loss", "val_loss_epoch"]

            for key in train_keys:
                if key in scalars:
                    metrics.train_loss_last = scalars[key]["last_value"]
                    metrics.train_loss_min = scalars[key]["min_value"]
                    break

            for key in val_keys:
                if key in scalars:
                    metrics.val_loss_last = scalars[key]["last_value"]
                    metrics.val_loss_min = scalars[key]["min_value"]
                    break

            # Epochs
            for key in ["epoch", "epochs", "trainer/global_step"]:
                if key in scalars:
                    if key == "trainer/global_step":
                        # Estimate epochs from steps
                        metrics.epochs_completed = None
                    else:
                        metrics.epochs_completed = int(scalars[key]["last_value"])
                    break

            # Learning rate
            for key in ["lr", "learning_rate", "lr-AdamW", "train/lr"]:
                if key in scalars:
                    metrics.final_lr = scalars[key]["last_value"]
                    break

            metrics.metrics = scalars
            return metrics

        def _harvest_from_log_files(self, exp_id: str) -> Optional[TrainingMetrics]:
            """Fallback: harvest metrics from SLURM log files."""
            import glob

            log_patterns = [
                self.experiments_dir / "logs" / "output" / f"{exp_id}_train_*.out",
            ]

            log_files = []
            for pattern in log_patterns:
                log_files.extend(glob.glob(str(pattern)))

            if not log_files:
                return None

            # Use most recent
            log_file = sorted(log_files, key=lambda x: Path(x).stat().st_mtime)[-1]

            try:
                with open(log_file, "r") as f:
                    content = f.read()

                metrics = TrainingMetrics(exp_id=exp_id)

                # Extract final epoch metrics from log
                # Look for patterns like "Epoch 29: train_loss=0.1234, val_loss=0.5678"
                epoch_matches = re.findall(
                    r"Epoch\s+(\d+).*?(?:train_loss|loss)[=:]\s*([\d.]+)",
                    content,
                    re.IGNORECASE,
                )
                if epoch_matches:
                    last_epoch, last_loss = epoch_matches[-1]
                    metrics.epochs_completed = int(last_epoch) + 1
                    metrics.train_loss_last = float(last_loss)

                val_matches = re.findall(
                    r"(?:val_loss|validation)[=:]\s*([\d.]+)", content, re.IGNORECASE
                )
                if val_matches:
                    metrics.val_loss_last = float(val_matches[-1])

                return metrics

            except Exception as e:
                print(f"Warning: Could not parse log file {log_file}: {e}")
                return None

        def generate_comparison_table(
            self, exp_ids: List[str], output_file: Optional[Path] = None
        ) -> Optional[Path]:
            """
            Generate a comparison table for multiple experiments.

            Returns CSV with training and evaluation metrics side by side.
            """
            import pandas as pd

            rows = []
            for exp_id in exp_ids:
                row = {"exp_id": exp_id}

                # Training metrics
                train_metrics = self.harvest_training_metrics(exp_id)
                if train_metrics:
                    row["train_loss"] = train_metrics.train_loss_last
                    row["val_loss"] = train_metrics.val_loss_last
                    row["epochs"] = train_metrics.epochs_completed

                # Evaluation metrics
                eval_metrics_list = self.harvest_evaluation_metrics(exp_id)
                for em in eval_metrics_list:
                    prefix = f"{em.eval_split}_" if em.eval_split != "unknown" else ""
                    row[f"{prefix}pdms"] = em.score
                    for k, v in em.metrics.items():
                        if k != "num_scenarios":
                            row[f"{prefix}{k}"] = v

                rows.append(row)

            if not rows:
                return None

            df = pd.DataFrame(rows)

            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.csvs_dir / f"comparison_{timestamp}.csv"

            df.to_csv(output_file, index=False)
            print(f"✓ Comparison table saved: {output_file}")
            return output_file

else:
    NavsimResultsHarvester = None


# =============================================================================
# NAVSIM Experiment Pruner (uses ExpFlow's ExperimentPruner)
# =============================================================================

if EXPFLOW_AVAILABLE and ExperimentPruner is not None:

    class NavsimPruner(ExperimentPruner):
        """
        NAVSIM-specific experiment pruner.

        Inherits from ExpFlow's ExperimentPruner with NAVSIM-specific paths.
        """

        def __init__(self):
            username = os.environ.get("USER", "ah7072")
            scratch = Path(f"/scratch/{username}")

            super().__init__(
                experiments_dir=scratch / "experiments" / "training",
                evaluations_dir=scratch / "experiments" / "evaluations",
                archive_dir=scratch / "experiments" / ".archive",
            )

        def _has_valid_checkpoint(
            self, exp_dir: Path, required_epochs: Optional[int] = None
        ) -> bool:
            """Override to check NAVSIM checkpoint patterns"""
            # NAVSIM uses lightning_logs/version_*/checkpoints/*.ckpt
            ckpt_patterns = [
                "lightning_logs/*/checkpoints/*.ckpt",
                "checkpoints/*.ckpt",
                "*.ckpt",
            ]

            for pattern in ckpt_patterns:
                matches = list(exp_dir.rglob(pattern))
                if matches:
                    if required_epochs is None:
                        return True
                    # Check for epoch in checkpoint name
                    for ckpt in matches:
                        if (
                            f"epoch={required_epochs}" in ckpt.name
                            or f"epoch_{required_epochs}" in ckpt.name
                        ):
                            return True
            return False

        def _has_valid_eval_results(
            self, exp_dir: Path, eval_dir: Optional[Path] = None
        ) -> bool:
            """Override to check NAVSIM evaluation result patterns"""
            # Check for NAVSIM-specific result files
            result_patterns = [
                "pdm_score*.csv",
                "*_results.csv",
                "metrics.json",
                "results.json",
            ]

            # Check in training dir
            for pattern in result_patterns:
                matches = list(exp_dir.rglob(pattern))
                for f in matches:
                    if f.stat().st_size > 0:
                        return True

            # Check evaluation directory
            if eval_dir and eval_dir.exists():
                for pattern in result_patterns:
                    matches = list(eval_dir.rglob(pattern))
                    for f in matches:
                        if f.stat().st_size > 0:
                            return True

            return False

        def _find_corresponding_eval_dir(self, train_dir: Path) -> Optional[Path]:
            """Find eval dir for a training run using NAVSIM naming conventions"""
            if not self.evaluations_dir or not self.evaluations_dir.exists():
                return None

            train_name = train_dir.name
            # Extract base experiment ID (e.g., B21 from B21_20260125_152818)
            base_exp_id = train_name.split("_")[0]

            # Look for eval directories matching this experiment
            eval_dirs = []
            for eval_dir in self.evaluations_dir.iterdir():
                if eval_dir.is_dir() and base_exp_id in eval_dir.name:
                    eval_dirs.append(eval_dir)

            if eval_dirs:
                # Return most recent
                return sorted(eval_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[
                    0
                ]

            return None

else:
    NavsimPruner = None


# =============================================================================
# NAVSIM Experiment Manager
# =============================================================================


class NavsimExperimentManager:
    """
    Experiment manager for NAVSIM thesis experiments.

    Key design: generates complete SLURM scripts matching your existing
    hand-written ones, with all NAVSIM-native parameters.
    """

    def __init__(self, project_root: Optional[str] = None):
        # Auto-detect paths - try to load from .hpc_config.yaml first
        self.username = os.environ.get("USER", "ah7072")
        self.scratch = Path(f"/scratch/{self.username}")

        # Project paths - defaults to navsim-ssl-city-generalization repo
        self.project_root = (
            Path(project_root)
            if project_root
            else self.scratch / "navsim-ssl-city-generalization"
        )

        # Load from .hpc_config.yaml if available
        if EXPFLOW_AVAILABLE:
            try:
                self.hpc_config = load_project_config(self.project_root)
                self.scratch = Path(self.hpc_config.scratch_dir)
                self.experiments_root = Path(self.hpc_config.experiments_dir)
                self.logs_dir = Path(self.hpc_config.logs_dir)
                self.checkpoints_dir = Path(self.hpc_config.checkpoints_dir)
            except Exception:
                # Fall back to defaults
                self.hpc_config = None
                self.experiments_root = self.scratch / "experiments"
                self.logs_dir = self.experiments_root / "logs"
                self.checkpoints_dir = self.experiments_root / "checkpoints"
        else:
            self.hpc_config = None
            self.experiments_root = self.scratch / "experiments"
            self.logs_dir = self.experiments_root / "logs"
            self.checkpoints_dir = self.experiments_root / "checkpoints"

        self.scripts_dir = self.project_root / "scripts" / "train"
        self.training_dir = self.experiments_root / "training"

        # ExpFlow integration directories
        self.configs_dir = self.project_root / "experiment_configs"
        self.templates_dir = self.project_root / "experiment_templates"
        self.generated_dir = self.project_root / "generated_scripts"
        self.metadata_db = self.configs_dir / "experiments.json"

        # Create directories
        for d in [
            self.configs_dir,
            self.templates_dir,
            self.generated_dir,
            self.logs_dir / "output",
            self.logs_dir / "error",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load experiment tracking database"""
        if self.metadata_db.exists():
            with open(self.metadata_db) as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save experiment tracking database"""
        with open(self.metadata_db, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _get_git_info(self) -> Dict[str, Any]:
        """Get current git commit info"""
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.project_root,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()[:8]
            )

            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=self.project_root,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            return {"git_commit": commit, "git_branch": branch}
        except:
            return {"git_commit": None, "git_branch": None}

    # -------------------------------------------------------------------------
    # Checkpoint Discovery (for Resume)
    # -------------------------------------------------------------------------

    def _find_latest_checkpoint(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Find the latest checkpoint for an experiment.

        NAVSIM checkpoints are stored in:
        /scratch/$USER/experiments/training/<exp_id>_<timestamp>/<run_timestamp>/lightning_logs/version_*/checkpoints/

        Returns:
            Dictionary with checkpoint info: {path, epoch, type} or None if not found
        """
        # Look for training directories matching this experiment
        training_dir = self.training_dir
        if not training_dir.exists():
            return None

        # Find all directories starting with exp_id
        exp_dirs = []
        for d in training_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{exp_id}_"):
                exp_dirs.append(d)

        if not exp_dirs:
            # Try exact match (for resumed experiments)
            exact_match = training_dir / exp_id
            if exact_match.exists():
                exp_dirs = [exact_match]

        if not exp_dirs:
            return None

        # Sort by modification time, most recent first
        exp_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Search for checkpoints in each experiment directory
        for exp_dir in exp_dirs:
            # NAVSIM structure: exp_dir/<run_timestamp>/lightning_logs/version_*/checkpoints/
            checkpoint_candidates = []

            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # Look in lightning_logs
                lightning_logs = run_dir / "lightning_logs"
                if lightning_logs.exists():
                    for version_dir in lightning_logs.iterdir():
                        if version_dir.is_dir() and version_dir.name.startswith(
                            "version_"
                        ):
                            ckpt_dir = version_dir / "checkpoints"
                            if ckpt_dir.exists():
                                for ckpt in ckpt_dir.glob("*.ckpt"):
                                    checkpoint_candidates.append(ckpt)

            if checkpoint_candidates:
                # Find the most recent checkpoint (by epoch number, then mtime)
                best_ckpt = None
                best_epoch = -1

                for ckpt in checkpoint_candidates:
                    epoch = self._extract_epoch_from_checkpoint(ckpt)
                    if epoch is not None and epoch > best_epoch:
                        best_epoch = epoch
                        best_ckpt = ckpt

                # Fallback to most recent by mtime if no epoch found
                if best_ckpt is None:
                    best_ckpt = max(
                        checkpoint_candidates, key=lambda p: p.stat().st_mtime
                    )
                    best_epoch = self._extract_epoch_from_checkpoint(best_ckpt)

                return {"path": str(best_ckpt), "epoch": best_epoch, "type": "latest"}

        return None

    def _extract_epoch_from_checkpoint(self, checkpoint_path: Path) -> Optional[int]:
        """Extract epoch number from checkpoint filename"""
        filename = checkpoint_path.stem

        # NAVSIM/PyTorch Lightning patterns: epoch=28-step=25723
        patterns = [
            r"epoch[=_](\d+)",
            r"epoch(\d+)",
            r"checkpoint[_-](\d+)",
            r"step[=_](\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    # -------------------------------------------------------------------------
    # Resume Experiment
    # -------------------------------------------------------------------------

    def resume_experiment(
        self,
        source_exp_id: str,
        new_exp_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        submit: bool = False,
        eval_only: bool = False,
        eval_type: str = "one_stage",
        dry_run: bool = False,
        **kwargs,
    ) -> str:
        """
        Create a new experiment that resumes from a previous experiment's checkpoint.

        Args:
            source_exp_id: Experiment ID to resume from
            new_exp_id: ID for the new resumed experiment (auto-generated if None)
            checkpoint_path: Specific checkpoint path (auto-detects latest if None)
            submit: If True, immediately submit the training job
            eval_only: If True, skip training and only submit eval (requires completed training)
            eval_type: Evaluation type (one_stage or two_stage)
            dry_run: If True, show what would be done without executing
            **kwargs: Additional config overrides

        Returns:
            The new experiment ID
        """
        # Check if source experiment exists in metadata or can find checkpoints
        source_meta = self.metadata.get(source_exp_id, {})
        source_config = source_meta.get("config", {})

        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_info = self._find_latest_checkpoint(source_exp_id)
            if checkpoint_info is None:
                print(f"Error: No checkpoint found for experiment {source_exp_id}")
                print(f"  Searched in: {self.training_dir}/{source_exp_id}_*")
                sys.exit(1)
            checkpoint_path = checkpoint_info["path"]
            resume_epoch = checkpoint_info["epoch"]
            print(f"Found checkpoint: {Path(checkpoint_path).name}")
            if resume_epoch is not None:
                print(f"  Resuming from epoch: {resume_epoch}")
        else:
            # Validate provided checkpoint exists
            if not Path(checkpoint_path).exists():
                print(f"Error: Checkpoint not found: {checkpoint_path}")
                sys.exit(1)
            resume_epoch = self._extract_epoch_from_checkpoint(Path(checkpoint_path))

        # Generate new experiment ID if not provided
        if new_exp_id is None:
            resume_count = source_meta.get("resume_count", 0) + 1
            new_exp_id = f"{source_exp_id}_resume{resume_count}"

            # Handle case where resumed experiment was itself resumed
            if source_meta.get("resume_from_exp_id"):
                original_exp_id = source_meta["resume_from_exp_id"]
                new_exp_id = f"{original_exp_id}_resume{resume_count}"

        # Check if new experiment ID already exists
        if new_exp_id in self.metadata:
            print(f"Error: Experiment {new_exp_id} already exists")
            print(
                f"  Specify a different --new-exp-id or delete the existing experiment"
            )
            sys.exit(1)

        # Create new config based on source config (or defaults if source not in metadata)
        if source_config:
            new_config = {**source_config}
        else:
            # Load from template or use defaults
            new_config = self._get_default_config()

        # Update config with resume info
        new_config.update(
            {
                "exp_id": new_exp_id,
                "description": f"Resume from {source_exp_id}: {new_config.get('description', '')}",
                "created_at": datetime.now().isoformat(),
                "resume_from_exp_id": source_exp_id,
                "resume_checkpoint_path": checkpoint_path,
                "resume_epoch": resume_epoch,
                **self._get_git_info(),
                **kwargs,  # Allow user to override any config values
            }
        )

        if dry_run:
            print(f"\n[DRY RUN] Would create resume experiment: {new_exp_id}")
            print(f"  Resuming from: {source_exp_id}")
            print(f"  Checkpoint: {Path(checkpoint_path).name}")
            if resume_epoch is not None:
                print(f"  Starting epoch: {resume_epoch + 1}")
            return new_exp_id

        # Save new config
        config_path = self.configs_dir / f"{new_exp_id}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

        # Create metadata entry
        self.metadata[new_exp_id] = {
            "exp_id": new_exp_id,
            "config": new_config,
            "status": "created",
            "train_script_path": None,
            "eval_script_path": None,
            "train_job_id": None,
            "eval_job_id": None,
            "results": {},
            "resume_from_exp_id": source_exp_id,
            "resume_checkpoint_path": checkpoint_path,
            "resume_epoch": resume_epoch,
            "resume_count": 0,
        }

        # Update source experiment's resume count
        if source_exp_id in self.metadata:
            self.metadata[source_exp_id]["resume_count"] = (
                self.metadata[source_exp_id].get("resume_count", 0) + 1
            )

        self._save_metadata()

        print(f"\n✓ Created resume experiment: {new_exp_id}")
        print(f"  Config: {config_path}")
        print(f"  Resuming from: {source_exp_id}")
        print(f"  Checkpoint: {Path(checkpoint_path).name}")
        if resume_epoch is not None:
            print(f"  Starting epoch: {resume_epoch + 1}")

        # Submit if requested
        if submit:
            print()
            self.submit_experiment(
                new_exp_id,
                train_only=False,
                eval_only=eval_only,
                eval_type=eval_type,
                dry_run=dry_run,
            )

        return new_exp_id

    def _get_default_config(self) -> Dict:
        """Get default experiment configuration"""
        return {
            "agent": "ijepa_planning_agent_v4",
            "backbone": "ijepa",
            "epochs": 30,
            "batch_size": 24,
            "learning_rate": 1e-4,
            "encoder_learning_rate": 3e-5,
            "trainable_fraction": 0.5,
            "vision_mode": "multi_per_view",
            "cache_name": "training_cache_ijepa_planning_agent_v4_6_cams",
            "partition": "l40s_public",
            "num_gpus": 4,
            "eval_split": "navtest",
            "eval_workers": 48,
            "eval_type": "one_stage",
        }

    # -------------------------------------------------------------------------
    # Experiment Creation
    # -------------------------------------------------------------------------

    def create_experiment(
        self,
        exp_id: str,
        template: Optional[str] = None,
        description: str = "",
        **overrides,
    ) -> Dict:
        """
        Create a new experiment configuration.
        """
        if exp_id in self.metadata:
            print(f"⚠ Experiment {exp_id} exists. Overwriting config.")

        # Start with defaults
        config = {
            "exp_id": exp_id,
            "description": description,
            "created_at": datetime.now().isoformat(),
            **self._get_git_info(),
        }

        # Load template if specified
        if template:
            template_path = self.templates_dir / f"{template}.yaml"
            if template_path.exists():
                with open(template_path) as f:
                    template_config = yaml.safe_load(f)
                    config.update(template_config)
                print(f"✓ Loaded template: {template}")
            else:
                print(f"⚠ Template not found: {template_path}")

        # Apply overrides
        config.update(overrides)
        # Set cross-attn params if present in overrides
        if "view_fusion_method" in overrides:
            config["view_fusion_method"] = overrides["view_fusion_method"]
        if "cross_attn_heads" in overrides:
            config["cross_attn_heads"] = overrides["cross_attn_heads"]
        if "cross_attn_layers" in overrides:
            config["cross_attn_layers"] = overrides["cross_attn_layers"]
        config["exp_id"] = exp_id  # Ensure exp_id isn't overwritten

        # Save config
        config_path = self.configs_dir / f"{exp_id}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Update metadata
        self.metadata[exp_id] = {
            "exp_id": exp_id,
            "config": config,
            "status": "created",
            "train_script_path": None,
            "eval_script_path": None,
            "train_job_id": None,
            "eval_job_id": None,
            "results": {},
        }
        self._save_metadata()

        print(f"✓ Created experiment: {exp_id}")
        print(f"  Config: {config_path}")
        return config

    # -------------------------------------------------------------------------
    # Training Script Generation
    # -------------------------------------------------------------------------

    def _generate_train_script(self, config: Dict) -> str:
        """
        Generate SLURM training script with ALL NAVSIM-native parameters.
        Matches the format of your hand-written scripts.

        Supports different agent types:
        - ego_status_mlp_agent: No vision, no cache, simple MLP
        - transfuser_agent: Vision + LiDAR, needs its own cache
        - ijepa_planning_agent_*: Vision with I-JEPA backbone
        """
        exp_id = config["exp_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract all config values with defaults
        agent = config.get("agent", "ijepa_planning_agent_v4")
        backbone = config.get("backbone", "ijepa")

        # Detect agent type for specialized script generation
        if agent == "ego_status_mlp_agent":
            return self._generate_ego_mlp_train_script(config, exp_id, timestamp)
        elif agent == "transfuser_agent":
            return self._generate_transfuser_train_script(config, exp_id, timestamp)

        # Default: I-JEPA style agent with vision
        # Fusion method
        view_fusion_method = config.get("view_fusion_method", "mean")
        cross_attn_heads = config.get("cross_attn_heads", 8)
        cross_attn_layers = config.get("cross_attn_layers", 2)
        partition = config.get("partition", "l40s_public")
        account = config.get("account", "torch_pr_68_general")
        num_gpus = config.get("num_gpus", 4)
        num_nodes = config.get("num_nodes", 1)
        cpus_per_task = config.get("cpus_per_task", 16)
        time_limit = config.get("time_limit", "48:00:00")

        # Training hyperparameters
        batch_size = config.get("batch_size", 48)
        learning_rate = config.get("learning_rate", 1e-4)
        encoder_learning_rate = config.get("encoder_learning_rate", 3e-5)
        trainable_fraction = config.get("trainable_fraction", 0.5)
        epochs = config.get("epochs", 30)
        num_workers = config.get("num_workers", 12)

        # Vision configuration
        vision_mode = config.get("vision_mode", "multi_per_view")
        camera_views = config.get("camera_views", ["cam_l0", "cam_f0", "cam_r0"])
        image_size = config.get("image_size", [224, 224])
        view_fusion_method = config.get("view_fusion_method", "mean")
        cross_attn_heads = config.get("cross_attn_heads", 8)
        cross_attn_layers = config.get("cross_attn_layers", 2)

        # Planning head configuration
        planning_head_type = config.get("planning_head_type", "mlp")
        transformer_head_hidden_dim = config.get("transformer_head_hidden_dim", 256)
        transformer_head_num_heads = config.get("transformer_head_num_heads", 8)
        transformer_head_num_layers = config.get("transformer_head_num_layers", 3)
        transformer_head_dropout = config.get("transformer_head_dropout", 0.1)
        transformer_head_lr_multiplier = config.get(
            "transformer_head_lr_multiplier", 1.0
        )

        # Cache configuration
        cache_name = config.get(
            "cache_name", "training_cache_ijepa_planning_agent_v3_v5"
        )
        use_legacy_cache_keys = str(config.get("use_legacy_cache_keys", True)).lower()
        cache_version = config.get("cache_version", "v3_224")

        # Trainer params
        precision = config.get("precision", "16-mixed")
        gradient_clip_val = config.get("gradient_clip_val", 1.0)
        accumulate_grad_batches = config.get("accumulate_grad_batches", 1)
        prefetch_factor = config.get("prefetch_factor", 4)

        # Train test split
        train_split = config.get("train_split", "navtrain")
        city = config.get("city")
        city_arg_str = f" +city={city}" if city else ""

        # Optional I-JEPA checkpoint
        ijepa_ckpt_path = config.get("ijepa_ckpt_path", "")
        ijepa_ckpt_which = config.get("ijepa_ckpt_which", "encoder")

        # Resume checkpoint path (for resuming interrupted training)
        resume_checkpoint_path = config.get("resume_checkpoint_path", "")

        # Format lists for shell
        vision_views_str = str(camera_views).replace("'", "").replace(" ", "")
        image_size_str = str(image_size).replace(" ", "")

        experiment_name = f"training/{exp_id}_{timestamp}"

        script = f"""#!/bin/bash
# =============================================================================
# Auto-generated by NAVSIM ExpFlow Manager
# Experiment: {exp_id}
# Description: {config.get("description", "")}
# Generated: {datetime.now().isoformat()}
# Git: {config.get("git_commit", "unknown")} ({config.get("git_branch", "unknown")})
# =============================================================================

# =============================================================================
# SLURM CONFIGURATION
# =============================================================================
#SBATCH --job-name={exp_id}_train
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --output={self.logs_dir}/output/{exp_id}_train_%j.out
#SBATCH --error={self.logs_dir}/error/{exp_id}_train_%j.err

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
echo "=============================================="
echo "ExpFlow Experiment: {exp_id}"
echo "{config.get('description', '')}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/{self.username}/navsim-ssl-city-generalization/navsim"
export OPENSCENE_DATA_ROOT="/scratch/{self.username}/data"
export NUPLAN_MAPS_ROOT="/scratch/{self.username}/data/maps"
export NAVSIM_EXP_ROOT="/scratch/{self.username}/experiments"
export DP_PREDS="none"

# Optional: local I-JEPA pretraining checkpoint
export IJEPA_CKPT_PATH="{ijepa_ckpt_path}"
export IJEPA_CKPT_WHICH="{ijepa_ckpt_which}"

# Resume checkpoint path (for resuming interrupted training)
export RESUME_CKPT_PATH="{resume_checkpoint_path}"

# SquashFS overlay for training cache
export CACHE_NAME="{cache_name}"
export CACHE_PATH="${{NAVSIM_EXP_ROOT}}/cache/${{CACHE_NAME}}"
export CACHE_OVERLAY="/scratch/{self.username}/experiments/cache/overlays/${{CACHE_NAME}}.sqsh"

# Container configuration
export CONTAINER="/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

# Threading
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# Training hyperparameters
export AGENT="{agent}"
export BACKBONE="{backbone}"
export BATCH_SIZE={batch_size}
export NUM_WORKERS={num_workers}
export EPOCHS={epochs}
export LEARNING_RATE={learning_rate}
export ENCODER_LEARNING_RATE={encoder_learning_rate}
export PC_TRAIN_FRACTION={trainable_fraction}
export VISION_MODE="{vision_mode}"
export VISION_VIEWS="{vision_views_str}"
export VISION_IMAGE_SIZE="{image_size_str}"
export USE_LEGACY_CACHE_KEYS={use_legacy_cache_keys}
export CACHE_VERSION="{cache_version}"
export EXPERIMENT_NAME="{experiment_name}"

# NCCL optimizations
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=2
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Multi-node DDP
export MASTER_PORT=12360
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo ""
echo "Configuration:"
echo "  Agent: $AGENT"
echo "  Backbone: $BACKBONE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Encoder LR: $ENCODER_LEARNING_RATE"
echo "  Trainable fraction: $PC_TRAIN_FRACTION"
echo "  Vision mode: $VISION_MODE"
echo "  Vision views: $VISION_VIEWS"
echo "  Cache: $CACHE_NAME"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Load conda
CONDA_ROOT="/scratch/{self.username}/miniconda3"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate navsim
else
    module purge || true
    module load anaconda3/2025.06 || true
    source $(conda info --base)/etc/profile.d/conda.sh || true
    conda activate navsim || true
fi

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Check overlay
if [ -f "${{CACHE_OVERLAY}}" ]; then
    echo "✓ Using SquashFS overlay for training cache"
    USE_OVERLAY=true
else
    echo "⚠ SquashFS overlay not found at ${{CACHE_OVERLAY}}"
    echo "  Falling back to regular directory: ${{CACHE_PATH}}"
    USE_OVERLAY=false
fi
echo ""

# GPU monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > {self.logs_dir}/output/{exp_id}_gpu_${{SLURM_JOB_ID}}.csv &
GPU_MONITOR_PID=$!

# =============================================================================
# TRAINING
# =============================================================================
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/train_{exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'TRAIN_SCRIPT_EOF'
#!/bin/bash
source /scratch/{self.username}/miniconda3/etc/profile.d/conda.sh
conda activate navsim
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

# Build extra args for optional I-JEPA checkpoint
EXTRA_AGENT_ARGS=""
if [ -n "${{IJEPA_CKPT_PATH:-}}" ]; then
    EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_path=${{IJEPA_CKPT_PATH}}"
    EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_which=${{IJEPA_CKPT_WHICH:-encoder}}"
fi

# Build resume args if resuming from checkpoint
RESUME_ARGS=""
if [ -n "${{RESUME_CKPT_PATH:-}}" ]; then
    RESUME_ARGS=" +ckpt_path='${{RESUME_CKPT_PATH}}'"
    echo "Resuming from checkpoint: ${{RESUME_CKPT_PATH}}"
fi

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_training.py \\
    agent=${{AGENT}} \\
    agent.learning_rate=${{LEARNING_RATE}} \\
    agent.encoder_learning_rate=${{ENCODER_LEARNING_RATE}} \\
    agent.trainable_ijepa_layers_fraction=${{PC_TRAIN_FRACTION}} \\
    agent.vision_mode=${{VISION_MODE}} \\
    agent.vision_views=${{VISION_VIEWS}} \\
    agent.vision_image_size=${{VISION_IMAGE_SIZE}} \\
    agent.view_fusion_method={view_fusion_method} \\
    agent.cross_attn_heads={cross_attn_heads} \\
    agent.cross_attn_layers={cross_attn_layers} \\
    agent.planning_head_type={planning_head_type} \\
    agent.transformer_head_hidden_dim={transformer_head_hidden_dim} \\
    agent.transformer_head_num_heads={transformer_head_num_heads} \\
    agent.transformer_head_num_layers={transformer_head_num_layers} \\
    agent.transformer_head_dropout={transformer_head_dropout} \\
    agent.transformer_head_lr_multiplier={transformer_head_lr_multiplier} \\
    agent.use_legacy_cache_keys=${{USE_LEGACY_CACHE_KEYS}} \\
    agent.cache_version=${{CACHE_VERSION}} \\
    experiment_name=${{EXPERIMENT_NAME}} \\
    train_test_split={train_split} \\
    cache_path=${{CACHE_PATH}} \\
    use_cache_without_dataset=true \\
    force_cache_computation=false \\
    trainer.params.max_epochs=${{EPOCHS}} \\
    trainer.params.precision={precision} \\
    trainer.params.accelerator=gpu \\
    trainer.params.strategy=ddp \\
    trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
    trainer.params.gradient_clip_val={gradient_clip_val} \\
    trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
    dataloader.params.batch_size=${{BATCH_SIZE}} \\
    dataloader.params.num_workers=${{NUM_WORKERS}} \\
    dataloader.params.prefetch_factor={prefetch_factor} \\
    dataloader.params.pin_memory=true \\
    ${{SMOKE:+trainer.params.limit_train_batches=1}} \\
    ${{SMOKE:+trainer.params.limit_val_batches=0}} \\
    ${{EXTRA_AGENT_ARGS}} \\
    ${{RESUME_ARGS}} \\
    {city_arg_str}
TRAIN_SCRIPT_EOF
    chmod +x "${{TEMP_SCRIPT}}"
    
    # Run inside apptainer with squashfs overlay
    srun --gres=gpu:{num_gpus} apptainer exec \\
        --nv \\
        --bind "${{CACHE_OVERLAY}}:${{CACHE_PATH}}:image-src=/" \\
        --bind /scratch/{self.username}:/scratch/{self.username} \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "NAVSIM_DEVKIT_ROOT=${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "OPENSCENE_DATA_ROOT=${{OPENSCENE_DATA_ROOT}}" \\
        --env "NUPLAN_MAPS_ROOT=${{NUPLAN_MAPS_ROOT}}" \\
        --env "NAVSIM_EXP_ROOT=${{NAVSIM_EXP_ROOT}}" \\
        --env "CACHE_PATH=${{CACHE_PATH}}" \\
        --env "EXPERIMENT_NAME=${{EXPERIMENT_NAME}}" \\
        --env "AGENT=${{AGENT}}" \\
        --env "PC_TRAIN_FRACTION=${{PC_TRAIN_FRACTION}}" \\
        --env "LEARNING_RATE=${{LEARNING_RATE}}" \\
        --env "ENCODER_LEARNING_RATE=${{ENCODER_LEARNING_RATE}}" \\
        --env "BATCH_SIZE=${{BATCH_SIZE}}" \\
        --env "NUM_WORKERS=${{NUM_WORKERS}}" \\
        --env "EPOCHS=${{EPOCHS}}" \\
        --env "VISION_MODE=${{VISION_MODE}}" \\
        --env "VISION_VIEWS=${{VISION_VIEWS}}" \\
        --env "VISION_IMAGE_SIZE=${{VISION_IMAGE_SIZE}}" \\
        --env "USE_LEGACY_CACHE_KEYS=${{USE_LEGACY_CACHE_KEYS}}" \\
        --env "CACHE_VERSION=${{CACHE_VERSION}}" \\
        --env "SLURM_JOB_NUM_NODES=${{SLURM_JOB_NUM_NODES}}" \\
        --env "MASTER_ADDR=${{MASTER_ADDR}}" \\
        --env "MASTER_PORT=${{MASTER_PORT}}" \\
        --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
        --env "NCCL_P2P_LEVEL=${{NCCL_P2P_LEVEL}}" \\
        --env "NCCL_NET_GDR_LEVEL=${{NCCL_NET_GDR_LEVEL}}" \\
        --env "IJEPA_CKPT_PATH=${{IJEPA_CKPT_PATH}}" \\
        --env "IJEPA_CKPT_WHICH=${{IJEPA_CKPT_WHICH}}" \\
        --env "RESUME_CKPT_PATH=${{RESUME_CKPT_PATH}}" \\
        "${{CONTAINER}}" \\
        bash "${{TEMP_SCRIPT}}"
    
    rm -f "${{TEMP_SCRIPT}}"
else
    # Run without overlay (fallback)
    EXTRA_AGENT_ARGS=""
    if [ -n "${{IJEPA_CKPT_PATH:-}}" ]; then
        EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_path=${{IJEPA_CKPT_PATH}}"
        EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_which=${{IJEPA_CKPT_WHICH:-encoder}}"
    fi
    
    # Build resume args if resuming from checkpoint
    RESUME_ARGS=""
    if [ -n "${{RESUME_CKPT_PATH:-}}" ]; then
        RESUME_ARGS=" +ckpt_path='${{RESUME_CKPT_PATH}}'"
        echo "Resuming from checkpoint: ${{RESUME_CKPT_PATH}}"
    fi
    
    srun --gres=gpu:{num_gpus} python navsim/planning/script/run_training.py \\
        agent=${{AGENT}} \\
        agent.learning_rate=${{LEARNING_RATE}} \\
        agent.encoder_learning_rate=${{ENCODER_LEARNING_RATE}} \\
        agent.trainable_ijepa_layers_fraction=${{PC_TRAIN_FRACTION}} \\
        agent.vision_mode=${{VISION_MODE}} \\
        agent.vision_views=${{VISION_VIEWS}} \\
        agent.vision_image_size=${{VISION_IMAGE_SIZE}} \\
        agent.view_fusion_method={view_fusion_method} \\
        agent.cross_attn_heads={cross_attn_heads} \\
        agent.cross_attn_layers={cross_attn_layers} \\
        agent.planning_head_type={planning_head_type} \\
        agent.transformer_head_hidden_dim={transformer_head_hidden_dim} \\
        agent.transformer_head_num_heads={transformer_head_num_heads} \\
        agent.transformer_head_num_layers={transformer_head_num_layers} \\
        agent.transformer_head_dropout={transformer_head_dropout} \\
        agent.transformer_head_lr_multiplier={transformer_head_lr_multiplier} \\
        agent.use_legacy_cache_keys=${{USE_LEGACY_CACHE_KEYS}} \\
        agent.cache_version=${{CACHE_VERSION}} \\
        experiment_name="${{EXPERIMENT_NAME}}" \\
        train_test_split={train_split} \\
        cache_path="${{CACHE_PATH}}" \\
        use_cache_without_dataset=true \\
        force_cache_computation=false \\
        trainer.params.max_epochs=${{EPOCHS}} \\
        trainer.params.precision={precision} \\
        trainer.params.accelerator=gpu \\
        trainer.params.strategy=ddp \\
        trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
        trainer.params.gradient_clip_val={gradient_clip_val} \\
        trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
        dataloader.params.batch_size=${{BATCH_SIZE}} \\
        dataloader.params.num_workers=${{NUM_WORKERS}} \\
        dataloader.params.prefetch_factor={prefetch_factor} \\
        dataloader.params.pin_memory=true \\
        ${{SMOKE:+trainer.params.limit_train_batches=1}} \\
        ${{SMOKE:+trainer.params.limit_val_batches=0}} \\
        ${{EXTRA_AGENT_ARGS}} \\
        ${{RESUME_ARGS}} \\
        {city_arg_str}
fi

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}"
echo "=============================================="

# Find and register checkpoint
CHECKPOINT_DIR=$(find "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}" -type d -name "checkpoints" 2>/dev/null | head -1)
if [ -n "$CHECKPOINT_DIR" ]; then
    BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/epoch*.ckpt 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/*.ckpt 2>/dev/null | head -1)
    fi
    
    if [ -n "$BEST_CKPT" ]; then
        echo "Found checkpoint: ${{BEST_CKPT}}"
        
        # Save checkpoint path to registry
        CKPT_REGISTRY="${{NAVSIM_EXP_ROOT}}/checkpoints"
        mkdir -p "${{CKPT_REGISTRY}}"
        echo "${{BEST_CKPT}}" > "${{CKPT_REGISTRY}}/{exp_id}.txt"
        echo "Checkpoint registered: ${{CKPT_REGISTRY}}/{exp_id}.txt"
        
        # Also save in experiment folder
        echo "${{BEST_CKPT}}" > "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}/checkpoint_path.txt"
    else
        echo "WARNING: No checkpoint found in $CHECKPOINT_DIR"
    fi
else
    echo "WARNING: Checkpoint directory not found"
fi

echo ""
echo "GPU utilization log: {self.logs_dir}/output/{exp_id}_gpu_${{SLURM_JOB_ID}}.csv"
echo "=============================================="

exit $TRAIN_EXIT_CODE
"""
        return script

    def _generate_ego_mlp_train_script(
        self, config: Dict, exp_id: str, timestamp: str
    ) -> str:
        """
        Generate training script for ego_status_mlp_agent.
        This agent uses only ego state (velocity, acceleration, driving_command).
        No vision, no LiDAR, no cache required.
        """
        partition = config.get("partition", "l40s_public")
        account = config.get("account", "torch_pr_68_general")
        num_gpus = config.get("num_gpus", 1)  # Light GPU usage
        num_nodes = config.get("num_nodes", 1)
        cpus_per_task = config.get("cpus_per_task", 8)
        time_limit = config.get("time_limit", "4:00:00")

        # Training hyperparameters
        batch_size = config.get("batch_size", 128)
        learning_rate = config.get("learning_rate", 1e-4)
        epochs = config.get("epochs", 50)
        num_workers = config.get("num_workers", 8)

        # Model hyperparameters
        hidden_layer_dim = config.get("hidden_layer_dim", 512)

        # Train test split
        train_split = config.get("train_split", "navtrain")

        experiment_name = f"training/{exp_id}_{timestamp}"

        script = f"""#!/bin/bash
# =============================================================================
# Auto-generated by NAVSIM ExpFlow Manager
# Experiment: {exp_id} (Ego-Status MLP Agent)
# Description: {config.get("description", "Ego-Status MLP baseline")}
# Generated: {datetime.now().isoformat()}
# Git: {config.get("git_commit", "unknown")} ({config.get("git_branch", "unknown")})
# =============================================================================

# =============================================================================
# SLURM CONFIGURATION
# =============================================================================
#SBATCH --job-name={exp_id}_train
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --output={self.logs_dir}/output/{exp_id}_train_%j.out
#SBATCH --error={self.logs_dir}/error/{exp_id}_train_%j.err

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
echo "=============================================="
echo "ExpFlow Experiment: {exp_id}"
echo "{config.get('description', 'Ego-Status MLP baseline')}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/{self.username}/navsim-ssl-city-generalization/navsim"
export OPENSCENE_DATA_ROOT="/scratch/{self.username}/data"
export NUPLAN_MAPS_ROOT="/scratch/{self.username}/data/maps"
export NAVSIM_EXP_ROOT="/scratch/{self.username}/experiments"
export DP_PREDS="none"

# Training hyperparameters
export AGENT="ego_status_mlp_agent"
export BATCH_SIZE={batch_size}
export NUM_WORKERS={num_workers}
export EPOCHS={epochs}
export LEARNING_RATE={learning_rate}
export HIDDEN_LAYER_DIM={hidden_layer_dim}
export EXPERIMENT_NAME="{experiment_name}"

echo ""
echo "Configuration:"
echo "  Agent: $AGENT"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Hidden layer dim: $HIDDEN_LAYER_DIM"
echo ""

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Load conda
CONDA_ROOT="/scratch/{self.username}/miniconda3"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate navsim
else
    module purge || true
    module load anaconda3/2025.06 || true
    source $(conda info --base)/etc/profile.d/conda.sh || true
    conda activate navsim || true
fi

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# GPU monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > {self.logs_dir}/output/{exp_id}_gpu_${{SLURM_JOB_ID}}.csv &
GPU_MONITOR_PID=$!

# =============================================================================
# TRAINING - Ego Status MLP Agent (no cache, no vision)
# =============================================================================
srun --gres=gpu:{num_gpus} python navsim/planning/script/run_training.py \\
    agent=ego_status_mlp_agent \\
    agent.lr=${{LEARNING_RATE}} \\
    agent.hidden_layer_dim=${{HIDDEN_LAYER_DIM}} \\
    experiment_name="${{EXPERIMENT_NAME}}" \\
    train_test_split={train_split} \\
    trainer.params.max_epochs=${{EPOCHS}} \\
    trainer.params.precision=16-mixed \\
    trainer.params.accelerator=gpu \\
    trainer.params.strategy=auto \\
    dataloader.params.batch_size=${{BATCH_SIZE}} \\
    dataloader.params.num_workers=${{NUM_WORKERS}} \\
    dataloader.params.prefetch_factor=2 \\
    dataloader.params.pin_memory=true \\
    ${{SMOKE:+trainer.params.limit_train_batches=1}} \\
    ${{SMOKE:+trainer.params.limit_val_batches=0}}

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}"
echo "=============================================="

# Find and register checkpoint
CHECKPOINT_DIR=$(find "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}" -type d -name "checkpoints" 2>/dev/null | head -1)
if [ -n "$CHECKPOINT_DIR" ]; then
    BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/epoch*.ckpt 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/*.ckpt 2>/dev/null | head -1)
    fi
    
    if [ -n "$BEST_CKPT" ]; then
        echo "Found checkpoint: ${{BEST_CKPT}}"
        
        # Save checkpoint path to registry
        CKPT_REGISTRY="${{NAVSIM_EXP_ROOT}}/checkpoints"
        mkdir -p "${{CKPT_REGISTRY}}"
        echo "${{BEST_CKPT}}" > "${{CKPT_REGISTRY}}/{exp_id}.txt"
        echo "Checkpoint registered: ${{CKPT_REGISTRY}}/{exp_id}.txt"
        
        # Also save in experiment folder
        echo "${{BEST_CKPT}}" > "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}/checkpoint_path.txt"
    else
        echo "WARNING: No checkpoint found in $CHECKPOINT_DIR"
    fi
else
    echo "WARNING: Checkpoint directory not found"
fi

echo ""
echo "GPU utilization log: {self.logs_dir}/output/{exp_id}_gpu_${{SLURM_JOB_ID}}.csv"
echo "=============================================="

exit $TRAIN_EXIT_CODE
"""
        return script

    def _generate_transfuser_train_script(
        self, config: Dict, exp_id: str, timestamp: str
    ) -> str:
        """
        Generate training script for transfuser_agent.
        TransFuser uses vision + LiDAR and needs its own cache.
        """
        partition = config.get("partition", "rtx8000_public")
        account = config.get("account", "torch_pr_68_general")
        num_gpus = config.get("num_gpus", 4)
        num_nodes = config.get("num_nodes", 1)
        cpus_per_task = config.get("cpus_per_task", 12)
        time_limit = config.get("time_limit", "24:00:00")

        # Training hyperparameters (paper-matched defaults)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 2e-4)  # Paper: 2e-4
        epochs = config.get("epochs", 31)  # Paper: 31
        num_workers = config.get("num_workers", 8)

        # Scheduler (MultiStep with milestones)
        lr_milestones = config.get("lr_milestones", [25, 30])
        lr_gamma = config.get("lr_gamma", 0.1)

        # Cache configuration
        cache_name = config.get("cache_name", "training_cache_transfuser")

        # Train test split
        train_split = config.get("train_split", "navtrain")

        experiment_name = f"training/{exp_id}_{timestamp}"

        # Format milestones for shell
        milestones_str = str(lr_milestones).replace(" ", "")

        script = f"""#!/bin/bash
# =============================================================================
# Auto-generated by NAVSIM ExpFlow Manager
# Experiment: {exp_id} (TransFuser Agent)
# Description: {config.get("description", "TransFuser baseline")}
# Generated: {datetime.now().isoformat()}
# Git: {config.get("git_commit", "unknown")} ({config.get("git_branch", "unknown")})
# =============================================================================

# =============================================================================
# SLURM CONFIGURATION
# =============================================================================
#SBATCH --job-name={exp_id}_train
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --output={self.logs_dir}/output/{exp_id}_train_%j.out
#SBATCH --error={self.logs_dir}/error/{exp_id}_train_%j.err

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
echo "=============================================="
echo "ExpFlow Experiment: {exp_id}"
echo "{config.get('description', 'TransFuser baseline')}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/{self.username}/navsim-ssl-city-generalization/navsim"
export OPENSCENE_DATA_ROOT="/scratch/{self.username}/data"
export NUPLAN_MAPS_ROOT="/scratch/{self.username}/data/maps"
export NAVSIM_EXP_ROOT="/scratch/{self.username}/experiments"
export DP_PREDS="none"

# SquashFS overlay for training cache
export CACHE_NAME="{cache_name}"
export CACHE_PATH="${{NAVSIM_EXP_ROOT}}/cache/${{CACHE_NAME}}"
export CACHE_OVERLAY="/scratch/{self.username}/experiments/cache/overlays/${{CACHE_NAME}}.sqsh"

# Container configuration
export CONTAINER="/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

# Training hyperparameters
export AGENT="transfuser_agent"
export BATCH_SIZE={batch_size}
export NUM_WORKERS={num_workers}
export EPOCHS={epochs}
export LEARNING_RATE={learning_rate}
export LR_MILESTONES="{milestones_str}"
export LR_GAMMA={lr_gamma}
export EXPERIMENT_NAME="{experiment_name}"

# NCCL optimizations
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=2
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Multi-node DDP
export MASTER_PORT=12360
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo ""
echo "Configuration:"
echo "  Agent: $AGENT"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  LR milestones: $LR_MILESTONES"
echo "  LR gamma: $LR_GAMMA"
echo "  Cache: $CACHE_NAME"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Load conda
CONDA_ROOT="/scratch/{self.username}/miniconda3"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate navsim
else
    module purge || true
    module load anaconda3/2025.06 || true
    source $(conda info --base)/etc/profile.d/conda.sh || true
    conda activate navsim || true
fi

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Check overlay
if [ -f "${{CACHE_OVERLAY}}" ]; then
    echo "✓ Using SquashFS overlay for training cache"
    USE_OVERLAY=true
else
    echo "⚠ SquashFS overlay not found at ${{CACHE_OVERLAY}}"
    echo "  Falling back to regular directory: ${{CACHE_PATH}}"
    USE_OVERLAY=false
fi
echo ""

# GPU monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > {self.logs_dir}/output/{exp_id}_gpu_${{SLURM_JOB_ID}}.csv &
GPU_MONITOR_PID=$!

# =============================================================================
# TRAINING - TransFuser Agent
# =============================================================================
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/train_{exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'TRAIN_SCRIPT_EOF'
#!/bin/bash
source /scratch/{self.username}/miniconda3/etc/profile.d/conda.sh
conda activate navsim
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_training.py \\
    agent=transfuser_agent \\
    agent.lr=${{LEARNING_RATE}} \\
    agent.config.lr_milestones=${{LR_MILESTONES}} \\
    agent.config.lr_gamma=${{LR_GAMMA}} \\
    experiment_name=${{EXPERIMENT_NAME}} \\
    train_test_split={train_split} \\
    cache_path=${{CACHE_PATH}} \\
    use_cache_without_dataset=true \\
    force_cache_computation=false \\
    trainer.params.max_epochs=${{EPOCHS}} \\
    trainer.params.precision=16-mixed \\
    trainer.params.accelerator=gpu \\
    trainer.params.strategy=ddp \\
    trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
    trainer.params.gradient_clip_val=1.0 \\
    dataloader.params.batch_size=${{BATCH_SIZE}} \\
    dataloader.params.num_workers=${{NUM_WORKERS}} \\
    dataloader.params.prefetch_factor=2 \\
    dataloader.params.pin_memory=true \\
    ${{SMOKE:+trainer.params.limit_train_batches=1}} \\
    ${{SMOKE:+trainer.params.limit_val_batches=0}}
TRAIN_SCRIPT_EOF
    chmod +x "${{TEMP_SCRIPT}}"
    
    # Run inside apptainer with squashfs overlay
    srun --gres=gpu:{num_gpus} apptainer exec \\
        --nv \\
        --bind "${{CACHE_OVERLAY}}:${{CACHE_PATH}}:image-src=/" \\
        --bind /scratch/{self.username}:/scratch/{self.username} \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "NAVSIM_DEVKIT_ROOT=${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "OPENSCENE_DATA_ROOT=${{OPENSCENE_DATA_ROOT}}" \\
        --env "NUPLAN_MAPS_ROOT=${{NUPLAN_MAPS_ROOT}}" \\
        --env "NAVSIM_EXP_ROOT=${{NAVSIM_EXP_ROOT}}" \\
        --env "CACHE_PATH=${{CACHE_PATH}}" \\
        --env "EXPERIMENT_NAME=${{EXPERIMENT_NAME}}" \\
        --env "LEARNING_RATE=${{LEARNING_RATE}}" \\
        --env "LR_MILESTONES=${{LR_MILESTONES}}" \\
        --env "LR_GAMMA=${{LR_GAMMA}}" \\
        --env "BATCH_SIZE=${{BATCH_SIZE}}" \\
        --env "NUM_WORKERS=${{NUM_WORKERS}}" \\
        --env "EPOCHS=${{EPOCHS}}" \\
        --env "SLURM_JOB_NUM_NODES=${{SLURM_JOB_NUM_NODES}}" \\
        --env "MASTER_ADDR=${{MASTER_ADDR}}" \\
        --env "MASTER_PORT=${{MASTER_PORT}}" \\
        --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
        --env "NCCL_P2P_LEVEL=${{NCCL_P2P_LEVEL}}" \\
        --env "NCCL_NET_GDR_LEVEL=${{NCCL_NET_GDR_LEVEL}}" \\
        "${{CONTAINER}}" \\
        bash "${{TEMP_SCRIPT}}"
    
    rm -f "${{TEMP_SCRIPT}}"
else
    # Run without overlay (fallback)
    srun --gres=gpu:{num_gpus} python navsim/planning/script/run_training.py \\
        agent=transfuser_agent \\
        agent.lr=${{LEARNING_RATE}} \\
        agent.config.lr_milestones=${{LR_MILESTONES}} \\
        agent.config.lr_gamma=${{LR_GAMMA}} \\
        experiment_name="${{EXPERIMENT_NAME}}" \\
        train_test_split={train_split} \\
        cache_path="${{CACHE_PATH}}" \\
        use_cache_without_dataset=true \\
        force_cache_computation=false \\
        trainer.params.max_epochs=${{EPOCHS}} \\
        trainer.params.precision=16-mixed \\
        trainer.params.accelerator=gpu \\
        trainer.params.strategy=ddp \\
        trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
        trainer.params.gradient_clip_val=1.0 \\
        dataloader.params.batch_size=${{BATCH_SIZE}} \\
        dataloader.params.num_workers=${{NUM_WORKERS}} \\
        dataloader.params.prefetch_factor=2 \\
        dataloader.params.pin_memory=true \\
        ${{SMOKE:+trainer.params.limit_train_batches=1}} \\
        ${{SMOKE:+trainer.params.limit_val_batches=0}}
fi

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}"
echo "=============================================="

# Find and register checkpoint
CHECKPOINT_DIR=$(find "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}" -type d -name "checkpoints" 2>/dev/null | head -1)
if [ -n "$CHECKPOINT_DIR" ]; then
    BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/epoch*.ckpt 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/*.ckpt 2>/dev/null | head -1)
    fi
    
    if [ -n "$BEST_CKPT" ]; then
        echo "Found checkpoint: ${{BEST_CKPT}}"
        
        # Save checkpoint path to registry
        CKPT_REGISTRY="${{NAVSIM_EXP_ROOT}}/checkpoints"
        mkdir -p "${{CKPT_REGISTRY}}"
        echo "${{BEST_CKPT}}" > "${{CKPT_REGISTRY}}/{exp_id}.txt"
        echo "Checkpoint registered: ${{CKPT_REGISTRY}}/{exp_id}.txt"
        
        # Also save in experiment folder
        echo "${{BEST_CKPT}}" > "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}/checkpoint_path.txt"
    else
        echo "WARNING: No checkpoint found in $CHECKPOINT_DIR"
    fi
else
    echo "WARNING: Checkpoint directory not found"
fi

echo ""
echo "GPU utilization log: {self.logs_dir}/output/{exp_id}_gpu_${{SLURM_JOB_ID}}.csv"
echo "=============================================="

exit $TRAIN_EXIT_CODE
"""
        return script

    # -------------------------------------------------------------------------
    # Evaluation Script Generation
    # -------------------------------------------------------------------------

    def _generate_eval_script(
        self, config: Dict, eval_type: str = "one_stage", city: str = None
    ) -> str:
        """
        Generate evaluation script with full NAVSIM-native parameters.

        Args:
            config: Experiment configuration
            eval_type: "one_stage" (default, non-reactive) or "two_stage" (reactive)
            city: Optional city filter for cross-city evaluation

        Supports both:
        - Trained agents (require checkpoint)
        - Rule-based agents (no checkpoint, e.g., constant_velocity_agent)
        """
        exp_id = config["exp_id"]
        agent = config.get("agent", "ijepa_planning_agent_v4")
        account = config.get("account", "torch_pr_68_general")
        eval_split = config.get("eval_split", "navtest")
        eval_workers = config.get("eval_workers", 48)
        eval_mem = config.get("eval_mem", "400GB")
        eval_time = config.get("eval_time", "06:00:00")
        use_multi_camera = str(config.get("use_multi_camera", True)).lower()
        requires_training = config.get("requires_training", True)

        # City-specific evaluation
        # Use navtest split with +city=X filter (city configs filter by log_names)
        if city:
            eval_split_actual = (
                eval_split  # Always use navtest, filtering happens via city config
            )
            city_suffix = f"_{city}"
            city_arg = f"+city={city}"
        else:
            eval_split_actual = eval_split
            city_suffix = ""
            city_arg = ""

        # Eval-specific settings
        traffic_agents = "non_reactive" if eval_type == "one_stage" else "reactive"
        eval_script_name = (
            "run_pdm_score_one_stage.py"
            if eval_type == "one_stage"
            else "run_pdm_score.py"
        )

        # For GPU-based eval (two_stage), we need GPUs
        if eval_type == "two_stage":
            partition = config.get("partition", "l40s_public")
            num_gpus = config.get("num_gpus", 4)
            gpu_line = f"#SBATCH --gres=gpu:{num_gpus}"
            partition_line = f"#SBATCH --partition={partition}"
            worker_type = "ray_distributed"
        else:
            # one_stage is CPU-only
            gpu_line = "# No GPU needed for one-stage (CPU-only)"
            partition_line = "#SBATCH --partition=cs"
            worker_type = "ray_distributed"

        # Checkpoint handling for rule-based vs trained agents
        if requires_training:
            checkpoint_block = f"""
# Checkpoint discovery (trained agent)
CKPT_REGISTRY="/scratch/{self.username}/experiments/checkpoints"

if [ -n "${{CHECKPOINT:-}}" ]; then
    echo "Using checkpoint from env: ${{CHECKPOINT}}"
elif [ -f "${{CKPT_REGISTRY}}/{exp_id}.txt" ]; then
    export CHECKPOINT=$(cat "${{CKPT_REGISTRY}}/{exp_id}.txt")
    echo "Using registered checkpoint: ${{CHECKPOINT}}"
else
    echo "ERROR: No checkpoint found for {exp_id}"
    exit 1
fi

# Validate checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
CHECKPOINT_ARG="agent.checkpoint_path='${{CHECKPOINT}}'"
"""
        else:
            checkpoint_block = f"""
# Rule-based agent - no checkpoint needed
echo "Rule-based agent: {agent} (no checkpoint required)"
CHECKPOINT_ARG=""
"""

        script = f"""#!/bin/bash
# =============================================================================
# Auto-generated Evaluation Script
# Experiment: {exp_id}{city_suffix}
# Type: {eval_type} PDM Score
# Agent: {agent} (requires_training={requires_training})
# City: {city if city else "all"}
# Generated: {datetime.now().isoformat()}
# =============================================================================

#SBATCH --job-name={exp_id}{city_suffix}_eval
#SBATCH --account={account}
{partition_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={eval_workers}
{gpu_line}
#SBATCH --mem={eval_mem}
#SBATCH --time={eval_time}
#SBATCH --output={self.logs_dir}/output/{exp_id}{city_suffix}_eval_%j.out
#SBATCH --error={self.logs_dir}/error/{exp_id}{city_suffix}_eval_%j.err
#SBATCH --requeue

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
export AGENT="{agent}"
export EVAL_SPLIT="{eval_split_actual}"
export MULTI_CAM={use_multi_camera}
export NUM_WORKERS={eval_workers}
export CITY_ARG="{city_arg}"

{checkpoint_block}

# Derive run name
export RUN_NAME="eval_{eval_split_actual}_{exp_id}_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# ENVIRONMENT
# =============================================================================
export NAVSIM_DEVKIT_ROOT="/scratch/{self.username}/navsim-ssl-city-generalization/navsim"
export OPENSCENE_DATA_ROOT="/scratch/{self.username}/data"
export NUPLAN_MAPS_ROOT="/scratch/{self.username}/data/maps"
export NAVSIM_EXP_ROOT="/scratch/{self.username}/navsim-ssl-city-generalization/experiments"
export OUTPUT_DIR="${{NAVSIM_EXP_ROOT}}/evaluations/${{RUN_NAME}}"

# Metric cache - use old path structure to match overlay contents
# The overlay was created with files at /scratch/ah7072/experiments/cache/navtest_metric_cache
export METRIC_CACHE="/scratch/{self.username}/experiments/cache/navtest_metric_cache"
export METRIC_CACHE_OVERLAY="/scratch/{self.username}/navsim-ssl-city-generalization/experiments/cache/overlays/navtest_metric_cache.sqsh"
export CONTAINER="/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

mkdir -p "${{OUTPUT_DIR}}"

echo "=============================================="
echo "PDM Score Evaluation ({eval_type})"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""
echo "Configuration:"
echo "  Agent: ${{AGENT}}"
echo "  Split: ${{EVAL_SPLIT}}"
echo "  City filter: ${{CITY_ARG}}"
echo "  Traffic: {traffic_agents}"
echo "  Workers: ${{NUM_WORKERS}}"
echo "  Output: ${{OUTPUT_DIR}}"
echo "=============================================="
echo ""

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Load conda (for non-overlay case and setup)
source /scratch/{self.username}/miniconda3/etc/profile.d/conda.sh
conda activate navsim

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"
export HYDRA_FULL_ERROR=1
export RAY_NUM_CPUS=${{NUM_WORKERS}}
export OMP_NUM_THREADS=1

# Check for SquashFS overlay
if [ -f "${{METRIC_CACHE_OVERLAY}}" ]; then
    echo "✓ Using SquashFS overlay: ${{METRIC_CACHE_OVERLAY}}"
    USE_OVERLAY=true
else
    echo "ℹ No overlay found, using directory cache"
    USE_OVERLAY=false
fi
echo ""

START_TIME=$(date +%s)

echo "Starting evaluation..."
echo ""

# =============================================================================
# RUN EVALUATION
# =============================================================================
if [ "$USE_OVERLAY" = true ]; then
    # Create temp script for apptainer execution
    TEMP_SCRIPT=$(mktemp /tmp/eval_{exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'EVAL_SCRIPT_EOF'
#!/bin/bash
source /scratch/{self.username}/miniconda3/etc/profile.d/conda.sh
conda activate navsim
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/{eval_script_name} \\
    train_test_split=${{EVAL_SPLIT}} \\
    experiment_name=${{RUN_NAME}} \\
    traffic_agents={traffic_agents} \\
    metric_cache_path=${{METRIC_CACHE}} \\
    output_dir=${{OUTPUT_DIR}} \\
    agent=${{AGENT}} \\
    ${{CHECKPOINT_ARG}} \\
    ${{CITY_ARG}} \\
    worker={worker_type} \\
    worker.threads_per_node=${{NUM_WORKERS}}
EVAL_SCRIPT_EOF
    chmod +x "${{TEMP_SCRIPT}}"
    
    apptainer exec \\
        --bind "${{METRIC_CACHE_OVERLAY}}:${{METRIC_CACHE}}:image-src=/" \\
        --bind /scratch/{self.username}:/scratch/{self.username} \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "NAVSIM_DEVKIT_ROOT=${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "OPENSCENE_DATA_ROOT=${{OPENSCENE_DATA_ROOT}}" \\
        --env "NUPLAN_MAPS_ROOT=${{NUPLAN_MAPS_ROOT}}" \\
        --env "NAVSIM_EXP_ROOT=${{NAVSIM_EXP_ROOT}}" \\
        --env "EVAL_SPLIT=${{EVAL_SPLIT}}" \\
        --env "RUN_NAME=${{RUN_NAME}}" \\
        --env "METRIC_CACHE=${{METRIC_CACHE}}" \\
        --env "OUTPUT_DIR=${{OUTPUT_DIR}}" \\
        --env "AGENT=${{AGENT}}" \\
        --env "NUM_WORKERS=${{NUM_WORKERS}}" \\
        --env "CHECKPOINT_ARG=${{CHECKPOINT_ARG}}" \\
        --env "CITY_ARG=${{CITY_ARG}}" \\
        "${{CONTAINER}}" \\
        bash "${{TEMP_SCRIPT}}"
    
    EVAL_EXIT_CODE=$?
    rm -f "${{TEMP_SCRIPT}}"
else
    python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/{eval_script_name} \\
        train_test_split=${{EVAL_SPLIT}} \\
        experiment_name=${{RUN_NAME}} \\
        traffic_agents={traffic_agents} \\
        metric_cache_path=${{METRIC_CACHE}} \\
        output_dir=${{OUTPUT_DIR}} \\
        agent=${{AGENT}} \\
        ${{CHECKPOINT_ARG}} \\
        ${{CITY_ARG}} \\
        worker={worker_type} \\
        worker.threads_per_node=${{NUM_WORKERS}}
    EVAL_EXIT_CODE=$?
fi

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "Time: $(date)"
echo "Runtime: ${{RUNTIME}} seconds ($((${{RUNTIME}} / 60)) mins)"
echo ""

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ Status: SUCCESS"
    echo ""
    echo "Results: ${{OUTPUT_DIR}}"
    
    # Parse results
    if ls "${{OUTPUT_DIR}}"/*.csv 1>/dev/null 2>&1; then
        echo ""
        echo "Results Summary:"
        python -c "
import pandas as pd
import glob
csv_files = glob.glob('${{OUTPUT_DIR}}/*.csv')
if csv_files:
    df = pd.read_csv(csv_files[0])
    print(f'  Scenarios: {{len(df)}}')
    print(f'  PDMS: {{df[\"pdm_score\"].mean():.4f}}')
    print(f'  NC (No Collision): {{df[\"no_at_fault_collision\"].mean():.4f}}')
    print(f'  DAC (Drivable Area): {{df[\"drivable_area_compliance\"].mean():.4f}}')
    print(f'  EP (Ego Progress): {{df[\"ego_progress\"].mean():.4f}}')
    print(f'  TTC (Time to Collision): {{df[\"time_to_collision\"].mean():.4f}}')
    print(f'  C (Comfort): {{df[\"comfort\"].mean():.4f}}')
" 2>/dev/null || echo "  (Run 'cat ${{OUTPUT_DIR}}/*.csv' to view)"
    fi
else
    echo "✗ Status: FAILED (exit code: $EVAL_EXIT_CODE)"
    echo ""
    echo "Check logs:"
    echo "  {self.logs_dir}/error/{exp_id}{city_suffix}_eval_${{SLURM_JOB_ID}}.err"
fi
echo "=============================================="

exit $EVAL_EXIT_CODE
"""
        return script

    # -------------------------------------------------------------------------
    # Job Submission
    # -------------------------------------------------------------------------

    def submit_experiment(
        self,
        exp_id: str,
        train_only: bool = False,
        eval_only: bool = False,
        eval_type: str = "one_stage",
        sweep_cities: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, str]:
        """
        Submit experiment to SLURM.

        Args:
            exp_id: Experiment ID
            train_only: Only submit training job
            eval_only: Only submit evaluation job(s)
            eval_type: "one_stage" or "two_stage"
            sweep_cities: Evaluate on all cities (boston, vegas, pittsburgh, singapore)
            dry_run: Generate scripts but don't submit
        """

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found.")
            print("Create it first: python navsim_manager.py new <exp_id>")
            sys.exit(1)

        meta = self.metadata[exp_id]
        config = meta["config"]
        requires_training = config.get("requires_training", True)

        # For rule-based agents, force eval_only
        if not requires_training:
            if not eval_only and not train_only:
                print(
                    f"ℹ Agent '{config.get('agent')}' is rule-based, running eval-only"
                )
                eval_only = True
            elif train_only:
                print(
                    f"Error: Agent '{config.get('agent')}' is rule-based and doesn't require training"
                )
                sys.exit(1)

        # Generate scripts
        train_script_path = self.generated_dir / f"train_{exp_id}.slurm"
        job_ids = {}

        # Training script (if needed)
        if not eval_only and requires_training:
            train_script = self._generate_train_script(config)
            with open(train_script_path, "w") as f:
                f.write(train_script)
            os.chmod(train_script_path, 0o755)
            meta["train_script_path"] = str(train_script_path)
            print(f"✓ Generated: {train_script_path}")

        # Evaluation script(s)
        if not train_only:
            if sweep_cities:
                cities = ["boston", "vegas", "pittsburgh", "singapore"]
                print(f"✓ Generating eval scripts for {len(cities)} cities")
            else:
                cities = [None]  # Single eval with default split

            for city in cities:
                city_suffix = f"_{city}" if city else ""
                eval_script_path = (
                    self.generated_dir / f"eval_{exp_id}{city_suffix}.slurm"
                )
                eval_script = self._generate_eval_script(
                    config, eval_type=eval_type, city=city
                )
                with open(eval_script_path, "w") as f:
                    f.write(eval_script)
                os.chmod(eval_script_path, 0o755)
                print(f"✓ Generated: {eval_script_path} (type: {eval_type})")

        if dry_run:
            print("\n[DRY RUN] Would submit:")
            if not eval_only and requires_training:
                print(f"  sbatch {train_script_path}")
            if not train_only:
                for city in cities:
                    city_suffix = f"_{city}" if city else ""
                    eval_path = self.generated_dir / f"eval_{exp_id}{city_suffix}.slurm"
                    dep = (
                        "--dependency=afterok:TRAIN_JOB "
                        if (not eval_only and requires_training)
                        else ""
                    )
                    print(f"  sbatch {dep}{eval_path}")
            return {}

        # Submit jobs
        try:
            # Training job (if needed)
            if not eval_only and requires_training:
                result = subprocess.run(
                    ["sbatch", str(train_script_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                train_job_id = result.stdout.strip().split()[-1]
                job_ids["train_job_id"] = train_job_id
                meta["train_job_id"] = train_job_id
                print(f"✓ Submitted training: {train_job_id}")

            # Evaluation job(s)
            if not train_only:
                eval_job_ids = []
                for city in cities:
                    city_suffix = f"_{city}" if city else ""
                    eval_script_path = (
                        self.generated_dir / f"eval_{exp_id}{city_suffix}.slurm"
                    )

                    cmd = ["sbatch"]
                    if "train_job_id" in job_ids:
                        cmd.extend(
                            ["--dependency", f"afterok:{job_ids['train_job_id']}"]
                        )
                    cmd.append(str(eval_script_path))

                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    eval_job_id = result.stdout.strip().split()[-1]
                    eval_job_ids.append(eval_job_id)
                    city_label = city if city else "default"
                    print(f"✓ Submitted eval ({city_label}): {eval_job_id}")

                job_ids["eval_job_ids"] = eval_job_ids
                meta["eval_job_ids"] = eval_job_ids

            meta["status"] = "submitted"
            meta["submitted_at"] = datetime.now().isoformat()
            self._save_metadata()

            return job_ids

        except subprocess.CalledProcessError as e:
            print(f"Error submitting: {e.stderr}")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Listing and Status
    # -------------------------------------------------------------------------

    def list_experiments(
        self, status: Optional[str] = None, tags: Optional[List[str]] = None
    ):
        """List all experiments"""

        # Get current SLURM jobs
        try:
            result = subprocess.run(
                ["squeue", "-u", self.username, "-h", "-o", "%.18i %.50j %.8T"],
                capture_output=True,
                text=True,
            )
            running_jobs = {}
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 3:
                    running_jobs[parts[0].strip()] = parts[2].strip()
        except:
            running_jobs = {}

        filtered = []
        for exp_id, meta in self.metadata.items():
            if status and meta.get("status") != status:
                continue
            config = meta.get("config", {})
            if tags and not any(t in config.get("tags", []) for t in tags):
                continue

            # Update status based on SLURM
            train_job = meta.get("train_job_id")
            if train_job and train_job in running_jobs:
                meta["status"] = running_jobs[train_job].lower()

            filtered.append((exp_id, meta))

        if not filtered:
            print("No experiments found")
            return

        print(
            f"\n{'ID':<15} {'Status':<12} {'Agent':<25} {'Epochs':<8} {'Description'}"
        )
        print("=" * 90)

        for exp_id, meta in sorted(filtered, key=lambda x: x[0]):
            config = meta.get("config", {})
            desc = config.get("description", "")[:30]
            agent = config.get("agent", "")[:23]
            epochs = config.get("epochs", "-")
            status_str = meta.get("status", "unknown")
            print(f"{exp_id:<15} {status_str:<12} {agent:<25} {epochs:<8} {desc}")

    def show_experiment(self, exp_id: str):
        """Show detailed experiment info"""

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            sys.exit(1)

        meta = self.metadata[exp_id]
        config = meta.get("config", {})

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_id}")
        print(f"{'='*60}")
        print(f"Status: {meta.get('status', 'unknown')}")
        print(f"Description: {config.get('description', '')}")
        print(f"Created: {config.get('created_at', '')}")
        print(
            f"Git: {config.get('git_commit', 'N/A')} ({config.get('git_branch', '')})"
        )
        print()

        print("Training Configuration:")
        for key in [
            "agent",
            "backbone",
            "epochs",
            "batch_size",
            "learning_rate",
            "encoder_learning_rate",
            "trainable_fraction",
            "vision_mode",
            "cache_name",
            "partition",
            "num_gpus",
        ]:
            if key in config:
                print(f"  {key}: {config[key]}")

        print()
        print("Evaluation Configuration:")
        for key in ["eval_split", "eval_workers", "eval_type"]:
            if key in config:
                print(f"  {key}: {config[key]}")

        print()
        print("Jobs:")
        print(f"  Train job: {meta.get('train_job_id', 'N/A')}")
        print(f"  Eval job: {meta.get('eval_job_id', 'N/A')}")

        # Check for registered checkpoint
        ckpt_file = self.checkpoints_dir / f"{exp_id}.txt"
        if ckpt_file.exists():
            print(f"  Checkpoint: {ckpt_file.read_text().strip()}")

        if meta.get("results"):
            print()
            print("Results:")
            for k, v in meta["results"].items():
                print(f"  {k}: {v}")

    def harvest_results(
        self, exp_id: str, generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Harvest results using ExpFlow's NavsimResultsHarvester.

        Collects training metrics from TensorBoard/logs and evaluation
        metrics from PDM Score CSVs.
        """
        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            return {}

        results = {}

        # Use NavsimResultsHarvester if available
        if EXPFLOW_AVAILABLE and NavsimResultsHarvester is not None:
            harvester = NavsimResultsHarvester(self.experiments_root)

            # Harvest training and evaluation metrics
            train_metrics, eval_metrics_list = harvester.harvest_experiment(
                exp_id, generate_plots=generate_plots
            )

            # Collect training metrics
            if train_metrics:
                if train_metrics.train_loss_last is not None:
                    results["train_loss"] = train_metrics.train_loss_last
                if train_metrics.val_loss_last is not None:
                    results["val_loss"] = train_metrics.val_loss_last
                if train_metrics.val_loss_min is not None:
                    results["val_loss_min"] = train_metrics.val_loss_min
                if train_metrics.epochs_completed is not None:
                    results["epochs"] = train_metrics.epochs_completed
                if train_metrics.plot_path:
                    results["plot_path"] = train_metrics.plot_path

            # Collect evaluation metrics
            for em in eval_metrics_list:
                prefix = (
                    f"{em.eval_split}_"
                    if em.eval_split and em.eval_split != "unknown"
                    else ""
                )
                if em.score is not None:
                    results[f"{prefix}pdms"] = em.score
                for k, v in em.metrics.items():
                    if k != "num_scenarios" and v is not None:
                        results[f"{prefix}{k}"] = v
                if em.csv_path:
                    results[f"{prefix}csv_path"] = em.csv_path
        else:
            # Fallback: original log parsing logic
            eval_logs = sorted(
                Path(self.logs_dir / "output").glob(f"{exp_id}_eval_*.out"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            if eval_logs:
                latest_log = eval_logs[0]
                print(f"Parsing: {latest_log}")

                with open(latest_log) as f:
                    content = f.read()

                # Extract PDMS
                score_match = re.search(
                    r"Final average score of valid results:\s*([0-9]+\.?[0-9]*)",
                    content,
                )
                if score_match:
                    results["pdms"] = float(score_match.group(1))

                if "pdms" not in results:
                    pdms_match = re.search(r"PDMS:\s*([0-9.]+)", content)
                    if pdms_match:
                        results["pdms"] = float(pdms_match.group(1))

        # Update metadata
        if results:
            self.metadata[exp_id]["results"] = results
            self._save_metadata()
            print(f"✓ Harvested results for {exp_id}:")
            for k, v in results.items():
                if not k.endswith("_path"):
                    print(f"  {k}: {v}")
        else:
            print(f"No results found for {exp_id}")

        return results

    def harvest_all(
        self, exp_ids: List[str] = None, generate_comparison: bool = True
    ) -> Path:
        """
        Harvest results for multiple experiments and generate comparison table.

        Args:
            exp_ids: List of experiment IDs (default: all experiments)
            generate_comparison: Whether to generate comparison CSV

        Returns:
            Path to comparison CSV
        """
        if exp_ids is None:
            exp_ids = list(self.metadata.keys())

        if not exp_ids:
            print("No experiments found")
            return None

        print(f"Harvesting results for {len(exp_ids)} experiments...")

        if EXPFLOW_AVAILABLE and NavsimResultsHarvester is not None:
            harvester = NavsimResultsHarvester(self.experiments_root)

            # Harvest all
            all_training, all_eval = harvester.harvest_all_experiments(
                exp_ids, generate_plots=True
            )

            # Update metadata for each experiment
            for tm in all_training:
                if tm.exp_id in self.metadata:
                    self.metadata[tm.exp_id]["results"] = self.metadata[tm.exp_id].get(
                        "results", {}
                    )
                    if tm.train_loss_last:
                        self.metadata[tm.exp_id]["results"][
                            "train_loss"
                        ] = tm.train_loss_last
                    if tm.val_loss_last:
                        self.metadata[tm.exp_id]["results"][
                            "val_loss"
                        ] = tm.val_loss_last

            for em in all_eval:
                if em.exp_id in self.metadata:
                    self.metadata[em.exp_id]["results"] = self.metadata[em.exp_id].get(
                        "results", {}
                    )
                    if em.score:
                        self.metadata[em.exp_id]["results"]["pdms"] = em.score

            self._save_metadata()

            # Generate comparison table
            if generate_comparison:
                return harvester.generate_comparison_table(exp_ids)
        else:
            # Fallback
            for exp_id in exp_ids:
                self.harvest_results(exp_id, generate_plots=False)
            return None

    # -------------------------------------------------------------------------
    # Results Collection (cross-city evaluation summary)
    # -------------------------------------------------------------------------

    def collect_results(self, exp_id: str, output_json: bool = False) -> Dict[str, Any]:
        """
        Collect and display cross-city evaluation results for an experiment.

        Parses evaluation logs to extract PDMS scores per city.

        Args:
            exp_id: Experiment ID
            output_json: If True, output JSON format

        Returns:
            Dictionary with results per city
        """
        results = {
            "experiment": exp_id,
            "timestamp": datetime.now().isoformat(),
            "cities": {},
            "total_scenarios": 0,
            "average_pdms": 0.0,
        }

        # Find all evaluation logs for this experiment
        log_pattern = self.logs_dir / "output" / f"{exp_id}_*_eval_*.out"
        log_files = list(self.logs_dir.glob(f"output/{exp_id}_*_eval_*.out"))

        if not log_files:
            # Try without city suffix
            log_files = list(self.logs_dir.glob(f"output/{exp_id}_eval_*.out"))

        if not log_files:
            print(f"No evaluation logs found for experiment: {exp_id}")
            print(f"  Searched: {log_pattern}")
            return results

        cities = ["boston", "vegas", "pittsburgh", "singapore"]
        total_scenarios = 0
        total_weighted_score = 0.0

        # Group logs by city, keep only most recent (highest job ID)
        city_logs = {}
        for log_file in log_files:
            # Determine which city this log is for
            city = None
            for c in cities:
                if c in log_file.name.lower():
                    city = c
                    break

            if not city:
                city = "all"

            # Extract job ID from filename (e.g., baseline_cv_sweep_boston_eval_1558135.out)
            job_match = re.search(r"_(\d+)\.out$", log_file.name)
            job_id = int(job_match.group(1)) if job_match else 0

            # Keep only the most recent log per city
            if city not in city_logs or job_id > city_logs[city][1]:
                city_logs[city] = (log_file, job_id)

        # Process only the most recent log per city
        for city, (log_file, job_id) in city_logs.items():
            # Parse the log file
            try:
                with open(log_file, "r") as f:
                    content = f.read()

                # Extract scenarios count (handle trailing period)
                scenarios_match = re.search(
                    r"Number of successful scenarios:\s*(\d+)\.?", content
                )
                scenarios = int(scenarios_match.group(1)) if scenarios_match else 0

                # Extract PDMS score (handle trailing period)
                score_match = re.search(
                    r"Final average score of valid results:\s*([\d.]+)\.?", content
                )
                if score_match:
                    score_str = score_match.group(1).rstrip(".")
                    pdms = float(score_str)
                else:
                    pdms = 0.0

                # Extract status
                status_match = re.search(r"Status:\s*(SUCCESS|FAILED)", content)
                status = status_match.group(1) if status_match else "UNKNOWN"

                results["cities"][city] = {
                    "scenarios": scenarios,
                    "pdms": pdms,
                    "status": status,
                    "log_file": str(log_file.name),
                    "job_id": job_id,
                }

                if status == "SUCCESS" and scenarios > 0:
                    total_scenarios += scenarios
                    total_weighted_score += pdms * scenarios

            except Exception as e:
                print(f"Warning: Could not parse {log_file}: {e}")

        # Calculate average
        results["total_scenarios"] = total_scenarios
        results["average_pdms"] = (
            total_weighted_score / total_scenarios if total_scenarios > 0 else 0.0
        )

        # Output
        if output_json:
            print(json.dumps(results, indent=2))
        else:
            self._print_results_table(results)

        return results

    def _print_results_table(self, results: Dict[str, Any]):
        """Print formatted results table."""
        print(f"\n{'='*55}")
        print(f"Experiment: {results['experiment']}")
        print(f"{'='*55}")
        print(f"{'City':<12} | {'Scenarios':>10} | {'PDMS':>8} | {'Status':<8}")
        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

        city_order = ["boston", "vegas", "pittsburgh", "singapore", "all"]
        for city in city_order:
            if city in results["cities"]:
                data = results["cities"][city]
                print(
                    f"{city.capitalize():<12} | {data['scenarios']:>10,} | {data['pdms']:>8.3f} | {data['status']:<8}"
                )

        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        print(
            f"{'Average':<12} | {results['total_scenarios']:>10,} | {results['average_pdms']:>8.3f} |"
        )
        print(f"{'='*55}\n")

    # -------------------------------------------------------------------------
    # Cache Building (delegates to NavsimCacheBuilder for ExpFlow inheritance)
    # -------------------------------------------------------------------------

    def build_cache_pipeline(
        self,
        cache_name: str,
        cache_type: str = "training",
        num_cams: int = 6,
        vision_views: List[str] = None,
        eval_split: str = "navtest",
        agent: str = "ijepa_planning_agent_v4",
        dry_run: bool = False,
        skip_squashfs: bool = False,
        skip_cleanup: bool = False,
    ) -> Dict[str, str]:
        """
        Build cache using ExpFlow's inherited pipeline: build → squashfs → cleanup.

        Delegates to NavsimCacheBuilder which inherits generic squashfs/cleanup
        from ExpFlow's BaseCacheBuilder, only implementing NAVSIM-specific build scripts.
        """
        if not EXPFLOW_AVAILABLE or NavsimCacheBuilder is None:
            print(
                "Error: ExpFlow not available. Install expflow-hpc for cache building."
            )
            print("  pip install -e /scratch/$USER/expflow-hpc")
            return {}

        # Initialize cache builder (inherits from ExpFlow)
        builder = NavsimCacheBuilder()

        # Set up cache params based on type
        if cache_type == "training":
            if vision_views is None:
                vision_views = (
                    ["cam_l0", "cam_f0", "cam_r0", "cam_l1", "cam_r1", "cam_b0"]
                    if num_cams == 6
                    else ["cam_l0", "cam_f0", "cam_r0"]
                )
            cache_params = {
                "num_cams": num_cams,
                "vision_views": vision_views,
                "agent": agent,
                "train_split": "navtrain",
            }
        else:  # metric
            cache_params = {
                "eval_split": eval_split,
            }

        # Create cache config using ExpFlow's CacheConfig dataclass
        config = builder.create_cache_config(
            cache_name=cache_name,
            cache_type=cache_type,
            description=f"NAVSIM {cache_type} cache",
            partition="cs",
            num_cpus=96,
            memory="256G",
            time_limit="48:00:00" if cache_type == "training" else "08:00:00",
            num_workers=72 if cache_type == "training" else 96,
            cache_params=cache_params,
        )

        # Use inherited pipeline: build → squashfs → cleanup
        # squashfs and cleanup scripts come from BaseCacheBuilder!
        return builder.build_cache_pipeline(
            cache_name,
            skip_squashfs=skip_squashfs,
            skip_cleanup=skip_cleanup,
            dry_run=dry_run,
        )


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NAVSIM Experiment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create experiment from template
  python navsim_manager.py new exp_b15 --template ijepa_mlp_v4 --description "Next run"
  
  # Override specific params
  python navsim_manager.py new exp_b16 --template ijepa_mlp_v4 --epochs 50 --batch-size 64
  
  # Preview before submitting
  python navsim_manager.py submit exp_b15 --dry-run
  
  # Submit with one-stage eval (default, CPU-only, non-reactive)
  python navsim_manager.py submit exp_b15 --eval-type one_stage
  
  # Submit with two-stage eval (GPU, reactive)
  python navsim_manager.py submit exp_b15 --eval-type two_stage
  
  # Submit training only
  python navsim_manager.py submit exp_b15 --train-only
  
  # Resume interrupted experiment
  python navsim_manager.py resume B22 --submit
  python navsim_manager.py resume B23 --submit --eval-type one_stage
  
  # List and show
  python navsim_manager.py list
  python navsim_manager.py show exp_b15
  python navsim_manager.py harvest exp_b15
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # new
    new_parser = subparsers.add_parser("new", help="Create new experiment")
    new_parser.add_argument("exp_id", help="Experiment ID")
    new_parser.add_argument("--template", "-t", help="Template to use")
    new_parser.add_argument("--description", "-d", default="", help="Description")
    new_parser.add_argument("--agent", help="Override agent")
    new_parser.add_argument("--epochs", type=int, help="Override epochs")
    new_parser.add_argument("--batch-size", type=int, help="Override batch size")
    new_parser.add_argument(
        "--learning-rate", type=float, help="Override learning rate"
    )
    new_parser.add_argument(
        "--trainable-fraction", type=float, help="Override trainable fraction"
    )
    new_parser.add_argument("--partition", help="Override partition")
    new_parser.add_argument(
        "--city", help="City for cross-city generalization (e.g. boston)"
    )
    new_parser.add_argument("--num-gpus", type=int, help="Override num GPUs")
    new_parser.add_argument(
        "--eval-split", help="Override eval split (navtest/navmini)"
    )
    new_parser.add_argument(
        "--eval-type",
        choices=["one_stage", "two_stage"],
        default="one_stage",
        help="Evaluation type (default: one_stage)",
    )
    new_parser.add_argument(
        "--view-fusion-method",
        choices=["mean", "concat_proj", "cross_attn"],
        help="View fusion method (mean, concat_proj, cross_attn)",
    )
    new_parser.add_argument(
        "--cross-attn-heads", type=int, default=8, help="Cross-attention heads"
    )
    new_parser.add_argument(
        "--cross-attn-layers", type=int, default=2, help="Cross-attention layers"
    )
    new_parser.add_argument(
        "--camera-views", nargs="+", help="Camera views (e.g., cam_l0 cam_f0 cam_r0)"
    )
    new_parser.add_argument(
        "--planning-head-type",
        choices=["mlp", "transformer"],
        default="mlp",
        help="Planning head type (mlp or transformer)",
    )
    new_parser.add_argument(
        "--transformer-head-hidden-dim",
        type=int,
        default=256,
        help="Transformer head hidden dim",
    )
    new_parser.add_argument(
        "--transformer-head-num-heads",
        type=int,
        default=8,
        help="Transformer head attention heads",
    )
    new_parser.add_argument(
        "--transformer-head-num-layers",
        type=int,
        default=3,
        help="Transformer head layers",
    )
    new_parser.add_argument(
        "--transformer-head-dropout",
        type=float,
        default=0.1,
        help="Transformer head dropout",
    )

    # submit
    submit_parser = subparsers.add_parser("submit", help="Submit experiment")
    submit_parser.add_argument("exp_id", help="Experiment ID")
    submit_parser.add_argument(
        "--train-only", action="store_true", help="Submit training only"
    )
    submit_parser.add_argument(
        "--eval-only", action="store_true", help="Submit evaluation only"
    )
    submit_parser.add_argument(
        "--eval-type",
        choices=["one_stage", "two_stage"],
        default="one_stage",
        help="Evaluation type: one_stage (CPU, non-reactive) or two_stage (GPU, reactive)",
    )
    submit_parser.add_argument(
        "--sweep-cities",
        action="store_true",
        help="Evaluate on all cities (boston, vegas, pittsburgh, singapore)",
    )
    submit_parser.add_argument(
        "--dry-run", action="store_true", help="Show scripts but don't submit"
    )

    # resume
    resume_parser = subparsers.add_parser(
        "resume", help="Resume interrupted experiment from checkpoint"
    )
    resume_parser.add_argument("source_exp_id", help="Experiment ID to resume from")
    resume_parser.add_argument(
        "--new-exp-id",
        help="ID for new resumed experiment (auto-generated if not provided)",
    )
    resume_parser.add_argument(
        "--checkpoint",
        help="Specific checkpoint path (auto-detects latest if not provided)",
    )
    resume_parser.add_argument(
        "--submit", action="store_true", help="Immediately submit the resumed training"
    )
    resume_parser.add_argument(
        "--eval-only", action="store_true", help="Skip training, only submit evaluation"
    )
    resume_parser.add_argument(
        "--eval-type",
        choices=["one_stage", "two_stage"],
        default="one_stage",
        help="Evaluation type (default: one_stage)",
    )
    resume_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # list
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")

    # show
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id", help="Experiment ID")

    # harvest
    harvest_parser = subparsers.add_parser(
        "harvest", help="Harvest results for one experiment"
    )
    harvest_parser.add_argument("exp_id", help="Experiment ID")
    harvest_parser.add_argument(
        "--no-plots", action="store_true", help="Skip plot generation"
    )

    # harvest-all
    harvest_all_parser = subparsers.add_parser(
        "harvest-all",
        help="Harvest results for all experiments and generate comparison",
    )
    harvest_all_parser.add_argument(
        "exp_ids", nargs="*", help="Experiment IDs (default: all)"
    )
    harvest_all_parser.add_argument(
        "--no-comparison", action="store_true", help="Skip comparison table generation"
    )

    # cache (with subcommands: build, list, show)
    cache_parser = subparsers.add_parser("cache", help="Manage training/metric caches")
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command", help="Cache commands"
    )

    # cache build
    cache_build_parser = cache_subparsers.add_parser("build", help="Build a new cache")
    cache_build_parser.add_argument(
        "cache_type", choices=["training", "metric"], help="Cache type"
    )
    cache_build_parser.add_argument(
        "cache_name",
        help="Cache name (e.g., training_cache_transfuser, navtest_metric_cache)",
    )
    cache_build_parser.add_argument(
        "--num-cams",
        type=int,
        default=3,
        choices=[3, 6],
        help="Number of cameras for training cache (default: 3)",
    )
    cache_build_parser.add_argument(
        "--agent",
        default="transfuser_agent",
        help="Agent config for training cache (default: transfuser_agent)",
    )
    cache_build_parser.add_argument(
        "--split",
        "--eval-split",
        dest="eval_split",
        default="navtest",
        help="Eval split for metric cache (default: navtest)",
    )
    cache_build_parser.add_argument(
        "--skip-squashfs", action="store_true", help="Skip SquashFS compression step"
    )
    cache_build_parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup step (keep original cache dir)",
    )
    cache_build_parser.add_argument(
        "--dry-run", action="store_true", help="Generate scripts but don't submit"
    )

    # cache list
    cache_list_parser = cache_subparsers.add_parser("list", help="List all caches")
    cache_list_parser.add_argument(
        "--type", dest="cache_type", help="Filter by cache type"
    )

    # cache show
    cache_show_parser = cache_subparsers.add_parser("show", help="Show cache details")
    cache_show_parser.add_argument("cache_name", help="Cache name to show")

    # templates
    templates_parser = subparsers.add_parser(
        "templates", help="List available templates"
    )

    # results
    results_parser = subparsers.add_parser(
        "results", help="Collect cross-city evaluation results"
    )
    results_parser.add_argument("exp_id", help="Experiment ID")
    results_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # prune
    prune_parser = subparsers.add_parser(
        "prune", help="Prune old/duplicate/invalid experiments"
    )
    prune_parser.add_argument(
        "--duplicates-only",
        action="store_true",
        help="Only prune duplicate experiments (keep most recent)",
    )
    prune_parser.add_argument(
        "--invalid-only",
        action="store_true",
        help="Only prune experiments without valid checkpoints/results",
    )
    prune_parser.add_argument(
        "--keep-n",
        type=int,
        default=1,
        help="Number of duplicate runs to keep (default: 1)",
    )
    prune_parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        default=True,
        help="Require valid checkpoint to keep experiment",
    )
    prune_parser.add_argument(
        "--require-eval",
        action="store_true",
        default=False,
        help="Require valid eval results to keep experiment",
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pruned without deleting",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = NavsimExperimentManager()

    if args.command == "new":
        overrides = {}
        if args.agent:
            overrides["agent"] = args.agent
        if args.epochs:
            overrides["epochs"] = args.epochs
        if args.batch_size:
            overrides["batch_size"] = args.batch_size
        if args.learning_rate:
            overrides["learning_rate"] = args.learning_rate
        if args.trainable_fraction:
            overrides["trainable_fraction"] = args.trainable_fraction
        if args.partition:
            overrides["partition"] = args.partition
        if args.city:
            overrides["city"] = args.city
        if args.num_gpus:
            overrides["num_gpus"] = args.num_gpus
        if args.eval_split:
            overrides["eval_split"] = args.eval_split
        if args.eval_type:
            overrides["eval_type"] = args.eval_type
        if args.view_fusion_method:
            overrides["view_fusion_method"] = args.view_fusion_method
        if args.cross_attn_heads:
            overrides["cross_attn_heads"] = args.cross_attn_heads
        if args.cross_attn_layers:
            overrides["cross_attn_layers"] = args.cross_attn_layers
        if args.camera_views:
            overrides["camera_views"] = args.camera_views
        if args.planning_head_type:
            overrides["planning_head_type"] = args.planning_head_type
        if args.transformer_head_hidden_dim:
            overrides["transformer_head_hidden_dim"] = args.transformer_head_hidden_dim
        if args.transformer_head_num_heads:
            overrides["transformer_head_num_heads"] = args.transformer_head_num_heads
        if args.transformer_head_num_layers:
            overrides["transformer_head_num_layers"] = args.transformer_head_num_layers
        if args.transformer_head_dropout:
            overrides["transformer_head_dropout"] = args.transformer_head_dropout
        manager.create_experiment(
            args.exp_id, args.template, args.description, **overrides
        )

    elif args.command == "submit":
        manager.submit_experiment(
            args.exp_id,
            train_only=args.train_only,
            eval_only=args.eval_only,
            eval_type=args.eval_type,
            sweep_cities=args.sweep_cities,
            dry_run=args.dry_run,
        )

    elif args.command == "resume":
        manager.resume_experiment(
            source_exp_id=args.source_exp_id,
            new_exp_id=args.new_exp_id,
            checkpoint_path=args.checkpoint,
            submit=args.submit,
            eval_only=args.eval_only,
            eval_type=args.eval_type,
            dry_run=args.dry_run,
        )

    elif args.command == "list":
        manager.list_experiments(args.status, args.tags)

    elif args.command == "show":
        manager.show_experiment(args.exp_id)

    elif args.command == "harvest":
        manager.harvest_results(args.exp_id, generate_plots=not args.no_plots)

    elif args.command == "harvest-all":
        exp_ids = args.exp_ids if args.exp_ids else None
        manager.harvest_all(exp_ids, generate_comparison=not args.no_comparison)

    elif args.command == "cache":
        if not EXPFLOW_AVAILABLE or NavsimCacheBuilder is None:
            print(
                "Error: ExpFlow not available. Install expflow-hpc for cache management."
            )
            print("  pip install -e /scratch/$USER/expflow-hpc")
            sys.exit(1)

        builder = NavsimCacheBuilder()

        if args.cache_command == "build":
            print(f"\n{'='*60}")
            print(
                f"Building {args.cache_type.upper()} Cache Pipeline: {args.cache_name}"
            )
            print(f"{'='*60}")
            if args.cache_type == "training":
                print(f"  Type: training (dataset features)")
                print(f"  Cameras: {args.num_cams}")
                print(f"  Agent: {args.agent}")
            else:
                print(f"  Type: metric (evaluation precomputation)")
                print(f"  Split: {args.eval_split}")
            print()
            manager.build_cache_pipeline(
                cache_name=args.cache_name,
                cache_type=args.cache_type,
                num_cams=args.num_cams,
                eval_split=args.eval_split,
                agent=args.agent,
                dry_run=args.dry_run,
                skip_squashfs=args.skip_squashfs,
                skip_cleanup=args.skip_cleanup,
            )

        elif args.cache_command == "list":
            builder.list_caches(cache_type=args.cache_type)

        elif args.cache_command == "show":
            builder.show_cache(args.cache_name)

        else:
            cache_parser.print_help()

    elif args.command == "templates":
        templates_dir = manager.templates_dir
        if templates_dir.exists():
            templates = list(templates_dir.glob("*.yaml"))
            if templates:
                print("\nAvailable templates:")
                for t in sorted(templates):
                    print(f"  {t.stem}")
            else:
                print("No templates found")
        else:
            print(f"Templates directory not found: {templates_dir}")

    elif args.command == "results":
        manager.collect_results(args.exp_id, output_json=args.json)

    elif args.command == "prune":
        if not EXPFLOW_AVAILABLE or NavsimPruner is None:
            print("Error: ExpFlow not available. Install expflow-hpc for pruning.")
            print("  pip install -e /scratch/$USER/expflow-hpc")
            sys.exit(1)

        pruner = NavsimPruner()

        if args.duplicates_only:
            pruned = pruner.prune_duplicates(keep_n=args.keep_n, dry_run=args.dry_run)
        elif args.invalid_only:
            pruned = pruner.prune_invalid(
                require_checkpoint=args.require_checkpoint,
                require_eval=args.require_eval,
                dry_run=args.dry_run,
            )
        else:
            # Full prune: duplicates + invalid
            pruned = pruner.prune_all(
                keep_n=args.keep_n,
                require_checkpoint=args.require_checkpoint,
                require_eval=args.require_eval,
                dry_run=args.dry_run,
            )

        # `pruner.prune_*` returns a PruneStats object. Print a concise summary.
        pruned_count = getattr(pruned, "pruned", None)
        freed_mb = getattr(pruned, "space_freed_mb", None)
        invalid_removed = getattr(pruned, "invalid_removed", None)
        duplicates_removed = getattr(pruned, "duplicates_removed", None)

        if args.dry_run:
            print(f"\n[DRY RUN] Summary: would prune {pruned_count} experiments")
        else:
            print(f"\n✓ Pruned {pruned_count} experiments")

        if freed_mb is not None:
            print(f"  Estimated space freed: {freed_mb:.1f} MB")
        if invalid_removed is not None and duplicates_removed is not None:
            print(
                f"  Invalid removed: {invalid_removed}, Duplicates removed: {duplicates_removed}"
            )


if __name__ == "__main__":
    main()
