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
    python navsim_manager.py cancel exp_b15
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

# ─── Fast path: `pretrain` command bypasses heavy expflow imports ────────────
# On congested login nodes, importing tensorboard/boto3 via expflow takes 30s+.
# The pretrain sub-system is self-contained and needs none of that.
if len(sys.argv) >= 2 and sys.argv[1] == "pretrain":
    # Defer to standalone pretrain CLI — defined at bottom of file, before main()
    pass  # PretrainManager is defined later; we still need argparse etc.
    _PRETRAIN_FAST_PATH = True
else:
    _PRETRAIN_FAST_PATH = False

# Try to use expflow for additional features (skip if pretrain fast-path)
if _PRETRAIN_FAST_PATH:
    EXPFLOW_AVAILABLE = False
    BaseCacheBuilder = None
    CacheConfig = None
    BaseResultsHarvester = None
    TrainingMetrics = None
    EvaluationMetrics = None
    ExperimentPruner = None
    PruneStats = None
    HPCConfig = None
else:
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
        BaseCacheBuilder = None
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

                # Custom loader that handles extra fields like custom_vars
                config_path = project_root / ".hpc_config.yaml"
                if config_path.exists():
                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)
                    # Remove custom_vars before passing to HPCConfig
                    config_data.pop("custom_vars", None)
                    hpc_config = HPCConfig(**config_data)
                else:
                    raise FileNotFoundError(
                        f"No .hpc_config.yaml found at {project_root}"
                    )

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

                # TransFuser and LAW have their own sensor config, no vision_views override
                if "transfuser" in agent or "law" in agent:
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
                elif "ego_status_mlp" in agent:
                    # Ego-status MLP doesn't use cameras, minimal config
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

# Environment setup (from hpc_config)
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="{getattr(self.hpc_config, 'navsim_devkit_root', f'/scratch/{self.username}/navsim-ssl-city-generalization/navsim')}"
export OPENSCENE_DATA_ROOT="{getattr(self.hpc_config, 'openscene_data_root', f'/scratch/{self.username}/data')}"
export NUPLAN_MAPS_ROOT="{getattr(self.hpc_config, 'nuplan_maps_root', f'/scratch/{self.username}/data/maps')}"
export NAVSIM_EXP_ROOT="{self.hpc_config.experiments_dir}"

mkdir -p "{config.cache_output_dir}"
cd "${{NAVSIM_DEVKIT_ROOT}}"

# Conda activation (from hpc_config)
CONDA_ROOT="{self.hpc_config.conda_root or f'/scratch/{self.username}/miniconda3'}"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate {self.hpc_config.conda_env or 'navsim'}
else
    module purge || true
    module load anaconda3/2025.06 || true
    source $(conda info --base)/etc/profile.d/conda.sh || true
    conda activate {self.hpc_config.conda_env or 'navsim'} || true
fi
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

# Environment setup (from hpc_config)
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="{getattr(self.hpc_config, 'navsim_devkit_root', f'/scratch/{self.username}/navsim-ssl-city-generalization/navsim')}"
export OPENSCENE_DATA_ROOT="{getattr(self.hpc_config, 'openscene_data_root', f'/scratch/{self.username}/data')}"
export NUPLAN_MAPS_ROOT="{getattr(self.hpc_config, 'nuplan_maps_root', f'/scratch/{self.username}/data/maps')}"
export NAVSIM_EXP_ROOT="{self.hpc_config.experiments_dir}"

mkdir -p "{config.cache_output_dir}"
cd "${{NAVSIM_DEVKIT_ROOT}}"

# Conda activation (from hpc_config)
CONDA_ROOT="{self.hpc_config.conda_root or f'/scratch/{self.username}/miniconda3'}"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate {self.hpc_config.conda_env or 'navsim'}
else
    module purge || true
    module load anaconda3/2025.06 || true
    source $(conda info --base)/etc/profile.d/conda.sh || true
    conda activate {self.hpc_config.conda_env or 'navsim'} || true
fi
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

            # Strict filtering: check if exp_id is an exact component of the directory name
            # This prevents A3 from matching A3-b
            filtered_files = []
            for f in result_files:
                path = Path(f)
                # Check parent directory name (e.g. eval_navtest_A3_...)
                d_name = path.parent.name
                if (
                    d_name == exp_id
                    or d_name.startswith(f"{exp_id}_")
                    or d_name.endswith(f"_{exp_id}")
                    or f"_{exp_id}_" in d_name
                ):
                    filtered_files.append(path)

            return filtered_files

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

        # Load from .hpc_config.yaml - custom loader that handles extra fields
        self.hpc_config = None
        self.custom_vars = {}
        config_path = self.project_root / ".hpc_config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)

                # Extract custom_vars before passing to HPCConfig
                self.custom_vars = config_data.pop("custom_vars", {}) or {}

                # Create HPCConfig with remaining fields
                if EXPFLOW_AVAILABLE:
                    from expflow.hpc_config import HPCConfig

                    self.hpc_config = HPCConfig(**config_data)
                    self.scratch = Path(self.hpc_config.scratch_dir)
                    self.experiments_root = Path(self.hpc_config.experiments_dir)
                    self.logs_dir = Path(self.hpc_config.logs_dir)
                    self.checkpoints_dir = Path(self.hpc_config.checkpoints_dir)
                else:
                    # Fallback: just use the raw config dict
                    self.hpc_config = type("HPCConfig", (), config_data)()
                    self.scratch = Path(
                        config_data.get("scratch_dir", f"/scratch/{self.username}")
                    )
                    self.experiments_root = Path(
                        config_data.get("experiments_dir", self.scratch / "experiments")
                    )
                    self.logs_dir = Path(
                        config_data.get("logs_dir", self.experiments_root / "logs")
                    )
                    self.checkpoints_dir = Path(
                        config_data.get(
                            "checkpoints_dir", self.experiments_root / "checkpoints"
                        )
                    )
            except Exception as e:
                print(f"Warning: Failed to load .hpc_config.yaml: {e}")
                self.experiments_root = self.scratch / "experiments"
                self.logs_dir = self.experiments_root / "logs"
                self.checkpoints_dir = self.experiments_root / "checkpoints"
        else:
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

    @property
    def results_storage(self):
        """
        Override BaseExperimentManager.results_storage to support multiple backends

        Supports:
        - SQLite (default): Local file-based storage
        - MongoDB: Remote cloud database (requires pymongo)
        - PostgreSQL: Remote SQL database (requires psycopg2-binary)

        Backend is selected via environment variables:
        - EXPFLOW_BACKEND: 'sqlite', 'mongodb', or 'postgresql'
        - EXPFLOW_CONNECTION_STRING: Connection string for remote databases

        Returns:
            ResultsStorage: Database backend instance
        """
        if not hasattr(self, "_results_storage"):
            from expflow.results_storage import ResultsStorage
            import os

            # Get backend configuration from environment
            backend = os.getenv("EXPFLOW_BACKEND", "sqlite")
            connection_string = os.getenv("EXPFLOW_CONNECTION_STRING")

            if backend == "mongodb":
                # MongoDB backend (remote)
                if not connection_string:
                    raise ValueError(
                        "EXPFLOW_CONNECTION_STRING required for MongoDB backend. "
                        "Get free tier at: https://www.mongodb.com/cloud/atlas"
                    )

                # Optional: allow TLS validation bypass for restricted HPC environments
                if os.getenv("EXPFLOW_MONGODB_TLS_INSECURE", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    if "tlsAllowInvalidCertificates=true" not in connection_string:
                        sep = "&" if "?" in connection_string else "?"
                        connection_string = (
                            f"{connection_string}{sep}tlsAllowInvalidCertificates=true"
                        )
                    print(
                        "[WARN] MongoDB TLS cert validation disabled via EXPFLOW_MONGODB_TLS_INSECURE"
                    )

                self._results_storage = ResultsStorage(
                    backend="mongodb",
                    connection_string=connection_string,
                    database="navsim_experiments",
                    collection="results",
                )
                print("[INFO] Using MongoDB backend")

            elif backend == "postgresql":
                # PostgreSQL backend (remote)
                if not connection_string:
                    raise ValueError(
                        "EXPFLOW_CONNECTION_STRING required for PostgreSQL backend"
                    )

                self._results_storage = ResultsStorage(
                    backend="postgresql",
                    connection_string=connection_string,
                    table_name="navsim_experiments",
                )
                print("[INFO] Using PostgreSQL backend")

            else:
                # SQLite backend (local, default)
                db_path = self.project_root / "experiments_results.db"
                self._results_storage = ResultsStorage(
                    backend="sqlite", path=str(db_path)
                )
                print(f"[INFO] Using SQLite backend: {db_path}")

        return self._results_storage

    @property
    def run_results_storage(self):
        """
        Storage backend for per-run experiment history.

        Uses a separate table/collection ('experiment_runs') from the main
        results_storage so run history is additive and never overwrites.
        Supports the same backends: SQLite, MongoDB, PostgreSQL.
        """
        if not hasattr(self, "_run_results_storage"):
            from expflow.results_storage import ResultsStorage

            backend = os.environ.get("EXPFLOW_BACKEND", "sqlite").lower()
            connection_string = os.environ.get("EXPFLOW_CONNECTION_STRING")

            if backend == "mongodb":
                self._run_results_storage = ResultsStorage(
                    backend="mongodb",
                    connection_string=connection_string,
                    database="navsim_experiments",
                    collection="experiment_runs",
                )
            elif backend == "postgresql":
                self._run_results_storage = ResultsStorage(
                    backend="postgresql",
                    connection_string=connection_string,
                    table_name="experiment_runs",
                )
            else:
                db_path = self.project_root / "experiment_runs.db"
                self._run_results_storage = ResultsStorage(
                    backend="sqlite", path=str(db_path)
                )

        return self._run_results_storage

    def store_comprehensive_results(self, exp_id: str) -> bool:
        """
        Store comprehensive experiment results with full metadata

        This method extracts all available data about an experiment and stores it
        in a structured format that matches the user's spreadsheet columns:
        - Research organization (phase, priority, notes)
        - Model architecture (name, backbone, type, freeze %)
        - Training configuration (city, data %, epochs, batch size, LR)
        - Evaluation results (per-city PDM scores, aggregates, breakdown metrics)
        - HPC/SLURM information (partition, GPUs, job IDs)
        - Git tracking (commit, branch, dirty status)

        Args:
            exp_id: Experiment identifier (e.g., 'exp_b20', 'A1')

        Returns:
            bool: True if successful, False otherwise

        Example:
            manager = NavsimExperimentManager()
            manager.store_comprehensive_results('exp_b20')
        """
        from datetime import datetime

        # Validate experiment exists
        if exp_id not in self.metadata:
            print(f"ERROR: Experiment {exp_id} not found in metadata")
            return False

        exp_meta = self.metadata[exp_id]
        _config_yaml = self.configs_dir / f"{exp_id}.yaml"
        if _config_yaml.exists():
            with open(_config_yaml) as _f:
                config = yaml.safe_load(_f) or {}
        else:
            config = exp_meta.get("config", {})

        # Harvest evaluation data using cross-city results parser
        try:
            cross_city_results = self.collect_results(exp_id, output_json=True)
        except Exception as e:
            print(f"WARNING: Could not collect cross-city results for {exp_id}: {e}")
            cross_city_results = {}

        city_results = (
            cross_city_results.get("cities", {})
            if isinstance(cross_city_results, dict)
            else {}
        )
        train_city_label = config.get("train_city") or config.get("city") or "all"
        train_city = (
            (config.get("train_city") or config.get("city") or "").strip().lower()
        )

        def _city_pdms(city_key: str):
            data = city_results.get(city_key)
            if not data:
                return None
            return data.get("pdms")

        def _in_dist(city_key: str):
            return bool(train_city) and train_city != "all" and train_city == city_key

        in_dist_pdms = (
            _city_pdms(train_city) if train_city and train_city != "all" else None
        )
        out_dist_scores = [
            d.get("pdms")
            for k, d in city_results.items()
            if k != train_city and d.get("pdms") is not None
        ]
        out_dist_avg = (
            (sum(out_dist_scores) / len(out_dist_scores)) if out_dist_scores else None
        )
        gen_gap_percent = None
        if in_dist_pdms not in (None, 0) and out_dist_avg is not None:
            gen_gap_percent = ((in_dist_pdms - out_dist_avg) / in_dist_pdms) * 100.0

        # Extract phase from experiment ID or config
        # (e.g., 'exp_b20' -> phase might be 'PHASE_1', 'A1' -> 'BASELINES')
        phase = config.get("phase", "UNKNOWN")
        if not phase or phase == "UNKNOWN":
            # Try to infer from exp_id
            if exp_id.startswith("A"):
                phase = "BASELINES"
            elif exp_id.startswith("P"):
                phase = "Transfuser"
            elif exp_id.startswith("F"):
                phase = "Latent Transfuser"
            elif exp_id.startswith("L"):
                phase = "LAW"
            elif exp_id.startswith("exp_b"):
                phase = "PHASE_1"
            elif exp_id.startswith("exp_c"):
                phase = "PHASE_2"

        agent = config.get("agent", "")
        architecture_name = config.get("architecture")
        # backbone_name is a display label, NOT the raw "backbone" config field
        # (which holds SSL method type like "ijepa", "dinov2", "mae", "resnet")
        _raw_ssl_methods = {"ijepa", "dinov2", "dino", "mae", "resnet"}
        backbone_name = config.get("backbone_name")  # explicit display name override
        if backbone_name is None:
            _raw_bb = config.get("backbone")
            # Only use raw value if it's NOT one of the method keys (those need inference)
            if _raw_bb and _raw_bb not in _raw_ssl_methods:
                backbone_name = _raw_bb
        backbone_type = config.get("backbone_type")
        freeze_percentage = config.get("freeze_percent", 0)

        if not architecture_name:
            if agent == "transfuser_agent":
                architecture_name = "TransFuser"
            elif agent == "diffusiondrive_ijepa_agent":
                architecture_name = "DiffusionDrive"
            elif agent == "ego_status_mlp_agent":
                architecture_name = "EgoStatusMLP"
            elif "constant_velocity" in agent:
                architecture_name = "ConstantVelocity"
            elif "law_agent" in agent:
                architecture_name = "LawAgent"
            else:
                architecture_name = "Unknown"

        if backbone_name is None:
            if agent in ("transfuser_agent", "diffusiondrive_ijepa_agent"):
                # Infer from backbone config field
                bb = config.get("backbone", "resnet")
                vit_be = config.get("vit_backend", "huggingface")
                vit_arch = config.get("vit_arch", "")
                vit_ps = config.get("vit_patch_size", 0)
                # Default backbone name map (when no vit_arch override)
                backbone_map_default = {
                    "resnet": "ResNet34",
                    "ijepa": "I-JEPA ViT-H/14",
                    "dinov2": "DINOv2 ViT-S/14",
                    "dino": "DINO ViT-B/16",
                    "mae": "MAE ViT-L/16",
                }
                # vit_arch override → use actual architecture
                vit_arch_label_map = {
                    "vit_tiny": "ViT-Ti",
                    "vit_small": "ViT-S",
                    "vit_base": "ViT-B",
                    "vit_large": "ViT-L",
                    "vit_huge": "ViT-H",
                    "vit_giant": "ViT-G",
                }
                ssl_label_map = {
                    "ijepa": "I-JEPA",
                    "dinov2": "DINOv2",
                    "dino": "DINO",
                    "mae": "MAE",
                }
                if vit_arch and bb in ssl_label_map:
                    arch_label = vit_arch_label_map.get(vit_arch, vit_arch)
                    patch_label = f"/{vit_ps}" if vit_ps else ""
                    backbone_name = f"{ssl_label_map[bb]} {arch_label}{patch_label}"
                    if vit_be == "custom":
                        backbone_name += " (nuScenes)"
                else:
                    backbone_name = backbone_map_default.get(bb, bb)
                    if bb != "resnet" and vit_be == "custom":
                        backbone_name += " (custom)"
            elif agent == "law_agent":
                backbone_name = "ResNet34"
            else:
                backbone_name = ""

        if backbone_type is None:
            if agent == "transfuser_agent":
                backbone_type = "RNN"
            elif agent == "diffusiondrive_ijepa_agent":
                backbone_type = "Diffusion"
            elif agent == "law_agent":
                backbone_type = "RNN"
            else:
                backbone_type = ""

        # Compute freeze percentage from backbone-specific trainable fraction
        if freeze_percentage == 0 and agent in (
            "transfuser_agent",
            "diffusiondrive_ijepa_agent",
        ):
            bb = config.get("backbone", "resnet")
            trainable_key_map = {
                "ijepa": "pc_trainable_ijepa_layers",
                "dinov2": "pc_trainable_dinov2_layers",
                "dino": "pc_trainable_dino_layers",
                "mae": "pc_trainable_mae_layers",
            }
            t_key = trainable_key_map.get(bb)
            if t_key and t_key in config:
                trainable_frac = float(config[t_key])
                freeze_percentage = round((1.0 - trainable_frac) * 100, 1)

        # Determine if latent mode
        is_latent = config.get("latent", False)

        # Compute effective batch size
        _num_gpus = config.get("num_gpus", 1)
        _batch_size = config.get("batch_size", 32)
        _accum = config.get("accumulate_grad_batches", 1)
        effective_batch_size = _num_gpus * _batch_size * _accum

        # Determine resolution type (square vs rectangular)
        use_native_res = config.get("use_native_resolution", True)
        if config.get("backbone", "resnet") == "resnet":
            resolution_type = None  # N/A for ResNet
            input_resolution = None
        elif use_native_res:
            resolution_type = "rect"
            input_resolution = "native (snapped to patch)"  # e.g. ~392x224
        else:
            resolution_type = "square"
            input_resolution = "224x224"

        # Determine weight source
        weight_source = "ImageNet"  # default
        for ckpt_key in [
            "ijepa_checkpoint_path",
            "dinov2_checkpoint_path",
            "mae_checkpoint_path",
        ]:
            ckpt_path = config.get(ckpt_key, "")
            if (
                "nuscenes" in ckpt_path.lower()
                or "_r/" in ckpt_path
                or "_s/" in ckpt_path
            ):
                weight_source = "nuScenes"
                break
            elif "IN1K" in ckpt_path or "imagenet" in ckpt_path.lower():
                weight_source = "ImageNet"
                break
        if config.get("backbone", "resnet") == "resnet":
            weight_source = "ImageNet (supervised)"

        # Build comprehensive data structure
        comprehensive_data = {
            # === Core Identifiers ===
            "exp_id": exp_id,
            "status": exp_meta.get("status", "unknown"),
            "created_at": config.get("created_at"),
            "submitted_at": config.get("submitted_at"),
            "completed_at": config.get("completed_at"),
            # === Research Organization ===
            "phase": phase,
            "priority": config.get("priority", "normal"),
            "notes": config.get("notes", config.get("description", "")),
            # === Model Architecture ===
            "architecture": {
                "name": architecture_name,
                "backbone": backbone_name,
                "backbone_type": backbone_type,
                "freeze_percentage": freeze_percentage,
                "latent": is_latent,
                "ssl_method": (
                    config.get("backbone", "resnet")
                    if config.get("backbone", "resnet") != "resnet"
                    else None
                ),
                "vit_arch": config.get("vit_arch") or None,
                "vit_patch_size": config.get("vit_patch_size") or None,
                "vit_backend": config.get("vit_backend") or None,
                "use_native_resolution": use_native_res,
                "resolution_type": resolution_type,
                "input_resolution": input_resolution,
                "weight_source": weight_source,
                "checkpoint_path": (
                    config.get("ijepa_checkpoint_path")
                    or config.get("dinov2_checkpoint_path")
                    or config.get("mae_checkpoint_path")
                    or None
                ),
                "checkpoint_key": (
                    config.get("ijepa_checkpoint_key")
                    or config.get("dinov2_checkpoint_key")
                    or config.get("mae_checkpoint_key")
                    or None
                ),
                # DiffusionDrive-specific fields
                "num_anchors": config.get("num_anchors") or None,
                "num_denoising_steps": config.get("num_denoising_steps") or None,
            },
            # === Training Configuration ===
            "training": {
                "city": train_city_label,
                "data_percentage": config.get("train_data_percent", 100),
                "epochs": config.get("epochs", 50),
                "batch_size": _batch_size,
                "learning_rate": config.get("learning_rate", 0.001),
                "num_gpus": _num_gpus,
                "accumulate_grad_batches": _accum,
                "effective_batch_size": effective_batch_size,
                "gradient_clip_val": config.get("gradient_clip_val"),
                "precision": config.get("precision", "16-mixed"),
                "num_workers": config.get("num_workers"),
                "cache_name": config.get("cache_name"),
                "train_split": config.get("train_split", "navtrain"),
            },
            # === Evaluation Results (Per-City) ===
            "evaluation": {
                # Individual city scores with sub-metrics
                "all": {
                    "pdm_score": _city_pdms("all"),
                    "in_distribution": False,
                    "scenarios": city_results.get("all", {}).get("scenarios"),
                    "NC": city_results.get("all", {}).get("NC"),
                    "DAC": city_results.get("all", {}).get("DAC"),
                    "EP": city_results.get("all", {}).get("EP"),
                    "TTC": city_results.get("all", {}).get("TTC"),
                    "C": city_results.get("all", {}).get("C"),
                },
                "boston": {
                    "pdm_score": _city_pdms("boston"),
                    "in_distribution": _in_dist("boston"),
                    "scenarios": city_results.get("boston", {}).get("scenarios"),
                    "NC": city_results.get("boston", {}).get("NC"),
                    "DAC": city_results.get("boston", {}).get("DAC"),
                    "EP": city_results.get("boston", {}).get("EP"),
                    "TTC": city_results.get("boston", {}).get("TTC"),
                    "C": city_results.get("boston", {}).get("C"),
                },
                "vegas": {
                    "pdm_score": _city_pdms("vegas"),
                    "in_distribution": _in_dist("vegas"),
                    "scenarios": city_results.get("vegas", {}).get("scenarios"),
                    "NC": city_results.get("vegas", {}).get("NC"),
                    "DAC": city_results.get("vegas", {}).get("DAC"),
                    "EP": city_results.get("vegas", {}).get("EP"),
                    "TTC": city_results.get("vegas", {}).get("TTC"),
                    "C": city_results.get("vegas", {}).get("C"),
                },
                "pittsburgh": {
                    "pdm_score": _city_pdms("pittsburgh"),
                    "in_distribution": _in_dist("pittsburgh"),
                    "scenarios": city_results.get("pittsburgh", {}).get("scenarios"),
                    "NC": city_results.get("pittsburgh", {}).get("NC"),
                    "DAC": city_results.get("pittsburgh", {}).get("DAC"),
                    "EP": city_results.get("pittsburgh", {}).get("EP"),
                    "TTC": city_results.get("pittsburgh", {}).get("TTC"),
                    "C": city_results.get("pittsburgh", {}).get("C"),
                },
                "singapore": {
                    "pdm_score": _city_pdms("singapore"),
                    "in_distribution": _in_dist("singapore"),
                    "scenarios": city_results.get("singapore", {}).get("scenarios"),
                    "NC": city_results.get("singapore", {}).get("NC"),
                    "DAC": city_results.get("singapore", {}).get("DAC"),
                    "EP": city_results.get("singapore", {}).get("EP"),
                    "TTC": city_results.get("singapore", {}).get("TTC"),
                    "C": city_results.get("singapore", {}).get("C"),
                },
                # Aggregate metrics
                "avg_pdm_score": cross_city_results.get("average_pdms"),
                "in_dist_pdm_score": in_dist_pdms,
                "out_dist_avg": out_dist_avg,
                "generalization_gap_percent": gen_gap_percent,
                "total_scenarios": cross_city_results.get("total_scenarios", 0),
                # Average breakdown scores (PDM components)
                "avg_nc": cross_city_results.get("avg_NC"),
                "avg_dac": cross_city_results.get("avg_DAC"),
                "avg_ep": cross_city_results.get("avg_EP"),
                "avg_ttc": cross_city_results.get("avg_TTC"),
                "avg_c": cross_city_results.get("avg_C"),
            },
            # === HPC/SLURM Information ===
            "slurm": {
                "partition": config.get("partition"),
                "gpu_constraint": config.get("gpu_constraint") or None,
                "num_gpus": _num_gpus,
                "num_nodes": config.get("num_nodes", 1),
                "cpus_per_task": config.get("cpus_per_task"),
                "mem": config.get("mem", "128G"),
                "time_limit": config.get("time_limit"),
                "account": config.get("account"),
                "train_job_id": exp_meta.get("train_job_id"),
                "eval_job_ids": exp_meta.get("eval_job_ids", {}),
            },
            # === Git Tracking ===
            "git": {
                "commit": config.get("git_commit"),
                "branch": config.get("git_branch"),
                "dirty": config.get("git_dirty", False),
            },
            # === Metadata ===
            "stored_at": datetime.now().isoformat(),
            "framework_version": "0.9.1",
        }

        # Store in database using results_storage property
        try:
            with self.results_storage as storage:
                success = storage.store(exp_id, comprehensive_data)

            if success:
                print(f"  [OK] Stored comprehensive results for {exp_id}")
            else:
                print(f"  [ERROR] Failed to store results for {exp_id}")

            return success

        except Exception as e:
            print(f"  [ERROR] Exception storing {exp_id}: {e}")
            return False

    def collect_all_results(
        self,
        status_filter: str = "completed",
        force_reharvest: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect all experiment results with comprehensive metadata

        This overrides BaseExperimentManager.collect_all_results() to use
        store_comprehensive_results() instead of basic storage.

        Args:
            status_filter: Only process experiments with this status
                           ('completed', 'all', 'running', etc.)
            force_reharvest: If True, re-harvest even if already in database
            verbose: If True, print progress messages

        Returns:
            Dict mapping exp_id to results dictionary

        Example:
            manager = NavsimExperimentManager()

            # Collect all completed experiments
            results = manager.collect_all_results()

            # Force re-harvest all experiments
            results = manager.collect_all_results(
                status_filter='all',
                force_reharvest=True
            )
        """
        results_collected = {}

        # Filter experiments by status
        experiments = []
        for exp_id, meta in self.metadata.items():
            exp_status = meta.get("status", "unknown")

            if status_filter == "all" or exp_status == status_filter:
                experiments.append(exp_id)

        if not experiments:
            if verbose:
                print(f"No experiments found with status '{status_filter}'")
            return {}

        if verbose:
            print(f"Collecting results from {len(experiments)} experiments...")
            print(f"Backend: {self.results_storage.backend_type}")

        # Process each experiment
        with self.results_storage as storage:
            for exp_id in experiments:
                # Check if already stored (unless force_reharvest)
                if not force_reharvest:
                    existing = storage.get(exp_id)
                    if existing:
                        if verbose:
                            print(f"  [SKIP] {exp_id} (already in database)")
                        results_collected[exp_id] = existing.get("evaluation", {})
                        continue

                # Harvest and store with comprehensive metadata
                if verbose:
                    print(f"  [HARVEST] {exp_id}...")

                success = self.store_comprehensive_results(exp_id)

                if success:
                    # Get stored data for return value
                    stored_data = storage.get(exp_id)
                    if stored_data:
                        results_collected[exp_id] = stored_data.get("evaluation", {})
                else:
                    results_collected[exp_id] = {}

        if verbose:
            print(f"\n[OK] Collected results from {len(results_collected)} experiments")
            if self.results_storage.backend_type == "sqlite":
                print(f"     Database: {self.results_storage.backend.db_path}")
            else:
                print(f"     Database: {self.results_storage.backend_type} (remote)")

        return results_collected

    # -------------------------------------------------------------------------
    # Run History Tracking (versioned per-run results)
    # -------------------------------------------------------------------------

    def _parse_eval_dir_metadata(self, dir_name: str) -> Dict[str, Any]:
        """
        Extract structured metadata from an evaluation directory name.

        Handles multiple naming patterns:
          - {exp_id}_eval_{split}_{city}_{date}_{time}_{job_id}
          - {exp_id}_eval_{split}_{date}_{time}_{job_id}
          - {exp_id}_eval_{split}_{city}_{date}_{time}  (older, no job_id)
          - {exp_id}_eval_{split}_{city}_{date}  (oldest format)

        Returns:
            Dict with keys: exp_id, eval_split, city, timestamp, slurm_job_id
        """
        cities = {"boston", "vegas", "pittsburgh", "singapore", "all"}
        parts = dir_name.split("_eval_")

        if len(parts) != 2:
            return {
                "exp_id": dir_name,
                "eval_split": None,
                "city": None,
                "timestamp": None,
                "slurm_job_id": None,
            }

        exp_id = parts[0]
        remainder = parts[
            1
        ]  # e.g., "navtest_boston_20260204_143053" or "navtest_20260305_232754_3469533"

        tokens = remainder.split("_")
        eval_split = tokens[0] if tokens else None  # "navtest"

        city = None
        date_start_idx = 1

        # Check if token after split is a city name
        if len(tokens) > 1 and tokens[1] in cities:
            city = tokens[1]
            date_start_idx = 2

        # Extract timestamp (YYYYMMDD_HHMMSS)
        timestamp = None
        slurm_job_id = None

        date_tokens = tokens[date_start_idx:]
        if len(date_tokens) >= 2:
            timestamp = f"{date_tokens[0]}_{date_tokens[1]}"
            # Remaining token(s) after timestamp are the SLURM job ID
            if len(date_tokens) >= 3:
                slurm_job_id = date_tokens[2]

        return {
            "exp_id": exp_id,
            "eval_split": eval_split,
            "city": city,
            "timestamp": timestamp,
            "slurm_job_id": slurm_job_id,
        }

    def _resolve_checkpoint_for_eval(self, exp_id: str) -> Dict[str, Any]:
        """
        Resolve checkpoint information for an experiment.

        Reads from experiments/checkpoints/{exp_id}.txt which stores the
        checkpoint path used by the evaluation.

        Returns:
            Dict with checkpoint_path and checkpoint_epoch
        """
        ckpt_file = self.experiments_root / "checkpoints" / f"{exp_id}.txt"
        result = {"checkpoint_path": None, "checkpoint_epoch": None}

        if ckpt_file.exists():
            try:
                ckpt_path = ckpt_file.read_text().strip()
                result["checkpoint_path"] = ckpt_path

                # Try to extract epoch from checkpoint filename
                # Pattern: epoch=49-step=12345.ckpt or last.ckpt
                epoch_match = re.search(r"epoch=(\d+)", ckpt_path)
                if epoch_match:
                    result["checkpoint_epoch"] = int(epoch_match.group(1))
                elif "last.ckpt" in ckpt_path:
                    result["checkpoint_epoch"] = -1  # Sentinel for "last"
            except Exception:
                pass

        return result

    def store_run_results(self, exp_id: str, force: bool = False) -> int:
        """
        Store all evaluation runs for a single experiment.

        Discovers every evaluation directory for exp_id, parses its CSV results,
        and stores a run record for each unique (exp_id, timestamp, city-group).
        Runs are grouped by timestamp so that a single training run that evaluated
        across multiple cities is stored as one run record.

        Args:
            exp_id: Experiment ID
            force: If True, overwrite existing run records

        Returns:
            Number of new run records stored
        """
        from datetime import datetime
        import pandas as pd

        if exp_id not in self.metadata:
            print(f"  [SKIP] {exp_id}: not in metadata")
            return 0

        exp_meta = self.metadata[exp_id]
        _config_yaml = self.configs_dir / f"{exp_id}.yaml"
        if _config_yaml.exists():
            with open(_config_yaml) as _f:
                config = yaml.safe_load(_f) or {}
        else:
            config = exp_meta.get("config", {})

        # Find all evaluation directories for this experiment
        raw_matches = list(self.experiments_root.glob(f"evaluations/*{exp_id}*"))
        eval_dirs = []
        for d in raw_matches:
            if not d.is_dir():
                continue
            d_name = d.name
            # Strict matching to avoid A3 matching A3-b etc.
            parsed = self._parse_eval_dir_metadata(d_name)
            if parsed["exp_id"] == exp_id:
                eval_dirs.append(d)

        if not eval_dirs:
            return 0

        # Group eval dirs by timestamp to combine per-city evals into one run
        runs_by_timestamp = {}
        for d in eval_dirs:
            meta = self._parse_eval_dir_metadata(d.name)
            ts = meta.get("timestamp", d.name)
            if ts not in runs_by_timestamp:
                runs_by_timestamp[ts] = {
                    "dirs": [],
                    "meta": meta,
                    "cities": {},
                }
            city = meta.get("city") or "all"
            runs_by_timestamp[ts]["dirs"].append(d)
            runs_by_timestamp[ts]["cities"][city] = d

        # Resolve checkpoint info
        ckpt_info = self._resolve_checkpoint_for_eval(exp_id)

        stored = 0

        with self.run_results_storage as storage:
            for ts, run_group in sorted(runs_by_timestamp.items()):
                # Build a unique run_id from (exp_id, timestamp, job_id)
                first_meta = run_group["meta"]
                job_id = first_meta.get("slurm_job_id", "")
                run_id = f"{exp_id}__{ts}"
                if job_id:
                    run_id += f"__{job_id}"

                # Check if already stored
                if not force:
                    existing = storage.get(run_id)
                    if existing:
                        continue

                # Parse evaluation results from CSV for each city
                city_results = {}
                for city, eval_dir in run_group["cities"].items():
                    try:
                        csv_files = sorted(
                            eval_dir.glob("*.csv"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if not csv_files:
                            continue

                        df = pd.read_csv(csv_files[0])
                        if "token" in df.columns:
                            df = df[df["token"] != "average_all_frames"]
                        if df.empty:
                            continue

                        # Extract metrics
                        col_map = {
                            "NC": "no_at_fault_collisions",
                            "DAC": "drivable_area_compliance",
                            "EP": "ego_progress",
                            "TTC": "time_to_collision_within_bound",
                            "C": "history_comfort",
                            "PDMS": "score",
                        }
                        metrics = {}
                        for short, col in col_map.items():
                            if col in df.columns:
                                vals = df[col].dropna()
                                if len(vals) > 0:
                                    metrics[short] = float(vals.mean())

                        # Comfort average
                        if (
                            "history_comfort" in df.columns
                            and "two_frame_extended_comfort" in df.columns
                        ):
                            comfort_avg = (
                                df["history_comfort"].fillna(1.0).mean()
                                + df["two_frame_extended_comfort"].fillna(1.0).mean()
                            ) / 2
                            metrics["C"] = float(comfort_avg)

                        city_results[city] = {
                            "pdms": metrics.get("PDMS", 0.0),
                            "scenarios": len(df),
                            "NC": metrics.get("NC"),
                            "DAC": metrics.get("DAC"),
                            "EP": metrics.get("EP"),
                            "TTC": metrics.get("TTC"),
                            "C": metrics.get("C"),
                            "csv_file": str(csv_files[0].name),
                            "eval_dir": str(eval_dir.name),
                        }
                    except Exception as e:
                        city_results[city] = {
                            "pdms": 0.0,
                            "error": str(e),
                            "eval_dir": str(eval_dir.name),
                        }

                if not city_results:
                    continue

                # Compute aggregate PDMS across cities (exclude "all" to avoid double-counting)
                per_city = {k: v for k, v in city_results.items() if k != "all"}
                source = per_city if per_city else city_results
                valid_scores = [
                    v["pdms"] for v in source.values() if v.get("pdms", 0) > 0
                ]
                avg_pdms = (
                    (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
                )

                # Build the run record
                run_record = {
                    "run_id": run_id,
                    "exp_id": exp_id,
                    "eval_timestamp": ts,
                    "harvested_at": datetime.now().isoformat(),
                    # Checkpoint info
                    "checkpoint_path": ckpt_info.get("checkpoint_path"),
                    "checkpoint_epoch": ckpt_info.get("checkpoint_epoch"),
                    # Git info (from config at creation/submit time)
                    "git_commit": config.get("git_commit"),
                    "git_branch": config.get("git_branch"),
                    "git_dirty": config.get("git_dirty"),
                    # SLURM info
                    "slurm_train_job_id": exp_meta.get("train_job_id"),
                    "slurm_eval_job_id": first_meta.get("slurm_job_id"),
                    "eval_job_ids": exp_meta.get("eval_job_ids", []),
                    # Config snapshot
                    "config": {
                        "agent": config.get("agent"),
                        "backbone": config.get("backbone"),
                        "epochs": config.get("epochs"),
                        "batch_size": config.get("batch_size"),
                        "learning_rate": config.get("learning_rate"),
                        "city": config.get("city", "all"),
                        "train_split": config.get("train_split"),
                        "eval_type": config.get("eval_type"),
                        "partition": config.get("partition"),
                        "num_gpus": config.get("num_gpus"),
                    },
                    # Results
                    "cities": city_results,
                    "avg_pdms": avg_pdms,
                    "total_scenarios": sum(
                        v.get("scenarios", 0) for v in city_results.values()
                    ),
                    # Status
                    "status": exp_meta.get("status", "unknown"),
                    "description": config.get("description", ""),
                }

                success = storage.store(run_id, run_record)
                if success:
                    stored += 1

        return stored

    def collect_all_run_results(
        self,
        exp_ids: List[str] = None,
        force: bool = False,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """
        Collect per-run results for all (or specified) experiments.

        Discovers every evaluation directory, groups by experiment and timestamp,
        and stores a run record for each. Existing runs are skipped unless force=True.

        Args:
            exp_ids: Specific experiment IDs to process (default: all)
            force: Re-harvest existing runs
            verbose: Print progress

        Returns:
            Dict mapping exp_id to number of new runs stored
        """
        if exp_ids is None:
            exp_ids = list(self.metadata.keys())

        if verbose:
            print(f"Collecting run history for {len(exp_ids)} experiments...")

        results = {}
        total_runs = 0

        for exp_id in sorted(exp_ids):
            count = self.store_run_results(exp_id, force=force)
            results[exp_id] = count
            total_runs += count
            if verbose and count > 0:
                print(f"  [OK] {exp_id}: {count} new runs stored")

        if verbose:
            print(
                f"\n[OK] Stored {total_runs} new runs across {len(results)} experiments"
            )

        return results

    def show_run_history(self, exp_id: str, output_json: bool = False) -> List[Dict]:
        """
        Show all stored runs for an experiment.

        Args:
            exp_id: Experiment ID
            output_json: If True, output JSON instead of table

        Returns:
            List of run records
        """
        with self.run_results_storage as storage:
            all_runs = storage.query()

        # Filter by exp_id
        runs = [r for r in all_runs if r.get("exp_id") == exp_id]
        runs.sort(key=lambda r: r.get("eval_timestamp", ""))

        if output_json:
            print(json.dumps(runs, indent=2))
            return runs

        if not runs:
            print(f"No run history found for {exp_id}")
            return []

        # Print formatted table
        print(f"\n{'='*90}")
        print(f"  Run History for {exp_id}  ({len(runs)} runs)")
        print(f"{'='*90}")
        print(f"  {'Run ID':<40} {'Checkpoint':<20} {'Git':<10} {'PDMS':>8} {'Date'}")
        print(f"  {'-'*40} {'-'*20} {'-'*10} {'-'*8} {'-'*16}")

        for run in runs:
            run_id = run.get("run_id", "?")
            # Shorten run_id for display
            display_id = run_id.replace(f"{exp_id}__", "")
            if len(display_id) > 38:
                display_id = display_id[:35] + "..."

            ckpt = run.get("checkpoint_path", "")
            if ckpt:
                ckpt_name = Path(ckpt).name
                if len(ckpt_name) > 18:
                    ckpt_name = ckpt_name[:15] + "..."
            else:
                ckpt_name = "-"

            git = run.get("git_commit", "-") or "-"
            pdms = run.get("avg_pdms", 0.0)
            pdms_str = f"{pdms:.4f}" if pdms else "-"

            ts = run.get("eval_timestamp", "")
            # Format timestamp YYYYMMDD_HHMMSS -> YYYY-MM-DD HH:MM
            if ts and len(ts) >= 15:
                date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"
            else:
                date_str = ts or "-"

            print(
                f"  {display_id:<40} {ckpt_name:<20} {git:<10} {pdms_str:>8} {date_str}"
            )

        # Print per-city breakdown for the latest run
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
        """Get current git commit info including dirty status"""
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

            # Check for uncommitted changes
            dirty = (
                subprocess.call(
                    ["git", "diff", "--quiet"],
                    cwd=self.project_root,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                )
                != 0
            )

            return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}
        except:
            return {"git_commit": None, "git_branch": None, "git_dirty": None}

    # -------------------------------------------------------------------------
    # ExpFlow v0.7.0 Helper Methods
    # -------------------------------------------------------------------------

    def _get_navsim_env_exports(self) -> str:
        """Generate NAVSIM environment variable exports from hpc_config"""
        # Read from custom_vars (parsed separately during init)
        navsim_devkit = self.custom_vars.get(
            "navsim_devkit_root",
            f"/scratch/{self.username}/navsim-ssl-city-generalization/navsim",
        )
        openscene_data = self.custom_vars.get(
            "openscene_data_root", f"/scratch/{self.username}/data"
        )
        nuplan_maps = self.custom_vars.get(
            "nuplan_maps_root", f"/scratch/{self.username}/data/maps"
        )

        if self.hpc_config:
            navsim_exp = self.hpc_config.experiments_dir
        else:
            navsim_exp = f"/scratch/{self.username}/experiments"

        return f'''export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="{navsim_devkit}"
export OPENSCENE_DATA_ROOT="{openscene_data}"
export NUPLAN_MAPS_ROOT="{nuplan_maps}"
export NAVSIM_EXP_ROOT="{navsim_exp}"
export DP_PREDS="none"'''

    def _generate_conda_activation(self, config: Dict[str, Any] = None) -> str:
        """
        Generate conda/environment activation commands (ExpFlow v0.7.0)
        Uses hpc_config for conda_root and conda_env
        """
        if self.hpc_config:
            conda_root = getattr(self.hpc_config, "conda_root", None)
            conda_env = getattr(self.hpc_config, "conda_env", "navsim")
            module_loads = getattr(self.hpc_config, "module_loads", [])
        else:
            conda_root = f"/scratch/{self.username}/miniconda3"
            conda_env = "navsim"
            module_loads = []

        # Override from config if provided
        if config:
            conda_root = config.get("conda_root", conda_root)
            conda_env = config.get("conda_env", conda_env)

        script_lines = []

        # Load modules first
        if module_loads:
            for module in module_loads:
                script_lines.append(f"module load {module}")

        # Conda activation with fallback
        if conda_root and conda_env:
            script_lines.extend(
                [
                    f'CONDA_ROOT="{conda_root}"',
                    'if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then',
                    '    source "${CONDA_ROOT}/etc/profile.d/conda.sh"',
                    f"    conda activate {conda_env}",
                    "else",
                    "    module purge || true",
                    "    module load anaconda3/2025.06 || true",
                    "    source $(conda info --base)/etc/profile.d/conda.sh || true",
                    f"    conda activate {conda_env} || true",
                    "fi",
                ]
            )

        return "\n".join(script_lines)

    def _generate_gpu_monitoring(self, exp_id: str, interval: int = None) -> str:
        """
        Generate GPU monitoring commands (ExpFlow v0.7.0)
        """
        if self.hpc_config:
            enable_monitoring = getattr(self.hpc_config, "enable_gpu_monitoring", True)
            default_interval = getattr(self.hpc_config, "gpu_monitor_interval", 60)
        else:
            enable_monitoring = True
            default_interval = 60

        if not enable_monitoring:
            return "# GPU monitoring disabled"

        if interval is None:
            interval = default_interval

        log_file = f"{self.logs_dir}/output/{exp_id}_gpu_${{SLURM_JOB_ID}}.csv"

        return f"""# Start GPU monitoring
nvidia-smi \\
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total \\
    --format=csv \\
    -l {interval} \\
    > {log_file} &
GPU_MONITOR_PID=$!

# Cleanup function
cleanup_gpu_monitor() {{
    if [ ! -z "${{GPU_MONITOR_PID}}" ]; then
        kill ${{GPU_MONITOR_PID}} 2>/dev/null || true
    fi
}}
trap cleanup_gpu_monitor EXIT"""

    def _get_nccl_env_vars(
        self, partition: str = None, preset: str = None
    ) -> Dict[str, str]:
        """
        Get NCCL optimization environment variables (ExpFlow v0.7.0)
        Auto-detects GPU type from partition name
        """
        if self.hpc_config:
            nccl_preset = preset or getattr(self.hpc_config, "nccl_preset", None)
            custom_vars = dict(getattr(self.hpc_config, "nccl_env_vars", {}))
        else:
            nccl_preset = preset
            custom_vars = {}

        # Auto-detect from partition if no preset
        if nccl_preset is None and partition:
            if "h200" in partition.lower():
                nccl_preset = "h200"
            elif "a100" in partition.lower():
                nccl_preset = "a100"
            elif "l40s" in partition.lower():
                nccl_preset = "l40s"
            elif "rtx8000" in partition.lower():
                nccl_preset = "rtx8000"

        # Preset configurations
        presets = {
            "h200": {
                "NCCL_IB_DISABLE": "0",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_NET_GDR_LEVEL": "2",
                "CUDA_LAUNCH_BLOCKING": "0",
                "TORCH_CUDNN_V8_API_ENABLED": "1",
            },
            "a100": {
                "NCCL_IB_DISABLE": "0",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_NET_GDR_LEVEL": "1",
                "CUDA_LAUNCH_BLOCKING": "0",
                "TORCH_CUDNN_V8_API_ENABLED": "1",
            },
            "l40s": {
                "NCCL_IB_DISABLE": "0",
                "NCCL_P2P_LEVEL": "SYS",
                "NCCL_NET_GDR_LEVEL": "1",
                "CUDA_LAUNCH_BLOCKING": "0",
            },
            "rtx8000": {
                "NCCL_IB_DISABLE": "0",
                "NCCL_P2P_LEVEL": "SYS",
                "NCCL_NET_GDR_LEVEL": "0",
                "CUDA_LAUNCH_BLOCKING": "0",
            },
        }

        # Start with preset, then override with custom
        nccl_vars = {}
        if nccl_preset and nccl_preset in presets:
            nccl_vars.update(presets[nccl_preset])
        nccl_vars.update(custom_vars)

        return nccl_vars

    def _generate_nccl_exports(self, partition: str = None) -> str:
        """Generate NCCL export statements"""
        nccl_vars = self._get_nccl_env_vars(partition=partition)
        if not nccl_vars:
            return "# No NCCL optimizations configured"

        lines = ["# NCCL optimizations (auto-detected from partition)"]
        for key, value in nccl_vars.items():
            lines.append(f"export {key}={value}")
        return "\n".join(lines)

    def _get_overlay_path(self, cache_name: str) -> str:
        """Get the path to a SquashFS overlay"""
        if self.hpc_config:
            overlay_dir = getattr(
                self.hpc_config,
                "overlay_cache_dir",
                f"{self.hpc_config.cache_dir}/overlays",
            )
        else:
            overlay_dir = f"/scratch/{self.username}/experiments/cache/overlays"
        return f"{overlay_dir}/{cache_name}.sqsh"

    def _check_overlay_availability(self, cache_name: str) -> bool:
        """Check if a SquashFS overlay exists"""
        overlay_path = self._get_overlay_path(cache_name)
        return Path(overlay_path).exists()

    def _get_container_image(self) -> Optional[str]:
        """Get the configured container image"""
        if self.hpc_config:
            return getattr(self.hpc_config, "container_image", None)
        return "/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

    def _get_scratch_dir(self) -> str:
        """Get the scratch directory"""
        if self.hpc_config:
            return self.hpc_config.scratch_dir
        return f"/scratch/{self.username}"

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
        elif agent == "diffusiondrive_ijepa_agent":
            return self._generate_diffusiondrive_train_script(config, exp_id, timestamp)
        elif agent == "law_agent":
            return self._generate_law_train_script(config, exp_id, timestamp)

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
        mem = config.get("mem", "128G")
        gpu_constraint = config.get("gpu_constraint", "")

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
{'#SBATCH --partition=' + partition if not gpu_constraint else '# Partition skipped (using --constraint)'}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --gres=gpu:{num_gpus}
{'#SBATCH --constraint="' + gpu_constraint + '"' if gpu_constraint else '# No GPU constraint'}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
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

# Environment setup (from hpc_config)
export HYDRA_FULL_ERROR=1
{self._get_navsim_env_exports()}
export DP_PREDS="none"

# Optional: local I-JEPA pretraining checkpoint
export IJEPA_CKPT_PATH="{ijepa_ckpt_path}"
export IJEPA_CKPT_WHICH="{ijepa_ckpt_which}"

# Resume checkpoint path (for resuming interrupted training)
export RESUME_CKPT_PATH="{resume_checkpoint_path}"

# SquashFS overlay for training cache
export CACHE_NAME="{cache_name}"
export CACHE_PATH="${{NAVSIM_EXP_ROOT}}/cache/${{CACHE_NAME}}"
export CACHE_OVERLAY="{self._get_overlay_path(cache_name)}"

# Container configuration
export CONTAINER="{self._get_container_image() or '/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif'}"

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

# NCCL optimizations (auto-detected from partition)
{self._generate_nccl_exports(partition)}

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

# Conda activation (from hpc_config)
{self._generate_conda_activation(config)}

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

# GPU monitoring (from hpc_config)
{self._generate_gpu_monitoring(exp_id)}

# =============================================================================
# TRAINING
# =============================================================================
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/train_{exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'TRAIN_SCRIPT_EOF'
#!/bin/bash
# Conda activation inside container
CONDA_ROOT="{self.hpc_config.conda_root if self.hpc_config else f'/scratch/{self.username}/miniconda3'}"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate {self.hpc_config.conda_env if self.hpc_config else 'navsim'}
fi
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
    trainer.params.strategy=ddp_find_unused_parameters_true \\
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
        --bind {self._get_scratch_dir()}:{self._get_scratch_dir()} \\
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
        trainer.params.strategy=ddp_find_unused_parameters_true \\
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
        mem = config.get("mem", "64G")

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
#SBATCH --mem={mem}
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

# Environment setup (via hpc_config)
{self._get_navsim_env_exports()}

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

# Conda activation (via hpc_config)
{self._generate_conda_activation(config)}

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# GPU monitoring with cleanup trap
{self._generate_gpu_monitoring(exp_id)}

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

# GPU monitoring cleanup is handled by trap

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

    def _generate_law_train_script(
        self, config: Dict, exp_id: str, timestamp: str
    ) -> str:
        """
        Generate training script for law_agent.
        LAW uses vision with optional world-model loss and supports cache usage.
        """
        partition = config.get("partition", "l40s_public")
        account = config.get("account", "torch_pr_68_general")
        num_gpus = config.get("num_gpus", 4)
        num_nodes = config.get("num_nodes", 1)
        cpus_per_task = config.get("cpus_per_task", 16)
        time_limit = config.get("time_limit", "24:00:00")
        mem = config.get("mem", "128G")
        gpu_constraint = config.get("gpu_constraint", "")

        # Training hyperparameters
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 1e-4)
        weight_decay = config.get("weight_decay", 0.0)
        epochs = config.get("epochs", 20)
        num_workers = config.get("num_workers", 12)

        # LAW-specific settings
        camera_width = config.get("camera_width", 640)
        camera_height = config.get("camera_height", 320)
        use_wm = str(config.get("use_wm", True)).lower()
        use_wm_training = str(config.get("use_wm_training", True)).lower()
        use_cosine_scheduler = str(config.get("use_cosine_scheduler", False)).lower()
        freeze_encoder = str(config.get("freeze_encoder", False)).lower()

        # LID depth encoding settings (replaces old PETR)
        num_views = config.get("num_views", 6)
        depth_num = config.get("depth_num", 64)
        position_range = config.get(
            "position_range", [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        )
        # Format list for Hydra: [a,b,c] without spaces to avoid parsing issues
        position_range_str = str(position_range).replace(" ", "")

        # LAW architecture settings
        encoder_lr_mult = config.get("encoder_lr_mult", 0.1)
        tf_d_model = config.get("tf_d_model", 256)
        tf_d_ffn = config.get("tf_d_ffn", 1024)
        tf_num_head = config.get("tf_num_head", 8)
        tf_dropout = config.get("tf_dropout", 0.0)
        traj_loss_weight = config.get("traj_loss_weight", 1.0)
        wm_loss_weight = config.get("wm_loss_weight", 0.2)
        wm_num_layers = config.get("wm_num_layers", 2)
        num_proposals = config.get("num_proposals", 8)
        image_architecture = config.get("image_architecture", "resnet34")
        pretrained_encoder = str(config.get("pretrained_encoder", True)).lower()
        encoder_weights_path = config.get("encoder_weights_path", None)
        min_lr_ratio = config.get("min_lr_ratio", 1e-3)

        # SSL Backbone configuration (same pattern as TransFuser)
        backbone = config.get("backbone", "resnet")
        use_native_resolution = config.get("use_native_resolution", True)
        vit_backend = config.get("vit_backend", "huggingface")
        backbone_hydra_args = f'agent.config.backbone="{backbone}"'
        backbone_hydra_args += (
            f" agent.config.use_native_resolution="
            f"{str(use_native_resolution).lower()}"
        )
        backbone_hydra_args += f' agent.config.vit_backend="{vit_backend}"'
        # Model paths (only override if explicitly set in experiment config)
        for key in [
            "ijepa_model_id",
            "dinov2_model_id",
            "dino_model_id",
            "mae_model_id",
            "ijepa_checkpoint_path",
            "dinov2_checkpoint_path",
            "mae_checkpoint_path",
        ]:
            if key in config:
                backbone_hydra_args += f' agent.config.{key}="{config[key]}"'
        # String config values (checkpoint keys)
        for key in [
            "ijepa_checkpoint_key",
            "dinov2_checkpoint_key",
            "mae_checkpoint_key",
        ]:
            if key in config:
                backbone_hydra_args += f" agent.config.{key}=" f'"{config[key]}"'
        # Trainable layer fractions
        for key in [
            "pc_trainable_ijepa_layers",
            "pc_trainable_dino_layers",
            "pc_trainable_dinov2_layers",
            "pc_trainable_mae_layers",
        ]:
            if key in config:
                backbone_hydra_args += f" agent.config.{key}={config[key]}"
        # Architecture overrides for custom ViT backend (e.g. ViT-S/14 nuScenes weights)
        if config.get("vit_arch"):
            backbone_hydra_args += f' agent.config.vit_arch="{config["vit_arch"]}"'
        if config.get("vit_patch_size"):
            backbone_hydra_args += (
                f" agent.config.vit_patch_size={config['vit_patch_size']}"
            )

        # Cache configuration
        use_cache = bool(config.get("use_cache", True))
        cache_name = config.get("cache_name", "law_navtrain_cache")
        train_split = config.get("train_split", "navtrain")

        # City filtering configuration
        city = config.get("city")
        city_config_args = ""
        if city:
            city_splits_dir = self.project_root / "city_splits" / "splits"
            train_logs_file = city_splits_dir / f"{city}_trainval.json"
            if train_logs_file.exists():
                # Use + prefix to add new config key (Hydra requirement)
                city_config_args = f'+city.train_logs_file="{train_logs_file}"'

        # Trainer params
        precision = config.get("precision", "16-mixed")
        # Handle null/None from YAML correctly (null → 0.0 = disabled in PL)
        gradient_clip_val_raw = config.get("gradient_clip_val")
        gradient_clip_val = (
            0.0 if gradient_clip_val_raw is None else float(gradient_clip_val_raw)
        )
        accumulate_grad_batches = config.get("accumulate_grad_batches", 1)
        prefetch_factor = config.get("prefetch_factor", 4)

        # Optional encoder weights path
        encoder_weights_hydra_arg = ""
        if encoder_weights_path:
            encoder_weights_hydra_arg = (
                f'    agent.config.encoder_weights_path="{encoder_weights_path}" \\\n'
            )

        experiment_name = f"training/{exp_id}_{timestamp}"

        cache_block = ""
        if use_cache:
            cache_block = """    cache_path=${CACHE_PATH} \\
    use_cache_without_dataset=true \\
    force_cache_computation=false \\
"""

        script = f"""#!/bin/bash
# =============================================================================
# Auto-generated by NAVSIM ExpFlow Manager
# Experiment: {exp_id} (LAW Agent)
# Description: {config.get("description", "LAW baseline")}
# Generated: {datetime.now().isoformat()}
# Git: {config.get("git_commit", "unknown")} ({config.get("git_branch", "unknown")})
# =============================================================================

# =============================================================================
# SLURM CONFIGURATION
# =============================================================================
#SBATCH --job-name={exp_id}_train
{'#SBATCH --partition=' + partition if not gpu_constraint else '# Partition skipped (using --constraint)'}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --gres=gpu:{num_gpus}
{'#SBATCH --constraint="' + gpu_constraint + '"' if gpu_constraint else '# No GPU constraint'}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --output={self.logs_dir}/output/{exp_id}_train_%j.out
#SBATCH --error={self.logs_dir}/error/{exp_id}_train_%j.err

echo "=============================================="
echo "ExpFlow Experiment: {exp_id} (LAW)"
echo "{config.get('description', 'LAW baseline')}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup (via hpc_config)
{self._get_navsim_env_exports()}

# Cache configuration
export CACHE_NAME="{cache_name}"
export CACHE_PATH="${{NAVSIM_EXP_ROOT}}/cache/${{CACHE_NAME}}"
export CACHE_OVERLAY="{self._get_overlay_path(cache_name)}"

# Container configuration
export CONTAINER="{self._get_container_image()}"

# Training hyperparameters
export AGENT="law_agent"
export BATCH_SIZE={batch_size}
export NUM_WORKERS={num_workers}
export EPOCHS={epochs}
export LEARNING_RATE={learning_rate}
export WEIGHT_DECAY={weight_decay}
export CAMERA_WIDTH={camera_width}
export CAMERA_HEIGHT={camera_height}
export USE_WM={use_wm}
export USE_WM_TRAINING={use_wm_training}
export USE_COSINE={use_cosine_scheduler}
export FREEZE_ENCODER={freeze_encoder}
export NUM_VIEWS={num_views}
export DEPTH_NUM={depth_num}
export POSITION_RANGE="{position_range_str}"
export ENCODER_LR_MULT={encoder_lr_mult}
export TF_D_MODEL={tf_d_model}
export TF_D_FFN={tf_d_ffn}
export TF_NUM_HEAD={tf_num_head}
export TF_DROPOUT={tf_dropout}
export TRAJ_LOSS_WEIGHT={traj_loss_weight}
export WM_LOSS_WEIGHT={wm_loss_weight}
export WM_NUM_LAYERS={wm_num_layers}
export NUM_PROPOSALS={num_proposals}
export IMAGE_ARCHITECTURE="{image_architecture}"
export PRETRAINED_ENCODER={pretrained_encoder}
export BACKBONE_ARGS="{backbone_hydra_args}"
export GRADIENT_CLIP_VAL={gradient_clip_val}
export EXPERIMENT_NAME="{experiment_name}"

# NCCL optimizations (auto-detected from partition)
{self._generate_nccl_exports(partition)}

# Multi-node DDP
# Use a dynamic port based on job ID to avoid conflicts on shared nodes
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo ""
echo "Configuration:"
echo "  Agent: $AGENT"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE (encoder: x$ENCODER_LR_MULT)"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Camera size: $CAMERA_WIDTH x $CAMERA_HEIGHT"
echo "  Backbone: $IMAGE_ARCHITECTURE (pretrained: $PRETRAINED_ENCODER)"
echo "  Views: $NUM_VIEWS, Proposals: $NUM_PROPOSALS"
echo "  Transformer: d_model=$TF_D_MODEL, d_ffn=$TF_D_FFN, heads=$TF_NUM_HEAD, dropout=$TF_DROPOUT"
echo "  World model: $USE_WM (train: $USE_WM_TRAINING, layers: $WM_NUM_LAYERS)"
echo "  Loss weights: traj=$TRAJ_LOSS_WEIGHT, wm=$WM_LOSS_WEIGHT"
echo "  Gradient clip: $GRADIENT_CLIP_VAL"
echo "  Cache: $CACHE_NAME"
echo ""

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Conda activation (via hpc_config)
{self._generate_conda_activation(config)}

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Clear stale __pycache__ to ensure fresh bytecode after git pull
find ${{NAVSIM_DEVKIT_ROOT}}/navsim/agents/law -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null || true

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

# GPU monitoring with cleanup trap
{self._generate_gpu_monitoring(exp_id)}

# =============================================================================
# TRAINING - LAW Agent
# =============================================================================
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/train_{exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'TRAIN_SCRIPT_EOF'
#!/bin/bash
# Conda activation inside container
CONDA_ROOT="{self.hpc_config.conda_root if self.hpc_config else f'/scratch/{self.username}/miniconda3'}"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate {self.hpc_config.conda_env if self.hpc_config else 'navsim'}
fi
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_training.py \\
    agent=law_agent \\
    ${{BACKBONE_ARGS}} \\
    agent.config.learning_rate=${{LEARNING_RATE}} \\
    agent.config.weight_decay=${{WEIGHT_DECAY}} \\
    agent.config.camera_width=${{CAMERA_WIDTH}} \\
    agent.config.camera_height=${{CAMERA_HEIGHT}} \\
    agent.config.use_wm=${{USE_WM}} \\
    agent.config.use_wm_training=${{USE_WM_TRAINING}} \\
    agent.config.use_cosine_scheduler=${{USE_COSINE}} \\
    agent.config.freeze_encoder=${{FREEZE_ENCODER}} \\
    agent.config.num_views=${{NUM_VIEWS}} \\
    agent.config.depth_num=${{DEPTH_NUM}} \\
    agent.config.position_range=${{POSITION_RANGE}} \\
    agent.config.encoder_lr_mult=${{ENCODER_LR_MULT}} \\
    agent.config.tf_d_model=${{TF_D_MODEL}} \\
    agent.config.tf_d_ffn=${{TF_D_FFN}} \\
    agent.config.tf_num_head=${{TF_NUM_HEAD}} \\
    agent.config.tf_dropout=${{TF_DROPOUT}} \\
    agent.config.traj_loss_weight=${{TRAJ_LOSS_WEIGHT}} \\
    agent.config.wm_loss_weight=${{WM_LOSS_WEIGHT}} \\
    agent.config.wm_num_layers=${{WM_NUM_LAYERS}} \\
    agent.config.num_proposals=${{NUM_PROPOSALS}} \\
    agent.config.image_architecture=${{IMAGE_ARCHITECTURE}} \\
    agent.config.pretrained_encoder=${{PRETRAINED_ENCODER}} \\
    agent.config.max_epochs=${{EPOCHS}} \\
    experiment_name="${{EXPERIMENT_NAME}}" \\
    train_test_split={train_split} \\
{cache_block}    {city_config_args + ' ' if city_config_args else ''}trainer.params.max_epochs=${{EPOCHS}} \
    trainer.params.precision={precision} \\
    trainer.params.accelerator=gpu \\
    trainer.params.strategy=ddp_find_unused_parameters_true \\
    trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
    trainer.params.gradient_clip_val=${{GRADIENT_CLIP_VAL}} \\
    trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
    dataloader.params.batch_size=${{BATCH_SIZE}} \\
    dataloader.params.num_workers=${{NUM_WORKERS}} \\
    dataloader.params.prefetch_factor={prefetch_factor} \\
    dataloader.params.pin_memory=true \\
    ${{SMOKE:+trainer.params.limit_train_batches=1}} \\
    ${{SMOKE:+trainer.params.limit_val_batches=0}}
TRAIN_SCRIPT_EOF
    chmod +x "${{TEMP_SCRIPT}}"

    # Run inside apptainer with squashfs overlay
    srun --gres=gpu:{num_gpus} apptainer exec \\
        --nv \\
        --bind "${{CACHE_OVERLAY}}:${{CACHE_PATH}}:image-src=/" \\
        --bind {self._get_scratch_dir()}:{self._get_scratch_dir()} \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "NAVSIM_DEVKIT_ROOT=${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "OPENSCENE_DATA_ROOT=${{OPENSCENE_DATA_ROOT}}" \\
        --env "NUPLAN_MAPS_ROOT=${{NUPLAN_MAPS_ROOT}}" \\
        --env "NAVSIM_EXP_ROOT=${{NAVSIM_EXP_ROOT}}" \\
        --env "CACHE_PATH=${{CACHE_PATH}}" \\
        --env "EXPERIMENT_NAME=${{EXPERIMENT_NAME}}" \\
        --env "LEARNING_RATE=${{LEARNING_RATE}}" \\
        --env "WEIGHT_DECAY=${{WEIGHT_DECAY}}" \\
        --env "CAMERA_WIDTH=${{CAMERA_WIDTH}}" \\
        --env "CAMERA_HEIGHT=${{CAMERA_HEIGHT}}" \\
        --env "USE_WM=${{USE_WM}}" \\
        --env "USE_WM_TRAINING=${{USE_WM_TRAINING}}" \\
        --env "USE_COSINE=${{USE_COSINE}}" \\
        --env "FREEZE_ENCODER=${{FREEZE_ENCODER}}" \\
        --env "BATCH_SIZE=${{BATCH_SIZE}}" \\
        --env "NUM_WORKERS=${{NUM_WORKERS}}" \\
        --env "EPOCHS=${{EPOCHS}}" \\
        --env "NUM_VIEWS=${{NUM_VIEWS}}" \\
        --env "DEPTH_NUM=${{DEPTH_NUM}}" \\
        --env "POSITION_RANGE=${{POSITION_RANGE}}" \\
        --env "ENCODER_LR_MULT=${{ENCODER_LR_MULT}}" \\
        --env "TF_D_MODEL=${{TF_D_MODEL}}" \\
        --env "TF_D_FFN=${{TF_D_FFN}}" \\
        --env "TF_NUM_HEAD=${{TF_NUM_HEAD}}" \\
        --env "TF_DROPOUT=${{TF_DROPOUT}}" \\
        --env "TRAJ_LOSS_WEIGHT=${{TRAJ_LOSS_WEIGHT}}" \\
        --env "WM_LOSS_WEIGHT=${{WM_LOSS_WEIGHT}}" \\
        --env "WM_NUM_LAYERS=${{WM_NUM_LAYERS}}" \\
        --env "NUM_PROPOSALS=${{NUM_PROPOSALS}}" \\
        --env "IMAGE_ARCHITECTURE=${{IMAGE_ARCHITECTURE}}" \\
        --env "PRETRAINED_ENCODER=${{PRETRAINED_ENCODER}}" \\
        --env "BACKBONE_ARGS=${{BACKBONE_ARGS}}" \\
        --env "GRADIENT_CLIP_VAL=${{GRADIENT_CLIP_VAL}}" \\
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
        agent=law_agent \\
        ${{BACKBONE_ARGS}} \\
        agent.config.learning_rate=${{LEARNING_RATE}} \\
        agent.config.weight_decay=${{WEIGHT_DECAY}} \\
        agent.config.camera_width=${{CAMERA_WIDTH}} \\
        agent.config.camera_height=${{CAMERA_HEIGHT}} \\
        agent.config.use_wm=${{USE_WM}} \\
        agent.config.use_wm_training=${{USE_WM_TRAINING}} \\
        agent.config.use_cosine_scheduler=${{USE_COSINE}} \\
        agent.config.freeze_encoder=${{FREEZE_ENCODER}} \\
        agent.config.num_views=${{NUM_VIEWS}} \\
        agent.config.depth_num=${{DEPTH_NUM}} \\
        agent.config.position_range=${{POSITION_RANGE}} \\
        agent.config.encoder_lr_mult=${{ENCODER_LR_MULT}} \\
        agent.config.tf_d_model=${{TF_D_MODEL}} \\
        agent.config.tf_d_ffn=${{TF_D_FFN}} \\
        agent.config.tf_num_head=${{TF_NUM_HEAD}} \\
        agent.config.tf_dropout=${{TF_DROPOUT}} \\
        agent.config.traj_loss_weight=${{TRAJ_LOSS_WEIGHT}} \\
        agent.config.wm_loss_weight=${{WM_LOSS_WEIGHT}} \\
        agent.config.wm_num_layers=${{WM_NUM_LAYERS}} \\
        agent.config.num_proposals=${{NUM_PROPOSALS}} \\
        agent.config.image_architecture=${{IMAGE_ARCHITECTURE}} \\
        agent.config.pretrained_encoder=${{PRETRAINED_ENCODER}} \\
        agent.config.max_epochs=${{EPOCHS}} \\
        experiment_name="${{EXPERIMENT_NAME}}" \\
        train_test_split={train_split} \\
    {cache_block}    {city_config_args + ' ' if city_config_args else ''}trainer.params.max_epochs=${{EPOCHS}} \
        trainer.params.precision={precision} \\
        trainer.params.accelerator=gpu \\
        trainer.params.strategy=ddp_find_unused_parameters_true \\
        trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
        trainer.params.gradient_clip_val=${{GRADIENT_CLIP_VAL}} \\
        trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
        dataloader.params.batch_size=${{BATCH_SIZE}} \\
        dataloader.params.num_workers=${{NUM_WORKERS}} \\
        dataloader.params.prefetch_factor={prefetch_factor} \\
        dataloader.params.pin_memory=true \\
        ${{SMOKE:+trainer.params.limit_train_batches=1}} \\
        ${{SMOKE:+trainer.params.limit_val_batches=0}}
fi

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
        echo "No checkpoint found."
    fi
else
    echo "No checkpoint directory found."
fi

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
        mem = config.get("mem", "128G")
        gpu_constraint = config.get("gpu_constraint", "")

        # Training hyperparameters (upstream NAVSIM defaults)
        batch_size = config.get("batch_size", 64)
        learning_rate = config.get("learning_rate", 2e-4)
        epochs = config.get("epochs", 100)
        num_workers = config.get("num_workers", 8)

        # Trainer params
        gradient_clip_val = config.get("gradient_clip_val", 0.0)
        accumulate_grad_batches = config.get("accumulate_grad_batches", 1)

        # Backbone configuration
        backbone = config.get("backbone", "resnet")
        use_native_resolution = config.get("use_native_resolution", True)
        vit_backend = config.get("vit_backend", "huggingface")
        backbone_agent_args = f'agent.config.backbone="{backbone}"'
        backbone_agent_args += (
            f" agent.config.use_native_resolution={str(use_native_resolution)}"
        )
        backbone_agent_args += f' agent.config.vit_backend="{vit_backend}"'
        # Model paths (only override if explicitly set in experiment config)
        for key in [
            "ijepa_model_id",
            "dinov2_model_id",
            "dino_model_id",
            "mae_model_id",
            "ijepa_checkpoint_path",
            "dinov2_checkpoint_path",
            "mae_checkpoint_path",
        ]:
            if key in config:
                backbone_agent_args += f' agent.config.{key}="{config[key]}"'
        # String config values (checkpoint keys)
        for key in [
            "ijepa_checkpoint_key",
            "dinov2_checkpoint_key",
            "mae_checkpoint_key",
        ]:
            if key in config:
                backbone_agent_args += f' agent.config.{key}="{config[key]}"'
        # Trainable layer fractions
        for key in [
            "pc_trainable_ijepa_layers",
            "pc_trainable_dino_layers",
            "pc_trainable_dinov2_layers",
            "pc_trainable_mae_layers",
        ]:
            if key in config:
                backbone_agent_args += f" agent.config.{key}={config[key]}"
        # Architecture overrides for custom ViT backend (e.g. ViT-S/14 nuScenes weights)
        if config.get("vit_arch"):
            backbone_agent_args += f' agent.config.vit_arch="{config["vit_arch"]}"'
        if config.get("vit_patch_size"):
            backbone_agent_args += (
                f" agent.config.vit_patch_size={config['vit_patch_size']}"
            )

        # Latent TransFuser mode (replaces LiDAR BEV with learned spatial encoding)
        latent = config.get("latent", False)
        if latent:
            backbone_agent_args += f" agent.config.latent={str(latent)}"

        # Cache configuration
        cache_name = config.get("cache_name", "training_cache_transfuser")

        # Train test split
        train_split = config.get("train_split", "navtrain")

        # City filtering configuration
        city = config.get("city", None)
        city_config_args = ""
        city_env_vars = ""
        if city:
            city_splits_dir = self.project_root / "city_splits" / "splits"
            train_logs_file = city_splits_dir / f"{city}_trainval.json"
            if train_logs_file.exists():
                # Use + prefix to add new config key (Hydra requirement)
                city_config_args = f'+city.train_logs_file="{train_logs_file}"'
                city_env_vars = f'--env "CITY={city}"'

        experiment_name = f"training/{exp_id}_{timestamp}"

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
{'#SBATCH --partition=' + partition if not gpu_constraint else '# Partition skipped (using --constraint)'}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --gres=gpu:{num_gpus}
{'#SBATCH --constraint="' + gpu_constraint + '"' if gpu_constraint else '# No GPU constraint'}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
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
# Environment setup (via hpc_config)
{self._get_navsim_env_exports()}

# SquashFS overlay for training cache
export CACHE_NAME="{cache_name}"
export CACHE_PATH="${{NAVSIM_EXP_ROOT}}/cache/${{CACHE_NAME}}"
export CACHE_OVERLAY="{self._get_overlay_path(cache_name)}"

# Container configuration
export CONTAINER="{self._get_container_image()}"

# Training hyperparameters
export AGENT="transfuser_agent"
export BATCH_SIZE={batch_size}
export NUM_WORKERS={num_workers}
export EPOCHS={epochs}
export LEARNING_RATE={learning_rate}
export GRADIENT_CLIP_VAL={gradient_clip_val}
export EXPERIMENT_NAME="{experiment_name}"

# Backbone configuration
export BACKBONE_ARGS="{backbone_agent_args}"

# NCCL optimizations (via hpc_config)
{self._generate_nccl_exports(partition)}

# Multi-node DDP
# Use a dynamic port based on job ID to avoid conflicts on shared nodes
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo ""
echo "Configuration:"
echo "  Agent: $AGENT"
echo "  Backbone: {backbone}"
echo "  ViT backend: {vit_backend}"
echo "  Native resolution: {use_native_resolution}"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient clip val: $GRADIENT_CLIP_VAL"
echo "  Cache: $CACHE_NAME"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Conda activation (via hpc_config)
{self._generate_conda_activation(config)}

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

# GPU monitoring with cleanup trap
{self._generate_gpu_monitoring(exp_id)}

# =============================================================================
# TRAINING - TransFuser Agent
# =============================================================================
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/train_{exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'TRAIN_SCRIPT_EOF'
#!/bin/bash
# Conda activation inside container
CONDA_ROOT="{self.hpc_config.conda_root if self.hpc_config else f'/scratch/{self.username}/miniconda3'}"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate {self.hpc_config.conda_env if self.hpc_config else 'navsim'}
fi
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_training.py \\
    agent=transfuser_agent \\
    agent.lr=${{LEARNING_RATE}} \\
    ${{BACKBONE_ARGS}} \\
    experiment_name=${{EXPERIMENT_NAME}} \\
    train_test_split={train_split} \\
    cache_path=${{CACHE_PATH}} \\
    use_cache_without_dataset=true \\
    force_cache_computation=false \\
    {city_config_args + ' ' if city_config_args else ''}trainer.params.max_epochs=${{EPOCHS}} \\
    trainer.params.precision=16-mixed \\
    trainer.params.accelerator=gpu \\
    trainer.params.strategy=ddp_find_unused_parameters_true \\
    trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
    trainer.params.gradient_clip_val=${{GRADIENT_CLIP_VAL}} \\
    trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
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
        --bind {self._get_scratch_dir()}:{self._get_scratch_dir()} \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "NAVSIM_DEVKIT_ROOT=${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "OPENSCENE_DATA_ROOT=${{OPENSCENE_DATA_ROOT}}" \\
        --env "NUPLAN_MAPS_ROOT=${{NUPLAN_MAPS_ROOT}}" \\
        --env "NAVSIM_EXP_ROOT=${{NAVSIM_EXP_ROOT}}" \\
        --env "CACHE_PATH=${{CACHE_PATH}}" \\
        --env "EXPERIMENT_NAME=${{EXPERIMENT_NAME}}" \\
        --env "LEARNING_RATE=${{LEARNING_RATE}}" \\
        --env "GRADIENT_CLIP_VAL=${{GRADIENT_CLIP_VAL}}" \\
        --env "BATCH_SIZE=${{BATCH_SIZE}}" \\
        --env "NUM_WORKERS=${{NUM_WORKERS}}" \\
        --env "EPOCHS=${{EPOCHS}}" \\
        --env "SLURM_JOB_NUM_NODES=${{SLURM_JOB_NUM_NODES}}" \\
        --env "MASTER_ADDR=${{MASTER_ADDR}}" \\
        --env "MASTER_PORT=${{MASTER_PORT}}" \\
        --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
        --env "NCCL_P2P_LEVEL=${{NCCL_P2P_LEVEL}}" \\
        --env "NCCL_NET_GDR_LEVEL=${{NCCL_NET_GDR_LEVEL}}" \\
        --env "BACKBONE_ARGS=${{BACKBONE_ARGS}}" \\
        "${{CONTAINER}}" \\
        bash "${{TEMP_SCRIPT}}"

    rm -f "${{TEMP_SCRIPT}}"
else
    # Run without overlay (fallback)
    srun --gres=gpu:{num_gpus} python navsim/planning/script/run_training.py \\
        agent=transfuser_agent \\
        agent.lr=${{LEARNING_RATE}} \\
        ${{BACKBONE_ARGS}} \\
        experiment_name="${{EXPERIMENT_NAME}}" \\
        train_test_split={train_split} \\
        cache_path="${{CACHE_PATH}}" \\
        use_cache_without_dataset=true \\
        force_cache_computation=false \\
        {city_config_args + ' ' if city_config_args else ''}trainer.params.max_epochs=${{EPOCHS}} \\
        trainer.params.precision=16-mixed \\
        trainer.params.accelerator=gpu \\
        trainer.params.strategy=ddp_find_unused_parameters_true \\
        trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
        trainer.params.gradient_clip_val=${{GRADIENT_CLIP_VAL}} \\
        trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
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

    def _generate_diffusiondrive_train_script(
        self, config: Dict, exp_id: str, timestamp: str
    ) -> str:
        """
        Generate training script for diffusiondrive_ijepa_agent.
        DiffusionDrive uses the same sensor config as TransFuser
        (cam_l0/f0/r0 + LiDAR BEV), so it reuses the transfuser cache.
        Paper reference: 8×GPU, total batch 512, AdamW lr=6e-4, 100 epochs.
        """
        partition = config.get("partition", "h200_tandon")
        account = config.get("account", "torch_pr_68_tandon_advanced")
        num_gpus = config.get("num_gpus", 4)
        num_nodes = config.get("num_nodes", 1)
        cpus_per_task = config.get("cpus_per_task", 16)
        time_limit = config.get("time_limit", "24:00:00")
        mem = config.get("mem", "128G")
        gpu_constraint = config.get("gpu_constraint", "")

        # Training hyperparameters — DD paper defaults
        batch_size = config.get("batch_size", 64)
        learning_rate = config.get("learning_rate", 6e-4)
        epochs = config.get("epochs", 100)
        num_workers = config.get("num_workers", 12)

        # Trainer params
        gradient_clip_val = config.get("gradient_clip_val", 1.0)
        accumulate_grad_batches = config.get("accumulate_grad_batches", 2)

        # DiffusionDrive-specific overrides
        num_anchors = config.get("num_anchors", 20)
        num_denoising_steps = config.get("num_denoising_steps", 2)
        plan_anchor_path = config.get(
            "plan_anchor_path",
            f"/scratch/{self.username}/data/models/kmeans_navsim_traj_{num_anchors}.npy",
        )
        # plan_anchor_path default is set in the Hydra YAML + dataclass;
        # only override via CLI if the user explicitly sets a non-default path
        dd_agent_args = ""
        default_anchor = (
            f"/scratch/{self.username}/data/models/kmeans_navsim_traj_{num_anchors}.npy"
        )
        if plan_anchor_path != default_anchor:
            dd_agent_args = f"+agent.config.plan_anchor_path='{plan_anchor_path}'"

        # Checkpoint path (for fine-tuning / resume)
        checkpoint_path = config.get("checkpoint_path", None)
        if checkpoint_path:
            dd_agent_args += f" agent.checkpoint_path='{checkpoint_path}'"

        # ── SSL backbone passthrough ──
        backbone = config.get("backbone", "resnet")
        if backbone != "resnet":
            dd_agent_args += f" agent.config.backbone={backbone}"
        vit_backend = config.get("vit_backend", "huggingface")
        if vit_backend != "huggingface":
            dd_agent_args += f" agent.config.vit_backend={vit_backend}"
        use_native_resolution = config.get("use_native_resolution", True)
        if not use_native_resolution:
            dd_agent_args += " agent.config.use_native_resolution=false"

        # Model IDs (only override if explicitly set and different from YAML defaults)
        for model_key in (
            "ijepa_model_id",
            "dinov2_model_id",
            "dino_model_id",
            "mae_model_id",
        ):
            model_val = config.get(model_key, None)
            if model_val:
                dd_agent_args += f" agent.config.{model_key}='{model_val}'"

        # Trainable fraction overrides
        for frac_key in (
            "pc_trainable_ijepa_layers",
            "pc_trainable_dino_layers",
            "pc_trainable_dinov2_layers",
            "pc_trainable_mae_layers",
        ):
            frac_val = config.get(frac_key, None)
            if frac_val is not None:
                dd_agent_args += f" agent.config.{frac_key}={frac_val}"

        # Custom ViT checkpoint paths
        for ckpt_key in (
            "ijepa_checkpoint_path",
            "dinov2_checkpoint_path",
            "mae_checkpoint_path",
        ):
            ckpt_val = config.get(ckpt_key, None)
            if ckpt_val:
                dd_agent_args += f" agent.config.{ckpt_key}='{ckpt_val}'"
        for key_key in (
            "ijepa_checkpoint_key",
            "dinov2_checkpoint_key",
            "mae_checkpoint_key",
        ):
            key_val = config.get(key_key, None)
            if key_val:
                dd_agent_args += f" agent.config.{key_key}='{key_val}'"

        # Architecture overrides
        vit_arch = config.get("vit_arch", "")
        if vit_arch:
            dd_agent_args += f" agent.config.vit_arch={vit_arch}"
        vit_patch_size = config.get("vit_patch_size", 0)
        if vit_patch_size > 0:
            dd_agent_args += f" agent.config.vit_patch_size={vit_patch_size}"

        # Cache configuration — reuses transfuser cache
        cache_name = config.get("cache_name", "transfuser_navtrain_cache")
        train_split = config.get("train_split", "navtrain")

        # City filtering
        city = config.get("city", None)
        city_config_args = ""
        if city:
            city_splits_dir = self.project_root / "city_splits" / "splits"
            train_logs_file = city_splits_dir / f"{city}_trainval.json"
            if train_logs_file.exists():
                city_config_args = f'+city.train_logs_file="{train_logs_file}"'

        experiment_name = f"training/{exp_id}_{timestamp}"

        script = f"""#!/bin/bash
# =============================================================================
# Auto-generated by NAVSIM ExpFlow Manager
# Experiment: {exp_id} (DiffusionDrive Agent)
# Description: {config.get("description", "DiffusionDrive baseline")}
# Generated: {datetime.now().isoformat()}
# Git: {config.get("git_commit", "unknown")} ({config.get("git_branch", "unknown")})
# Paper: DiffusionDrive (CVPR 2025) — 88.1 PDMS, ResNet-34 + LiDAR
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
{'#SBATCH --constraint="' + gpu_constraint + '"' if gpu_constraint else '# No GPU constraint'}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --output={self.logs_dir}/output/{exp_id}_train_%j.out
#SBATCH --error={self.logs_dir}/error/{exp_id}_train_%j.err

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
echo "=============================================="
echo "DiffusionDrive Experiment: {exp_id}"
echo "{config.get('description', 'DiffusionDrive baseline')}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
{self._get_navsim_env_exports()}

# SquashFS overlay for training cache
export CACHE_NAME="{cache_name}"
export CACHE_PATH="${{NAVSIM_EXP_ROOT}}/cache/${{CACHE_NAME}}"
export CACHE_OVERLAY="{self._get_overlay_path(cache_name)}"

# Container configuration
export CONTAINER="{self._get_container_image()}"

# Training hyperparameters
export AGENT="diffusiondrive_ijepa_agent"
export BATCH_SIZE={batch_size}
export NUM_WORKERS={num_workers}
export EPOCHS={epochs}
export LEARNING_RATE={learning_rate}
export GRADIENT_CLIP_VAL={gradient_clip_val}
export EXPERIMENT_NAME="{experiment_name}"
export DD_AGENT_ARGS="{dd_agent_args}"

# NCCL optimizations
{self._generate_nccl_exports(partition)}

# Multi-node DDP
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo ""
echo "Configuration:"
echo "  Agent: $AGENT"
echo "  Anchors: {num_anchors} (hardcoded)  Denoising steps: {num_denoising_steps} (hardcoded)"
echo "  Epochs: $EPOCHS"
echo "  Batch size per GPU: $BATCH_SIZE  Accumulate: {accumulate_grad_batches}  Effective: $(( {batch_size} * {num_gpus} * {num_nodes} * {accumulate_grad_batches} ))"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient clip: $GRADIENT_CLIP_VAL"
echo "  Cache: $CACHE_NAME"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Conda activation
{self._generate_conda_activation(config)}

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

# GPU monitoring with cleanup trap
{self._generate_gpu_monitoring(exp_id)}

# =============================================================================
# TRAINING - DiffusionDrive Agent
# =============================================================================
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/train_{exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'TRAIN_SCRIPT_EOF'
#!/bin/bash
# Conda activation inside container
CONDA_ROOT="{self.hpc_config.conda_root if self.hpc_config else f'/scratch/{self.username}/miniconda3'}"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate {self.hpc_config.conda_env if self.hpc_config else 'navsim'}
fi
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_training.py \\
    agent=diffusiondrive_ijepa_agent \\
    agent.lr=${{LEARNING_RATE}} \\
    ${{DD_AGENT_ARGS}} \\
    experiment_name=${{EXPERIMENT_NAME}} \\
    train_test_split={train_split} \\
    cache_path=${{CACHE_PATH}} \\
    use_cache_without_dataset=true \\
    force_cache_computation=false \\
    {city_config_args + ' ' if city_config_args else ''}trainer.params.max_epochs=${{EPOCHS}} \\
    trainer.params.precision=16-mixed \\
    trainer.params.accelerator=gpu \\
    trainer.params.strategy=ddp_find_unused_parameters_true \\
    trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
    trainer.params.gradient_clip_val=${{GRADIENT_CLIP_VAL}} \\
    trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
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
        --bind {self._get_scratch_dir()}:{self._get_scratch_dir()} \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "NAVSIM_DEVKIT_ROOT=${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "OPENSCENE_DATA_ROOT=${{OPENSCENE_DATA_ROOT}}" \\
        --env "NUPLAN_MAPS_ROOT=${{NUPLAN_MAPS_ROOT}}" \\
        --env "NAVSIM_EXP_ROOT=${{NAVSIM_EXP_ROOT}}" \\
        --env "CACHE_PATH=${{CACHE_PATH}}" \\
        --env "EXPERIMENT_NAME=${{EXPERIMENT_NAME}}" \\
        --env "LEARNING_RATE=${{LEARNING_RATE}}" \\
        --env "GRADIENT_CLIP_VAL=${{GRADIENT_CLIP_VAL}}" \\
        --env "BATCH_SIZE=${{BATCH_SIZE}}" \\
        --env "NUM_WORKERS=${{NUM_WORKERS}}" \\
        --env "EPOCHS=${{EPOCHS}}" \\
        --env "DD_AGENT_ARGS=${{DD_AGENT_ARGS}}" \\
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
        agent=diffusiondrive_ijepa_agent \\
        agent.lr=${{LEARNING_RATE}} \\
        ${{DD_AGENT_ARGS}} \\
        experiment_name="${{EXPERIMENT_NAME}}" \\
        train_test_split={train_split} \\
        cache_path="${{CACHE_PATH}}" \\
        use_cache_without_dataset=true \\
        force_cache_computation=false \\
        {city_config_args + ' ' if city_config_args else ''}trainer.params.max_epochs=${{EPOCHS}} \\
        trainer.params.precision=16-mixed \\
        trainer.params.accelerator=gpu \\
        trainer.params.strategy=ddp_find_unused_parameters_true \\
        trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
        trainer.params.gradient_clip_val=${{GRADIENT_CLIP_VAL}} \\
        trainer.params.accumulate_grad_batches={accumulate_grad_batches} \\
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
        CKPT_REGISTRY="${{NAVSIM_EXP_ROOT}}/checkpoints"
        mkdir -p "${{CKPT_REGISTRY}}"
        echo "${{BEST_CKPT}}" > "${{CKPT_REGISTRY}}/{exp_id}.txt"
        echo "Checkpoint registered: ${{CKPT_REGISTRY}}/{exp_id}.txt"
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

        # Backbone configuration for TransFuser agent
        backbone_eval_args = ""
        if agent == "transfuser_agent":
            backbone = config.get("backbone", "resnet")
            use_native_resolution = config.get("use_native_resolution", True)
            vit_backend = config.get("vit_backend", "huggingface")
            backbone_eval_args = f'agent.config.backbone="{backbone}" agent.config.use_native_resolution={str(use_native_resolution)}'
            backbone_eval_args += f' agent.config.vit_backend="{vit_backend}"'
            for key in [
                "ijepa_model_id",
                "dinov2_model_id",
                "dino_model_id",
                "mae_model_id",
                "ijepa_checkpoint_path",
                "dinov2_checkpoint_path",
                "mae_checkpoint_path",
            ]:
                if key in config:
                    backbone_eval_args += f' agent.config.{key}="{config[key]}"'
            for key in [
                "ijepa_checkpoint_key",
                "dinov2_checkpoint_key",
                "mae_checkpoint_key",
            ]:
                if key in config:
                    backbone_eval_args += f' agent.config.{key}="{config[key]}"'
            for key in [
                "pc_trainable_ijepa_layers",
                "pc_trainable_dino_layers",
                "pc_trainable_dinov2_layers",
                "pc_trainable_mae_layers",
            ]:
                if key in config:
                    backbone_eval_args += f" agent.config.{key}={config[key]}"
            # Architecture overrides for custom ViT backend
            if config.get("vit_arch"):
                backbone_eval_args += f' agent.config.vit_arch="{config["vit_arch"]}"'
            if config.get("vit_patch_size"):
                backbone_eval_args += (
                    f" agent.config.vit_patch_size={config['vit_patch_size']}"
                )
            # Latent TransFuser mode
            latent = config.get("latent", False)
            if latent:
                backbone_eval_args += f" agent.config.latent={str(latent)}"
        elif agent == "law_agent":
            # LAW agent needs architecture + backbone params for checkpoint loading
            # (tensor shapes and encoder implementation must match training)
            backbone = config.get("backbone", "resnet")
            use_native_resolution = config.get("use_native_resolution", True)
            vit_backend = config.get("vit_backend", "huggingface")
            law_eval_parts = [
                f'agent.config.backbone="{backbone}"',
                f"agent.config.use_native_resolution={str(use_native_resolution).lower()}",
                f'agent.config.vit_backend="{vit_backend}"',
            ]
            for key in [
                "ijepa_model_id",
                "dinov2_model_id",
                "dino_model_id",
                "mae_model_id",
                "ijepa_checkpoint_path",
                "dinov2_checkpoint_path",
                "mae_checkpoint_path",
            ]:
                if key in config:
                    law_eval_parts.append(f'agent.config.{key}="{config[key]}"')
            for key in [
                "ijepa_checkpoint_key",
                "dinov2_checkpoint_key",
                "mae_checkpoint_key",
            ]:
                if key in config:
                    law_eval_parts.append(f'agent.config.{key}="{config[key]}"')
            for key in [
                "pc_trainable_ijepa_layers",
                "pc_trainable_dino_layers",
                "pc_trainable_dinov2_layers",
                "pc_trainable_mae_layers",
            ]:
                if key in config:
                    law_eval_parts.append(f"agent.config.{key}={config[key]}")
            # Architecture overrides for custom ViT backend
            if config.get("vit_arch"):
                law_eval_parts.append(f'agent.config.vit_arch="{config["vit_arch"]}"')
            if config.get("vit_patch_size"):
                law_eval_parts.append(
                    f"agent.config.vit_patch_size={config['vit_patch_size']}"
                )

            # Keep existing LAW shape overrides for non-default experiments
            for key, default in [
                ("camera_width", 640),
                ("camera_height", 320),
                ("num_views", 6),
                ("image_architecture", "resnet34"),
                ("num_proposals", 8),
                ("tf_d_model", 256),
                ("depth_num", 64),
            ]:
                val = config.get(key, default)
                if val != default:
                    if isinstance(val, str):
                        law_eval_parts.append(f'agent.config.{key}="{val}"')
                    else:
                        law_eval_parts.append(f"agent.config.{key}={val}")

            backbone_eval_args = " ".join(law_eval_parts)
        elif agent == "diffusiondrive_ijepa_agent":
            # DiffusionDrive needs backbone config + plan_anchor_path at eval time
            # Without backbone args, model defaults to ResNet-34 and checkpoint loading fails
            backbone = config.get("backbone", "resnet")
            use_native_resolution = config.get("use_native_resolution", True)
            vit_backend = config.get("vit_backend", "huggingface")
            dd_eval_parts = []

            # Only add backbone overrides if not default resnet
            if backbone != "resnet":
                dd_eval_parts.append(f"agent.config.backbone={backbone}")
                dd_eval_parts.append(f"agent.config.vit_backend={vit_backend}")
                if not use_native_resolution:
                    dd_eval_parts.append(f"agent.config.use_native_resolution=false")
                for key in [
                    "ijepa_checkpoint_path",
                    "dinov2_checkpoint_path",
                    "mae_checkpoint_path",
                ]:
                    if key in config:
                        dd_eval_parts.append(f"agent.config.{key}='{config[key]}'")
                for key in [
                    "ijepa_checkpoint_key",
                    "dinov2_checkpoint_key",
                    "mae_checkpoint_key",
                ]:
                    if key in config:
                        dd_eval_parts.append(f"agent.config.{key}='{config[key]}'")
                for key in [
                    "pc_trainable_ijepa_layers",
                    "pc_trainable_dino_layers",
                    "pc_trainable_dinov2_layers",
                    "pc_trainable_mae_layers",
                ]:
                    if key in config:
                        dd_eval_parts.append(f"agent.config.{key}={config[key]}")
                if config.get("vit_arch"):
                    dd_eval_parts.append(f"agent.config.vit_arch={config['vit_arch']}")
                if config.get("vit_patch_size"):
                    dd_eval_parts.append(
                        f"agent.config.vit_patch_size={config['vit_patch_size']}"
                    )

            # plan_anchor_path — only override if non-default
            num_anchors = config.get("num_anchors", 20)
            plan_anchor_path = config.get(
                "plan_anchor_path",
                f"/scratch/{self.username}/data/models/kmeans_navsim_traj_{num_anchors}.npy",
            )
            default_anchor = f"/scratch/{self.username}/data/models/kmeans_navsim_traj_{num_anchors}.npy"
            if plan_anchor_path != default_anchor:
                dd_eval_parts.append(
                    f"+agent.config.plan_anchor_path='{plan_anchor_path}'"
                )

            backbone_eval_args = " ".join(dd_eval_parts)
        eval_split = config.get("eval_split", "navtest")
        eval_workers = config.get("eval_workers", None)
        eval_mem = config.get("eval_mem", None)
        eval_time = config.get("eval_time", None)
        use_multi_camera = str(config.get("use_multi_camera", True)).lower()
        requires_training = config.get("requires_training", True)

        # Smart eval worker scaling for large ViT backbones.
        # Each Ray worker loads its own model copy into RAM.
        # ViT-S checkpoint ~0.3 GB, ViT-H ~2.5 GB (frozen) to ~7.5 GB (fully finetuned).
        # Scale workers + memory based on backbone size and trainable fraction.
        # Per admin guidance: standard nodes have 128 cores / 500 GB; request proportionally.
        _large_vit_backbones = {"ijepa", "dinov2", "dino", "mae"}
        backbone = config.get("backbone", "resnet")
        vit_arch = config.get("vit_arch", "")
        trainable_frac = float(config.get("trainable_fraction", 0.0))
        # Also check backbone-specific pc values as fallback
        if trainable_frac == 0.0:
            for key in [
                "pc_trainable_ijepa_layers",
                "pc_trainable_dino_layers",
                "pc_trainable_dinov2_layers",
                "pc_trainable_mae_layers",
            ]:
                val = float(config.get(key, 0.0))
                if val > trainable_frac:
                    trainable_frac = val

        # Determine model size class for resource scaling
        _small_archs = {"vit_tiny", "vit_small"}  # ViT-S/14 ~22M params, ~0.3 GB
        is_small_vit = vit_arch in _small_archs

        if backbone in _large_vit_backbones and eval_workers is None:
            if is_small_vit:
                # ViT-S: ~0.3 GB/worker → 32 workers × 0.3 = ~10 GB — very light
                eval_workers = 32
            elif trainable_frac > 0.5:
                eval_workers = 16
            elif trainable_frac > 0.0:
                eval_workers = 20
            else:
                eval_workers = 24

        # Apply defaults for anything not set by config or auto-scaling
        if eval_workers is None:
            eval_workers = 32
        if eval_mem is None:
            # Scale memory proportionally to workers and model size.
            # CPU nodes: 128 cores / 500 GB → ~4 GB per core.
            # Each Ray worker loads model + NAVSIM data pipeline + metric computation.
            # Per admin guidance: avoid 400GB requests; keep under ~200G to share nodes.
            if backbone in _large_vit_backbones:
                if is_small_vit:
                    # ViT-S: ~0.3 GB model + ~3 GB data/worker → ~3.5 GB/worker
                    eval_mem = f"{min(eval_workers * 4 + 32, 200)}G"  # 32w → 160G
                elif trainable_frac > 0.5:
                    # Large finetuned ViT: ~7.5 GB model per worker
                    eval_mem = f"{min(eval_workers * 9 + 32, 200)}G"  # 16w → 176G
                else:
                    # Large frozen ViT: ~2.5 GB model + ~3 GB data/worker
                    eval_mem = f"{min(eval_workers * 6 + 32, 200)}G"  # 24w → 176G
            else:
                # ResNet or other small backbone
                eval_mem = f"{min(eval_workers * 4 + 32, 200)}G"  # 32w → 160G
        if eval_time is None:
            # Large ViT with fewer workers needs more wall time
            if backbone in _large_vit_backbones:
                eval_time = "12:00:00"
            else:
                eval_time = "06:00:00"

        # City-specific evaluation
        # Use navtest split with +city=X filter (city configs filter by log_names)
        if city and city != "all":
            eval_split_actual = (
                eval_split  # Always use navtest, filtering happens via city config
            )
            city_suffix = f"_{city}"
            city_arg = f"+city={city}"
        else:
            eval_split_actual = eval_split
            city_suffix = "_all" if city == "all" else ""
            city_arg = ""

        # Eval-specific settings
        traffic_agents = "non_reactive" if eval_type == "one_stage" else "reactive"
        eval_script_name = (
            "run_pdm_score_one_stage.py"
            if eval_type == "one_stage"
            else "run_pdm_score.py"
        )

        # For GPU-based eval (two_stage), we need GPUs
        gpu_constraint = config.get("gpu_constraint", "")
        if eval_type == "two_stage":
            partition = config.get("partition", "l40s_public")
            num_gpus = config.get("num_gpus", 4)
            gpu_line = f"#SBATCH --gres=gpu:{num_gpus}"
            if gpu_constraint:
                partition_line = "# Partition skipped (using --constraint)"
            else:
                partition_line = f"#SBATCH --partition={partition}"
            constraint_line = (
                f'#SBATCH --constraint="{gpu_constraint}"'
                if gpu_constraint
                else "# No GPU constraint"
            )
            worker_type = "ray_distributed"
        else:
            # one_stage is CPU-only
            gpu_line = "# No GPU needed for one-stage (CPU-only)"
            constraint_line = "# CPU-only job"
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
{constraint_line}
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
export BACKBONE_EVAL_ARGS="{backbone_eval_args}"

{checkpoint_block}

# Derive run name: exp_id_eval_split_city_date_job_id
export RUN_NAME="{exp_id}_eval_{eval_split_actual}{city_suffix}_$(date +%Y%m%d_%H%M%S)_${{SLURM_JOB_ID}}"

# =============================================================================
# ENVIRONMENT
# =============================================================================
# Environment setup (via hpc_config)
{self._get_navsim_env_exports()}
export OUTPUT_DIR="${{NAVSIM_EXP_ROOT}}/evaluations/${{RUN_NAME}}"

# Metric cache - use old path structure to match overlay contents
# The overlay was created with files at /scratch/ah7072/experiments/cache/navtest_metric_cache
export METRIC_CACHE="{self._get_scratch_dir()}/experiments/cache/navtest_metric_cache"
export METRIC_CACHE_OVERLAY="{self._get_overlay_path('navtest_metric_cache')}"
export CONTAINER="{self._get_container_image()}"

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

# Conda activation (via hpc_config)
{self._generate_conda_activation(config)}

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
# Conda activation inside container
CONDA_ROOT="{self.hpc_config.conda_root if self.hpc_config else f'/scratch/{self.username}/miniconda3'}"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate {self.hpc_config.conda_env if self.hpc_config else 'navsim'}
fi
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
    ${{BACKBONE_EVAL_ARGS}} \\
    ${{CITY_ARG}} \\
    worker={worker_type} \\
    worker.threads_per_node=${{NUM_WORKERS}}
EVAL_SCRIPT_EOF
    chmod +x "${{TEMP_SCRIPT}}"
    
    apptainer exec \\
        --bind "${{METRIC_CACHE_OVERLAY}}:${{METRIC_CACHE}}:image-src=/" \\
        --bind {self._get_scratch_dir()}:{self._get_scratch_dir()} \\
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
        --env "BACKBONE_EVAL_ARGS=${{BACKBONE_EVAL_ARGS}}" \\
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
        ${{BACKBONE_EVAL_ARGS}} \\
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
                cities = ["boston", "vegas", "pittsburgh", "singapore", "all"]
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

    def cancel_experiment(
        self, exp_id: str, include_matching_name: bool = True, dry_run: bool = False
    ) -> List[str]:
        """
        Cancel all SLURM jobs related to an experiment ID.

        Finds job IDs from metadata and (optionally) live SLURM jobs whose
        names match the experiment ID prefix.
        """
        job_ids = set()

        meta = self.metadata.get(exp_id, {})
        train_job = meta.get("train_job_id")
        eval_job = meta.get("eval_job_id")
        eval_jobs = meta.get("eval_job_ids", [])

        for jid in [train_job, eval_job]:
            if jid:
                job_ids.add(str(jid))

        for jid in eval_jobs or []:
            if jid:
                job_ids.add(str(jid))

        def _is_related_job(name: str) -> bool:
            if name == exp_id:
                return True
            if not name.startswith(exp_id):
                return False
            if len(name) == len(exp_id):
                return True
            return name[len(exp_id)] in {"_", "-", "."}

        if include_matching_name:
            try:
                result = subprocess.run(
                    ["squeue", "-u", self.username, "-h", "-o", "%i %j"],
                    capture_output=True,
                    text=True,
                    check=True,
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
                print(f"Warning: could not query squeue: {e.stderr}")

        if not job_ids:
            print(f"No related SLURM jobs found for {exp_id}")
            return []

        job_ids_list = sorted(job_ids)

        if dry_run:
            print("[DRY RUN] Would cancel jobs:")
            print("  " + " ".join(job_ids_list))
            return job_ids_list

        try:
            subprocess.run(["scancel", *job_ids_list], check=True)
            print(f"✓ Canceled {len(job_ids_list)} jobs for {exp_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error cancelling jobs: {e.stderr}")
            sys.exit(1)

        if exp_id in self.metadata:
            meta["status"] = "canceled"
            meta["canceled_at"] = datetime.now().isoformat()
            self._save_metadata()

        return job_ids_list

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
            "vit_backend",
            "use_native_resolution",
            "ijepa_checkpoint_path",
            "ijepa_checkpoint_key",
            "epochs",
            "batch_size",
            "learning_rate",
            "encoder_learning_rate",
            "trainable_fraction",
            "vision_mode",
            "ijepa_model_id",
            "dinov2_model_id",
            "dino_model_id",
            "mae_model_id",
            "pc_trainable_ijepa_layers",
            "pc_trainable_dino_layers",
            "pc_trainable_dinov2_layers",
            "pc_trainable_mae_layers",
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

        Parses evaluation logs and CSV files to extract PDMS scores and sub-scores per city.

        Args:
            exp_id: Experiment ID
            output_json: If True, output JSON format

        Returns:
            Dictionary with results per city including sub-scores
        """
        import pandas as pd
        from pathlib import Path

        results = {
            "experiment": exp_id,
            "timestamp": datetime.now().isoformat(),
            "cities": {},
            "total_scenarios": 0,
            "average_pdms": 0.0,
        }

        def _pick_latest_log(patterns: List[str]) -> Optional[Path]:
            candidates = []
            for pattern in patterns:
                candidates.extend(list(Path(self.logs_dir / "output").glob(pattern)))
            if not candidates:
                return None

            def _job_id(path: Path) -> int:
                match = re.search(r"_(\d+)\.out$", path.name)
                return int(match.group(1)) if match else 0

            return max(candidates, key=lambda p: (_job_id(p), p.stat().st_mtime))

        def _parse_eval_log(log_file: Path) -> Dict[str, Any]:
            content = log_file.read_text()
            scenarios_match = re.search(
                r"Number of successful scenarios:\s*(\d+)\.?", content
            )
            score_match = re.search(
                r"Final average score of valid results:\s*([0-9]+\.?[0-9]*)",
                content,
            )
            if not score_match:
                score_match = re.search(r"PDMS:\s*([0-9.]+)", content)

            status_match = re.search(r"Status:\s*(SUCCESS|FAILED)", content)
            status = status_match.group(1) if status_match else "UNKNOWN"

            scenarios = int(scenarios_match.group(1)) if scenarios_match else 0
            pdms = float(score_match.group(1)) if score_match else 0.0
            if status == "UNKNOWN" and pdms > 0:
                status = "SUCCESS"

            return {
                "scenarios": scenarios,
                "pdms": pdms,
                "status": status,
                "log_file": str(log_file.name),
            }

        # Find all evaluation directories for this experiment
        # Strict filtering: avoid substring matches (e.g. A3 matching A3-b)
        raw_matches = list(self.experiments_root.glob(f"evaluations/*{exp_id}*"))
        eval_dirs = []
        for d in raw_matches:
            d_name = d.name
            if (
                d_name == exp_id
                or d_name.startswith(f"{exp_id}_")
                or d_name.endswith(f"_{exp_id}")
                or f"_{exp_id}_" in d_name
            ):
                eval_dirs.append(d)

        if not eval_dirs:
            print(f"No evaluation results found for experiment: {exp_id}")
            return results

        cities = ["boston", "vegas", "pittsburgh", "singapore"]
        total_scenarios = 0
        total_weighted_score = 0.0

        # Group eval dirs by city, keep only most recent valid CSV
        city_eval_dirs = {}
        city_candidates = {}
        for eval_dir in eval_dirs:
            city = None
            for c in cities:
                if c in eval_dir.name.lower():
                    city = c
                    break
            if not city:
                city = "all"
            city_candidates.setdefault(city, []).append(eval_dir)

        for city, candidates in city_candidates.items():
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for eval_dir in candidates:
                csv_files = sorted(
                    eval_dir.glob("*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if csv_files:
                    city_eval_dirs[city] = eval_dir
                    break

        # Process each city's evaluation
        for city, eval_dir in city_eval_dirs.items():
            try:
                # Find CSV result file
                csv_files = sorted(
                    eval_dir.glob("*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not csv_files:
                    raise FileNotFoundError("No CSV files found in eval dir")

                csv_file = csv_files[0]  # Take most recent CSV
                df = pd.read_csv(csv_file)

                # Filter out the summary row (NAVSIM appends an 'average_all_frames' row)
                if "token" in df.columns:
                    df = df[df["token"] != "average_all_frames"]
                if df.empty:
                    raise ValueError("CSV has no valid rows after filtering")

                # Calculate metrics from CSV columns
                # NAVSIM column names
                col_map = {
                    "NC": "no_at_fault_collisions",
                    "DAC": "drivable_area_compliance",
                    "DDC": "driving_direction_compliance",
                    "TLC": "traffic_light_compliance",
                    "EP": "ego_progress",
                    "TTC": "time_to_collision_within_bound",
                    "C": "history_comfort",  # or average of comfort columns
                    "PDMS": "score",
                }

                # Calculate means
                metrics = {}
                for short_name, col_name in col_map.items():
                    if col_name in df.columns:
                        # Handle NaN values
                        valid_vals = df[col_name].dropna()
                        if len(valid_vals) > 0:
                            metrics[short_name] = float(valid_vals.mean())

                # For comfort, average history_comfort and two_frame_extended_comfort
                if (
                    "history_comfort" in df.columns
                    and "two_frame_extended_comfort" in df.columns
                ):
                    comfort_avg = (
                        df["history_comfort"].fillna(1.0).mean()
                        + df["two_frame_extended_comfort"].fillna(1.0).mean()
                    ) / 2
                    metrics["C"] = float(comfort_avg)

                scenarios = len(df)
                pdms = metrics.get("PDMS", 0.0)

                # Determine status from log
                status = "SUCCESS" if pdms > 0 else "UNKNOWN"
                log_file = eval_dir / "run_pdm_score_one_stage.log"
                if log_file.exists():
                    with open(log_file) as f:
                        content = f.read()
                    if "Status: SUCCESS" in content:
                        status = "SUCCESS"
                    elif "Status: FAILED" in content:
                        status = "FAILED"

                results["cities"][city] = {
                    "scenarios": scenarios,
                    "pdms": pdms,
                    "status": status,
                    "eval_dir": str(eval_dir.name),
                    # Sub-scores
                    "NC": metrics.get("NC", None),
                    "DAC": metrics.get("DAC", None),
                    "EP": metrics.get("EP", None),
                    "TTC": metrics.get("TTC", None),
                    "C": metrics.get("C", None),
                }

                if status == "SUCCESS" and scenarios > 0:
                    # Only count per-city results for weighted average
                    # (avoid double-counting when "all" eval also exists)
                    if city != "all":
                        total_scenarios += scenarios
                        total_weighted_score += pdms * scenarios

            except Exception as e:
                # Fallback to eval logs when CSV is missing or malformed
                log_patterns = [
                    f"{exp_id}_{city}_eval_*.out",
                    f"{exp_id}_eval_*.out",
                ]
                latest_log = _pick_latest_log(log_patterns)
                if latest_log is None:
                    print(f"Warning: Could not parse {eval_dir}: {e}")
                    continue

                log_metrics = _parse_eval_log(latest_log)
                results["cities"][city] = {
                    "scenarios": log_metrics["scenarios"],
                    "pdms": log_metrics["pdms"],
                    "status": log_metrics["status"],
                    "eval_dir": str(eval_dir.name),
                    "log_file": log_metrics["log_file"],
                    "NC": None,
                    "DAC": None,
                    "EP": None,
                    "TTC": None,
                    "C": None,
                }

                if log_metrics["status"] == "SUCCESS" and log_metrics["scenarios"] > 0:
                    if city != "all":
                        total_scenarios += log_metrics["scenarios"]
                        total_weighted_score += (
                            log_metrics["pdms"] * log_metrics["scenarios"]
                        )

        # Calculate averages
        # If we have per-city results, use scenario-weighted average
        # If only "all" eval exists, use its PDMS directly
        if total_scenarios > 0:
            results["total_scenarios"] = total_scenarios
            results["average_pdms"] = total_weighted_score / total_scenarios
        elif "all" in results["cities"]:
            all_data = results["cities"]["all"]
            results["total_scenarios"] = all_data.get("scenarios", 0)
            results["average_pdms"] = all_data.get("pdms", 0.0)
        else:
            results["total_scenarios"] = 0
            results["average_pdms"] = 0.0

        # Calculate average sub-scores (per-city only, exclude "all" to avoid double-counting)
        per_city_results = {k: v for k, v in results["cities"].items() if k != "all"}
        source = per_city_results if per_city_results else results["cities"]
        for metric in ["NC", "DAC", "EP", "TTC", "C"]:
            values = [
                c.get(metric) for c in source.values() if c.get(metric) is not None
            ]
            if values:
                results[f"avg_{metric}"] = sum(values) / len(values)

        # Output
        if output_json:
            print(json.dumps(results, indent=2))
        else:
            self._print_results_table(results)

        return results

    def _print_results_table(self, results: Dict[str, Any]):
        """Print formatted results table with sub-scores."""
        print(f"\n{'='*95}")
        print(f"Experiment: {results['experiment']}")
        print(f"{'='*95}")
        print(
            f"{'City':<12} | {'Scenarios':>8} | {'PDMS':>6} | {'NC':>5} | {'DAC':>5} | {'EP':>5} | {'TTC':>5} | {'C':>5} | {'Status':<8}"
        )
        print(
            f"{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}"
        )

        city_order = ["boston", "vegas", "pittsburgh", "singapore", "all"]
        for city in city_order:
            if city in results["cities"]:
                d = results["cities"][city]
                nc = f"{d['NC']:.3f}" if d.get("NC") is not None else "  -  "
                dac = f"{d['DAC']:.3f}" if d.get("DAC") is not None else "  -  "
                ep = f"{d['EP']:.3f}" if d.get("EP") is not None else "  -  "
                ttc = f"{d['TTC']:.3f}" if d.get("TTC") is not None else "  -  "
                c = f"{d['C']:.3f}" if d.get("C") is not None else "  -  "
                print(
                    f"{city.capitalize():<12} | {d['scenarios']:>8,} | {d['pdms']:>6.3f} | {nc} | {dac} | {ep} | {ttc} | {c} | {d['status']:<8}"
                )

        print(
            f"{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}"
        )
        # Print averages
        avg_nc = f"{results.get('avg_NC', 0):.3f}" if "avg_NC" in results else "  -  "
        avg_dac = (
            f"{results.get('avg_DAC', 0):.3f}" if "avg_DAC" in results else "  -  "
        )
        avg_ep = f"{results.get('avg_EP', 0):.3f}" if "avg_EP" in results else "  -  "
        avg_ttc = (
            f"{results.get('avg_TTC', 0):.3f}" if "avg_TTC" in results else "  -  "
        )
        avg_c = f"{results.get('avg_C', 0):.3f}" if "avg_C" in results else "  -  "
        print(
            f"{'Average':<12} | {results['total_scenarios']:>8,} | {results['average_pdms']:>6.3f} | {avg_nc} | {avg_dac} | {avg_ep} | {avg_ttc} | {avg_c} |"
        )
        print(f"{'='*95}")

    def export_results_csv(self, exp_id: str, output_path: str = None) -> str:
        """Export results to CSV format compatible with experiment tracking spreadsheet."""
        results = self.collect_results(exp_id, output_json=False)

        if output_path is None:
            output_path = self.experiments_root / f"{exp_id}_results.csv"

        rows = []
        for city, data in results["cities"].items():
            rows.append(
                {
                    "City": city.capitalize(),
                    "Scenarios": data["scenarios"],
                    "PDMS": data["pdms"],
                    "NC": data.get("NC"),
                    "DAC": data.get("DAC"),
                    "EP": data.get("EP"),
                    "TTC": data.get("TTC"),
                    "C": data.get("C"),
                    "Status": data["status"],
                }
            )

        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported results to: {output_path}")
        return str(output_path)

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


def cmd_collect_results(args, manager):
    """
    Collect all experiment results into database

    Usage:
        python navsim_manager.py collect-results
        python navsim_manager.py collect-results --status all --force
    """
    # Override backend if specified
    if args.backend:
        import os

        os.environ["EXPFLOW_BACKEND"] = args.backend
        # Clear cached property
        if hasattr(manager, "_results_storage"):
            delattr(manager, "_results_storage")

    if args.after_current:
        try:
            squeue = subprocess.run(
                ["squeue", "-u", manager.username, "-h", "-o", "%i"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error querying squeue: {e.stderr}")
            return

        job_ids = [j.strip() for j in squeue.stdout.split() if j.strip()]

        if not job_ids:
            print("No active jobs found; collecting results now.")
        else:
            dependency = "afterok:" + ":".join(job_ids)
            account = None
            if manager.hpc_config:
                account = getattr(manager.hpc_config, "account", None)
            if not account:
                account = "torch_pr_68_general"

            export_vars = [
                "ALL",
                "EXPFLOW_BACKEND",
                "EXPFLOW_CONNECTION_STRING",
                "EXPFLOW_MONGODB_TLS_INSECURE",
            ]
            export_str = ",".join(export_vars)
            output_log = f"{manager.logs_dir}/output/collect_results_%j.out"
            error_log = f"{manager.logs_dir}/error/collect_results_%j.err"

            cmd = [
                "sbatch",
                f"--job-name=navsim_collect_results",
                f"--account={account}",
                f"--dependency={dependency}",
                f"--export={export_str}",
                f"--output={output_log}",
                f"--error={error_log}",
                "--wrap",
                "source /scratch/ah7072/miniconda3/etc/profile.d/conda.sh && "
                "conda activate navsim && "
                f"cd {manager.project_root} && "
                f"python navsim_manager.py collect-results --status={args.status}"
                + (" --force" if args.force else ""),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                print(f"Submitted collect-results job after current jobs: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting collect-results job: {e.stderr}")
            return

    print(f"Collecting results (status: {args.status})...")

    results = manager.collect_all_results(
        status_filter=args.status, force_reharvest=args.force
    )

    print(f"\n[SUCCESS] Collected {len(results)} experiments")


def cmd_export_results(args, manager):
    """
    Export results to spreadsheet format

    Usage:
        python navsim_manager.py export-results
        python navsim_manager.py export-results --format json --output results.json
    """
    import pandas as pd

    # Get all experiments from database
    with manager.results_storage as storage:
        filters = {}
        if args.status:
            filters["status"] = args.status

        all_experiments = storage.query(**filters) if filters else storage.query()

    if not all_experiments:
        print("No experiments found!")
        return

    # Convert to spreadsheet format (matching user's columns)
    rows = []
    for exp in all_experiments:
        eval_data = exp.get("evaluation", {})
        arch_data = exp.get("architecture", {})
        train_data = exp.get("training", {})

        row = {
            "Phase": exp.get("phase", ""),
            "ID": exp.get("exp_id", ""),
            "Architecture": arch_data.get("name", ""),
            "Backbone": arch_data.get("backbone", ""),
            "Backbone_Type": arch_data.get("backbone_type", ""),
            "Train_City": train_data.get("city", ""),
            "Train_Data_%": train_data.get("data_percentage", ""),
            "Freeze_%": arch_data.get("freeze_percentage", ""),
            "Epochs": train_data.get("epochs", ""),
            "Test_All": eval_data.get("all", {}).get("pdm_score", ""),
            "Test_Boston": eval_data.get("boston", {}).get("pdm_score", ""),
            "Test_Vegas": eval_data.get("vegas", {}).get("pdm_score", ""),
            "Test_Pittsburgh": eval_data.get("pittsburgh", {}).get("pdm_score", ""),
            "Test_Singapore": eval_data.get("singapore", {}).get("pdm_score", ""),
            "Avg_PDMS": eval_data.get("avg_pdm_score", ""),
            "In_Dist_PDMS": eval_data.get("in_dist_pdm_score", ""),
            "Out_Dist_Avg": eval_data.get("out_dist_avg", ""),
            "Gen_Gap_%": eval_data.get("generalization_gap_percent", ""),
            "NC": eval_data.get("nc", ""),
            "DAC": eval_data.get("dac", ""),
            "EP": eval_data.get("ep", ""),
            "TTC": eval_data.get("ttc", ""),
            "C": eval_data.get("c", ""),
            "Status": exp.get("status", ""),
            "Priority": exp.get("priority", ""),
            "Notes": exp.get("notes", ""),
        }
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    df = df.sort_values(["Phase", "ID"])

    # Generate output path
    output_file = args.output
    if not output_file:
        output_file = f"navsim_results.{args.format}"

    # Export
    if args.format == "csv":
        df.to_csv(output_file, index=False)
    else:  # json
        df.to_json(output_file, orient="records", indent=2)

    print(f"[SUCCESS] Exported {len(df)} experiments to {output_file}")
    print(f"          Columns: {len(df.columns)}")


def cmd_collect_run_results(args, manager):
    """
    Collect versioned per-run results into the run history database.

    Usage:
        python navsim_manager.py collect-run-results
        python navsim_manager.py collect-run-results --force
        python navsim_manager.py collect-run-results --exp A10 A11
    """
    exp_ids = args.exp if hasattr(args, "exp") and args.exp else None

    results = manager.collect_all_run_results(
        exp_ids=exp_ids,
        force=args.force,
        verbose=True,
    )

    total = sum(results.values())
    print(f"\n[SUMMARY] {total} new run records across {len(results)} experiments")


def cmd_show_run_history(args, manager):
    """
    Show the run history for a specific experiment.

    Usage:
        python navsim_manager.py show-run-history A10
        python navsim_manager.py show-run-history A10 --json
    """
    manager.show_run_history(args.exp_id, output_json=args.json)


# =============================================================================
# SSL Pretraining Manager (navtrain ViT-S/14)
# =============================================================================


class PretrainManager:
    """
    Manages the 3×5 SSL pretraining matrix on navtrain data.

    Methods: I-JEPA, DINOv2, MAE (all ViT-S/14)
    Datasets: all, boston, singapore, loo_boston, loo_singapore

    Usage via navsim_manager.py:
        python navsim_manager.py pretrain prepare-data
        python navsim_manager.py pretrain generate
        python navsim_manager.py pretrain submit --dry-run
        python navsim_manager.py pretrain status
    """

    METHODS = ["ijepa", "dinov2", "mae"]
    DATASETS = ["all", "boston", "singapore", "loo_boston", "loo_singapore"]

    DATASET_DESCRIPTIONS = {
        "all": "All 4 cities (LV+B+P+S)",
        "boston": "Boston only",
        "singapore": "Singapore only",
        "loo_boston": "Leave-out Boston (LV+P+S)",
        "loo_singapore": "Leave-out Singapore (LV+B+P)",
    }

    DEFAULT_BATCH_SIZES = {
        "ijepa": 512,
        "mae": 768,
        "dinov2": 384,
    }

    def __init__(self):
        self.username = os.environ.get("USER", "ah7072")
        self.scratch = Path(f"/scratch/{self.username}")
        self.backbones_root = self.scratch / "backbones"
        self.data_root = self.scratch / "data"
        self.pickles_dir = self.data_root / "navtrain_pickles"
        self.models_dir = self.data_root / "models" / "navtrain_pretrained"
        self.configs_dir = self.backbones_root / "configs" / "navtrain"
        self.scripts_dir = self.backbones_root / "pretraining_scripts" / "navtrain"
        self.logs_dir = self.scratch / "experiments" / "logs" / "pretraining"

        # Pretraining code roots (local copies under ah7072)
        self.pretrain_root = self.backbones_root
        self.conda_env = (
            "/scratch/fn2174/envs/ijepa"  # shared env — deps installed here
        )

    # ── prepare-data ─────────────────────────────────────────────────────

    def prepare_data(
        self,
        dry_run: bool = False,
        verify: bool = False,
        account: str = "torch_pr_106_tandon_advanced",
        time_limit: str = "02:00:00",
        submit: bool = True,
    ):
        """Generate and submit a Slurm job to extract navtrain images into pickles."""
        extraction_script = self.backbones_root / "utils" / "prepare_navtrain_data.py"
        if not extraction_script.exists():
            print(f"Error: Extraction script not found at {extraction_script}")
            return False

        sensor_blobs = self.data_root / "sensor_blobs" / "trainval"
        splits_dir = (
            self.scratch / "navsim-ssl-city-generalization" / "city_splits" / "splits"
        )

        flags = ""
        if verify:
            flags += " --verify"

        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=pt_prepare_data
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={time_limit}
#SBATCH --account={account}
#SBATCH --output={self.logs_dir}/pt_prepare_data_%j.out
#SBATCH --error={self.logs_dir}/pt_prepare_data_%j.err

set -euo pipefail

source /scratch/{self.username}/miniconda3/etc/profile.d/conda.sh
conda activate navsim

echo "=== Navtrain Image Extraction ==="
echo "  Host:   $(hostname)"
echo "  Date:   $(date)"
echo "  Output: {self.pickles_dir}"

python {extraction_script} \\
    --sensor-blobs-dir {sensor_blobs} \\
    --splits-dir {splits_dir} \\
    --output-dir {self.pickles_dir}{flags}

echo ""
echo "=== Extraction complete ==="
ls -lh {self.pickles_dir}/navtrain_*.pkl 2>/dev/null || echo "No pickles found"
"""
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.scripts_dir / "pt_prepare_data.sh"
        with open(script_path, "w") as f:
            f.write(slurm_script)
        os.chmod(script_path, 0o755)

        print(f"Generated: {script_path}")

        if dry_run:
            print(f"[DRY RUN] Would run: sbatch {script_path}")
            return True

        if submit:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"Submitted data extraction job: {job_id}")
                print(f"  Monitor: squeue -j {job_id}")
                print(
                    f"  Log:     tail -f {self.logs_dir}/pt_prepare_data_{job_id}.out"
                )
                print(f"\nOnce complete, run:")
                print(f"  python navsim_manager.py pretrain generate")
                print(f"  python navsim_manager.py pretrain submit")
                return True
            else:
                print(f"sbatch failed: {result.stderr.strip()}")
                return False
        else:
            print(f"Script ready. Submit with: sbatch {script_path}")
            return True

    # ── generate ─────────────────────────────────────────────────────────

    def generate_scripts(
        self,
        methods: list = None,
        datasets: list = None,
        epochs: int = 200,
        batch_size: int = None,
        account: str = "torch_pr_106_tandon_advanced",
        time_limit: str = "48:00:00",
        partition: str = None,
    ):
        """Generate Slurm scripts for the pretraining matrix."""
        methods = methods or self.METHODS
        datasets = datasets or self.DATASETS

        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for method in methods:
            for dataset in datasets:
                script_path = self._generate_one_script(
                    method, dataset, epochs, batch_size, account, time_limit, partition
                )
                generated.append((method, dataset, script_path))

        # Print summary
        print(f"\nGenerated {len(generated)} Slurm scripts:")
        print(f"{'Method':<10} {'Dataset':<20} {'Script'}")
        print("-" * 80)
        for method, dataset, path in generated:
            print(f"{method:<10} {dataset:<20} {path.name}")
        print(f"\nAll scripts in: {self.scripts_dir}")
        return generated

    def _generate_one_script(
        self, method, dataset, epochs, batch_size, account, time_limit, partition
    ):
        """Generate a single Slurm pretraining script."""
        job_name = f"pt_{method}_{dataset}"
        bs = batch_size or self.DEFAULT_BATCH_SIZES[method]
        output_dir = self.models_dir / f"{method}_{dataset}"
        pkl_path = self.pickles_dir / f"navtrain_{dataset}.pkl"

        slurm_header = self._slurm_header(job_name, account, time_limit, partition)
        env_setup = self._env_setup()

        if method == "ijepa":
            train_cmd = self._ijepa_command(dataset, output_dir, epochs, bs)
        elif method == "mae":
            train_cmd = self._mae_command(pkl_path, output_dir, epochs, bs)
        elif method == "dinov2":
            train_cmd = self._dinov2_command(pkl_path, output_dir, epochs, bs)
        else:
            raise ValueError(f"Unknown method: {method}")

        script_content = f"""{slurm_header}

set -euo pipefail

{env_setup}

# ── Job info ──
echo "{'='*60}"
echo "SSL Pretraining: {method.upper()} on navtrain_{dataset}"
echo "  Method:    {method}"
echo "  Dataset:   navtrain_{dataset} ({self.DATASET_DESCRIPTIONS[dataset]})"
echo "  Epochs:    {epochs}"
echo "  Batch:     {bs}"
echo "  Output:    {output_dir}"
echo "  Host:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Date:      $(date)"
echo "{'='*60}"

# Verify pickle exists
if [ ! -f "{pkl_path}" ]; then
    echo "ERROR: Pickle not found: {pkl_path}"
    echo "Run: python navsim_manager.py pretrain prepare-data"
    exit 1
fi

# Image count
python -c "import pickle; d=pickle.load(open('{pkl_path}','rb')); print(f'Dataset: {{len(d):,}} images')"

mkdir -p "{output_dir}"

{train_cmd}

echo ""
echo "Pretraining complete: {method} on navtrain_{dataset}"
echo "Checkpoint: {output_dir}"
"""
        script_path = self.scripts_dir / f"{job_name}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        return script_path

    def _slurm_header(self, job_name, account, time_limit, partition):
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            "#SBATCH --gres=gpu:h200:1",
            '#SBATCH --constraint="h200"',
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=256G",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --account={account}",
            f"#SBATCH --output={self.logs_dir}/{job_name}_%j.out",
            f"#SBATCH --error={self.logs_dir}/{job_name}_%j.err",
        ]
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        return "\n".join(lines)

    def _env_setup(self):
        return f"""# Conda env (Fatimeh's ijepa env has all deps)
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate {self.conda_env}

# PYTHONPATH for pretraining repos
export PYTHONPATH="{self.pretrain_root}/mae:{self.pretrain_root}/dinov2:{self.pretrain_root}/ijepa/src:{self.pretrain_root}/utils:${{PYTHONPATH:-}}"

# Thread settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"""

    def _ijepa_command(self, dataset, output_dir, epochs, batch_size):
        config_path = self.configs_dir / f"ijepa_{dataset}.yaml"
        runtime_config = output_dir / "config.yaml"
        return f"""# I-JEPA: generate runtime config from template
sed -e "s|folder:.*|folder: {output_dir}/|" \\
    -e "s|epochs:.*|epochs: {epochs}|" \\
    -e "s|batch_size:.*|batch_size: {batch_size}|" \\
    "{config_path}" > "{runtime_config}"

echo "I-JEPA config: {runtime_config}"
cd "{self.pretrain_root}"

python -u pretraining_scripts/pretrain_ijepa_vits14.py \\
    --config "{runtime_config}" \\
    2>&1 | tee "{output_dir}/training_console.log"
"""

    def _mae_command(self, pkl_path, output_dir, epochs, batch_size):
        return f"""cd "{self.pretrain_root}"

python -u pretraining_scripts/pretrain_mae_vits14.py \\
    --model mae_vit_small_patch14 \\
    --batch_size {batch_size} \\
    --epochs {epochs} \\
    --warmup_epochs 40 \\
    --lr 1.5e-4 \\
    --weight_decay 0.05 \\
    --data_path "" \\
    --pkl_path "{pkl_path}" \\
    --output_dir "{output_dir}" \\
    --log_dir "{output_dir}" \\
    --num_workers 8 \\
    --prefetch_factor 8 \\
    --persistent_workers \\
    2>&1 | tee "{output_dir}/training_console.log"
"""

    def _dinov2_command(self, pkl_path, output_dir, epochs, batch_size):
        return f"""cd "{self.pretrain_root}"

# DINOv2: disable xFormers (H200 compatibility)
export XFORMERS_DISABLED=1

python -u pretraining_scripts/pretrain_dinov2_vits14.py \\
    --train-dataset "NuScenes:split=TRAIN:root=:extra={pkl_path}" \\
    --epochs {epochs} \\
    --batch-size-per-gpu {batch_size} \\
    --base-lr 0.002 \\
    --warmup-epochs 10 \\
    --weight-decay 0.04 \\
    --weight-decay-end 0.4 \\
    --output-dir "{output_dir}" \\
    2>&1 | tee "{output_dir}/training_console.log"
"""

    # ── submit ───────────────────────────────────────────────────────────

    def submit_jobs(self, methods=None, datasets=None, dry_run=False):
        """Submit generated Slurm scripts."""
        methods = methods or self.METHODS
        datasets = datasets or self.DATASETS

        submitted = []
        for method in methods:
            for dataset in datasets:
                script_path = self.scripts_dir / f"pt_{method}_{dataset}.sh"
                if not script_path.exists():
                    print(f"  SKIP (no script): {method} x {dataset}")
                    print(f"         Run: python navsim_manager.py pretrain generate")
                    continue

                pkl_path = self.pickles_dir / f"navtrain_{dataset}.pkl"
                if not pkl_path.exists():
                    print(f"  SKIP (no data): {method} x {dataset}")
                    print(
                        f"         Run: python navsim_manager.py pretrain prepare-data"
                    )
                    continue

                if dry_run:
                    print(f"  [DRY RUN] sbatch {script_path}")
                    submitted.append((method, dataset, None))
                else:
                    result = subprocess.run(
                        ["sbatch", str(script_path)], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        job_id = result.stdout.strip().split()[-1]
                        print(f"  Submitted: {method} x {dataset} -> Job {job_id}")
                        submitted.append((method, dataset, job_id))
                    else:
                        print(
                            f"  FAILED: {method} x {dataset}: {result.stderr.strip()}"
                        )

        print(
            f"\n{'[DRY RUN] Would submit' if dry_run else 'Submitted'} {len(submitted)} jobs"
        )
        return submitted

    # ── status ───────────────────────────────────────────────────────────

    def show_status(self):
        """Show status of all pretraining runs in a matrix view."""
        print(f"\n{'='*72}")
        print("SSL Pretraining Matrix — ViT-S/14 on navtrain")
        print(f"{'='*72}")

        # Data status
        print(f"\n--- Pickle Datasets ({self.pickles_dir}) ---")
        for ds in self.DATASETS:
            pkl = self.pickles_dir / f"navtrain_{ds}.pkl"
            if pkl.exists():
                import pickle as pkl_mod

                with open(pkl, "rb") as f:
                    count = len(pkl_mod.load(f))
                size_mb = pkl.stat().st_size / (1024 * 1024)
                print(f"  navtrain_{ds:<20} {count:>8,} images  ({size_mb:.1f} MB)")
            else:
                print(f"  navtrain_{ds:<20} NOT FOUND")

        # Checkpoint matrix
        print(f"\n--- Checkpoint Status ---")
        header = f"{'Method':<10}"
        for ds in self.DATASETS:
            header += f" {ds:>15}"
        print(header)
        print("-" * (10 + 16 * len(self.DATASETS)))

        for method in self.METHODS:
            row = f"{method:<10}"
            for ds in self.DATASETS:
                ckpt_dir = self.models_dir / f"{method}_{ds}"
                if ckpt_dir.exists():
                    # Check for final checkpoint
                    ckpts = list(ckpt_dir.glob("*.pth*"))
                    if ckpts:
                        # Find latest checkpoint and extract epoch info
                        latest = max(ckpts, key=lambda p: p.stat().st_mtime)
                        row += f" {'DONE':>15}"
                    else:
                        # Directory exists but no checkpoint — running or failed
                        log_exists = (ckpt_dir / "training_console.log").exists()
                        row += f" {'RUNNING' if log_exists else 'EMPTY':>15}"
                else:
                    row += f" {'—':>15}"
            print(row)

        # Slurm job status
        print(f"\n--- Active Slurm Jobs ---")
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "-u",
                    self.username,
                    "-o",
                    "%.8i %.30j %.8T %.10M %.6D %R",
                    "--name",
                    ",".join(
                        f"pt_{m}_{d}" for m in self.METHODS for d in self.DATASETS
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                print(result.stdout.strip())
            else:
                print("  No active pretraining jobs")
        except Exception:
            print("  (squeue unavailable)")

    # ── list ─────────────────────────────────────────────────────────────

    def list_configs(self):
        """List available pretraining datasets and configs."""
        print("\n--- Pretraining Matrix (3 methods x 5 datasets = 15 runs) ---\n")
        print(f"{'Dataset':<22} {'Description':<30} {'Zero-shot for'}")
        print("-" * 72)
        zs_map = {
            "all": "None (backbone sees all cities)",
            "boston": "LV, Pittsburgh, Singapore",
            "singapore": "LV, Boston, Pittsburgh",
            "loo_boston": "Boston",
            "loo_singapore": "Singapore",
        }
        for ds in self.DATASETS:
            print(f"navtrain_{ds:<17} {self.DATASET_DESCRIPTIONS[ds]:<30} {zs_map[ds]}")

        print(f"\n{'Method':<10} {'Architecture':<15} {'Objective'}")
        print("-" * 50)
        print(f"{'ijepa':<10} {'ViT-S/14':<15} Latent prediction (JEPA)")
        print(f"{'dinov2':<10} {'ViT-S/14':<15} Self-distillation + DINO loss")
        print(f"{'mae':<10} {'ViT-S/14':<15} Masked pixel reconstruction")

        # Show I-JEPA configs
        print(f"\n--- I-JEPA YAML Configs ({self.configs_dir}) ---")
        if self.configs_dir.exists():
            for f in sorted(self.configs_dir.glob("ijepa_*.yaml")):
                print(f"  {f.name}")
        else:
            print("  (not generated yet)")

        # Show generated scripts
        print(f"\n--- Generated Slurm Scripts ({self.scripts_dir}) ---")
        if self.scripts_dir.exists():
            scripts = sorted(self.scripts_dir.glob("pt_*.sh"))
            if scripts:
                for s in scripts:
                    print(f"  {s.name}")
            else:
                print("  (none — run: python navsim_manager.py pretrain generate)")
        else:
            print("  (none — run: python navsim_manager.py pretrain generate)")


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

    # Cancel all jobs related to an experiment
    python navsim_manager.py cancel exp_b15
  
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
    # Training & architecture args (useful for LAW and other agents)
    new_parser.add_argument(
        "--weight-decay", type=float, help="Weight decay (0=Adam, >0=AdamW)"
    )
    new_parser.add_argument(
        "--encoder-lr-mult", type=float, help="Encoder LR multiplier (default: 0.1)"
    )
    new_parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=None,
        help="Freeze backbone encoder",
    )
    new_parser.add_argument(
        "--use-wm", action="store_true", default=None, help="Enable world model"
    )
    new_parser.add_argument(
        "--no-wm", dest="use_wm", action="store_false", help="Disable world model"
    )
    new_parser.add_argument(
        "--wm-loss-weight", type=float, help="WM loss weight (default: 0.2)"
    )
    new_parser.add_argument(
        "--traj-loss-weight", type=float, help="Trajectory loss weight (default: 1.0)"
    )
    new_parser.add_argument(
        "--gradient-clip-val", type=float, help="Gradient clip value (0=disabled)"
    )
    new_parser.add_argument(
        "--use-cosine-scheduler",
        action="store_true",
        default=None,
        help="Use cosine annealing scheduler",
    )
    new_parser.add_argument(
        "--camera-width", type=int, help="Camera input width (default: 640)"
    )
    new_parser.add_argument(
        "--camera-height", type=int, help="Camera input height (default: 320)"
    )
    new_parser.add_argument(
        "--image-architecture", type=str, help="Backbone architecture (resnet34, etc.)"
    )
    new_parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet", "ijepa", "dino", "dinov2", "mae"],
        help="Backbone type (resnet, ijepa, dino, dinov2, mae)",
    )
    new_parser.add_argument(
        "--encoder-weights-path", type=str, help="Path to custom encoder weights"
    )
    new_parser.add_argument(
        "--precision", type=str, help="Training precision (16-mixed, bf16-mixed, 32)"
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

    # cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel experiment jobs")
    cancel_parser.add_argument("exp_id", help="Experiment ID")
    cancel_parser.add_argument(
        "--dry-run", action="store_true", help="Show jobs but don't cancel"
    )
    cancel_parser.add_argument(
        "--no-name-match",
        action="store_true",
        help="Only cancel jobs recorded in metadata (skip squeue name matching)",
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

    # collect-results (v0.8.0)
    collect_results_parser = subparsers.add_parser(
        "collect-results", help="Collect all experiment results into database (v0.8.0)"
    )
    collect_results_parser.add_argument(
        "--status", default="completed", help="Filter by status (default: completed)"
    )
    collect_results_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-harvest even if already in database",
    )
    collect_results_parser.add_argument(
        "--backend",
        choices=["sqlite", "mongodb", "postgresql"],
        help="Override backend (default: from environment)",
    )
    collect_results_parser.add_argument(
        "--after-current",
        action="store_true",
        help="Submit a dependent job that runs after current Slurm jobs finish",
    )

    # export-results (v0.8.0)
    export_results_parser = subparsers.add_parser(
        "export-results", help="Export results to spreadsheet format (v0.8.0)"
    )
    export_results_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Export format (default: csv)",
    )
    export_results_parser.add_argument(
        "--output", help="Output file path (default: auto-generated)"
    )
    export_results_parser.add_argument("--status", help="Filter by status")

    # collect-run-results (v0.9.0 - versioned run history)
    collect_run_results_parser = subparsers.add_parser(
        "collect-run-results",
        help="Collect versioned per-run results into run history database",
    )
    collect_run_results_parser.add_argument(
        "--exp",
        nargs="+",
        help="Specific experiment IDs to process (default: all)",
    )
    collect_run_results_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-harvest existing run records",
    )

    # show-run-history (v0.9.0)
    show_run_history_parser = subparsers.add_parser(
        "show-run-history",
        help="Show versioned run history for an experiment",
    )
    show_run_history_parser.add_argument("exp_id", help="Experiment ID")
    show_run_history_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

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

    # ─── pretrain: SSL backbone pretraining on navtrain ──────────────────
    pretrain_parser = subparsers.add_parser(
        "pretrain", help="SSL backbone pretraining on navtrain (3 methods x 5 datasets)"
    )
    pretrain_subparsers = pretrain_parser.add_subparsers(
        dest="pretrain_command", help="Pretraining sub-commands"
    )

    # pretrain prepare-data
    pretrain_prep = pretrain_subparsers.add_parser(
        "prepare-data", help="Submit Slurm job to extract navtrain images into pickles"
    )
    pretrain_prep.add_argument(
        "--dry-run", action="store_true", help="Generate script but don't submit"
    )
    pretrain_prep.add_argument(
        "--verify", action="store_true", help="Also verify random sample of images"
    )
    pretrain_prep.add_argument(
        "--no-submit", action="store_true", help="Only generate script, don't sbatch"
    )

    # pretrain generate
    pretrain_gen = pretrain_subparsers.add_parser(
        "generate", help="Generate Slurm scripts for pretraining jobs"
    )
    pretrain_gen.add_argument(
        "--methods",
        nargs="+",
        default=["ijepa", "dinov2", "mae"],
        choices=["ijepa", "dinov2", "mae"],
        help="SSL methods to generate scripts for (default: all 3)",
    )
    pretrain_gen.add_argument(
        "--datasets",
        nargs="+",
        default=["all", "boston", "singapore", "loo_boston", "loo_singapore"],
        help="Datasets to use (default: all 5)",
    )
    pretrain_gen.add_argument(
        "--epochs", type=int, default=200, help="Training epochs (default: 200)"
    )
    pretrain_gen.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: method-specific — ijepa:512, mae:768, dinov2:384)",
    )
    pretrain_gen.add_argument(
        "--partition",
        type=str,
        default=None,
        help="Slurm partition (default: from .hpc_config.yaml)",
    )
    pretrain_gen.add_argument(
        "--account",
        type=str,
        default="torch_pr_106_tandon_advanced",
        help="Slurm account",
    )
    pretrain_gen.add_argument(
        "--time", type=str, default="48:00:00", help="Slurm time limit"
    )

    # pretrain submit
    pretrain_sub = pretrain_subparsers.add_parser(
        "submit", help="Submit pretraining Slurm jobs"
    )
    pretrain_sub.add_argument(
        "--methods",
        nargs="+",
        default=["ijepa", "dinov2", "mae"],
        choices=["ijepa", "dinov2", "mae"],
        help="SSL methods to submit (default: all 3)",
    )
    pretrain_sub.add_argument(
        "--datasets",
        nargs="+",
        default=["all", "boston", "singapore", "loo_boston", "loo_singapore"],
        help="Datasets to submit for (default: all 5)",
    )
    pretrain_sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )

    # pretrain status
    pretrain_status = pretrain_subparsers.add_parser(
        "status", help="Show status of all pretraining jobs"
    )

    # pretrain list
    pretrain_list = pretrain_subparsers.add_parser(
        "list", help="List available pretraining datasets and configs"
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
            # Also set specific backbone trainable layers for SSL models
            overrides["pc_trainable_ijepa_layers"] = args.trainable_fraction
            overrides["pc_trainable_dino_layers"] = args.trainable_fraction
            overrides["pc_trainable_dinov2_layers"] = args.trainable_fraction
            overrides["pc_trainable_mae_layers"] = args.trainable_fraction
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
        # LAW and general training/architecture overrides
        if args.weight_decay is not None:
            overrides["weight_decay"] = args.weight_decay
        if args.encoder_lr_mult is not None:
            overrides["encoder_lr_mult"] = args.encoder_lr_mult
        if args.freeze_encoder is not None:
            overrides["freeze_encoder"] = args.freeze_encoder
        if args.use_wm is not None:
            overrides["use_wm"] = args.use_wm
            overrides["use_wm_training"] = args.use_wm
        if hasattr(args, "wm_loss_weight") and args.wm_loss_weight is not None:
            overrides["wm_loss_weight"] = args.wm_loss_weight
        if hasattr(args, "traj_loss_weight") and args.traj_loss_weight is not None:
            overrides["traj_loss_weight"] = args.traj_loss_weight
        if hasattr(args, "gradient_clip_val") and args.gradient_clip_val is not None:
            overrides["gradient_clip_val"] = args.gradient_clip_val
        if args.use_cosine_scheduler is not None:
            overrides["use_cosine_scheduler"] = args.use_cosine_scheduler
        if hasattr(args, "camera_width") and args.camera_width is not None:
            overrides["camera_width"] = args.camera_width
        if hasattr(args, "camera_height") and args.camera_height is not None:
            overrides["camera_height"] = args.camera_height
        if hasattr(args, "image_architecture") and args.image_architecture is not None:
            overrides["image_architecture"] = args.image_architecture
        if (
            hasattr(args, "encoder_weights_path")
            and args.encoder_weights_path is not None
        ):
            overrides["encoder_weights_path"] = args.encoder_weights_path
        if hasattr(args, "precision") and args.precision is not None:
            overrides["precision"] = args.precision
        if hasattr(args, "backbone") and args.backbone is not None:
            overrides["backbone"] = args.backbone
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

    elif args.command == "cancel":
        manager.cancel_experiment(
            args.exp_id,
            include_matching_name=not args.no_name_match,
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

    elif args.command == "collect-results":
        cmd_collect_results(args, manager)

    elif args.command == "export-results":
        cmd_export_results(args, manager)

    elif args.command == "collect-run-results":
        cmd_collect_run_results(args, manager)

    elif args.command == "show-run-history":
        cmd_show_run_history(args, manager)

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

    elif args.command == "pretrain":
        pt = PretrainManager()

        if args.pretrain_command == "prepare-data":
            pt.prepare_data(
                dry_run=args.dry_run,
                verify=args.verify,
                submit=not args.no_submit,
            )

        elif args.pretrain_command == "generate":
            pt.generate_scripts(
                methods=args.methods,
                datasets=args.datasets,
                epochs=args.epochs,
                batch_size=args.batch_size,
                account=args.account,
                time_limit=args.time,
                partition=args.partition,
            )

        elif args.pretrain_command == "submit":
            pt.submit_jobs(
                methods=args.methods,
                datasets=args.datasets,
                dry_run=args.dry_run,
            )

        elif args.pretrain_command == "status":
            pt.show_status()

        elif args.pretrain_command == "list":
            pt.list_configs()

        else:
            # No sub-command: show help
            print("\nSSL Pretraining Manager — ViT-S/14 on navtrain")
            print("=" * 50)
            print("\nWorkflow:")
            print("  1. python navsim_manager.py pretrain prepare-data")
            print("  2. python navsim_manager.py pretrain generate")
            print("  3. python navsim_manager.py pretrain submit --dry-run")
            print("  4. python navsim_manager.py pretrain submit")
            print("  5. python navsim_manager.py pretrain status")
            print("\nOther:")
            print("  python navsim_manager.py pretrain list")


if __name__ == "__main__":
    main()
