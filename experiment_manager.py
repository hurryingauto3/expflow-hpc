#!/usr/bin/env python3
"""
Experiment Manager for NavSim I-JEPA Planning Agents

Automates the experiment workflow:
1. Define experiments in YAML configs
2. Generate SLURM scripts programmatically
3. Submit jobs with automatic metadata tracking
4. Harvest evaluation results
5. Export to master CSV

Usage:
    # Create a new experiment config
    python experiment_manager.py new --template ijepa_mlp --name b15 --description "Test new LR"

    # Submit an experiment (generates script, logs metadata, submits to SLURM)
    python experiment_manager.py submit b15

    # Harvest results after completion
    python experiment_manager.py harvest b15

    # Export all results to CSV
    python experiment_manager.py export results_master.csv

    # List all experiments
    python experiment_manager.py list

    # Show experiment details
    python experiment_manager.py show b15
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    # Required fields
    exp_id: str
    description: str
    agent: str  # e.g., "ijepa_planning_agent_v3", "ijepa_planning_agent_v4"

    # Training configuration
    batch_size: int = 48
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 3e-5
    trainable_fraction: float = 0.5
    epochs: int = 30
    data_percent: int = 100

    # Agent-specific
    use_multi_camera: bool = True
    camera_views: List[str] = field(default_factory=lambda: ["cam_l0", "cam_f0", "cam_r0"])
    vision_mode: str = "multi_per_view"  # For V4: multi_per_view, front_preprocessed
    image_size: List[int] = field(default_factory=lambda: [224, 224])

    # Architecture details
    backbone: str = "ijepa"  # ijepa, vit, resnet, dino, dinov2
    model_id: Optional[str] = None  # For ViT: "google/vit-huge-patch14-224-in21k"

    # Resource configuration
    partition: str = "l40s_public"
    num_gpus: int = 4
    num_nodes: int = 1
    cpus_per_task: int = 16
    time_limit: str = "48:00:00"
    account: str = "torch_pr_68_tandon_advanced"

    # Cache and paths
    cache_name: str = "training_cache_ijepa_planning_agent_v3_v5"
    use_cache_overlay: bool = True

    # Evaluation
    eval_split: str = "navtest"
    eval_workers: int = 48

    # Metadata (auto-populated)
    created_at: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    train_job_id: Optional[str] = None
    eval_job_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    pdm_score: Optional[float] = None

    # Git tracking
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None

    # Tags for organization
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentMetadata:
    """Runtime metadata for tracking experiments"""
    exp_id: str
    config: ExperimentConfig
    status: str  # created, submitted, training, evaluating, completed, failed
    train_script_path: Optional[str] = None
    eval_script_path: Optional[str] = None
    run_id: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Experiment Manager Class
# =============================================================================

class ExperimentManager:
    """Manages experiment lifecycle from creation to results export"""

    def __init__(self, workspace_root: str = "/scratch/ah7072"):
        self.workspace = Path(workspace_root)
        self.experiments_dir = self.workspace / "experiments"
        self.configs_dir = Path(__file__).parent / "experiment_configs"
        self.scripts_dir = Path(__file__).parent
        self.templates_dir = Path(__file__).parent / "experiment_templates"
        self.metadata_db = self.configs_dir / "experiments.json"
        self.checkpoints_registry = self.experiments_dir / "checkpoints"

        # Create necessary directories
        self.configs_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        self.checkpoints_registry.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata database
        self._load_metadata()

    def _load_metadata(self):
        """Load experiment metadata database"""
        if self.metadata_db.exists():
            with open(self.metadata_db, 'r') as f:
                data = json.load(f)
                self.metadata = {
                    exp_id: ExperimentMetadata(**meta)
                    for exp_id, meta in data.items()
                }
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save experiment metadata database"""
        data = {
            exp_id: {
                **asdict(meta),
                'config': asdict(meta.config)
            }
            for exp_id, meta in self.metadata.items()
        }
        with open(self.metadata_db, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_git_info(self) -> Dict[str, Any]:
        """Get current git commit info"""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.scripts_dir.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.scripts_dir.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Check if working tree is dirty
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.scripts_dir.parent,
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

    def create_experiment(
        self,
        exp_id: str,
        template: str = "ijepa_mlp",
        description: str = "",
        **kwargs
    ) -> ExperimentConfig:
        """Create a new experiment configuration"""

        # Check if experiment already exists
        if exp_id in self.metadata:
            print(f"Warning: Experiment {exp_id} already exists. Overwriting.")

        # Load template or create from scratch
        template_path = self.templates_dir / f"{template}.yaml"
        if template_path.exists():
            with open(template_path, 'r') as f:
                template_config = yaml.safe_load(f)
        else:
            template_config = {}

        # Merge template with kwargs
        config_dict = {
            "exp_id": exp_id,
            "description": description,
            "created_at": datetime.now().isoformat(),
            **self._get_git_info(),
            **template_config,
            **kwargs
        }

        config = ExperimentConfig(**config_dict)

        # Save config as YAML
        config_path = self.configs_dir / f"{exp_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)

        # Create metadata entry
        self.metadata[exp_id] = ExperimentMetadata(
            exp_id=exp_id,
            config=config,
            status="created"
        )
        self._save_metadata()

        print(f"Created experiment: {exp_id}")
        print(f"Config saved to: {config_path}")

        return config

    def _generate_train_script(self, config: ExperimentConfig) -> str:
        """Generate SLURM training script from config"""

        # Determine partition settings based on config
        if config.partition == "h200_tandon":
            partition_line = f"#SBATCH --partition={config.partition}"
            gres_line = f"#SBATCH --gres=gpu:{config.num_gpus}"
            ntasks_line = f"#SBATCH --ntasks-per-node={config.num_gpus}"
        else:  # l40s_public
            partition_line = f"#SBATCH --partition={config.partition}"
            gres_line = f"#SBATCH --gres=gpu:{config.num_gpus}"
            ntasks_line = f"#SBATCH --ntasks-per-node={config.num_gpus}"

        # Build agent-specific arguments
        agent_args = []
        if config.agent == "ijepa_planning_agent_v4":
            agent_args.extend([
                f"agent.vision_mode={config.vision_mode}",
                f"agent.vision_views=[{','.join(config.camera_views)}]",
                f"agent.vision_image_size=[{','.join(map(str, config.image_size))}]",
                "agent.vision_crop=center",
                "agent.vision_resample=bicubic",
                "agent.vision_normalize=true",
            ])

        if config.model_id:
            agent_args.append(f"agent.ijepa_model_id={config.model_id}")

        agent_args_str = " \\\n    ".join(agent_args) if agent_args else ""
        if agent_args_str:
            agent_args_str = " \\\n    " + agent_args_str

        script = f'''#!/bin/bash
# =============================================================================
# Auto-generated SLURM Training Script
# =============================================================================
# Experiment ID: {config.exp_id}
# Description: {config.description}
# Generated: {datetime.now().isoformat()}
# Git commit: {config.git_commit or "unknown"}
# =============================================================================

{partition_line}
{ntasks_line}
{gres_line}
#SBATCH --account={config.account}
#SBATCH --nodes={config.num_nodes}
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem=0
#SBATCH --job-name={config.exp_id}_train
#SBATCH --time={config.time_limit}
#SBATCH --requeue
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_{config.exp_id}_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_{config.exp_id}_%j.err

echo "=============================================="
echo "Experiment: {config.exp_id}"
echo "{config.description}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_refactor"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Optional I-JEPA checkpoint
export IJEPA_CKPT_PATH="${{IJEPA_CKPT_PATH:-}}"
export IJEPA_CKPT_WHICH="${{IJEPA_CKPT_WHICH:-encoder}}"

EXTRA_AGENT_ARGS=""
if [ -n "${{IJEPA_CKPT_PATH}}" ]; then
    EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_path=${{IJEPA_CKPT_PATH}}"
    EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_which=${{IJEPA_CKPT_WHICH}}"
fi

# Cache configuration
export CACHE_NAME="{config.cache_name}"
export CACHE_PATH="${{NAVSIM_EXP_ROOT}}/cache/${{CACHE_NAME}}"
export CACHE_OVERLAY="/scratch/ah7072/overlays/${{CACHE_NAME}}.sqsh"

# Container
export CONTAINER="/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

# Threading
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# Training hyperparameters
export BATCH_SIZE={config.batch_size}
export LEARNING_RATE={config.learning_rate}
export ENCODER_LEARNING_RATE={config.encoder_learning_rate}
export NUM_WORKERS=12

# NCCL optimizations
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=2

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Multi-node DDP
export MASTER_PORT=12360
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

# Experiment naming
export EXPERIMENT_NAME="training/exp_{config.exp_id}_${{RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"

mkdir -p /scratch/ah7072/experiments/logs/output
mkdir -p /scratch/ah7072/experiments/logs/error

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Load conda
CONDA_ROOT="/scratch/$USER/miniconda3"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate navsim
else
    module purge || true
    module load anaconda3/2025.06 || true
    if command -v conda &> /dev/null; then
        source $(conda info --base)/etc/profile.d/conda.sh || true
        conda activate navsim || source activate navsim || true
    fi
fi

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"

echo "Configuration:"
echo "  Agent: {config.agent}"
echo "  Backbone: {config.backbone}"
echo "  Batch size: {config.batch_size}"
echo "  Learning rate: {config.learning_rate}"
echo "  Encoder LR: {config.encoder_learning_rate}"
echo "  Trainable fraction: {config.trainable_fraction}"
echo "  Multi-camera: {str(config.use_multi_camera).lower()}"
echo "  Epochs: {config.epochs}"
echo "  GPUs: {config.num_gpus}"
echo ""

# Check overlay
if [ -f "${{CACHE_OVERLAY}}" ]; then
    echo "✓ Using SquashFS overlay for cache"
    USE_OVERLAY=true
else
    echo "⚠ No overlay, using directory cache"
    USE_OVERLAY=false
fi

# GPU monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > /scratch/ah7072/experiments/logs/output/gpu_util_{config.exp_id}_${{SLURM_JOB_ID}}.csv &
GPU_MONITOR_PID=$!

# Training
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/train_{config.exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'SCRIPT_EOF'
#!/bin/bash
source /scratch/ah7072/miniconda3/etc/profile.d/conda.sh
conda activate navsim
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

EXTRA_AGENT_ARGS=""
if [ -n "${{IJEPA_CKPT_PATH:-}}" ]; then
    EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_path=${{IJEPA_CKPT_PATH}}"
    EXTRA_AGENT_ARGS+=" agent.ijepa_ckpt_which=${{IJEPA_CKPT_WHICH:-encoder}}"
fi

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_training.py \\
    agent={config.agent} \\
    agent.trainable_ijepa_layers_fraction={config.trainable_fraction} \\
    agent.use_multi_camera={str(config.use_multi_camera).lower()} \\
    agent.learning_rate=${{LEARNING_RATE}} \\
    agent.encoder_learning_rate=${{ENCODER_LEARNING_RATE}}{agent_args_str} \\
    experiment_name=${{EXPERIMENT_NAME}} \\
    train_test_split=navtrain \\
    cache_path=${{CACHE_PATH}} \\
    use_cache_without_dataset=true \\
    force_cache_computation=false \\
    trainer.params.max_epochs={config.epochs} \\
    trainer.params.accelerator=gpu \\
    trainer.params.strategy=ddp \\
    trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
    trainer.params.gradient_clip_val=1.0 \\
    trainer.params.accumulate_grad_batches=1 \\
    dataloader.params.batch_size=${{BATCH_SIZE}} \\
    dataloader.params.num_workers=${{NUM_WORKERS}} \\
    dataloader.params.prefetch_factor=4 \\
    dataloader.params.pin_memory=true \\
    ${{EXTRA_AGENT_ARGS}}
SCRIPT_EOF
    chmod +x "${{TEMP_SCRIPT}}"

    srun --gres=gpu:{config.num_gpus} apptainer exec \\
        --nv \\
        --bind "${{CACHE_OVERLAY}}:${{CACHE_PATH}}:image-src=/" \\
        --bind /scratch/ah7072:/scratch/ah7072 \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "NAVSIM_DEVKIT_ROOT=${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "OPENSCENE_DATA_ROOT=${{OPENSCENE_DATA_ROOT}}" \\
        --env "NUPLAN_MAPS_ROOT=${{NUPLAN_MAPS_ROOT}}" \\
        --env "NAVSIM_EXP_ROOT=${{NAVSIM_EXP_ROOT}}" \\
        --env "CACHE_PATH=${{CACHE_PATH}}" \\
        --env "EXPERIMENT_NAME=${{EXPERIMENT_NAME}}" \\
        --env "BATCH_SIZE=${{BATCH_SIZE}}" \\
        --env "LEARNING_RATE=${{LEARNING_RATE}}" \\
        --env "ENCODER_LEARNING_RATE=${{ENCODER_LEARNING_RATE}}" \\
        --env "NUM_WORKERS=${{NUM_WORKERS}}" \\
        --env "SLURM_JOB_NUM_NODES=${{SLURM_JOB_NUM_NODES}}" \\
        --env "MASTER_ADDR=${{MASTER_ADDR}}" \\
        --env "MASTER_PORT=${{MASTER_PORT}}" \\
        --env "NCCL_IB_DISABLE=${{NCCL_IB_DISABLE}}" \\
        --env "NCCL_P2P_LEVEL=${{NCCL_P2P_LEVEL}}" \\
        --env "NCCL_NET_GDR_LEVEL=${{NCCL_NET_GDR_LEVEL}}" \\
        --env "IJEPA_CKPT_PATH=${{IJEPA_CKPT_PATH}}" \\
        --env "IJEPA_CKPT_WHICH=${{IJEPA_CKPT_WHICH}}" \\
        "${{CONTAINER}}" \\
        bash "${{TEMP_SCRIPT}}"

    rm -f "${{TEMP_SCRIPT}}"
else
    srun --gres=gpu:{config.num_gpus} python navsim/planning/script/run_training.py \\
        agent={config.agent} \\
        agent.trainable_ijepa_layers_fraction={config.trainable_fraction} \\
        agent.use_multi_camera={str(config.use_multi_camera).lower()} \\
        agent.learning_rate=${{LEARNING_RATE}} \\
        agent.encoder_learning_rate=${{ENCODER_LEARNING_RATE}}{agent_args_str} \\
        experiment_name="${{EXPERIMENT_NAME}}" \\
        train_test_split=navtrain \\
        cache_path="${{CACHE_PATH}}" \\
        use_cache_without_dataset=true \\
        force_cache_computation=false \\
        trainer.params.max_epochs={config.epochs} \\
        trainer.params.accelerator=gpu \\
        trainer.params.strategy=ddp \\
        trainer.params.num_nodes=${{SLURM_JOB_NUM_NODES}} \\
        trainer.params.gradient_clip_val=1.0 \\
        trainer.params.accumulate_grad_batches=1 \\
        dataloader.params.batch_size=${{BATCH_SIZE}} \\
        dataloader.params.num_workers=${{NUM_WORKERS}} \\
        dataloader.params.prefetch_factor=4 \\
        dataloader.params.pin_memory=true \\
        ${{EXTRA_AGENT_ARGS}}
fi

kill $GPU_MONITOR_PID 2>/dev/null || true
TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=============================================="

# Find checkpoint
CHECKPOINT_DIR=$(find "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}" -type d -name "checkpoints" 2>/dev/null | head -1)
if [ -n "$CHECKPOINT_DIR" ]; then
    BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/epoch*.ckpt 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT=$(ls -t "${{CHECKPOINT_DIR}}"/*.ckpt 2>/dev/null | head -1)
    fi

    if [ -n "$BEST_CKPT" ]; then
        echo "Checkpoint: ${{BEST_CKPT}}"

        CKPT_REGISTRY="${{NAVSIM_EXP_ROOT}}/checkpoints"
        mkdir -p "${{CKPT_REGISTRY}}"

        REL_CKPT=${{BEST_CKPT#${{NAVSIM_EXP_ROOT}}/}}
        CKPT_ENTRY=${{REL_CKPT:-${{BEST_CKPT}}}}

        if [ -n "${{RUN_ID:-}}" ]; then
            echo "${{CKPT_ENTRY}}" > "${{CKPT_REGISTRY}}/${{RUN_ID}}.txt"
            echo "Saved to registry: ${{CKPT_REGISTRY}}/${{RUN_ID}}.txt"
        else
            echo "${{CKPT_ENTRY}}" > "${{CKPT_REGISTRY}}/${{SLURM_JOB_ID}}.txt"
            echo "Saved to registry: ${{CKPT_REGISTRY}}/${{SLURM_JOB_ID}}.txt"
        fi

        echo "${{CKPT_ENTRY}}" > "${{NAVSIM_EXP_ROOT}}/${{EXPERIMENT_NAME}}/checkpoint_path.txt"
    fi
fi

exit $TRAIN_EXIT_CODE
'''
        return script

    def _generate_eval_script(self, config: ExperimentConfig) -> str:
        """Generate SLURM evaluation script from config"""

        # Determine agent config based on version
        agent_config = config.agent
        multi_cam = str(config.use_multi_camera).lower()

        script = f'''#!/bin/bash
#SBATCH --job-name={config.exp_id}_eval
#SBATCH --account={config.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/eval_{config.exp_id}_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/eval_{config.exp_id}_%j.err
#SBATCH --requeue

# =============================================================================
# Auto-generated Evaluation Script
# =============================================================================
# Experiment: {config.exp_id}
# Description: {config.description}
# Generated: {datetime.now().isoformat()}
# =============================================================================

export AGENT="{agent_config}"
export EVAL_SPLIT="{config.eval_split}"
export MULTI_CAM={multi_cam}
export NUM_WORKERS={config.eval_workers}

# Checkpoint discovery
CKPT_REGISTRY="/scratch/ah7072/experiments/checkpoints"

if [ -n "${{CHECKPOINT:-}}" ]; then
    echo "Using checkpoint from env: ${{CHECKPOINT}}"
elif [ -n "${{RUN_ID:-}}" ]; then
    CKPT_FILE="${{CKPT_REGISTRY}}/${{RUN_ID}}.txt"
    if [ -f "$CKPT_FILE" ]; then
        export CHECKPOINT=$(cat "$CKPT_FILE")
        echo "Using checkpoint for run ${{RUN_ID}}: ${{CHECKPOINT}}"
    else
        echo "ERROR: No checkpoint found for RUN_ID: ${{RUN_ID}}"
        exit 1
    fi
else
    echo "ERROR: No CHECKPOINT or RUN_ID specified"
    exit 1
fi

# Derive run name
export RUN_NAME="eval_${{EVAL_SPLIT}}_exp_{config.exp_id}"

# Environment
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_refactor"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export OUTPUT_DIR="${{NAVSIM_EXP_ROOT}}/evaluations/${{RUN_NAME}}"
export METRIC_CACHE="${{NAVSIM_EXP_ROOT}}/cache/${{EVAL_SPLIT}}_metric_cache"
export METRIC_CACHE_OVERLAY="/scratch/ah7072/overlays/${{EVAL_SPLIT}}_metric_cache.sqsh"
export CONTAINER="/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

# Prepend NAVSIM_EXP_ROOT if relative path
if [[ "${{CHECKPOINT}}" != /* ]]; then
    CHECKPOINT="${{NAVSIM_EXP_ROOT}}/${{CHECKPOINT}}"
fi

mkdir -p "${{OUTPUT_DIR}}"
mkdir -p /scratch/ah7072/experiments/logs/output
mkdir -p /scratch/ah7072/experiments/logs/error

echo "=============================================="
echo "Evaluation: {config.exp_id}"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Checkpoint: ${{CHECKPOINT}}"
echo "Output: ${{OUTPUT_DIR}}"
echo "=============================================="

# Validate checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

cd "${{NAVSIM_DEVKIT_ROOT}}"

# Load conda
CONDA_ROOT="/scratch/$USER/miniconda3"
if [ -f "${{CONDA_ROOT}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_ROOT}}/etc/profile.d/conda.sh"
    conda activate navsim
else
    module purge || true
    module load anaconda3/2025.06 || true
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate navsim
fi

export PYTHONPATH="${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}"
export HYDRA_FULL_ERROR=1

# Check overlay
if [ -f "${{METRIC_CACHE_OVERLAY}}" ]; then
    echo "✓ Using SquashFS overlay for metric cache"
    USE_OVERLAY=true
else
    echo "⚠ No overlay, using directory cache"
    USE_OVERLAY=false
fi

START_TIME=$(date +%s)

echo "Starting evaluation at $(date)"

# Run evaluation
if [ "$USE_OVERLAY" = true ]; then
    TEMP_SCRIPT=$(mktemp /tmp/eval_{config.exp_id}_XXXXXX.sh)
    cat > "${{TEMP_SCRIPT}}" << 'SCRIPT_EOF'
#!/bin/bash
source /scratch/ah7072/miniconda3/etc/profile.d/conda.sh
conda activate navsim
export PYTHONPATH=${{NAVSIM_DEVKIT_ROOT}}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1

python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_pdm_score_one_stage.py \\
    train_test_split=${{EVAL_SPLIT}} \\
    experiment_name=${{RUN_NAME}} \\
    traffic_agents=non_reactive \\
    metric_cache_path=${{METRIC_CACHE}} \\
    output_dir=${{OUTPUT_DIR}} \\
    agent=${{AGENT}} \\
    "agent.checkpoint_path='${{CHECKPOINT_PATH}}'" \\
    agent.use_multi_camera=${{MULTI_CAM}} \\
    worker=ray_distributed \\
    worker.threads_per_node=${{NUM_WORKERS}}
SCRIPT_EOF
    chmod +x "${{TEMP_SCRIPT}}"

    apptainer exec \\
        --bind "${{METRIC_CACHE_OVERLAY}}:${{METRIC_CACHE}}:image-src=/" \\
        --bind /scratch/ah7072:/scratch/ah7072 \\
        --bind /tmp:/tmp \\
        --pwd "${{NAVSIM_DEVKIT_ROOT}}" \\
        --env "CHECKPOINT_PATH=${{CHECKPOINT}}" \\
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
        --env "MULTI_CAM=${{MULTI_CAM}}" \\
        "${{CONTAINER}}" \\
        bash "${{TEMP_SCRIPT}}"

    rm -f "${{TEMP_SCRIPT}}"
else
    python ${{NAVSIM_DEVKIT_ROOT}}/navsim/planning/script/run_pdm_score_one_stage.py \\
        train_test_split="${{EVAL_SPLIT}}" \\
        experiment_name="${{RUN_NAME}}" \\
        traffic_agents=non_reactive \\
        metric_cache_path="${{METRIC_CACHE}}" \\
        output_dir="${{OUTPUT_DIR}}" \\
        agent="${{AGENT}}" \\
        agent.checkpoint_path="'${{CHECKPOINT}}'" \\
        agent.use_multi_camera=${{MULTI_CAM}} \\
        worker=ray_distributed \\
        worker.threads_per_node=${{NUM_WORKERS}}
fi

EVAL_EXIT_CODE=$?
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "Runtime: $((RUNTIME / 60)) minutes"
echo "Exit code: ${{EVAL_EXIT_CODE}}"
echo ""

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS"
    echo "Results: ${{OUTPUT_DIR}}"

    # Parse results
    if ls "${{OUTPUT_DIR}}"/*.csv 1> /dev/null 2>&1; then
        python -c "
import pandas as pd
import glob
csv_file = glob.glob('${{OUTPUT_DIR}}/*.csv')[0]
df = pd.read_csv(csv_file)
print(f'  PDM Score: {{df[\\"pdm_score\\"].mean():.4f}}')
print(f'  No Collision: {{df[\\"no_collision\\"].mean():.4f}}')
print(f'  Drivable Area: {{df[\\"drivable_area_compliance\\"].mean():.4f}}')
print(f'  Ego Progress: {{df[\\"ego_progress\\"].mean():.4f}}')
" 2>/dev/null || echo "  (CSV parsing failed)"
    fi
else
    echo "✗ FAILED"
fi

echo "=============================================="

exit $EVAL_EXIT_CODE
'''
        return script

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
            print("Run 'python experiment_manager.py new' to create it first.")
            sys.exit(1)

        meta = self.metadata[exp_id]
        config = meta.config

        # Generate scripts
        generated_scripts_dir = self.scripts_dir / "generated"
        generated_scripts_dir.mkdir(exist_ok=True)

        train_script_path = generated_scripts_dir / f"train_{exp_id}.slurm"
        eval_script_path = generated_scripts_dir / f"eval_{exp_id}.slurm"

        if not eval_only:
            train_script = self._generate_train_script(config)
            with open(train_script_path, 'w') as f:
                f.write(train_script)
            meta.train_script_path = str(train_script_path)
            print(f"Generated training script: {train_script_path}")

        if not train_only:
            eval_script = self._generate_eval_script(config)
            with open(eval_script_path, 'w') as f:
                f.write(eval_script)
            meta.eval_script_path = str(eval_script_path)
            print(f"Generated evaluation script: {eval_script_path}")

        if dry_run:
            print("\n[DRY RUN] Would submit the following jobs:")
            if not eval_only:
                print(f"  Training: sbatch {train_script_path}")
            if not train_only:
                print(f"  Evaluation: sbatch --dependency=afterok:TRAIN_JOB_ID {eval_script_path}")
            return {}

        # Submit to SLURM using train_and_eval.sh wrapper
        wrapper_script = self.scripts_dir / "train_and_eval.sh"

        cmd = [str(wrapper_script)]

        if not eval_only:
            cmd.extend(["--train", str(train_script_path)])

        if not train_only:
            cmd.extend(["--eval", str(eval_script_path)])

        print(f"\nSubmitting experiment {exp_id}...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print(result.stdout)

            # Parse job IDs from output
            train_job_id = None
            eval_job_id = None
            run_id = None

            for line in result.stdout.split('\n'):
                if "Submitted batch job" in line and "train" in line.lower():
                    train_job_id = line.split()[-1]
                elif "Submitted batch job" in line and "eval" in line.lower():
                    eval_job_id = line.split()[-1]
                elif "Run ID:" in line:
                    run_id = line.split(":")[-1].strip()

            # Update metadata
            meta.status = "submitted"
            meta.submitted_at = datetime.now().isoformat()
            meta.train_job_id = train_job_id
            meta.eval_job_id = eval_job_id
            meta.run_id = run_id

            config.submitted_at = meta.submitted_at
            config.train_job_id = train_job_id
            config.eval_job_id = eval_job_id

            self._save_metadata()

            print(f"\n✓ Experiment {exp_id} submitted successfully")
            if train_job_id:
                print(f"  Training job: {train_job_id}")
            if eval_job_id:
                print(f"  Evaluation job: {eval_job_id}")
            if run_id:
                print(f"  Run ID: {run_id}")

            return {
                "train_job_id": train_job_id,
                "eval_job_id": eval_job_id,
                "run_id": run_id
            }

        except subprocess.CalledProcessError as e:
            print(f"Error submitting jobs:")
            print(e.stderr)
            sys.exit(1)

    def harvest_results(self, exp_id: str) -> Dict[str, Any]:
        """Harvest evaluation results for an experiment"""

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            sys.exit(1)

        meta = self.metadata[exp_id]
        config = meta.config

        # Find evaluation directory
        eval_dir_pattern = f"eval_{config.eval_split}_exp_{exp_id}_*"
        eval_dirs = list((self.experiments_dir / "evaluations").glob(eval_dir_pattern))

        if not eval_dirs:
            print(f"No evaluation results found for {exp_id}")
            print(f"Looking for: {eval_dir_pattern}")
            return {}

        # Use the most recent evaluation
        eval_dir = sorted(eval_dirs, key=lambda p: p.stat().st_mtime)[-1]

        print(f"Found evaluation directory: {eval_dir}")

        # Parse CSV results
        csv_files = list(eval_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV results found in {eval_dir}")
            return {}

        csv_file = csv_files[0]
        print(f"Parsing results from: {csv_file}")

        try:
            import pandas as pd
            df = pd.read_csv(csv_file)

            results = {
                "pdm_score": float(df["pdm_score"].mean()),
                "no_collision": float(df["no_collision"].mean()),
                "drivable_area_compliance": float(df["drivable_area_compliance"].mean()),
                "ego_progress": float(df["ego_progress"].mean()),
                "time_to_collision": float(df["time_to_collision"].mean()),
                "comfort": float(df["comfort"].mean()),
                "num_scenarios": len(df),
                "eval_directory": str(eval_dir),
                "harvested_at": datetime.now().isoformat()
            }

            # Update metadata
            meta.results = results
            meta.status = "completed"
            meta.completed_at = results["harvested_at"]

            config.pdm_score = results["pdm_score"]
            config.completed_at = results["harvested_at"]

            self._save_metadata()

            print(f"\n✓ Results harvested for {exp_id}")
            print(f"  PDM Score: {results['pdm_score']:.4f}")
            print(f"  No Collision: {results['no_collision']:.4f}")
            print(f"  Drivable Area: {results['drivable_area_compliance']:.4f}")

            return results

        except Exception as e:
            print(f"Error parsing results: {e}")
            return {}

    def export_results(self, output_file: str = "results_master.csv"):
        """Export all experiment results to CSV"""

        import pandas as pd

        records = []
        for exp_id, meta in self.metadata.items():
            config = meta.config

            record = {
                "exp_id": exp_id,
                "description": config.description,
                "agent": config.agent,
                "backbone": config.backbone,
                "data_percent": config.data_percent,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "encoder_lr": config.encoder_learning_rate,
                "trainable_fraction": config.trainable_fraction,
                "epochs": config.epochs,
                "multi_camera": config.use_multi_camera,
                "num_gpus": config.num_gpus,
                "pdm_score": config.pdm_score if config.pdm_score else None,
                "status": meta.status,
                "train_job_id": config.train_job_id,
                "eval_job_id": meta.eval_job_id,
                "git_commit": config.git_commit,
                "created_at": config.created_at,
                "submitted_at": config.submitted_at,
                "completed_at": config.completed_at,
                "tags": ",".join(config.tags) if config.tags else "",
                "notes": config.notes
            }

            # Add detailed results if available
            if meta.results:
                record.update({
                    "no_collision": meta.results.get("no_collision"),
                    "drivable_area": meta.results.get("drivable_area_compliance"),
                    "ego_progress": meta.results.get("ego_progress"),
                    "time_to_collision": meta.results.get("time_to_collision"),
                    "comfort": meta.results.get("comfort"),
                })

            records.append(record)

        df = pd.DataFrame(records)

        # Sort by PDM score (descending), then by exp_id
        df = df.sort_values(by=["pdm_score", "exp_id"], ascending=[False, True])

        output_path = Path(output_file)
        df.to_csv(output_path, index=False)

        print(f"✓ Exported {len(records)} experiments to {output_path}")
        print(f"\nTop 5 experiments:")
        print(df[["exp_id", "pdm_score", "status", "description"]].head(5).to_string(index=False))

    def list_experiments(self, status: Optional[str] = None, tags: Optional[List[str]] = None):
        """List all experiments with optional filtering"""

        filtered = []
        for exp_id, meta in self.metadata.items():
            if status and meta.status != status:
                continue
            if tags and not any(tag in meta.config.tags for tag in tags):
                continue
            filtered.append((exp_id, meta))

        if not filtered:
            print("No experiments found")
            return

        print(f"\nFound {len(filtered)} experiments:")
        print(f"{'ID':<10} {'Status':<12} {'PDM':<8} {'Description':<50}")
        print("-" * 85)

        for exp_id, meta in sorted(filtered, key=lambda x: x[0]):
            config = meta.config
            pdm = f"{config.pdm_score:.4f}" if config.pdm_score else "N/A"
            desc = config.description[:47] + "..." if len(config.description) > 50 else config.description
            print(f"{exp_id:<10} {meta.status:<12} {pdm:<8} {desc:<50}")

    def show_experiment(self, exp_id: str):
        """Show detailed information about an experiment"""

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            sys.exit(1)

        meta = self.metadata[exp_id]
        config = meta.config

        print(f"\n{'='*70}")
        print(f"Experiment: {exp_id}")
        print(f"{'='*70}")
        print(f"\nDescription: {config.description}")
        print(f"Status: {meta.status}")
        print(f"\nConfiguration:")
        print(f"  Agent: {config.agent}")
        print(f"  Backbone: {config.backbone}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Encoder LR: {config.encoder_learning_rate}")
        print(f"  Trainable fraction: {config.trainable_fraction}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Multi-camera: {config.use_multi_camera}")
        print(f"  GPUs: {config.num_gpus}")
        print(f"  Partition: {config.partition}")

        if config.pdm_score:
            print(f"\nResults:")
            print(f"  PDM Score: {config.pdm_score:.4f}")
            if meta.results:
                print(f"  No Collision: {meta.results.get('no_collision', 'N/A')}")
                print(f"  Drivable Area: {meta.results.get('drivable_area_compliance', 'N/A')}")
                print(f"  Ego Progress: {meta.results.get('ego_progress', 'N/A')}")

        print(f"\nTimeline:")
        print(f"  Created: {config.created_at}")
        if config.submitted_at:
            print(f"  Submitted: {config.submitted_at}")
        if config.completed_at:
            print(f"  Completed: {config.completed_at}")

        if config.train_job_id:
            print(f"\nJobs:")
            print(f"  Training: {config.train_job_id}")
            if meta.eval_job_id:
                print(f"  Evaluation: {meta.eval_job_id}")

        if config.git_commit:
            print(f"\nGit:")
            print(f"  Commit: {config.git_commit[:8]}")
            print(f"  Branch: {config.git_branch}")
            if config.git_dirty:
                print(f"  ⚠ Working tree was dirty")

        if config.tags:
            print(f"\nTags: {', '.join(config.tags)}")

        if config.notes:
            print(f"\nNotes: {config.notes}")

        print(f"\n{'='*70}\n")


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NavSim Experiment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # New experiment
    new_parser = subparsers.add_parser("new", help="Create a new experiment")
    new_parser.add_argument("--exp-id", required=True, help="Experiment ID (e.g., b15)")
    new_parser.add_argument("--template", default="ijepa_mlp", help="Template name")
    new_parser.add_argument("--description", required=True, help="Experiment description")
    new_parser.add_argument("--agent", help="Agent name (overrides template)")
    new_parser.add_argument("--batch-size", type=int, help="Batch size")
    new_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    new_parser.add_argument("--backbone", help="Backbone name")
    new_parser.add_argument("--tags", nargs="+", help="Tags for organization")

    # Submit experiment
    submit_parser = subparsers.add_parser("submit", help="Submit experiment to SLURM")
    submit_parser.add_argument("exp_id", help="Experiment ID")
    submit_parser.add_argument("--train-only", action="store_true", help="Only submit training")
    submit_parser.add_argument("--eval-only", action="store_true", help="Only submit evaluation")
    submit_parser.add_argument("--dry-run", action="store_true", help="Generate scripts without submitting")

    # Harvest results
    harvest_parser = subparsers.add_parser("harvest", help="Harvest evaluation results")
    harvest_parser.add_argument("exp_id", help="Experiment ID")

    # Export results
    export_parser = subparsers.add_parser("export", help="Export all results to CSV")
    export_parser.add_argument("output", nargs="?", default="results_master.csv", help="Output CSV file")

    # List experiments
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")

    # Show experiment
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id", help="Experiment ID")

    # Resource advisor
    resource_parser = subparsers.add_parser("resources", help="Get resource recommendations")
    resource_parser.add_argument("--exp-id", help="Experiment ID to optimize for")
    resource_parser.add_argument("--global-batch", type=int, default=192, help="Target global batch size")
    resource_parser.add_argument("--use-gemini", action="store_true", help="Use Gemini API for AI suggestions")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize manager
    manager = ExperimentManager()

    # Execute command
    if args.command == "new":
        kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ["command", "exp_id", "template", "description"]}
        manager.create_experiment(
            exp_id=args.exp_id,
            template=args.template,
            description=args.description,
            **kwargs
        )

    elif args.command == "submit":
        manager.submit_experiment(
            exp_id=args.exp_id,
            train_only=args.train_only,
            eval_only=args.eval_only,
            dry_run=args.dry_run
        )

    elif args.command == "harvest":
        manager.harvest_results(args.exp_id)

    elif args.command == "export":
        manager.export_results(args.output)

    elif args.command == "list":
        manager.list_experiments(status=args.status, tags=args.tags)

    elif args.command == "show":
        manager.show_experiment(args.exp_id)

    elif args.command == "resources":
        # Import resource advisor
        try:
            from resource_advisor import ResourceAdvisor
        except ImportError:
            print("Error: resource_advisor.py not found")
            print("Make sure resource_advisor.py is in the same directory")
            sys.exit(1)

        advisor = ResourceAdvisor()

        # Load experiment config if provided
        exp_config = None
        if args.exp_id and args.exp_id in manager.metadata:
            exp_config = asdict(manager.metadata[args.exp_id].config)

        # Get recommendations
        recommendations = advisor.get_recommendations(
            exp_config=exp_config,
            target_global_batch=args.global_batch
        )

        # Get Gemini suggestion if requested
        gemini_suggestion = None
        if args.use_gemini:
            partition_stats = advisor.get_queue_status()
            gemini_suggestion = advisor.get_gemini_suggestion(
                partition_stats,
                exp_config
            )

        # Print results
        advisor.print_status()
        advisor.print_recommendations(recommendations, gemini_suggestion)


if __name__ == "__main__":
    main()
