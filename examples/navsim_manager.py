#!/usr/bin/env python3
"""
Example: NAVSIM Experiment Manager with Resume Support

Complete experiment manager for NAVSIM planning agents with built-in resume functionality.
Demonstrates how to use ExpFlow's checkpoint resumption features.
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from expflow.hpcexp_core import BaseExperimentManager, BaseExperimentConfig
from expflow.hpc_config import load_project_config


# =============================================================================
# NAVSIM Configuration
# =============================================================================

@dataclass
class NavsimConfig(BaseExperimentConfig):
    """Configuration for NAVSIM planning agent experiments"""

    # Agent and model architecture
    agent: str = "ijepa_planning_agent_v4"
    backbone: str = "ijepa"  # ijepa, resnet50, vit, etc.
    num_cams: int = 6  # 3-cam or 6-cam setup

    # Training data
    train_split: str = "navtrain"
    val_split: str = "navval"
    cache_path: Optional[str] = None  # Path to SquashFS cache

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    optimizer: str = "adamw"
    weight_decay: float = 0.01

    # Checkpoint settings
    checkpoint_interval: int = 5  # Save every N epochs
    save_best_only: bool = False
    metric_for_best: str = "val_loss"  # Metric to track for best checkpoint

    # Evaluation
    eval_split: str = "navtest"
    eval_type: str = "two_stage"  # one_stage, two_stage

    # Container settings
    container_path: str = "/scratch/ah7072/containers/navsim_latest.sif"


# =============================================================================
# NAVSIM Experiment Manager
# =============================================================================

class NavsimExperimentManager(BaseExperimentManager):
    """Experiment manager for NAVSIM planning agents with resume support"""

    def _generate_train_script(self, config: dict) -> str:
        """Generate SLURM training script with resume support"""

        # Check if this is a resumed experiment
        is_resume = config.get("resume_checkpoint_path") is not None
        checkpoint_arg = ""
        start_epoch_arg = ""

        if is_resume:
            checkpoint_path = config["resume_checkpoint_path"]
            resume_epoch = config.get("resume_epoch", 0)
            checkpoint_arg = f"--resume {checkpoint_path}"
            start_epoch_arg = f"--start-epoch {resume_epoch + 1}"

            print(f"Configuring resume from checkpoint: {Path(checkpoint_path).name}")
            print(f"Will start training from epoch {resume_epoch + 1}")

        # Prepare cache overlay if using SquashFS cache
        cache_overlay = ""
        if config.get("cache_path"):
            cache_overlay = f"""
# Mount SquashFS cache with overlay
export CACHE_BASE="{config['cache_path']}"
export OVERLAY_DIR="{self.cache_dir}/overlays/{config['agent']}"
mkdir -p "$OVERLAY_DIR"

# Create overlay mount point
export CACHE_MOUNT="/tmp/navsim_cache_$SLURM_JOB_ID"
mkdir -p "$CACHE_MOUNT"

# Mount with overlay for read-write access
apptainer exec --overlay "$CACHE_BASE:ro" \\
    --overlay "$OVERLAY_DIR" \\
    --bind "$CACHE_MOUNT:/cache" \\
    {config.get('container_path')} \\
    bash -c "
"""
            container_close = '"'
        else:
            container_close = ""

        script = f'''#!/bin/bash
# =============================================================================
# Auto-generated NAVSIM Training Script
# =============================================================================
# Experiment ID: {config['exp_id']}
# Description: {config['description']}
{"# RESUME: From " + config.get('resume_from_exp_id', '') + " (epoch " + str(config.get('resume_epoch', 0)) + ")" if is_resume else ""}
# Generated: {config.get('created_at', 'unknown')}
# =============================================================================

#SBATCH --partition={config.get('partition', 'gpu')}
#SBATCH --gres=gpu:{config.get('num_gpus', 4)}
#SBATCH --nodes={config.get('num_nodes', 1)}
#SBATCH --ntasks-per-node={config.get('num_gpus', 4)}
#SBATCH --cpus-per-task={config.get('cpus_per_task', 16)}
#SBATCH --mem=0
#SBATCH --time={config.get('time_limit', '48:00:00')}
#SBATCH --account={config.get('account', 'default')}
#SBATCH --job-name={config['exp_id']}_train
#SBATCH --output={self.logs_dir}/output/train_{config['exp_id']}_%j.out
#SBATCH --error={self.logs_dir}/error/train_{config['exp_id']}_%j.err

echo "=============================================="
echo "Experiment: {config['exp_id']}"
echo "{config['description']}"
{"echo 'RESUME: From checkpoint at epoch " + str(config.get('resume_epoch', 0)) + "'" if is_resume else ""}
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Conda environment
module load anaconda3/2025.06
source activate navsim

# Paths
export CHECKPOINT_DIR="{self.checkpoints_dir}/{config['exp_id']}"
mkdir -p "$CHECKPOINT_DIR"

export LOG_DIR="{self.logs_dir}/tensorboard/{config['exp_id']}"
mkdir -p "$LOG_DIR"

# Configuration
echo "Configuration:"
echo "  Agent: {config.get('agent', 'ijepa_planning_agent_v4')}"
echo "  Backbone: {config.get('backbone', 'ijepa')}"
echo "  Cameras: {config.get('num_cams', 6)}"
echo "  Batch size: {config.get('batch_size', 32)}"
echo "  Learning rate: {config.get('learning_rate', 0.001)}"
echo "  Epochs: {config.get('epochs', 100)}"
echo "  GPUs: {config.get('num_gpus', 4)}"
{"echo '  Resume from: " + str(config.get('resume_checkpoint_path', '')) + "'" if is_resume else ""}
echo ""

{cache_overlay}

# Training command
python -m torch.distributed.launch \\
    --nproc_per_node={config.get('num_gpus', 4)} \\
    --nnodes={config.get('num_nodes', 1)} \\
    --node_rank=$SLURM_NODEID \\
    train_navsim.py \\
        --agent {config.get('agent', 'ijepa_planning_agent_v4')} \\
        --backbone {config.get('backbone', 'ijepa')} \\
        --num-cams {config.get('num_cams', 6)} \\
        --train-split {config.get('train_split', 'navtrain')} \\
        --val-split {config.get('val_split', 'navval')} \\
        --batch-size {config.get('batch_size', 32)} \\
        --lr {config.get('learning_rate', 0.001)} \\
        --epochs {config.get('epochs', 100)} \\
        --optimizer {config.get('optimizer', 'adamw')} \\
        --weight-decay {config.get('weight_decay', 0.01)} \\
        --checkpoint-dir "$CHECKPOINT_DIR" \\
        --checkpoint-interval {config.get('checkpoint_interval', 5)} \\
        --log-dir "$LOG_DIR" \\
        {checkpoint_arg} \\
        {start_epoch_arg}

{container_close}

TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
fi
echo "=============================================="

exit $TRAIN_EXIT_CODE
'''

        return script

    def _generate_eval_script(self, config: dict) -> str:
        """Generate SLURM evaluation script"""

        # Find the best checkpoint for evaluation
        checkpoint_path = config.get("resume_checkpoint_path")
        if not checkpoint_path:
            # Use best checkpoint from this experiment
            checkpoint_path = f"{self.checkpoints_dir}/{config['exp_id']}/checkpoint_best.pth"

        script = f'''#!/bin/bash
# =============================================================================
# Auto-generated NAVSIM Evaluation Script
# =============================================================================
# Experiment ID: {config['exp_id']}
# Description: {config['description']}
# Generated: {config.get('created_at', 'unknown')}
# =============================================================================

#SBATCH --partition={config.get('partition', 'gpu')}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account={config.get('account', 'default')}
#SBATCH --job-name={config['exp_id']}_eval
#SBATCH --output={self.logs_dir}/output/eval_{config['exp_id']}_%j.out
#SBATCH --error={self.logs_dir}/error/eval_{config['exp_id']}_%j.err

echo "=============================================="
echo "Evaluation: {config['exp_id']}"
echo "{config['description']}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Environment
export PYTHONUNBUFFERED=1

# Conda environment
module load anaconda3/2025.06
source activate navsim

# Paths
export RESULTS_DIR="{self.results_dir}/{config['exp_id']}"
mkdir -p "$RESULTS_DIR"

# Configuration
echo "Configuration:"
echo "  Agent: {config.get('agent', 'ijepa_planning_agent_v4')}"
echo "  Eval split: {config.get('eval_split', 'navtest')}"
echo "  Eval type: {config.get('eval_type', 'two_stage')}"
echo "  Checkpoint: {checkpoint_path}"
echo ""

# Evaluation command
python eval_navsim.py \\
    --agent {config.get('agent', 'ijepa_planning_agent_v4')} \\
    --backbone {config.get('backbone', 'ijepa')} \\
    --checkpoint {checkpoint_path} \\
    --eval-split {config.get('eval_split', 'navtest')} \\
    --eval-type {config.get('eval_type', 'two_stage')} \\
    --output-dir "$RESULTS_DIR"

EVAL_EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Results saved to: $RESULTS_DIR"
else
    echo "Evaluation failed with exit code $EVAL_EXIT_CODE"
fi
echo "=============================================="

exit $EVAL_EXIT_CODE
'''

        return script

    def harvest_results(self, exp_id: str) -> Dict[str, Any]:
        """
        Harvest NAVSIM evaluation results

        Parses PDM scores and training metrics from logs
        """
        results = {}

        # Find results file
        results_dir = self.results_dir / exp_id
        results_file = results_dir / "results.json"

        if results_file.exists():
            import json
            with open(results_file) as f:
                eval_results = json.load(f)
                results["pdm_score"] = eval_results.get("pdm_score")
                results["open_loop_score"] = eval_results.get("open_loop_score")

        # Parse training logs for best validation loss
        log_dir = self.logs_dir / "tensorboard" / exp_id
        if log_dir.exists():
            # Parse TensorBoard logs (implementation depends on your logging format)
            # This is a placeholder - implement based on your actual log format
            results["best_val_loss"] = None

        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NAVSIM Experiment Manager with Resume Support")
    parser.add_argument("command", choices=["new", "submit", "resume", "list", "show", "status", "harvest"])

    # Common arguments
    parser.add_argument("--exp-id", help="Experiment ID")

    # New experiment arguments
    parser.add_argument("--template", help="Template to use for new experiment")
    parser.add_argument("--description", help="Experiment description")

    # Resume arguments
    parser.add_argument("--source-exp", help="Source experiment ID to resume from")
    parser.add_argument("--checkpoint", help="Specific checkpoint path (optional, auto-detects if not provided)")
    parser.add_argument("--new-exp-id", help="New experiment ID for resumed run (optional, auto-generated)")

    # Submit arguments
    parser.add_argument("--train-only", action="store_true", help="Submit only training job")
    parser.add_argument("--eval-only", action="store_true", help="Submit only evaluation job")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts without submitting")

    # List arguments
    parser.add_argument("--status-filter", help="Filter by status")

    args = parser.parse_args()

    # Load HPC configuration
    try:
        hpc_config = load_project_config()
    except FileNotFoundError:
        print("Error: Not in a project directory")
        print("Run 'expflow init <project>' first")
        sys.exit(1)

    # Initialize manager
    manager = NavsimExperimentManager(hpc_config)

    # Execute command
    if args.command == "new":
        if not args.exp_id:
            print("Error: --exp-id required for 'new' command")
            sys.exit(1)

        manager.create_experiment(
            exp_id=args.exp_id,
            template=args.template,
            description=args.description or ""
        )

    elif args.command == "submit":
        if not args.exp_id:
            print("Error: --exp-id required for 'submit' command")
            sys.exit(1)

        manager.submit_experiment(
            exp_id=args.exp_id,
            train_only=args.train_only,
            eval_only=args.eval_only,
            dry_run=args.dry_run
        )

    elif args.command == "resume":
        if not args.source_exp:
            print("Error: --source-exp required for 'resume' command")
            sys.exit(1)

        new_exp_id = manager.resume_experiment(
            source_exp_id=args.source_exp,
            new_exp_id=args.new_exp_id,
            checkpoint_path=args.checkpoint
        )

        print(f"\nNext steps:")
        print(f"  1. Review config: cat experiment_configs/{new_exp_id}.yaml")
        print(f"  2. Submit job: python {sys.argv[0]} submit --exp-id {new_exp_id}")

    elif args.command == "list":
        manager.list_experiments(status=args.status_filter)

    elif args.command == "show":
        if not args.exp_id:
            print("Error: --exp-id required for 'show' command")
            sys.exit(1)

        manager.show_experiment(args.exp_id)

    elif args.command == "status":
        manager.status()

    elif args.command == "harvest":
        if not args.exp_id:
            print("Error: --exp-id required for 'harvest' command")
            sys.exit(1)

        results = manager.harvest_results(args.exp_id)
        print(f"\nResults for {args.exp_id}:")
        for key, value in results.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
