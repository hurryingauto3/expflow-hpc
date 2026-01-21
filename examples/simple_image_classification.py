#!/usr/bin/env python3
"""
Example: Simple Image Classification Experiment Manager

Shows how to adapt the generic HPC framework for image classification tasks.
Works with any model (ResNet, ViT, etc.) and any dataset (ImageNet, CIFAR, etc.)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import framework
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpcexp_core import BaseExperimentManager, BaseExperimentConfig
from hpc_config import load_project_config
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Custom Configuration
# =============================================================================

@dataclass
class ImageClassificationConfig(BaseExperimentConfig):
    """Configuration specific to image classification"""

    # Model architecture
    model: str = "resnet50"  # resnet50, vit_b_16, efficientnet_b0, etc.
    pretrained: bool = True

    # Dataset
    dataset: str = "imagenet"  # imagenet, cifar10, cifar100
    data_path: str = "/scratch/USERID/data"
    num_classes: int = 1000

    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    epochs: int = 90

    # Optimization
    optimizer: str = "sgd"  # sgd, adam, adamw
    scheduler: str = "cosine"  # cosine, step, multistep
    warmup_epochs: int = 5

    # Augmentation
    augmentation: List[str] = field(default_factory=lambda: ["random_crop", "random_flip"])

    # Evaluation
    eval_batch_size: int = 512
    test_split: str = "val"


# =============================================================================
# Custom Experiment Manager
# =============================================================================

class ImageClassificationManager(BaseExperimentManager):
    """Experiment manager for image classification tasks"""

    def _generate_train_script(self, config: dict) -> str:
        """Generate SLURM training script"""

        # Replace USERID with actual username
        data_path = config.get('data_path', '/scratch/USERID/data')
        data_path = data_path.replace('USERID', self.hpc_config.username)

        script = f'''#!/bin/bash
# =============================================================================
# Auto-generated Training Script for Image Classification
# =============================================================================
# Experiment ID: {config['exp_id']}
# Description: {config['description']}
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
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Conda environment (modify as needed)
module load anaconda3/2025.06
source activate pytorch

# Data paths
export DATA_ROOT="{data_path}"
export CHECKPOINT_DIR="{self.checkpoints_dir}/{config['exp_id']}"
mkdir -p "$CHECKPOINT_DIR"

# Model configuration
export MODEL="{config.get('model', 'resnet50')}"
export DATASET="{config.get('dataset', 'imagenet')}"
export BATCH_SIZE={config.get('batch_size', 256)}
export LEARNING_RATE={config.get('learning_rate', 0.1)}
export EPOCHS={config.get('epochs', 90)}

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  GPUs: {config.get('num_gpus', 4)}"
echo ""

# Training script (example - modify for your codebase)
python -m torch.distributed.launch \\
    --nproc_per_node={config.get('num_gpus', 4)} \\
    --nnodes={config.get('num_nodes', 1)} \\
    --node_rank=$SLURM_NODEID \\
    train.py \\
        --model "$MODEL" \\
        --dataset "$DATASET" \\
        --data-path "$DATA_ROOT" \\
        --batch-size $BATCH_SIZE \\
        --lr $LEARNING_RATE \\
        --epochs $EPOCHS \\
        --optimizer {config.get('optimizer', 'sgd')} \\
        --momentum {config.get('momentum', 0.9)} \\
        --weight-decay {config.get('weight_decay', 0.0001)} \\
        --scheduler {config.get('scheduler', 'cosine')} \\
        --warmup-epochs {config.get('warmup_epochs', 5)} \\
        --output-dir "$CHECKPOINT_DIR" \\
        --save-every 10

TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=============================================="

# Find best checkpoint
BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_*.pth 2>/dev/null | head -1)
fi

if [ -n "$BEST_CKPT" ]; then
    echo "Best checkpoint: $BEST_CKPT"
    echo "$BEST_CKPT" > "{self.checkpoints_dir}/{config['exp_id']}.txt"
fi

exit $TRAIN_EXIT_CODE
'''
        return script

    def _generate_eval_script(self, config: dict) -> str:
        """Generate SLURM evaluation script"""

        data_path = config.get('data_path', '/scratch/USERID/data')
        data_path = data_path.replace('USERID', self.hpc_config.username)

        script = f'''#!/bin/bash
#SBATCH --job-name={config['exp_id']}_eval
#SBATCH --account={config.get('account', 'default')}
#SBATCH --partition={config.get('partition', 'gpu')}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output={self.logs_dir}/output/eval_{config['exp_id']}_%j.out
#SBATCH --error={self.logs_dir}/error/eval_{config['exp_id']}_%j.err

# =============================================================================
# Auto-generated Evaluation Script
# =============================================================================

echo "=============================================="
echo "Evaluation: {config['exp_id']}"
echo "=============================================="

# Environment
module load anaconda3/2025.06
source activate pytorch

export DATA_ROOT="{data_path}"
export RESULTS_DIR="{self.experiments_dir}/results"
mkdir -p "$RESULTS_DIR"

# Find checkpoint
CKPT_FILE="{self.checkpoints_dir}/{config['exp_id']}.txt"
if [ -f "$CKPT_FILE" ]; then
    CHECKPOINT=$(cat "$CKPT_FILE")
else
    echo "ERROR: No checkpoint found"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"

# Evaluation
python evaluate.py \\
    --model {config.get('model', 'resnet50')} \\
    --dataset {config.get('dataset', 'imagenet')} \\
    --data-path "$DATA_ROOT" \\
    --checkpoint "$CHECKPOINT" \\
    --batch-size {config.get('eval_batch_size', 512)} \\
    --split {config.get('test_split', 'val')} \\
    --output "$RESULTS_DIR/{config['exp_id']}_results.json"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation complete"
    echo "Results: $RESULTS_DIR/{config['exp_id']}_results.json"
else
    echo "✗ Evaluation failed"
fi

exit $EXIT_CODE
'''
        return script

    def harvest_results(self, exp_id: str):
        """Harvest evaluation results"""

        if exp_id not in self.metadata:
            print(f"Error: Experiment {exp_id} not found")
            return {}

        results_file = self.experiments_dir / "results" / f"{exp_id}_results.json"

        if not results_file.exists():
            print(f"No results found: {results_file}")
            return {}

        print(f"Parsing results from: {results_file}")

        import json
        with open(results_file) as f:
            results = json.load(f)

        # Update metadata
        self.metadata[exp_id]["results"] = results
        self.metadata[exp_id]["status"] = "completed"
        self.metadata[exp_id]["completed_at"] = str(Path(results_file).stat().st_mtime)
        self._save_metadata()

        print(f"\n✓ Results harvested for {exp_id}")
        print(f"  Top-1 Accuracy: {results.get('top1_acc', 'N/A')}")
        print(f"  Top-5 Accuracy: {results.get('top5_acc', 'N/A')}")
        print(f"  Test Loss: {results.get('test_loss', 'N/A')}")

        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Image Classification Experiment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # New experiment
    new_parser = subparsers.add_parser("new", help="Create new experiment")
    new_parser.add_argument("--exp-id", required=True)
    new_parser.add_argument("--template", default="resnet_baseline")
    new_parser.add_argument("--description", required=True)
    new_parser.add_argument("--model", help="Model architecture")
    new_parser.add_argument("--dataset", help="Dataset name")
    new_parser.add_argument("--batch-size", type=int)
    new_parser.add_argument("--learning-rate", type=float)
    new_parser.add_argument("--epochs", type=int)
    new_parser.add_argument("--tags", nargs="+")

    # Submit
    submit_parser = subparsers.add_parser("submit", help="Submit experiment")
    submit_parser.add_argument("exp_id")
    submit_parser.add_argument("--train-only", action="store_true")
    submit_parser.add_argument("--eval-only", action="store_true")
    submit_parser.add_argument("--dry-run", action="store_true")

    # Harvest
    harvest_parser = subparsers.add_parser("harvest", help="Harvest results")
    harvest_parser.add_argument("exp_id")

    # List
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status")
    list_parser.add_argument("--tags", nargs="+")

    # Show
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id")

    # Export
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument("output", nargs="?", default="results.csv")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load config and create manager
    try:
        config = load_project_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun: hpcexp init <project_name>")
        sys.exit(1)

    manager = ImageClassificationManager(config)

    # Execute command
    if args.command == "new":
        kwargs = {
            k: v for k, v in vars(args).items()
            if v is not None and k not in ["command", "exp_id", "template", "description"]
        }
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

    elif args.command == "list":
        manager.list_experiments(status=args.status, tags=args.tags)

    elif args.command == "show":
        manager.show_experiment(args.exp_id)

    elif args.command == "export":
        manager.export_results(args.output)


if __name__ == "__main__":
    main()
