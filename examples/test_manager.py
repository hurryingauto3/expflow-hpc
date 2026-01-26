#!/usr/bin/env python3
"""
Simple experiment manager for testing expflow.
Only handles experiment creation and submission - monitoring is done via expflow CLI.
"""

import argparse
import sys
from dataclasses import dataclass

from expflow.hpcexp_core import BaseExperimentManager, BaseExperimentConfig
from expflow.hpc_config import load_project_config


@dataclass
class SimpleConfig(BaseExperimentConfig):
    """Simple test configuration"""
    model: str = "resnet50"
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10


class SimpleManager(BaseExperimentManager):
    """Simple experiment manager for testing"""

    def _generate_train_script(self, config: dict) -> str:
        script = f'''#!/bin/bash
#SBATCH --job-name={config['exp_id']}_train
#SBATCH --partition={config.get('partition', 'l40s_public')}
#SBATCH --account={config.get('account', 'default')}
#SBATCH --gres=gpu:{config.get('num_gpus', 1)}
#SBATCH --cpus-per-task={config.get('cpus_per_task', 4)}
#SBATCH --time={config.get('time_limit', '01:00:00')}
#SBATCH --output={self.logs_dir}/output/train_{config['exp_id']}_%j.out
#SBATCH --error={self.logs_dir}/error/train_{config['exp_id']}_%j.err

echo "=============================================="
echo "Experiment: {config['exp_id']}"
echo "Description: {config.get('description', 'N/A')}"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Configuration
echo "Model: {config.get('model', 'resnet50')}"
echo "Batch size: {config.get('batch_size', 32)}"
echo "Learning rate: {config.get('learning_rate', 0.001)}"
echo "Epochs: {config.get('epochs', 10)}"

# Simulate training with some output
for i in $(seq 1 5); do
    echo "[Epoch $i/5] Training... loss=0.$((RANDOM % 100))"
    sleep 2
done

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "=============================================="
'''
        return script

    def _generate_eval_script(self, config: dict) -> str:
        script = f'''#!/bin/bash
#SBATCH --job-name={config['exp_id']}_eval
#SBATCH --partition={config.get('partition', 'l40s_public')}
#SBATCH --account={config.get('account', 'default')}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output={self.logs_dir}/output/eval_{config['exp_id']}_%j.out
#SBATCH --error={self.logs_dir}/error/eval_{config['exp_id']}_%j.err

echo "=============================================="
echo "Evaluating: {config['exp_id']}"
echo "=============================================="

# Simulate evaluation
echo "Loading checkpoint..."
sleep 2
echo "Running evaluation..."
sleep 3
echo "Accuracy: 0.$((RANDOM % 100 + 80))"

echo ""
echo "Evaluation complete at $(date)"
'''
        return script

    def harvest_results(self, exp_id: str):
        print(f"Harvesting results for {exp_id}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Simple Test Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This manager handles experiment creation and submission.
For monitoring, use the expflow CLI:
  expflow status              # Show all experiments and jobs
  expflow logs <exp_id>       # View logs
  expflow tail <exp_id>       # Follow logs live
  expflow cancel <exp_id>     # Cancel jobs
        """
    )
    subparsers = parser.add_subparsers(dest="cmd")

    # new
    p = subparsers.add_parser("new", help="Create experiment")
    p.add_argument("--exp-id", required=True)
    p.add_argument("--description", default="Test experiment")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=10)

    # submit
    p = subparsers.add_parser("submit", help="Submit experiment")
    p.add_argument("exp_id")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--train-only", action="store_true")

    # show (for viewing config before submit)
    p = subparsers.add_parser("show", help="Show experiment config")
    p.add_argument("exp_id")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    # Load config and create manager
    try:
        config = load_project_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're in a project directory with .hpc_config.yaml")
        sys.exit(1)

    manager = SimpleManager(config)

    if args.cmd == "new":
        manager.create_experiment(
            exp_id=args.exp_id,
            description=args.description,
            model=args.model,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )

    elif args.cmd == "submit":
        manager.submit_experiment(
            exp_id=args.exp_id,
            dry_run=args.dry_run,
            train_only=args.train_only
        )

    elif args.cmd == "show":
        manager.show_experiment(args.exp_id)


if __name__ == "__main__":
    main()
