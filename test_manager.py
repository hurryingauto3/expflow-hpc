#!/usr/bin/env python3
"""
Simple test script for the experiment manager.
Copy this to your project directory and run it.
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
echo "=============================================="

# Configuration
echo "Model: {config.get('model', 'resnet50')}"
echo "Batch size: {config.get('batch_size', 32)}"
echo "Learning rate: {config.get('learning_rate', 0.001)}"
echo "Epochs: {config.get('epochs', 10)}"

# Your training code would go here
# python train.py --model {config.get('model')} ...

echo "Training complete"
'''
        return script

    def _generate_eval_script(self, config: dict) -> str:
        script = f'''#!/bin/bash
#SBATCH --job-name={config['exp_id']}_eval
#SBATCH --partition={config.get('partition', 'l40s_public')}
#SBATCH --account={config.get('account', 'default')}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output={self.logs_dir}/output/eval_{config['exp_id']}_%j.out
#SBATCH --error={self.logs_dir}/error/eval_{config['exp_id']}_%j.err

echo "Evaluating experiment: {config['exp_id']}"

# Your evaluation code would go here

echo "Evaluation complete"
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
Commands:
  new       Create a new experiment
  submit    Submit experiment to SLURM
  list      List all experiments
  show      Show experiment details
  status    Show running jobs and experiment status
  logs      View experiment output logs
  tail      Follow experiment logs in real-time
  cancel    Cancel running jobs
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

    # list
    subparsers.add_parser("list", help="List experiments")

    # show
    p = subparsers.add_parser("show", help="Show experiment details")
    p.add_argument("exp_id")

    # status - NEW
    subparsers.add_parser("status", help="Show running jobs and experiment status")

    # logs - NEW
    p = subparsers.add_parser("logs", help="View experiment logs")
    p.add_argument("exp_id")
    p.add_argument("--type", choices=["train", "eval"], default="train", help="Log type")
    p.add_argument("--lines", "-n", type=int, default=50, help="Number of lines to show")
    p.add_argument("--errors", "-e", action="store_true", help="Show error log instead")

    # tail - NEW
    p = subparsers.add_parser("tail", help="Follow experiment logs in real-time")
    p.add_argument("exp_id")
    p.add_argument("--type", choices=["train", "eval"], default="train", help="Log type")
    p.add_argument("--errors", "-e", action="store_true", help="Follow error log instead")

    # cancel - NEW
    p = subparsers.add_parser("cancel", help="Cancel running jobs")
    p.add_argument("exp_id")
    p.add_argument("--type", choices=["train", "eval"], help="Cancel specific job type")

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

    elif args.cmd == "list":
        manager.list_experiments()

    elif args.cmd == "show":
        manager.show_experiment(args.exp_id)

    elif args.cmd == "status":
        manager.status()

    elif args.cmd == "logs":
        manager.logs(args.exp_id, log_type=args.type, lines=args.lines, errors=args.errors)

    elif args.cmd == "tail":
        manager.tail_logs(args.exp_id, log_type=args.type, errors=args.errors)

    elif args.cmd == "cancel":
        manager.cancel(args.exp_id, job_type=args.type)


if __name__ == "__main__":
    main()
