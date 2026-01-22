#!/usr/bin/env python3
"""
Example: Partition-Aware Experiment Manager

Demonstrates how to use PartitionValidator to automatically select
the best partition-account combination for experiments.
"""

import argparse
from expflow import (
    BaseExperimentManager,
    load_project_config,
    PartitionValidator
)


class PartitionAwareManager(BaseExperimentManager):
    """
    Example manager that automatically handles partition-account selection
    """

    def __init__(self, hpc_config):
        super().__init__(hpc_config)

        # Initialize partition validator
        self.validator = PartitionValidator()
        print("Detecting partition access...")
        self.validator.detect_partition_access()
        print("Partition access detected!\n")

    def create_experiment_smart(self,
                               exp_id: str,
                               description: str = "",
                               gpu_type: str = None,
                               num_gpus: int = 1,
                               **kwargs):
        """
        Create experiment with automatic partition-account selection

        Args:
            exp_id: Experiment ID
            description: Description
            gpu_type: Desired GPU type ('H200', 'L40s', etc.)
            num_gpus: Number of GPUs
            **kwargs: Other experiment parameters
        """

        # Auto-select partition and account
        result = self.validator.auto_select_partition(
            gpu_type=gpu_type,
            account=kwargs.get('account'),
            prefer_public=kwargs.get('prefer_public', True)
        )

        if not result:
            raise ValueError(
                f"No accessible partitions found for GPU type: {gpu_type}"
            )

        partition, account = result

        print(f"Auto-selected:")
        print(f"  Partition: {partition}")
        print(f"  Account: {account}")

        # Create experiment with auto-selected values
        return self.create_experiment(
            exp_id=exp_id,
            description=description,
            partition=partition,
            account=account,
            num_gpus=num_gpus,
            **kwargs
        )

    def _generate_train_script(self, config):
        """Generate training SLURM script"""

        # Validate partition-account combination
        is_valid = self.validator.validate_partition_account(
            config.get('partition'),
            config.get('account', self.hpc_config.default_account)
        )

        if not is_valid:
            raise ValueError(
                f"Invalid partition-account combination: "
                f"{config.get('partition')} + {config.get('account')}"
            )

        return f'''#!/bin/bash
#SBATCH --job-name={config['exp_id']}_train
#SBATCH --partition={config['partition']}
#SBATCH --account={config.get('account', self.hpc_config.default_account)}
#SBATCH --nodes={config.get('num_nodes', 1)}
#SBATCH --gres=gpu:{config['num_gpus']}
#SBATCH --time={config.get('time_limit', '48:00:00')}
#SBATCH --output={self.logs_dir}/output/{config['exp_id']}_train_%j.log
#SBATCH --error={self.logs_dir}/error/{config['exp_id']}_train_%j.err

# Auto-validated partition-account combination
# Partition: {config['partition']}
# Account: {config.get('account', self.hpc_config.default_account)}

echo "Starting training for experiment: {config['exp_id']}"
echo "Partition: {config['partition']}"
echo "Account: {config.get('account', self.hpc_config.default_account)}"

# Your training code here
python train.py --exp-id {config['exp_id']}
'''

    def _generate_eval_script(self, config):
        """Generate evaluation SLURM script"""
        return f'''#!/bin/bash
#SBATCH --job-name={config['exp_id']}_eval
#SBATCH --partition={config['partition']}
#SBATCH --account={config.get('account', self.hpc_config.default_account)}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "Evaluation for: {config['exp_id']}"
python evaluate.py --exp-id {config['exp_id']}
'''

    def harvest_results(self, exp_id):
        """Harvest experiment results"""
        return {"status": "completed", "exp_id": exp_id}


def main():
    parser = argparse.ArgumentParser(
        description="Partition-Aware Experiment Manager Example"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Show access map
    show_parser = subparsers.add_parser(
        "show-access",
        help="Show partition access map"
    )

    # Create experiment (smart)
    new_parser = subparsers.add_parser(
        "new",
        help="Create new experiment with auto partition selection"
    )
    new_parser.add_argument("--exp-id", required=True)
    new_parser.add_argument("--description", default="")
    new_parser.add_argument(
        "--gpu-type",
        choices=['H200', 'L40s', 'RTX8000'],
        help="Desired GPU type"
    )
    new_parser.add_argument("--num-gpus", type=int, default=4)

    # Submit experiment
    submit_parser = subparsers.add_parser("submit", help="Submit experiment")
    submit_parser.add_argument("exp_id")

    args = parser.parse_args()

    # Load config
    try:
        config = load_project_config()
    except FileNotFoundError:
        print("Error: Run 'expflow init <project>' first")
        return

    # Create manager
    manager = PartitionAwareManager(config)

    if args.command == "show-access":
        manager.validator.print_access_map()

    elif args.command == "new":
        manager.create_experiment_smart(
            exp_id=args.exp_id,
            description=args.description,
            gpu_type=args.gpu_type,
            num_gpus=args.num_gpus
        )
        print(f"\nExperiment {args.exp_id} created successfully!")
        print(f"Submit with: python {__file__} submit {args.exp_id}")

    elif args.command == "submit":
        manager.submit_experiment(args.exp_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
