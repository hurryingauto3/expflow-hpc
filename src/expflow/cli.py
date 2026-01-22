#!/usr/bin/env python3
"""
ExpFlow CLI - Command-line interface
"""

import argparse
import sys
from pathlib import Path

from .hpc_config import initialize_project, load_project_config, HPCEnvironment
from datetime import datetime


def cmd_init(args):
    """Initialize a new HPC experiment project"""
    print(f"Initializing project: {args.project_name}")
    config = initialize_project(args.project_name)

    print("\n" + "=" * 70)
    print(" Project ready!")
    print("=" * 70)
    print(f" Location: {config.project_root}")
    print(f" Next: cd {config.project_root}")
    print(f" Docs: https://github.com/hurryingauto3/expflow-hpc/docs")
    print()


def cmd_info(args):
    """Show HPC environment information"""
    print("=" * 70)
    print("HPC Environment Information")
    print("=" * 70)
    print(f"Username: {HPCEnvironment.get_username()}")
    print(f"Home: {HPCEnvironment.get_home_dir()}")
    print(f"Scratch: {HPCEnvironment.get_scratch_dir()}")
    print(f"Cluster: {HPCEnvironment.detect_cluster()}")

    accounts = HPCEnvironment.get_slurm_accounts()
    if accounts:
        print(f"SLURM Accounts: {', '.join(accounts)}")

    partitions = HPCEnvironment.get_available_partitions()
    if partitions:
        print(f"Partitions: {', '.join(partitions[:10])}")

    print("=" * 70)


def cmd_config(args):
    """Show project configuration"""
    try:
        config = load_project_config(args.project_root)
        print("=" * 70)
        print("Project Configuration")
        print("=" * 70)
        for key, value in config.to_dict().items():
            if isinstance(value, list):
                print(f"{key}: [{len(value)} items]")
            elif isinstance(value, str) and len(value) > 60:
                print(f"{key}: {value[:60]}...")
            else:
                print(f"{key}: {value}")
        print("=" * 70)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'expflow init <project_name>' first")
        sys.exit(1)


def cmd_resources(args):
    """Show resource recommendations"""
    try:
        from .resource_advisor import ResourceAdvisor

        advisor = ResourceAdvisor()

        if args.status:
            advisor.print_status()
        else:
            # Load project config for recommendations
            try:
                config = load_project_config(args.project_root)
                exp_config = None
            except:
                exp_config = None

            recommendations = advisor.get_recommendations(
                exp_config=exp_config, target_global_batch=args.global_batch
            )

            gemini_suggestion = None
            if args.use_gemini:
                partition_stats = advisor.get_queue_status()
                gemini_suggestion = advisor.get_gemini_suggestion(
                    partition_stats, exp_config
                )

            advisor.print_status()
            advisor.print_recommendations(recommendations, gemini_suggestion)

    except ImportError:
        print("Error: resource_advisor module not found")
        sys.exit(1)


def cmd_template(args):
    """Create a template experiment configuration"""
    try:
        config = load_project_config()
    except:
        print("Error: Run 'expflow init <project_name>' first")
        sys.exit(1)

    template_dir = Path(config.project_root) / "experiment_templates"
    template_path = template_dir / f"{args.name}.yaml"

    if template_path.exists() and not args.force:
        print(f"Error: Template {args.name} already exists")
        print("Use --force to overwrite")
        sys.exit(1)

    # Create basic template
    template_content = f"""# Experiment Template: {args.name}
# Created: {datetime.now().isoformat()}

description: "TODO: Add description"

# Add your project-specific parameters here
# model: your_model
# dataset: your_dataset
# batch_size: 256
# learning_rate: 0.1

# Resource configuration
partition: {config.default_partition}
num_gpus: 4
num_nodes: 1
cpus_per_task: 16
time_limit: "48:00:00"

# Tags
tags:
  - {args.name}
"""

    with open(template_path, "w") as f:
        f.write(template_content)

    print(f" Created template: {template_path}")
    print(f"  Edit this file to customize your experiment parameters")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ExpFlow - HPC Experiment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  expflow init my-research          # Initialize new project
  expflow info                      # Show HPC environment
  expflow resources --status        # Check GPU availability
  expflow template baseline         # Create experiment template

For full docs: https://github.com/hurryingauto3/expflow-hpc
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new project")
    init_parser.add_argument("project_name", help="Project name")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show HPC environment info")

    # Config command
    config_parser = subparsers.add_parser("config", help="Show project configuration")
    config_parser.add_argument("--project-root", help="Project root directory")

    # Resources command
    resources_parser = subparsers.add_parser("resources", help="Resource advisor")
    resources_parser.add_argument(
        "--status", action="store_true", help="Show current status only"
    )
    resources_parser.add_argument(
        "--global-batch", type=int, default=192, help="Target global batch"
    )
    resources_parser.add_argument(
        "--use-gemini", action="store_true", help="Use Gemini API"
    )
    resources_parser.add_argument("--project-root", help="Project root directory")

    # Template command
    template_parser = subparsers.add_parser(
        "template", help="Create experiment template"
    )
    template_parser.add_argument("name", help="Template name")
    template_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to command handlers
    if args.command == "init":
        cmd_init(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "resources":
        cmd_resources(args)
    elif args.command == "template":
        cmd_template(args)


if __name__ == "__main__":
    main()
