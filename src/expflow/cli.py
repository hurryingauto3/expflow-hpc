#!/usr/bin/env python3
"""
ExpFlow CLI - Command-line interface
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from .hpc_config import initialize_project, load_project_config, HPCEnvironment
from datetime import datetime


# =============================================================================
# Experiment Management Helpers
# =============================================================================

def _load_experiments_db(project_root: Path) -> Dict:
    """Load experiments metadata database"""
    db_path = project_root / "experiment_configs" / "experiments.json"
    if db_path.exists():
        with open(db_path) as f:
            return json.load(f)
    return {}


def _save_experiments_db(project_root: Path, metadata: Dict):
    """Save experiments metadata database"""
    db_path = project_root / "experiment_configs" / "experiments.json"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def _get_slurm_jobs() -> Dict[str, Dict]:
    """Get current SLURM jobs for this user"""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h",
             "-o", "%.18i %.9P %.50j %.8T %.10M %.6D %R"],
            capture_output=True, text=True
        )
        jobs = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                job_id = parts[0].strip()
                jobs[job_id] = {
                    "job_id": job_id,
                    "partition": parts[1].strip(),
                    "name": parts[2].strip(),
                    "state": parts[3].strip(),
                    "time": parts[4].strip() if len(parts) > 4 else "",
                    "nodes": parts[5].strip() if len(parts) > 5 else "",
                    "nodelist": parts[6].strip() if len(parts) > 6 else ""
                }
        return jobs
    except Exception:
        return {}


def _find_log_file(logs_dir: Path, exp_id: str, log_type: str = "train",
                   errors: bool = False) -> Optional[Path]:
    """Find the most recent log file for an experiment"""
    subdir = "error" if errors else "output"
    ext = "err" if errors else "out"
    pattern = f"{log_type}_{exp_id}_*.{ext}"

    log_dir = logs_dir / subdir
    if not log_dir.exists():
        return None

    matches = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def cmd_init(args):
    """Initialize a new HPC experiment project"""

    if args.interactive:
        # Interactive mode with menus
        from .interactive_init import interactive_init
        config = interactive_init(args.project_name)
        # Create directories and save config
        _finalize_project_setup(config)
    elif args.quick:
        # Quick mode with smart defaults (no prompts)
        from .interactive_init import quick_init
        print(f"Quick setup for: {args.project_name}")
        config = quick_init(args.project_name)
        print(f"  Account: {config.default_account}")
        print(f"  Partition: {config.default_partition}")
        # Create directories and save config
        _finalize_project_setup(config)
    else:
        # Legacy auto-detect mode
        print(f"Initializing project: {args.project_name}")
        config = initialize_project(args.project_name)

    print("\n" + "=" * 70)
    print(" Project ready!")
    print("=" * 70)
    print(f" Location: {config.project_root}")
    print(f" Next: cd {config.project_root}")
    print(f" Docs: https://github.com/hurryingauto3/expflow-hpc/docs")
    print()


def _finalize_project_setup(config):
    """Create project directories and save config"""
    from pathlib import Path

    print(f"\nCreating project directories...")
    for directory in [
        config.experiments_dir,
        f"{config.logs_dir}/output",
        f"{config.logs_dir}/error",
        config.cache_dir,
        config.checkpoints_dir
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   {directory}")

    # Save config
    config_save_path = f"{config.project_root}/.hpc_config.yaml"
    config.save(config_save_path)
    print(f"\n Configuration saved to {config_save_path}")


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


def cmd_partitions(args):
    """Show partition-account access map"""
    try:
        from .partition_validator import PartitionValidator

        print("Detecting partition access (this may take 10-30 seconds)...")
        validator = PartitionValidator()

        accounts = HPCEnvironment.get_slurm_accounts()
        partition_map = validator.detect_partition_access(accounts)

        if args.json:
            import json
            print(json.dumps(partition_map, indent=2))
        else:
            validator.print_access_map()

            # Show auto-selection example
            if not args.quiet:
                print("\nAuto-Selection Examples:")
                print("-" * 70)

                # Try to auto-select H200
                result = validator.auto_select_partition(gpu_type="H200")
                if result:
                    partition, account = result
                    print(f"  For H200 GPU: partition={partition}, account={account}")

                # Try to auto-select L40s
                result = validator.auto_select_partition(gpu_type="L40s")
                if result:
                    partition, account = result
                    print(f"  For L40s GPU: partition={partition}, account={account}")

    except ImportError:
        print("Error: partition_validator module not found")
        sys.exit(1)


def cmd_template(args):
    """Create a template experiment configuration"""
    try:
        config = load_project_config()
    except:
        print("Error: Run 'expflow init <project_name>' first")
        sys.exit(1)

    template_dir = Path(config.project_root) / "experiment_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
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
account: {config.default_account}
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


# =============================================================================
# Experiment Management Commands
# =============================================================================

def cmd_status(args):
    """Show experiment status and running jobs"""
    try:
        config = load_project_config()
    except FileNotFoundError:
        print("Error: Not in a project directory. Run 'expflow init' first.")
        sys.exit(1)

    project_root = Path(config.project_root)
    metadata = _load_experiments_db(project_root)
    slurm_jobs = _get_slurm_jobs()

    # Build job_id to experiment mapping
    job_to_exp = {}
    for exp_id, meta in metadata.items():
        if meta.get("train_job_id"):
            job_to_exp[meta["train_job_id"]] = (exp_id, "train")
        if meta.get("eval_job_id"):
            job_to_exp[meta["eval_job_id"]] = (exp_id, "eval")

    print(f"\n{'='*80}")
    print("Experiment Status")
    print(f"{'='*80}")

    # Show running/pending jobs
    active_jobs = []
    for job_id, job_info in slurm_jobs.items():
        if job_id in job_to_exp:
            exp_id, job_type = job_to_exp[job_id]
            active_jobs.append({
                "exp_id": exp_id,
                "job_type": job_type,
                "job_id": job_id,
                **job_info
            })

    if active_jobs:
        print(f"\nActive Jobs ({len(active_jobs)}):")
        print(f"{'Experiment':<15} {'Type':<6} {'JobID':<10} {'State':<10} {'Time':<12} {'Node'}")
        print("-" * 80)
        for job in active_jobs:
            print(f"{job['exp_id']:<15} {job['job_type']:<6} {job['job_id']:<10} "
                  f"{job['state']:<10} {job['time']:<12} {job.get('nodelist', '')}")
    else:
        print("\nNo active jobs")

    # Show recent experiments
    if metadata:
        print(f"\nRecent Experiments:")
        print(f"{'ID':<15} {'Status':<12} {'Train Job':<12} {'Eval Job':<12} {'Description'}")
        print("-" * 80)

        sorted_exps = sorted(
            metadata.items(),
            key=lambda x: x[1].get("config", {}).get("created_at", ""),
            reverse=True
        )[:10]

        for exp_id, meta in sorted_exps:
            exp_config = meta.get("config", {})
            desc = exp_config.get("description", "")[:30]
            train_job = meta.get("train_job_id", "-")
            eval_job = meta.get("eval_job_id", "-")
            status = meta.get("status", "unknown")
            print(f"{exp_id:<15} {status:<12} {train_job:<12} {eval_job:<12} {desc}")
    else:
        print("\nNo experiments yet")

    print(f"{'='*80}\n")


def cmd_list(args):
    """List all experiments"""
    try:
        config = load_project_config()
    except FileNotFoundError:
        print("Error: Not in a project directory. Run 'expflow init' first.")
        sys.exit(1)

    project_root = Path(config.project_root)
    metadata = _load_experiments_db(project_root)

    if not metadata:
        print("No experiments found")
        return

    # Filter by status if provided
    filtered = []
    for exp_id, meta in metadata.items():
        if args.status and meta.get("status") != args.status:
            continue
        filtered.append((exp_id, meta))

    if not filtered:
        print(f"No experiments found with status: {args.status}")
        return

    print(f"\nFound {len(filtered)} experiments:")
    print(f"{'ID':<20} {'Status':<12} {'Description':<45}")
    print("-" * 80)

    for exp_id, meta in sorted(filtered, key=lambda x: x[0]):
        exp_config = meta.get("config", {})
        desc = exp_config.get("description", "")
        if len(desc) > 42:
            desc = desc[:42] + "..."
        status = meta.get("status", "unknown")
        print(f"{exp_id:<20} {status:<12} {desc:<45}")


def cmd_logs(args):
    """View experiment logs"""
    try:
        config = load_project_config()
    except FileNotFoundError:
        print("Error: Not in a project directory. Run 'expflow init' first.")
        sys.exit(1)

    project_root = Path(config.project_root)
    metadata = _load_experiments_db(project_root)
    logs_dir = Path(config.logs_dir)

    if args.exp_id not in metadata:
        print(f"Error: Experiment '{args.exp_id}' not found")
        print(f"Use 'expflow list' to see available experiments")
        sys.exit(1)

    log_file = _find_log_file(logs_dir, args.exp_id, args.type, args.errors)
    file_type = "error" if args.errors else "output"

    if not log_file:
        print(f"No {file_type} log found for '{args.exp_id}' ({args.type})")
        print(f"  Looked in: {logs_dir}/{file_type}/")
        return

    print(f"\n{'='*70}")
    print(f"Log: {log_file.name}")
    print(f"{'='*70}\n")

    try:
        with open(log_file, 'r') as f:
            content = f.readlines()
            if len(content) > args.lines:
                print(f"[Showing last {args.lines} lines of {len(content)} total]\n")
                content = content[-args.lines:]
            for line in content:
                print(line, end='')
    except Exception as e:
        print(f"Error reading log: {e}")

    print(f"\n{'='*70}")
    print(f"Full log: {log_file}")
    print(f"{'='*70}\n")


def cmd_tail(args):
    """Tail experiment logs in real-time"""
    try:
        config = load_project_config()
    except FileNotFoundError:
        print("Error: Not in a project directory. Run 'expflow init' first.")
        sys.exit(1)

    project_root = Path(config.project_root)
    metadata = _load_experiments_db(project_root)
    logs_dir = Path(config.logs_dir)

    if args.exp_id not in metadata:
        print(f"Error: Experiment '{args.exp_id}' not found")
        sys.exit(1)

    log_file = _find_log_file(logs_dir, args.exp_id, args.type, args.errors)

    if not log_file:
        print(f"No log found for '{args.exp_id}' ({args.type})")
        return

    print(f"Tailing: {log_file}")
    print("Press Ctrl+C to stop\n")
    print("-" * 70)

    try:
        subprocess.run(["tail", "-f", str(log_file)])
    except KeyboardInterrupt:
        print("\n[Stopped]")


def cmd_cancel(args):
    """Cancel running jobs for an experiment"""
    try:
        config = load_project_config()
    except FileNotFoundError:
        print("Error: Not in a project directory. Run 'expflow init' first.")
        sys.exit(1)

    project_root = Path(config.project_root)
    metadata = _load_experiments_db(project_root)

    if args.exp_id not in metadata:
        print(f"Error: Experiment '{args.exp_id}' not found")
        sys.exit(1)

    meta = metadata[args.exp_id]
    cancelled = []

    jobs_to_cancel = []
    if args.type in (None, "train") and meta.get("train_job_id"):
        jobs_to_cancel.append(("train", meta["train_job_id"]))
    if args.type in (None, "eval") and meta.get("eval_job_id"):
        jobs_to_cancel.append(("eval", meta["eval_job_id"]))

    if not jobs_to_cancel:
        print(f"No jobs to cancel for '{args.exp_id}'")
        return

    for jtype, job_id in jobs_to_cancel:
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                cancelled.append(f"{jtype} ({job_id})")
            else:
                print(f"  Warning: Could not cancel {jtype} job {job_id}")
        except Exception as e:
            print(f"  Error cancelling {jtype} job: {e}")

    if cancelled:
        print(f"Cancelled jobs for '{args.exp_id}': {', '.join(cancelled)}")
        meta["status"] = "cancelled"
        _save_experiments_db(project_root, metadata)


def cmd_prune(args):
    """Prune duplicate and invalid experiments"""
    try:
        config = load_project_config()
    except FileNotFoundError:
        print("Error: Not in a project directory. Run 'expflow init' first.")
        sys.exit(1)

    from .pruner import ExperimentPruner

    # Determine directories
    experiments_dir = Path(config.experiments_dir)

    # Check if experiments_dir has training and evaluations subdirs
    training_dir = experiments_dir / "training"
    evaluations_dir = experiments_dir / "evaluations"

    # Use subdirectories if they exist, otherwise use experiments_dir directly
    if training_dir.exists() and training_dir.is_dir():
        target_dir = training_dir
    else:
        target_dir = experiments_dir

    eval_dir = evaluations_dir if evaluations_dir.exists() else None

    # Initialize pruner
    pruner = ExperimentPruner(
        experiments_dir=target_dir,
        evaluations_dir=eval_dir,
        archive_dir=experiments_dir.parent / ".archive" / "experiments"
    )

    # Perform pruning based on mode
    if args.mode == "duplicates":
        stats = pruner.prune_duplicates(
            keep_n=args.keep,
            dry_run=args.dry_run,
            verbose=True
        )
    elif args.mode == "invalid":
        stats = pruner.prune_invalid(
            require_checkpoint=not args.no_checkpoint_check,
            require_eval=not args.no_eval_check,
            required_epochs=args.required_epochs,
            dry_run=args.dry_run,
            verbose=True
        )
    else:  # all
        stats = pruner.prune_all(
            keep_n=args.keep,
            require_checkpoint=not args.no_checkpoint_check,
            require_eval=not args.no_eval_check,
            required_epochs=args.required_epochs,
            dry_run=args.dry_run,
            verbose=True
        )


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
  expflow status                    # Show experiments and running jobs
  expflow logs exp001               # View experiment logs
  expflow tail exp001               # Follow logs in real-time
  expflow cancel exp001             # Cancel running jobs
  expflow prune --dry-run           # Preview cleanup of duplicates/invalid experiments

For full docs: https://github.com/hurryingauto3/expflow-hpc
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new project")
    init_parser.add_argument("project_name", help="Project name")
    init_parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive setup with menus (recommended)"
    )
    init_parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick setup with smart defaults (no prompts)"
    )

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

    # Partitions command
    partitions_parser = subparsers.add_parser(
        "partitions", help="Show partition-account access map"
    )
    partitions_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    partitions_parser.add_argument(
        "--quiet", action="store_true", help="Don't show auto-selection examples"
    )

    # Template command
    template_parser = subparsers.add_parser(
        "template", help="Create experiment template"
    )
    template_parser.add_argument("name", help="Template name")
    template_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing"
    )

    # Status command
    subparsers.add_parser("status", help="Show experiments and running SLURM jobs")

    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments")
    list_parser.add_argument(
        "--status", help="Filter by status (created, submitted, completed, failed)"
    )

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View experiment logs")
    logs_parser.add_argument("exp_id", help="Experiment ID")
    logs_parser.add_argument(
        "--type", choices=["train", "eval"], default="train",
        help="Log type (default: train)"
    )
    logs_parser.add_argument(
        "-n", "--lines", type=int, default=50,
        help="Number of lines to show (default: 50)"
    )
    logs_parser.add_argument(
        "-e", "--errors", action="store_true",
        help="Show error log instead of output"
    )

    # Tail command
    tail_parser = subparsers.add_parser("tail", help="Follow experiment logs in real-time")
    tail_parser.add_argument("exp_id", help="Experiment ID")
    tail_parser.add_argument(
        "--type", choices=["train", "eval"], default="train",
        help="Log type (default: train)"
    )
    tail_parser.add_argument(
        "-e", "--errors", action="store_true",
        help="Follow error log instead of output"
    )

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel running jobs")
    cancel_parser.add_argument("exp_id", help="Experiment ID")
    cancel_parser.add_argument(
        "--type", choices=["train", "eval"],
        help="Cancel specific job type only"
    )

    # Prune command
    prune_parser = subparsers.add_parser(
        "prune",
        help="Clean up duplicate and invalid experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  expflow prune --dry-run              # Preview what would be pruned
  expflow prune --mode duplicates      # Remove duplicates only
  expflow prune --mode invalid         # Remove invalid only
  expflow prune --keep 2               # Keep 2 most recent of each experiment
  expflow prune --required-epochs 50   # Require checkpoint with >= 50 epochs
        """
    )
    prune_parser.add_argument(
        "--mode",
        choices=["all", "duplicates", "invalid"],
        default="all",
        help="Pruning mode (default: all)"
    )
    prune_parser.add_argument(
        "--keep",
        type=int,
        default=1,
        help="Number of most recent runs to keep per experiment (default: 1)"
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be pruned without actually deleting"
    )
    prune_parser.add_argument(
        "--no-checkpoint-check",
        action="store_true",
        help="Don't check for valid checkpoints"
    )
    prune_parser.add_argument(
        "--no-eval-check",
        action="store_true",
        help="Don't check for evaluation results"
    )
    prune_parser.add_argument(
        "--required-epochs",
        type=int,
        help="Require checkpoint with at least this many epochs"
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
    elif args.command == "partitions":
        cmd_partitions(args)
    elif args.command == "template":
        cmd_template(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "tail":
        cmd_tail(args)
    elif args.command == "cancel":
        cmd_cancel(args)
    elif args.command == "prune":
        cmd_prune(args)


if __name__ == "__main__":
    main()
