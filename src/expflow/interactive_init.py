#!/usr/bin/env python3
"""
Interactive Initialization for ExpFlow

Provides a CLI menu-based setup experience with intelligent recommendations
"""

import sys
from typing import Dict, List, Optional, Tuple
from .hpc_config import HPCEnvironment, HPCConfig
from .partition_validator import PartitionValidator


def interactive_init(project_name: str) -> HPCConfig:
    """
    Interactive initialization with user preferences

    Returns configured HPCConfig
    """

    print("\n" + "="*70)
    print("ExpFlow Interactive Setup")
    print("="*70)

    # Step 1: Auto-detect environment
    print("\n[1/4] Detecting HPC environment...")
    username = HPCEnvironment.get_username()
    scratch = HPCEnvironment.get_scratch_dir()
    cluster = HPCEnvironment.detect_cluster()

    print(f"  Cluster: {cluster}")
    print(f"  Username: {username}")
    print(f"  Scratch: {scratch}")

    # Step 2: Account selection with recommendations
    print("\n[2/4] Selecting SLURM account...")
    accounts = HPCEnvironment.get_slurm_accounts()

    if not accounts:
        print("  WARNING: No SLURM accounts detected")
        default_account = input("  Enter account name (or press Enter for 'default'): ").strip() or "default"
    elif len(accounts) == 1:
        default_account = accounts[0]
        print(f"  Auto-selected: {default_account}")
    else:
        default_account = _select_account(accounts)

    # Step 3: GPU/Partition preference with intelligent recommendations
    print("\n[3/4] Selecting default GPU partition...")
    print("  Analyzing GPU partition access...")

    validator = PartitionValidator()
    # Use optimized detection (only known GPU partitions)
    partition_map = validator.detect_partition_access(accounts, filter_known_gpus=True)

    if not partition_map:
        print("  WARNING: No GPU partitions detected")
        print("  Using fallback partition. You can edit .hpc_config.yaml later.")
        default_partition = "gpu"
    else:
        default_partition = _select_partition_interactive(
            validator, partition_map, default_account
        )

    # Step 4: Additional preferences
    print("\n[4/4] Additional settings...")
    preferences = _get_additional_preferences()

    # Create config
    config = HPCConfig(
        username=username,
        user_home=HPCEnvironment.get_home_dir(),
        scratch_dir=scratch,
        project_name=project_name,
        project_root=f"{scratch}/{project_name}",
        default_account=default_account,
        default_partition=default_partition,
        cluster_name=cluster,
        available_partitions=list(partition_map.keys()) if partition_map else [],
        default_time_limit=preferences.get('time_limit', '48:00:00'),
    )

    # Summary
    print("\n" + "="*70)
    print("Configuration Summary")
    print("="*70)
    print(f"  Project: {project_name}")
    print(f"  Location: {config.project_root}")
    print(f"  Account: {default_account}")
    print(f"  Default GPU: {default_partition}")
    print(f"  Time Limit: {config.default_time_limit}")
    print("="*70)

    confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes', '']:
        print("Setup cancelled.")
        sys.exit(0)

    return config


def _select_account(accounts: List[str]) -> str:
    """
    Interactive account selection with recommendations
    """
    print(f"\n  Available accounts ({len(accounts)}):")

    # Analyze and recommend
    recommendations = []
    for i, account in enumerate(accounts, 1):
        # Heuristic: prefer general/public accounts for default
        is_general = 'general' in account.lower()
        is_public = 'public' in account.lower()

        rec_marker = ""
        if is_general:
            rec_marker = " [RECOMMENDED: Broadest access]"
            recommendations.append((i, "Broadest access"))
        elif is_public:
            rec_marker = " [Public access]"

        print(f"    {i}. {account}{rec_marker}")

    # Recommend first general account, or first overall
    recommended_idx = 1
    for i, account in enumerate(accounts, 1):
        if 'general' in account.lower():
            recommended_idx = i
            break

    while True:
        choice = input(f"\n  Select account [1-{len(accounts)}] (default: {recommended_idx}): ").strip()

        if not choice:
            return accounts[recommended_idx - 1]

        try:
            idx = int(choice)
            if 1 <= idx <= len(accounts):
                return accounts[idx - 1]
            else:
                print(f"  Invalid choice. Please enter 1-{len(accounts)}")
        except ValueError:
            print(f"  Invalid input. Please enter a number 1-{len(accounts)}")


def _select_partition_interactive(
    validator: PartitionValidator,
    partition_map: Dict[str, List[str]],
    selected_account: str
) -> str:
    """
    Interactive partition selection with intelligent recommendations
    """

    # Filter partitions accessible by selected account
    accessible_partitions = []
    for partition, accounts in partition_map.items():
        if selected_account in accounts:
            accessible_partitions.append(partition)

    if not accessible_partitions:
        print(f"  WARNING: No partitions accessible with account '{selected_account}'")
        return "gpu"

    # Categorize by GPU type
    gpu_categories = {
        'H200': [],
        'L40s': [],
        'A100': [],
        'RTX8000': [],
        'Other': []
    }

    for partition in accessible_partitions:
        info = validator.partition_map.get(partition)
        if info and info.gpu_type:
            if info.gpu_type in gpu_categories:
                gpu_categories[info.gpu_type].append(partition)
            else:
                gpu_categories['Other'].append(partition)
        else:
            gpu_categories['Other'].append(partition)

    # Display by category
    print(f"\n  Accessible partitions with account '{selected_account}':\n")

    partition_list = []
    idx = 1

    for gpu_type in ['H200', 'L40s', 'A100', 'RTX8000', 'Other']:
        partitions = gpu_categories[gpu_type]
        if not partitions:
            continue

        if gpu_type != 'Other':
            print(f"  {gpu_type} GPUs:")
        else:
            print(f"  Other:")

        for partition in partitions:
            is_public = 'public' in partition.lower()

            # Recommendation logic
            rec_marker = ""
            if gpu_type == 'L40s' and is_public:
                rec_marker = " [RECOMMENDED: Best availability]"
            elif gpu_type == 'H200' and is_public:
                rec_marker = " [RECOMMENDED: Powerful & available]"
            elif is_public:
                rec_marker = " [Public access]"

            info = validator.partition_map.get(partition)
            gpu_str = f" ({info.gpu_type})" if info and info.gpu_type else ""

            print(f"    {idx}. {partition}{gpu_str}{rec_marker}")
            partition_list.append(partition)
            idx += 1

        print()

    # Determine recommended partition
    recommended_idx = 1
    for i, partition in enumerate(partition_list, 1):
        if 'l40s_public' in partition.lower():
            recommended_idx = i
            break
        elif 'h200_public' in partition.lower() and recommended_idx == 1:
            recommended_idx = i

    while True:
        choice = input(f"  Select partition [1-{len(partition_list)}] (default: {recommended_idx}): ").strip()

        if not choice:
            return partition_list[recommended_idx - 1]

        try:
            idx = int(choice)
            if 1 <= idx <= len(partition_list):
                return partition_list[idx - 1]
            else:
                print(f"  Invalid choice. Please enter 1-{len(partition_list)}")
        except ValueError:
            print(f"  Invalid input. Please enter a number 1-{len(partition_list)}")


def _get_additional_preferences() -> Dict[str, str]:
    """
    Get additional user preferences
    """

    # Time limit
    print("\n  Default time limit for jobs:")
    print("    1. 6 hours")
    print("    2. 12 hours")
    print("    3. 24 hours")
    print("    4. 48 hours [RECOMMENDED]")
    print("    5. 72 hours")
    print("    6. Custom")

    time_limits = ["06:00:00", "12:00:00", "24:00:00", "48:00:00", "72:00:00"]

    while True:
        choice = input("\n  Select time limit [1-6] (default: 4): ").strip()

        if not choice or choice == '4':
            time_limit = "48:00:00"
            break

        try:
            idx = int(choice)
            if 1 <= idx <= 5:
                time_limit = time_limits[idx - 1]
                break
            elif idx == 6:
                custom = input("  Enter time limit (HH:MM:SS): ").strip()
                if custom:
                    time_limit = custom
                    break
            else:
                print("  Invalid choice. Please enter 1-6")
        except ValueError:
            print("  Invalid input. Please enter a number 1-6")

    return {
        'time_limit': time_limit
    }


def quick_init(project_name: str) -> HPCConfig:
    """
    Quick non-interactive initialization with smart defaults
    """
    username = HPCEnvironment.get_username()
    scratch = HPCEnvironment.get_scratch_dir()
    cluster = HPCEnvironment.detect_cluster()

    accounts = HPCEnvironment.get_slurm_accounts()
    # Prefer general account
    default_account = accounts[0] if accounts else "default"
    for acc in accounts:
        if 'general' in acc.lower():
            default_account = acc
            break

    # Smart partition selection
    partitions = HPCEnvironment.get_available_partitions()
    partition_preferences = ['l40s_public', 'h200_public', 'rtx8000', 'a100_public']
    default_partition = "gpu"

    for pref in partition_preferences:
        if pref in partitions:
            default_partition = pref
            break

    if default_partition == "gpu" and partitions:
        default_partition = partitions[0]

    config = HPCConfig(
        username=username,
        user_home=HPCEnvironment.get_home_dir(),
        scratch_dir=scratch,
        project_name=project_name,
        project_root=f"{scratch}/{project_name}",
        default_account=default_account,
        default_partition=default_partition,
        cluster_name=cluster,
        available_partitions=partitions,
        default_time_limit="48:00:00"
    )

    return config
