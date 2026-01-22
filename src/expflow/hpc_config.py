#!/usr/bin/env python3
"""
HPC Environment Configuration
Auto-detects user environment and provides HPC-agnostic configuration
"""

import os
import pwd
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class HPCConfig:
    """HPC cluster configuration (auto-detected or user-specified)"""

    # User information (auto-detected)
    username: str
    user_home: str
    scratch_dir: str

    # Project configuration
    project_name: str
    project_root: str

    # SLURM defaults
    default_account: str
    default_partition: str
    default_time_limit: str = "48:00:00"

    # Paths
    experiments_dir: str = None
    logs_dir: str = None
    cache_dir: str = None
    checkpoints_dir: str = None

    # Optional: Container settings
    container_image: Optional[str] = None
    use_apptainer: bool = False

    # Cluster-specific settings
    cluster_name: str = "greene"  # greene, perlmutter, summit, etc.
    available_partitions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Auto-populate paths if not provided"""
        if self.experiments_dir is None:
            self.experiments_dir = f"{self.scratch_dir}/{self.project_name}/experiments"
        if self.logs_dir is None:
            self.logs_dir = f"{self.experiments_dir}/logs"
        if self.cache_dir is None:
            self.cache_dir = f"{self.experiments_dir}/cache"
        if self.checkpoints_dir is None:
            self.checkpoints_dir = f"{self.experiments_dir}/checkpoints"

    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class HPCEnvironment:
    """Auto-detect HPC environment details"""

    @staticmethod
    def get_username() -> str:
        """Get current username"""
        try:
            return pwd.getpwuid(os.getuid()).pw_name
        except:
            return os.getenv("USER", "unknown")

    @staticmethod
    def get_home_dir() -> str:
        """Get user home directory"""
        return str(Path.home())

    @staticmethod
    def get_scratch_dir() -> str:
        """Auto-detect scratch directory"""
        username = HPCEnvironment.get_username()

        # Common HPC scratch locations
        candidates = [
            f"/scratch/{username}",
            f"/scratch/users/{username}",
            f"/scratch/work/{username}",
            f"/global/scratch/{username}",
            f"/gpfs/scratch/{username}",
            os.getenv("SCRATCH"),
            os.getenv("SCRATCHDIR"),
        ]

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate

        # Fallback: use home directory
        return str(Path.home())

    @staticmethod
    def detect_cluster() -> str:
        """Detect which HPC cluster we're on"""
        hostname = subprocess.check_output(["hostname"], text=True).strip().lower()

        if "greene" in hostname or "hpc.nyu.edu" in hostname:
            return "greene"
        elif "perlmutter" in hostname:
            return "perlmutter"
        elif "summit" in hostname:
            return "summit"
        elif "frontera" in hostname:
            return "frontera"
        else:
            return "unknown"

    @staticmethod
    def get_slurm_accounts() -> List[str]:
        """Get available SLURM accounts for user"""
        try:
            # Use format to get Account field explicitly
            result = subprocess.run(
                ["sacctmgr", "show", "associations",
                 f"user={HPCEnvironment.get_username()}",
                 "format=Account", "-n"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                accounts = []
                for line in result.stdout.split('\n'):
                    account = line.strip()
                    # Filter out empty lines and default 'users' account
                    if account and account != 'users':
                        accounts.append(account)
                # Return unique accounts, preserving order
                seen = set()
                unique_accounts = []
                for acc in accounts:
                    if acc not in seen:
                        seen.add(acc)
                        unique_accounts.append(acc)
                return unique_accounts
        except:
            pass

        return []

    @staticmethod
    def get_available_partitions() -> List[str]:
        """Get available SLURM partitions"""
        try:
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%R"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                partitions = []
                for line in result.stdout.split('\n'):
                    partition = line.strip()
                    if partition:
                        partitions.append(partition)
                # Return unique partitions, preserving order
                seen = set()
                unique_partitions = []
                for part in partitions:
                    if part not in seen:
                        seen.add(part)
                        unique_partitions.append(part)
                return unique_partitions
        except:
            pass

        return []

    @staticmethod
    def create_default_config(project_name: str) -> HPCConfig:
        """Create default configuration by auto-detecting environment"""

        username = HPCEnvironment.get_username()
        home_dir = HPCEnvironment.get_home_dir()
        scratch_dir = HPCEnvironment.get_scratch_dir()
        cluster = HPCEnvironment.detect_cluster()

        # Try to get SLURM accounts
        accounts = HPCEnvironment.get_slurm_accounts()
        default_account = accounts[0] if accounts else "default"

        # Get available partitions
        partitions = HPCEnvironment.get_available_partitions()
        default_partition = partitions[0] if partitions else "gpu"

        config = HPCConfig(
            username=username,
            user_home=home_dir,
            scratch_dir=scratch_dir,
            project_name=project_name,
            project_root=f"{scratch_dir}/{project_name}",
            default_account=default_account,
            default_partition=default_partition,
            cluster_name=cluster,
            available_partitions=partitions
        )

        return config


def initialize_project(project_name: str, config_path: Optional[str] = None) -> HPCConfig:
    """
    Initialize a new HPC experiment project

    Args:
        project_name: Name of your project (e.g., 'navsim', 'llm-training')
        config_path: Optional path to existing config file

    Returns:
        HPCConfig object
    """

    if config_path and Path(config_path).exists():
        print(f"Loading existing config from {config_path}")
        return HPCConfig.load(config_path)

    print("Auto-detecting HPC environment...")
    config = HPCEnvironment.create_default_config(project_name)

    print(f"\n Detected HPC Environment:")
    print(f"  Cluster: {config.cluster_name}")
    print(f"  Username: {config.username}")
    print(f"  Scratch: {config.scratch_dir}")
    print(f"  Default Account: {config.default_account}")
    print(f"  Default Partition: {config.default_partition}")
    print(f"  Available Partitions: {', '.join(config.available_partitions[:5])}")

    # Create project directories
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

    return config


def load_project_config(project_root: Optional[str] = None) -> HPCConfig:
    """
    Load project configuration

    Args:
        project_root: Project root directory (defaults to current directory)

    Returns:
        HPCConfig object
    """

    if project_root is None:
        project_root = os.getcwd()

    config_path = Path(project_root) / ".hpc_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"No HPC config found at {config_path}\n"
            f"Run: hpcexp init <project_name> to create one"
        )

    return HPCConfig.load(str(config_path))


if __name__ == "__main__":
    """Test configuration detection"""
    import sys

    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = "test_project"

    print("="*70)
    print("HPC Environment Auto-Detection Test")
    print("="*70)

    config = initialize_project(project_name)

    print("\n" + "="*70)
    print("Configuration:")
    print("="*70)
    for key, value in config.to_dict().items():
        if isinstance(value, list):
            print(f"{key}: [{len(value)} items]")
        else:
            print(f"{key}: {value}")
