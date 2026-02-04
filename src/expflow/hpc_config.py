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

    # Container settings
    container_image: Optional[str] = None
    use_apptainer: bool = False
    container_bind_mounts: List[str] = field(default_factory=list)

    # Conda/Environment settings
    conda_root: Optional[str] = None  # Auto-detected if None
    conda_env: Optional[str] = None  # Environment name to activate
    module_loads: List[str] = field(default_factory=list)  # e.g., ["anaconda3/2025.06"]

    # SquashFS overlay settings
    overlay_cache_dir: Optional[str] = None  # Where .sqsh overlays are stored

    # GPU monitoring
    enable_gpu_monitoring: bool = False
    gpu_monitor_interval: int = 60  # seconds

    # NCCL optimization presets (by GPU type)
    nccl_preset: Optional[str] = None  # 'h200', 'a100', 'l40s', 'rtx8000', or None
    nccl_env_vars: Dict[str, str] = field(default_factory=dict)  # Custom NCCL vars

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
        if self.overlay_cache_dir is None:
            self.overlay_cache_dir = f"{self.cache_dir}/overlays"

        # Auto-detect conda root if not provided
        if self.conda_root is None:
            self.conda_root = HPCEnvironment.detect_conda_root()

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
            # Use format with explicit width to prevent truncation
            result = subprocess.run(
                ["sacctmgr", "show", "associations",
                 f"user={HPCEnvironment.get_username()}",
                 "format=Account%40", "-n"],  # %40 = 40 character width
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
    def detect_conda_root() -> Optional[str]:
        """Auto-detect conda installation root"""
        # Try common locations
        username = HPCEnvironment.get_username()
        candidates = [
            f"/scratch/{username}/miniconda3",
            f"/scratch/{username}/anaconda3",
            f"{Path.home()}/miniconda3",
            f"{Path.home()}/anaconda3",
            "/opt/conda",
            "/usr/local/conda",
        ]

        # Also check CONDA_PREFIX environment variable
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            # Go up to the conda root (remove /envs/env_name if present)
            conda_root = Path(conda_prefix)
            while conda_root.name in ["envs", "bin"]:
                conda_root = conda_root.parent
            candidates.insert(0, str(conda_root))

        for candidate in candidates:
            conda_sh = Path(candidate) / "etc" / "profile.d" / "conda.sh"
            if conda_sh.exists():
                return candidate

        return None

    @staticmethod
    def detect_container_image() -> Optional[str]:
        """Auto-detect default container image on cluster"""
        cluster = HPCEnvironment.detect_cluster()

        # NYU Greene default containers
        if cluster == "greene":
            # Check for common containers
            candidates = [
                "/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif",
                "/share/apps/images/cuda12.1.0-cudnn8.9.0-ubuntu22.04.sif",
                "/share/apps/images/pytorch-2.1.0-cuda12.1.sif",
            ]
            for candidate in candidates:
                if Path(candidate).exists():
                    return candidate

        return None

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

        # Smart partition selection based on known preferences
        # NYU Greene specific: prefer public accessible partitions
        default_partition = "gpu"  # fallback
        if partitions:
            # Ordered by preference (most broadly accessible first)
            partition_preferences = [
                'l40s_public',      # L40s, broadly accessible
                'h200_public',      # H200, broadly accessible
                'rtx8000',          # RTX8000, broadly accessible
                'a100_public',      # A100, broadly accessible
                'h200_tandon',      # H200, Tandon only
                'h200_courant',     # H200, Courant only
                'h200_cds',         # H200, CDS only
            ]

            # Select first preferred partition that exists
            for pref in partition_preferences:
                if pref in partitions:
                    default_partition = pref
                    break

            # If no known partition found, use first available
            if default_partition == "gpu":
                default_partition = partitions[0]

        # Auto-detect container and conda
        container_image = HPCEnvironment.detect_container_image()
        conda_root = HPCEnvironment.detect_conda_root()

        config = HPCConfig(
            username=username,
            user_home=home_dir,
            scratch_dir=scratch_dir,
            project_name=project_name,
            project_root=f"{scratch_dir}/{project_name}",
            default_account=default_account,
            default_partition=default_partition,
            cluster_name=cluster,
            available_partitions=partitions,
            container_image=container_image,
            conda_root=conda_root
        )

        return config

    @staticmethod
    def _quick_test_partition(partition: str, account: str) -> bool:
        """Quick test if partition-account combination works"""
        try:
            # GPU partitions that require --gres
            gpu_partitions = {'h200_public', 'h200_tandon', 'h200_bpeher',
                            'l40s_public', 'rtx8000', 'a100_public'}

            cmd = ["sbatch", "--test-only", "-p", partition, "-A", account,
                   "-N1", "-t", "1:00:00"]

            if partition in gpu_partitions:
                cmd.extend(["--gres=gpu:1"])

            cmd.extend(["--wrap=hostname"])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)

            # Success if would schedule
            return result.returncode == 0 and "to start at" in result.stdout
        except:
            return False


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
