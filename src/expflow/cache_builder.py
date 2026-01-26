#!/usr/bin/env python3
"""
Cache Building Framework for ExpFlow

Provides generic infrastructure for building and managing caches on HPC:
1. Initial cache generation (CPU-heavy, parallel)
2. SquashFS compression (inode optimization)
3. Cleanup of original cache directory

Usage:
    manager = MyCacheManager()
    manager.build_cache("training_cache_v1", cache_type="training")
    manager.squashfs_cache("training_cache_v1")
    manager.cleanup_cache("training_cache_v1")
"""

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .hpc_config import HPCConfig


# =============================================================================
# Cache Configuration
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for cache building"""

    # Identifiers
    cache_name: str
    cache_type: str  # "training", "metric", "validation", etc.
    description: str = ""

    # Paths
    cache_output_dir: Optional[str] = None  # Where cache will be built
    overlay_output_dir: Optional[str] = None  # Where squashfs will be saved

    # Resource configuration for cache building
    partition: str = "cpu"  # Usually CPU-only for caching
    account: str = "default"
    num_cpus: int = 128
    memory: str = "256G"
    time_limit: str = "48:00:00"

    # Cache building parameters (project-specific)
    num_workers: int = 96
    force_rebuild: bool = True
    cache_params: Dict[str, Any] = field(default_factory=dict)

    # SquashFS options
    squashfs_compression: str = "zstd"
    squashfs_block_size: int = 1048576  # 1MB
    squashfs_processors: int = 32

    # Container (if using Apptainer/Singularity)
    container_image: Optional[str] = None
    use_container: bool = False

    # Metadata
    created_at: Optional[str] = None
    build_job_id: Optional[str] = None
    squashfs_job_id: Optional[str] = None
    cleanup_job_id: Optional[str] = None
    status: str = "created"  # created, building, built, compressing, compressed, cleaned


# =============================================================================
# Base Cache Builder
# =============================================================================

class BaseCacheBuilder(ABC):
    """
    Base class for cache building operations.

    Provides a 3-stage pipeline:
    1. Build cache (data extraction/preprocessing)
    2. Compress to SquashFS (inode optimization)
    3. Cleanup original directory

    Subclass this to implement project-specific cache building.
    """

    def __init__(self, hpc_config: HPCConfig):
        """
        Initialize cache builder

        Args:
            hpc_config: HPC configuration object
        """
        self.hpc_config = hpc_config

        # Setup paths
        self.project_root = Path(hpc_config.project_root)
        self.cache_base_dir = Path(hpc_config.cache_dir)
        self.logs_dir = Path(hpc_config.logs_dir)

        # Cache-specific directories - keep everything under experiments/cache
        self.overlay_dir = self.cache_base_dir / "overlays"  # experiments/cache/overlays
        self.cache_configs_dir = self.project_root / "cache_configs"
        self.cache_scripts_dir = self.project_root / "generated_scripts" / "cache"
        self.cache_metadata_db = self.cache_configs_dir / "caches.json"

        # Create directories
        for directory in [
            self.cache_configs_dir,
            self.cache_scripts_dir,
            self.overlay_dir,
            self.cache_base_dir,
            self.logs_dir / "caching"
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self._load_cache_metadata()

    def _load_cache_metadata(self):
        """Load cache tracking database"""
        if self.cache_metadata_db.exists():
            with open(self.cache_metadata_db, 'r') as f:
                self.cache_metadata = json.load(f)
        else:
            self.cache_metadata = {}

    def _save_cache_metadata(self):
        """Save cache tracking database"""
        with open(self.cache_metadata_db, 'w') as f:
            json.dump(self.cache_metadata, f, indent=2, default=str)

    # -------------------------------------------------------------------------
    # Abstract Methods (implement in subclass)
    # -------------------------------------------------------------------------

    @abstractmethod
    def _generate_cache_build_script(self, config: CacheConfig) -> str:
        """
        Generate the cache building script.

        This is project-specific - implement your cache generation logic here.

        Args:
            config: Cache configuration

        Returns:
            SLURM script as string
        """
        pass

    @abstractmethod
    def get_cache_script_command(self, config: CacheConfig) -> str:
        """
        Get the Python command to run for cache building.

        Example:
            return f"python run_dataset_caching.py split={config.cache_params['split']}"

        Args:
            config: Cache configuration

        Returns:
            Python command string
        """
        pass

    # -------------------------------------------------------------------------
    # SquashFS Generation (generic, rarely needs override)
    # -------------------------------------------------------------------------

    def _generate_squashfs_script(self, config: CacheConfig) -> str:
        """
        Generate SquashFS compression script.

        This is generic and rarely needs to be overridden.
        """
        cache_name = config.cache_name
        source_path = Path(config.cache_output_dir) if config.cache_output_dir else self.cache_base_dir / cache_name
        output_file = Path(config.overlay_output_dir) / f"{cache_name}.sqsh" if config.overlay_output_dir else self.overlay_dir / f"{cache_name}.sqsh"

        container = config.container_image or "/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

        script = f'''#!/bin/bash
# =============================================================================
# SquashFS Cache Builder - Auto-generated by ExpFlow
# Cache: {cache_name}
# Description: {config.description}
# Generated: {datetime.now().isoformat()}
# =============================================================================

#SBATCH --job-name=sqsh_{cache_name[:20]}
#SBATCH --account={config.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.squashfs_processors}
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output={self.logs_dir}/caching/squashfs_{cache_name}_%j.out
#SBATCH --error={self.logs_dir}/caching/squashfs_{cache_name}_%j.err

set -e

echo "=============================================="
echo "SquashFS Cache Compression"
echo "=============================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "  Cache: {cache_name}"
echo "  Source: {source_path}"
echo "  Output: {output_file}"
echo ""

# Validate source exists
if [[ ! -d "{source_path}" ]]; then
    echo "ERROR: Source directory not found: {source_path}"
    exit 1
fi

# Check if output already exists
if [[ -f "{output_file}" ]]; then
    echo "ERROR: Output file already exists: {output_file}"
    echo "Delete it first if you want to rebuild:"
    echo "  rm {output_file}"
    exit 1
fi

# Show source stats
echo "Source directory stats:"
du -sh "{source_path}"
find "{source_path}" -type f | wc -l | xargs -I {{}} echo "  Files: {{}}"
echo ""

# Build SquashFS
echo "Building SquashFS image..."
apptainer exec --pwd {self.hpc_config.scratch_dir} "{container}" \\
    mksquashfs "{source_path}" "{output_file}" \\
    -noappend \\
    -comp {config.squashfs_compression} \\
    -b {config.squashfs_block_size} \\
    -processors {config.squashfs_processors}

echo ""
echo "=============================================="
echo "Compression completed: $(date)"
echo "=============================================="
echo ""
echo "Output file:"
ls -lh "{output_file}"
echo ""
echo "Compression ratio:"
SOURCE_SIZE=$(du -sb "{source_path}" | cut -f1)
OUTPUT_SIZE=$(stat -f%z "{output_file}" 2>/dev/null || stat -c%s "{output_file}")
RATIO=$(echo "scale=2; $OUTPUT_SIZE * 100 / $SOURCE_SIZE" | bc)
echo "  Original: $(numfmt --to=iec-i --suffix=B $SOURCE_SIZE)"
echo "  Compressed: $(numfmt --to=iec-i --suffix=B $OUTPUT_SIZE)"
echo "  Ratio: ${{RATIO}}%"
echo ""
echo "Usage:"
echo "  Mount with: --bind {output_file}:{source_path}:image-src=/"
echo ""
echo "Next steps:"
echo "  1. Verify overlay works in training"
echo "  2. Run cleanup: sbatch {self.cache_scripts_dir}/cleanup_{cache_name}.slurm"
echo "=============================================="
'''
        return script

    # -------------------------------------------------------------------------
    # Cleanup Script (generic)
    # -------------------------------------------------------------------------

    def _generate_cleanup_script(self, config: CacheConfig) -> str:
        """Generate cleanup script to remove original cache directory"""

        cache_name = config.cache_name
        source_path = Path(config.cache_output_dir) if config.cache_output_dir else self.cache_base_dir / cache_name
        output_file = Path(config.overlay_output_dir) / f"{cache_name}.sqsh" if config.overlay_output_dir else self.overlay_dir / f"{cache_name}.sqsh"

        script = f'''#!/bin/bash
# =============================================================================
# Cache Cleanup - Auto-generated by ExpFlow
# Cache: {cache_name}
# Generated: {datetime.now().isoformat()}
# =============================================================================

#SBATCH --job-name=rm_{cache_name[:20]}
#SBATCH --account={config.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output={self.logs_dir}/caching/cleanup_{cache_name}_%j.out
#SBATCH --error={self.logs_dir}/caching/cleanup_{cache_name}_%j.err

set -e

echo "=============================================="
echo "Cache Cleanup"
echo "=============================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Target directory: {source_path}"
echo ""

# Verify the squashfs was created successfully
if [[ ! -f "{output_file}" ]]; then
    echo "ERROR: SquashFS file not found: {output_file}"
    echo "The compression job may have failed. Aborting cleanup."
    exit 1
fi

echo "✓ SquashFS file exists: {output_file}"
ls -lh "{output_file}"
echo ""

# Check directory exists before attempting removal
if [[ ! -d "{source_path}" ]]; then
    echo "WARNING: Directory does not exist: {source_path}"
    echo "Nothing to remove."
    exit 0
fi

# Show what we're about to remove
echo "Directory stats before removal:"
du -sh "{source_path}" || true
find "{source_path}" -type f | wc -l | xargs -I {{}} echo "  Files: {{}}"
echo ""

echo "Starting removal..."
rm -rf "{source_path}"

echo ""
echo "=============================================="
echo "Cleanup completed: $(date)"
echo "=============================================="
echo ""
echo "Freed disk space. SquashFS overlay at:"
echo "  {output_file}"
echo ""
'''
        return script

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def create_cache_config(
        self,
        cache_name: str,
        cache_type: str,
        description: str = "",
        **kwargs
    ) -> CacheConfig:
        """
        Create a new cache configuration.

        Args:
            cache_name: Unique cache identifier
            cache_type: Type of cache (training, metric, etc.)
            description: Cache description
            **kwargs: Additional config parameters

        Returns:
            CacheConfig object
        """

        # Set defaults from HPC config
        defaults = {
            "account": self.hpc_config.default_account,
            "cache_output_dir": str(self.cache_base_dir / cache_name),
            "overlay_output_dir": str(self.overlay_dir),
        }

        config_dict = {
            "cache_name": cache_name,
            "cache_type": cache_type,
            "description": description,
            "created_at": datetime.now().isoformat(),
            **defaults,
            **kwargs
        }

        config = CacheConfig(**config_dict)

        # Save config
        config_path = self.cache_configs_dir / f"{cache_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Update metadata
        self.cache_metadata[cache_name] = {
            "cache_name": cache_name,
            "cache_type": cache_type,
            "config": config_dict,
            "status": "created",
            "build_job_id": None,
            "squashfs_job_id": None,
            "cleanup_job_id": None,
        }
        self._save_cache_metadata()

        print(f"✓ Created cache config: {cache_name}")
        print(f"  Config: {config_path}")
        return config

    def build_cache(
        self,
        cache_name: str,
        dry_run: bool = False,
        wait_for: Optional[str] = None
    ) -> Optional[str]:
        """
        Submit cache building job.

        Args:
            cache_name: Cache identifier
            dry_run: If True, generate script but don't submit
            wait_for: Optional job ID to wait for before starting

        Returns:
            Job ID if submitted, None if dry run
        """

        if cache_name not in self.cache_metadata:
            print(f"Error: Cache {cache_name} not found.")
            print("Create it first with create_cache_config()")
            return None

        meta = self.cache_metadata[cache_name]
        config = CacheConfig(**meta["config"])

        # Generate script
        script = self._generate_cache_build_script(config)
        script_path = self.cache_scripts_dir / f"build_{cache_name}.slurm"

        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        print(f"✓ Generated: {script_path}")

        if dry_run:
            print(f"\n[DRY RUN] Would submit: sbatch {script_path}")
            return None

        # Submit
        cmd = ["sbatch"]
        if wait_for:
            cmd.extend(["--dependency", f"afterok:{wait_for}"])
        cmd.append(str(script_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]

            meta["build_job_id"] = job_id
            meta["status"] = "building"
            self._save_cache_metadata()

            print(f"✓ Submitted cache build: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            print(f"Error submitting: {e.stderr}")
            return None

    def squashfs_cache(
        self,
        cache_name: str,
        dry_run: bool = False,
        wait_for: Optional[str] = None
    ) -> Optional[str]:
        """
        Submit SquashFS compression job.

        Args:
            cache_name: Cache identifier
            dry_run: If True, generate script but don't submit
            wait_for: Optional job ID to wait for (usually the build job)

        Returns:
            Job ID if submitted, None if dry run
        """

        if cache_name not in self.cache_metadata:
            print(f"Error: Cache {cache_name} not found.")
            return None

        meta = self.cache_metadata[cache_name]
        config = CacheConfig(**meta["config"])

        # Generate script
        script = self._generate_squashfs_script(config)
        script_path = self.cache_scripts_dir / f"squashfs_{cache_name}.slurm"

        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        print(f"✓ Generated: {script_path}")

        if dry_run:
            print(f"\n[DRY RUN] Would submit: sbatch {script_path}")
            return None

        # Submit
        cmd = ["sbatch"]
        if wait_for:
            cmd.extend(["--dependency", f"afterok:{wait_for}"])
        elif meta.get("build_job_id"):
            cmd.extend(["--dependency", f"afterok:{meta['build_job_id']}"])
        cmd.append(str(script_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]

            meta["squashfs_job_id"] = job_id
            meta["status"] = "compressing"
            self._save_cache_metadata()

            print(f"✓ Submitted SquashFS compression: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            print(f"Error submitting: {e.stderr}")
            return None

    def cleanup_cache(
        self,
        cache_name: str,
        dry_run: bool = False,
        wait_for: Optional[str] = None
    ) -> Optional[str]:
        """
        Submit cleanup job to remove original cache directory.

        Args:
            cache_name: Cache identifier
            dry_run: If True, generate script but don't submit
            wait_for: Optional job ID to wait for (usually squashfs job)

        Returns:
            Job ID if submitted, None if dry run
        """

        if cache_name not in self.cache_metadata:
            print(f"Error: Cache {cache_name} not found.")
            return None

        meta = self.cache_metadata[cache_name]
        config = CacheConfig(**meta["config"])

        # Generate script
        script = self._generate_cleanup_script(config)
        script_path = self.cache_scripts_dir / f"cleanup_{cache_name}.slurm"

        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        print(f"✓ Generated: {script_path}")

        if dry_run:
            print(f"\n[DRY RUN] Would submit: sbatch {script_path}")
            return None

        # Submit
        cmd = ["sbatch"]
        if wait_for:
            cmd.extend(["--dependency", f"afterok:{wait_for}"])
        elif meta.get("squashfs_job_id"):
            cmd.extend(["--dependency", f"afterok:{meta['squashfs_job_id']}"])
        cmd.append(str(script_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]

            meta["cleanup_job_id"] = job_id
            self._save_cache_metadata()

            print(f"✓ Submitted cleanup: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            print(f"Error submitting: {e.stderr}")
            return None

    def build_cache_pipeline(
        self,
        cache_name: str,
        skip_squashfs: bool = False,
        skip_cleanup: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Optional[str]]:
        """
        Run complete cache building pipeline: build → squashfs → cleanup.

        Jobs are chained with SLURM dependencies.

        Args:
            cache_name: Cache identifier
            skip_squashfs: Skip SquashFS compression
            skip_cleanup: Skip cleanup step
            dry_run: Generate scripts but don't submit

        Returns:
            Dictionary with job IDs: {build, squashfs, cleanup}
        """

        job_ids = {}

        # Step 1: Build cache
        build_job = self.build_cache(cache_name, dry_run=dry_run)
        job_ids["build"] = build_job

        if skip_squashfs:
            return job_ids

        # Step 2: SquashFS (wait for build)
        squashfs_job = self.squashfs_cache(
            cache_name,
            dry_run=dry_run,
            wait_for=build_job
        )
        job_ids["squashfs"] = squashfs_job

        if skip_cleanup:
            return job_ids

        # Step 3: Cleanup (wait for squashfs)
        cleanup_job = self.cleanup_cache(
            cache_name,
            dry_run=dry_run,
            wait_for=squashfs_job
        )
        job_ids["cleanup"] = cleanup_job

        return job_ids

    def list_caches(self, cache_type: Optional[str] = None):
        """List all caches"""

        if not self.cache_metadata:
            print("No caches found")
            return

        filtered = []
        for name, meta in self.cache_metadata.items():
            if cache_type and meta.get("cache_type") != cache_type:
                continue
            filtered.append((name, meta))

        if not filtered:
            print(f"No caches found" + (f" of type '{cache_type}'" if cache_type else ""))
            return

        print(f"\n{'Name':<40} {'Type':<15} {'Status':<12} {'Build Job'}")
        print("=" * 90)

        for name, meta in sorted(filtered):
            cache_type_str = meta.get("cache_type", "unknown")[:14]
            status = meta.get("status", "unknown")[:11]
            build_job = meta.get("build_job_id", "-")
            print(f"{name:<40} {cache_type_str:<15} {status:<12} {build_job}")

    def show_cache(self, cache_name: str):
        """Show detailed cache information"""

        if cache_name not in self.cache_metadata:
            print(f"Error: Cache {cache_name} not found")
            return

        meta = self.cache_metadata[cache_name]
        config = meta.get("config", {})

        print(f"\n{'='*60}")
        print(f"Cache: {cache_name}")
        print(f"{'='*60}")
        print(f"Type: {meta.get('cache_type', 'unknown')}")
        print(f"Status: {meta.get('status', 'unknown')}")
        print(f"Description: {config.get('description', '')}")
        print(f"Created: {config.get('created_at', '')}")
        print()

        print("Configuration:")
        print(f"  Output: {config.get('cache_output_dir', '')}")
        print(f"  Overlay: {config.get('overlay_output_dir', '')}")
        print(f"  Partition: {config.get('partition', '')}")
        print(f"  CPUs: {config.get('num_cpus', '')}")
        print(f"  Memory: {config.get('memory', '')}")
        print(f"  Workers: {config.get('num_workers', '')}")
        print()

        print("Jobs:")
        print(f"  Build: {meta.get('build_job_id', 'N/A')}")
        print(f"  SquashFS: {meta.get('squashfs_job_id', 'N/A')}")
        print(f"  Cleanup: {meta.get('cleanup_job_id', 'N/A')}")
