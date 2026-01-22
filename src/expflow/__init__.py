"""
ExpFlow - HPC Experiment Manager

Lightweight experiment tracking for SLURM-based HPC clusters.
Auto-detects environment, generates SLURM scripts, tracks experiments.
"""

__version__ = "0.3.3"

from .hpcexp_core import BaseExperimentManager, BaseExperimentConfig, ExperimentMetadata
from .hpc_config import HPCConfig, HPCEnvironment, initialize_project, load_project_config
from .resource_advisor import ResourceAdvisor, PartitionInfo, ResourceRecommendation
from .partition_validator import PartitionValidator, validate_job_config
from .interactive_init import interactive_init, quick_init

__all__ = [
    "BaseExperimentManager",
    "BaseExperimentConfig",
    "ExperimentMetadata",
    "HPCConfig",
    "HPCEnvironment",
    "initialize_project",
    "load_project_config",
    "ResourceAdvisor",
    "PartitionInfo",
    "ResourceRecommendation",
    "PartitionValidator",
    "validate_job_config",
    "interactive_init",
    "quick_init",
]
