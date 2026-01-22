"""
ExpFlow - HPC Experiment Manager

Lightweight experiment tracking for SLURM-based HPC clusters.
Auto-detects environment, generates SLURM scripts, tracks experiments.
"""

__version__ = "0.1.0"

from .hpcexp_core import BaseExperimentManager, BaseExperimentConfig, ExperimentMetadata
from .hpc_config import HPCConfig, HPCEnvironment, initialize_project, load_project_config
from .resource_advisor import ResourceAdvisor, PartitionInfo, ResourceRecommendation

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
]
