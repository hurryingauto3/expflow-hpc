"""
ExpFlow - HPC Experiment Manager

Lightweight experiment tracking for SLURM-based HPC clusters.
Auto-detects environment, generates SLURM scripts, tracks experiments.
"""

__version__ = "0.10.1"

from .analysis_pipeline import (
    AnalysisPipeline,
    PipelineExperiment,
    PipelineResult,
    PipelineSummary,
)
from .cache_builder import BaseCacheBuilder, CacheConfig
from .checkpoint_validator import CheckpointResolver, CheckpointValidator, ValidationReport
from .eval_advisor import DefaultEvalResourceAdvisor, EvalResourceAdvisor
from .eval_log_parser import EvalLogParser

# Phase-2: execution backends (laptop / CI / SLURM)
from .execution import (
    ExecutionBackend,
    LocalBackend,
    SlurmBackend,
    auto_detect_backend,
)
from .experiment_series import CheckpointSpec, ExperimentConfig, ExperimentSeries
from .hpc_config import (
    HPCConfig,
    HPCEnvironment,
    initialize_project,
    load_project_config,
)
from .hpcexp_core import (
    BaseExperimentConfig,
    BaseExperimentManager,
    BatchPreview,
    ConsistencyReport,
    ExperimentMetadata,
)
from .interactive_init import interactive_init, quick_init
from .matrix_builder import MatrixExperimentBuilder
from .partition_validator import PartitionValidator, validate_job_config
from .pruner import ExperimentPruner, PruneStats
from .resource_advisor import PartitionInfo, ResourceAdvisor, ResourceRecommendation

# Phase-1 abstractions (extracted from navsim_manager.py)
from .result_record import BaseRecordEnricher, NoopRecordEnricher, ResultRecordBuilder
from .results_harvester import (
    BaseResultsHarvester,
    EvaluationMetrics,
    TrainingMetrics,
)
from .results_storage import (
    MongoDBBackend,
    PostgreSQLBackend,
    ResultsQueryAPI,
    ResultsStorage,
    SQLiteBackend,
    export_to_csv,
    export_to_json,
)
from .run_history import AttemptGrouping
from .scope_aggregator import BaseScopeAggregator, NullScopeAggregator
from .script_utils import assert_safe_identifier, git_worktree_block, quote_bash

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
    "BaseCacheBuilder",
    "CacheConfig",
    "BaseResultsHarvester",
    "TrainingMetrics",
    "EvaluationMetrics",
    "ExperimentPruner",
    "PruneStats",
    "ConsistencyReport",
    "BatchPreview",
    "ExperimentSeries",
    "ExperimentConfig",
    "CheckpointSpec",
    "CheckpointValidator",
    "CheckpointResolver",
    "ValidationReport",
    "AnalysisPipeline",
    "PipelineExperiment",
    "PipelineResult",
    "PipelineSummary",
    "ResultsStorage",
    "ResultsQueryAPI",
    "SQLiteBackend",
    "MongoDBBackend",
    "PostgreSQLBackend",
    "export_to_json",
    "export_to_csv",
    "ResultRecordBuilder",
    "BaseRecordEnricher",
    "NoopRecordEnricher",
    "BaseScopeAggregator",
    "NullScopeAggregator",
    "AttemptGrouping",
    "quote_bash",
    "assert_safe_identifier",
    "git_worktree_block",
    "EvalLogParser",
    "EvalResourceAdvisor",
    "DefaultEvalResourceAdvisor",
    "MatrixExperimentBuilder",
    "ExecutionBackend",
    "LocalBackend",
    "SlurmBackend",
    "auto_detect_backend",
]
