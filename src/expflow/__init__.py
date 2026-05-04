"""
ExpFlow - HPC Experiment Manager

Lightweight experiment tracking for SLURM-based HPC clusters.
Auto-detects environment, generates SLURM scripts, tracks experiments.
"""

__version__ = "0.10.1"

from .hpcexp_core import (
    BaseExperimentManager,
    BaseExperimentConfig,
    ExperimentMetadata,
    ConsistencyReport,
    BatchPreview,
)
from .hpc_config import (
    HPCConfig,
    HPCEnvironment,
    initialize_project,
    load_project_config,
)
from .resource_advisor import ResourceAdvisor, PartitionInfo, ResourceRecommendation
from .partition_validator import PartitionValidator, validate_job_config
from .interactive_init import interactive_init, quick_init
from .cache_builder import BaseCacheBuilder, CacheConfig
from .results_harvester import (
    BaseResultsHarvester,
    TrainingMetrics,
    EvaluationMetrics,
)
from .pruner import ExperimentPruner, PruneStats
from .experiment_series import ExperimentSeries, ExperimentConfig, CheckpointSpec
from .checkpoint_validator import CheckpointValidator, CheckpointResolver, ValidationReport
from .analysis_pipeline import (
    AnalysisPipeline,
    PipelineExperiment,
    PipelineResult,
    PipelineSummary,
)
from .results_storage import (
    ResultsStorage,
    ResultsQueryAPI,
    SQLiteBackend,
    MongoDBBackend,
    PostgreSQLBackend,
    export_to_json,
    export_to_csv,
)

# Phase-1 abstractions (extracted from navsim_manager.py)
from .result_record import ResultRecordBuilder, BaseRecordEnricher, NoopRecordEnricher
from .scope_aggregator import BaseScopeAggregator, NullScopeAggregator
from .run_history import AttemptGrouping
from .script_utils import quote_bash, assert_safe_identifier, git_worktree_block
from .eval_log_parser import EvalLogParser
from .eval_advisor import EvalResourceAdvisor, DefaultEvalResourceAdvisor
from .matrix_builder import MatrixExperimentBuilder

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
]
