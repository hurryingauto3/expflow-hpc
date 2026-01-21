#!/usr/bin/env python3
"""
HPC Resource Advisor for NYU Greene Cluster

Analyzes SLURM queue, available resources, and experiment requirements
to suggest optimal partition and GPU allocation.

Features:
- Real-time SLURM queue analysis
- Wait time estimation
- Resource availability checking
- Cost-efficiency scoring
- Optional Gemini API for intelligent suggestions

Usage:
    # Basic analysis
    python resource_advisor.py analyze

    # For specific experiment
    python resource_advisor.py suggest --exp-id b15

    # With Gemini API suggestions
    python resource_advisor.py suggest --exp-id b15 --use-gemini

    # Check current queue status
    python resource_advisor.py status
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PartitionInfo:
    """Information about a SLURM partition"""
    name: str
    total_nodes: int
    available_nodes: int
    total_gpus: int
    available_gpus: int
    gpu_type: str
    gpus_per_node: int
    queue_length: int
    avg_wait_time_mins: Optional[float] = None
    priority: str = "normal"  # high, normal, low


@dataclass
class ResourceRecommendation:
    """Resource allocation recommendation"""
    partition: str
    num_gpus: int
    num_nodes: int
    estimated_wait_mins: float
    reason: str
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0  # Higher is better

    # Reproducibility considerations
    global_batch_size: Optional[int] = None
    per_gpu_batch_size: Optional[int] = None
    gradient_accumulation: int = 1
    precision_recommendation: str = "bf16"


# =============================================================================
# Resource Advisor Class
# =============================================================================

class ResourceAdvisor:
    """Analyzes HPC resources and provides recommendations"""

    # NYU Greene partition configurations
    PARTITIONS = {
        "l40s_public": {
            "gpu_type": "L40S",
            "gpus_per_node": 4,
            "total_nodes": 10,  # Approximate
            "arch": "Ada Lovelace",
            "memory_gb": 48,
            "fp8_support": False,
            "priority": "normal"
        },
        "h200_tandon": {
            "gpu_type": "H200",
            "gpus_per_node": 2,
            "total_nodes": 5,  # Approximate
            "arch": "Hopper",
            "memory_gb": 141,
            "fp8_support": True,
            "priority": "normal"
        },
        "a100_public": {
            "gpu_type": "A100",
            "gpus_per_node": 4,
            "total_nodes": 20,
            "arch": "Ampere",
            "memory_gb": 80,
            "fp8_support": False,
            "priority": "normal"
        }
    }

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.cache_dir = Path.home() / ".cache" / "hpc_advisor"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_queue_status(self) -> Dict[str, PartitionInfo]:
        """Query SLURM for current partition status"""

        partition_stats = {}

        for partition_name, config in self.PARTITIONS.items():
            try:
                # Get partition info
                sinfo_cmd = [
                    "sinfo",
                    "-p", partition_name,
                    "-o", "%D %T",
                    "--noheader"
                ]

                result = subprocess.run(
                    sinfo_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode != 0:
                    continue

                # Parse sinfo output
                lines = result.stdout.strip().split('\n')
                total_nodes = 0
                available_nodes = 0

                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        count = int(parts[0])
                        state = parts[1]
                        total_nodes += count
                        if "idle" in state.lower() or "mix" in state.lower():
                            available_nodes += count

                # Get queue length
                squeue_cmd = [
                    "squeue",
                    "-p", partition_name,
                    "-h",
                    "-o", "%i"
                ]

                result = subprocess.run(
                    squeue_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                queue_length = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

                # Calculate GPU availability
                total_gpus = total_nodes * config["gpus_per_node"]
                available_gpus = available_nodes * config["gpus_per_node"]

                # Estimate wait time (simple heuristic)
                avg_wait_mins = None
                if available_gpus > 0:
                    avg_wait_mins = 0
                elif queue_length > 0:
                    # Rough estimate: 30 mins per queued job ahead
                    avg_wait_mins = queue_length * 30

                partition_stats[partition_name] = PartitionInfo(
                    name=partition_name,
                    total_nodes=total_nodes or config["total_nodes"],
                    available_nodes=available_nodes,
                    total_gpus=total_gpus or config["total_nodes"] * config["gpus_per_node"],
                    available_gpus=available_gpus,
                    gpu_type=config["gpu_type"],
                    gpus_per_node=config["gpus_per_node"],
                    queue_length=queue_length,
                    avg_wait_time_mins=avg_wait_mins,
                    priority=config["priority"]
                )

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                # Fallback to config defaults if SLURM query fails
                partition_stats[partition_name] = PartitionInfo(
                    name=partition_name,
                    total_nodes=config["total_nodes"],
                    available_nodes=0,
                    total_gpus=config["total_nodes"] * config["gpus_per_node"],
                    available_gpus=0,
                    gpu_type=config["gpu_type"],
                    gpus_per_node=config["gpus_per_node"],
                    queue_length=0,
                    avg_wait_time_mins=None,
                    priority=config["priority"]
                )

        return partition_stats

    def calculate_global_batch_requirements(
        self,
        target_global_batch: int,
        num_gpus: int,
        max_per_gpu_batch: int = 48
    ) -> Tuple[int, int]:
        """Calculate per-GPU batch and gradient accumulation"""

        # Try to fit global batch without accumulation
        per_gpu_batch = target_global_batch // num_gpus

        if per_gpu_batch <= max_per_gpu_batch:
            return per_gpu_batch, 1

        # Need gradient accumulation
        # Start with max per-GPU batch and accumulate
        per_gpu_batch = max_per_gpu_batch
        grad_accum = (target_global_batch + (per_gpu_batch * num_gpus - 1)) // (per_gpu_batch * num_gpus)

        return per_gpu_batch, grad_accum

    def score_recommendation(
        self,
        partition_info: PartitionInfo,
        num_gpus: int,
        exp_config: Optional[Dict] = None
    ) -> float:
        """Score a resource allocation (higher is better)"""

        score = 100.0

        # Availability penalty
        if partition_info.available_gpus < num_gpus:
            score -= 30

        # Queue length penalty
        score -= min(partition_info.queue_length * 2, 30)

        # Wait time penalty
        if partition_info.avg_wait_time_mins:
            score -= min(partition_info.avg_wait_time_mins / 10, 20)

        # Efficiency bonus for using full nodes
        if num_gpus == partition_info.gpus_per_node:
            score += 10

        # Multi-node penalty (more complex, more failure modes)
        num_nodes = (num_gpus + partition_info.gpus_per_node - 1) // partition_info.gpus_per_node
        if num_nodes > 1:
            score -= (num_nodes - 1) * 5

        # Match experiment requirements if provided
        if exp_config:
            # Prefer partition that matches current config
            if exp_config.get("partition") == partition_info.name:
                score += 15

            # Prefer GPU count that matches
            if exp_config.get("num_gpus") == num_gpus:
                score += 10

        return max(score, 0)

    def get_recommendations(
        self,
        exp_config: Optional[Dict] = None,
        target_global_batch: int = 192,  # 48 * 4 GPUs (B11 config)
        max_per_gpu_batch: int = 48
    ) -> List[ResourceRecommendation]:
        """Generate resource recommendations"""

        partition_stats = self.get_queue_status()
        recommendations = []

        # GPU configurations to consider
        gpu_configs = [
            (4, 1),  # 4 GPUs, 1 node (L40S standard)
            (2, 1),  # 2 GPUs, 1 node (H200 standard)
            (8, 2),  # 8 GPUs, 2 nodes
            (1, 1),  # 1 GPU (testing)
        ]

        for partition_name, partition_info in partition_stats.items():
            partition_config = self.PARTITIONS[partition_name]

            for num_gpus, num_nodes in gpu_configs:
                # Skip if doesn't fit partition
                if num_gpus > partition_info.gpus_per_node * num_nodes:
                    continue

                # Calculate batch requirements
                per_gpu_batch, grad_accum = self.calculate_global_batch_requirements(
                    target_global_batch,
                    num_gpus,
                    max_per_gpu_batch
                )

                # Build recommendation
                warnings = []

                # Check reproducibility concerns
                if exp_config:
                    orig_gpus = exp_config.get("num_gpus", 4)
                    orig_partition = exp_config.get("partition", "l40s_public")

                    if num_gpus != orig_gpus:
                        warnings.append(
                            f"GPU count changed from {orig_gpus} to {num_gpus}. "
                            "Verify global batch size matches for reproducibility."
                        )

                    if partition_name != orig_partition:
                        orig_gpu_type = self.PARTITIONS[orig_partition]["gpu_type"]
                        new_gpu_type = partition_config["gpu_type"]
                        warnings.append(
                            f"GPU type changed from {orig_gpu_type} to {new_gpu_type}. "
                            "May affect precision/performance. Consider locking precision mode."
                        )

                if grad_accum > 1:
                    warnings.append(
                        f"Gradient accumulation ({grad_accum}x) required to maintain global batch. "
                        "May affect training dynamics slightly."
                    )

                # Precision recommendation
                precision = "bf16"
                if partition_config["fp8_support"]:
                    precision = "bf16 or fp8 (H200 supports FP8, but changes training algorithm)"

                # Build reason
                reason_parts = []
                if partition_info.available_gpus >= num_gpus:
                    reason_parts.append(f"✓ {partition_info.available_gpus} GPUs available now")
                else:
                    reason_parts.append(f"⚠ Only {partition_info.available_gpus} GPUs available ({num_gpus} needed)")

                if partition_info.queue_length == 0:
                    reason_parts.append("empty queue")
                else:
                    reason_parts.append(f"{partition_info.queue_length} jobs queued")

                if num_gpus == partition_info.gpus_per_node:
                    reason_parts.append("full node (efficient)")

                reason = f"{partition_config['gpu_type']} ({', '.join(reason_parts)})"

                # Calculate score
                score = self.score_recommendation(partition_info, num_gpus, exp_config)

                # Estimate wait time
                estimated_wait = partition_info.avg_wait_time_mins or 0

                rec = ResourceRecommendation(
                    partition=partition_name,
                    num_gpus=num_gpus,
                    num_nodes=num_nodes,
                    estimated_wait_mins=estimated_wait,
                    reason=reason,
                    warnings=warnings,
                    score=score,
                    global_batch_size=target_global_batch,
                    per_gpu_batch_size=per_gpu_batch,
                    gradient_accumulation=grad_accum,
                    precision_recommendation=precision
                )

                recommendations.append(rec)

        # Sort by score (descending)
        recommendations.sort(key=lambda r: r.score, reverse=True)

        return recommendations

    def get_gemini_suggestion(
        self,
        partition_stats: Dict[str, PartitionInfo],
        exp_config: Optional[Dict] = None
    ) -> Optional[str]:
        """Use Gemini API for intelligent resource suggestion"""

        if not self.gemini_api_key:
            return None

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Build context
            context = "Current HPC Resource Status:\n\n"
            for name, info in partition_stats.items():
                context += f"{name} ({info.gpu_type}):\n"
                context += f"  Available: {info.available_gpus}/{info.total_gpus} GPUs\n"
                context += f"  Queue: {info.queue_length} jobs\n"
                context += f"  Est. wait: {info.avg_wait_time_mins or 'N/A'} mins\n\n"

            if exp_config:
                context += f"\nExperiment Requirements:\n"
                context += f"  Current partition: {exp_config.get('partition', 'unknown')}\n"
                context += f"  Current GPUs: {exp_config.get('num_gpus', 'unknown')}\n"
                context += f"  Batch size: {exp_config.get('batch_size', 'unknown')}\n"
                context += f"  Training: {exp_config.get('description', 'N/A')}\n"

            prompt = f"""{context}

Based on the current HPC resource availability, suggest the best partition and GPU allocation.
Consider:
1. Current availability and wait times
2. Reproducibility if switching from current setup
3. Efficiency (full node utilization)
4. Global batch size maintenance

Provide a concise recommendation (2-3 sentences) with reasoning.
"""

            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"Gemini API error: {e}"

    def print_status(self):
        """Print current resource status"""

        partition_stats = self.get_queue_status()

        print("\n" + "="*70)
        print("HPC Resource Status - NYU Greene Cluster")
        print("="*70)
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for name, info in partition_stats.items():
            config = self.PARTITIONS[name]

            print(f"📊 {name.upper()}")
            print(f"   GPU Type: {info.gpu_type} ({config['arch']}, {config['memory_gb']}GB)")
            print(f"   Available: {info.available_gpus}/{info.total_gpus} GPUs")
            print(f"   Nodes: {info.available_nodes}/{info.total_nodes} available")
            print(f"   Queue: {info.queue_length} jobs")

            if info.avg_wait_time_mins is not None:
                if info.avg_wait_time_mins == 0:
                    print(f"   Wait Time: ✓ Ready now")
                else:
                    print(f"   Wait Time: ~{info.avg_wait_time_mins:.0f} minutes")

            # Status indicator
            if info.available_gpus >= 4:
                status = "🟢 Good availability"
            elif info.available_gpus > 0:
                status = "🟡 Limited availability"
            else:
                status = "🔴 No GPUs available"
            print(f"   Status: {status}\n")

        print("="*70 + "\n")

    def print_recommendations(
        self,
        recommendations: List[ResourceRecommendation],
        gemini_suggestion: Optional[str] = None,
        top_n: int = 5
    ):
        """Print resource recommendations"""

        print("\n" + "="*70)
        print("Resource Recommendations")
        print("="*70 + "\n")

        if gemini_suggestion:
            print("💡 AI Suggestion:")
            print(f"   {gemini_suggestion}\n")
            print("-"*70 + "\n")

        print(f"Top {top_n} Recommendations (by score):\n")

        for i, rec in enumerate(recommendations[:top_n], 1):
            print(f"{i}. {rec.partition.upper()} - {rec.num_gpus} GPUs ({rec.num_nodes} node{'s' if rec.num_nodes > 1 else ''})")
            print(f"   Score: {rec.score:.1f}/100")
            print(f"   Reason: {rec.reason}")
            print(f"   Est. Wait: {rec.estimated_wait_mins:.0f} minutes")
            print(f"   Batch Config: {rec.per_gpu_batch_size}/GPU × {rec.num_gpus} GPUs", end="")
            if rec.gradient_accumulation > 1:
                print(f" × {rec.gradient_accumulation} accum", end="")
            print(f" = {rec.global_batch_size} global")
            print(f"   Precision: {rec.precision_recommendation}")

            if rec.warnings:
                print(f"   ⚠ Warnings:")
                for warning in rec.warnings:
                    print(f"      - {warning}")

            print()

        print("="*70 + "\n")

        # Print reproducibility checklist
        print("Reproducibility Checklist when switching GPUs:")
        print("  ✓ Lock global batch size (not per-GPU)")
        print("  ✓ Schedule LR by steps (not epochs)")
        print("  ✓ Set precision explicitly (bf16/fp16/fp32)")
        print("  ✓ Set TF32 mode explicitly")
        print("  ✓ Use deterministic settings if critical")
        print("  ✓ Run ≥3 seeds for statistical confidence")
        print()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HPC Resource Advisor for NYU Greene",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current resource status")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze resources and show recommendations")
    analyze_parser.add_argument("--global-batch", type=int, default=192, help="Target global batch size")
    analyze_parser.add_argument("--use-gemini", action="store_true", help="Use Gemini API for suggestions")

    # Suggest command
    suggest_parser = subparsers.add_parser("suggest", help="Suggest resources for specific experiment")
    suggest_parser.add_argument("--exp-id", help="Experiment ID")
    suggest_parser.add_argument("--config", help="Path to experiment config YAML")
    suggest_parser.add_argument("--global-batch", type=int, default=192, help="Target global batch size")
    suggest_parser.add_argument("--use-gemini", action="store_true", help="Use Gemini API")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize advisor
    advisor = ResourceAdvisor()

    if args.command == "status":
        advisor.print_status()

    elif args.command == "analyze":
        advisor.print_status()

        recommendations = advisor.get_recommendations(
            target_global_batch=args.global_batch
        )

        gemini_suggestion = None
        if args.use_gemini:
            partition_stats = advisor.get_queue_status()
            gemini_suggestion = advisor.get_gemini_suggestion(partition_stats)

        advisor.print_recommendations(recommendations, gemini_suggestion)

    elif args.command == "suggest":
        # Load experiment config
        exp_config = None

        if args.exp_id:
            config_path = Path("experiment_configs") / f"{args.exp_id}.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    exp_config = yaml.safe_load(f)
        elif args.config:
            with open(args.config) as f:
                exp_config = yaml.safe_load(f)

        recommendations = advisor.get_recommendations(
            exp_config=exp_config,
            target_global_batch=args.global_batch
        )

        gemini_suggestion = None
        if args.use_gemini:
            partition_stats = advisor.get_queue_status()
            gemini_suggestion = advisor.get_gemini_suggestion(
                partition_stats,
                exp_config
            )

        advisor.print_recommendations(recommendations, gemini_suggestion)


if __name__ == "__main__":
    main()
