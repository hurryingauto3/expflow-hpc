#!/usr/bin/env python3
"""
Partition-Account Validator and Auto-Selector

Automatically detects which accounts can access which partitions,
validates partition-account combinations, and intelligently selects
the best account for a given partition.
"""

import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PartitionInfo:
    """Information about a partition and its requirements"""
    name: str
    requires_gpu: bool = False
    gpu_type: Optional[str] = None
    accessible_accounts: List[str] = None

    def __post_init__(self):
        if self.accessible_accounts is None:
            self.accessible_accounts = []


class PartitionValidator:
    """
    Validates partition-account combinations and auto-selects accounts

    Usage:
        validator = PartitionValidator()
        validator.detect_partition_access()

        # Get best account for partition
        account = validator.get_account_for_partition("h200_tandon")

        # Validate combination
        is_valid = validator.validate_partition_account("h200_public", "torch_pr_68_general")
    """

    def __init__(self, username: Optional[str] = None):
        """Initialize validator"""
        self.username = username
        self.partition_map: Dict[str, PartitionInfo] = {}
        self.account_partition_map: Dict[str, List[str]] = {}

        # NYU Greene partition rules (known requirements)
        self.known_gpu_only_partitions = {
            'h200_public', 'h200_tandon', 'h200_bpeher',
            'l40s_public', 'rtx8000'
        }

        self.gpu_type_map = {
            'h200_public': 'H200',
            'h200_tandon': 'H200',
            'h200_bpeher': 'H200',
            'l40s_public': 'L40s',
            'rtx8000': 'RTX8000',
        }

    def detect_partition_access(self, accounts: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Auto-detect which accounts can access which partitions

        Args:
            accounts: List of accounts to test. If None, auto-detects from sacctmgr

        Returns:
            Dict mapping partition -> list of accounts that can access it
        """
        if accounts is None:
            accounts = self._get_user_accounts()

        partitions = self._get_partitions()

        print(f"Testing partition access for {len(accounts)} accounts × {len(partitions)} partitions...")

        partition_access = {}

        for partition in partitions:
            accessible_by = []
            requires_gpu = partition in self.known_gpu_only_partitions

            for account in accounts:
                if self._test_partition_access(partition, account, requires_gpu):
                    accessible_by.append(account)

            if accessible_by:
                partition_access[partition] = accessible_by

                # Store partition info
                self.partition_map[partition] = PartitionInfo(
                    name=partition,
                    requires_gpu=requires_gpu,
                    gpu_type=self.gpu_type_map.get(partition),
                    accessible_accounts=accessible_by
                )

        # Build reverse mapping (account -> partitions)
        self.account_partition_map = {}
        for partition, accounts_list in partition_access.items():
            for account in accounts_list:
                if account not in self.account_partition_map:
                    self.account_partition_map[account] = []
                self.account_partition_map[account].append(partition)

        return partition_access

    def _get_user_accounts(self) -> List[str]:
        """Get user's SLURM accounts"""
        try:
            import os
            username = self.username or os.getenv('USER')

            result = subprocess.run(
                ["sacctmgr", "show", "associations",
                 f"user={username}",
                 "format=Account", "-n"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                accounts = []
                for line in result.stdout.split('\n'):
                    account = line.strip()
                    if account and account != 'users':
                        accounts.append(account)
                # Deduplicate
                return list(dict.fromkeys(accounts))
        except:
            pass

        return []

    def _get_partitions(self) -> List[str]:
        """Get available partitions"""
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
                # Deduplicate
                return list(dict.fromkeys(partitions))
        except:
            pass

        return []

    def _test_partition_access(self, partition: str, account: str, requires_gpu: bool = False) -> bool:
        """
        Test if an account can access a partition

        Uses sbatch --test-only to verify access without submitting
        """
        try:
            cmd = [
                "sbatch", "--test-only",
                "-p", partition,
                "-A", account,
                "-N1",
                "-t", "1:00:00"
            ]

            # Add GPU requirement if needed
            if requires_gpu:
                cmd.extend(["--gres=gpu:1"])

            cmd.extend(["--wrap=hostname"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            # Success if job would be scheduled
            if result.returncode == 0 and "to start at" in result.stdout:
                return True

            # Check for access denied errors
            if "is not valid for this job" in result.stderr:
                return False

        except:
            pass

        return False

    def get_account_for_partition(self, partition: str) -> Optional[str]:
        """
        Get the best account for accessing a partition

        Returns:
            Best account name, or None if partition not accessible
        """
        if partition not in self.partition_map:
            # Try to detect on-demand
            accounts = self._get_user_accounts()
            requires_gpu = partition in self.known_gpu_only_partitions

            for account in accounts:
                if self._test_partition_access(partition, account, requires_gpu):
                    return account

            return None

        accessible = self.partition_map[partition].accessible_accounts

        if not accessible:
            return None

        # Prefer specific accounts over general ones
        # E.g., prefer "torch_pr_68_tandon_advanced" for "h200_tandon"
        partition_lower = partition.lower()

        for account in accessible:
            if any(keyword in account.lower() for keyword in partition_lower.split('_')):
                return account

        # Otherwise return first accessible account
        return accessible[0]

    def validate_partition_account(self, partition: str, account: str) -> bool:
        """
        Validate if an account can access a partition

        Returns:
            True if valid, False otherwise
        """
        if partition not in self.partition_map:
            # Test on-demand
            requires_gpu = partition in self.known_gpu_only_partitions
            return self._test_partition_access(partition, account, requires_gpu)

        return account in self.partition_map[partition].accessible_accounts

    def get_accessible_partitions(self, account: str) -> List[str]:
        """
        Get all partitions accessible by an account

        Returns:
            List of partition names
        """
        return self.account_partition_map.get(account, [])

    def print_access_map(self):
        """Print a human-readable access map"""
        print("\n" + "="*70)
        print("Partition Access Map")
        print("="*70)

        for partition, info in sorted(self.partition_map.items()):
            gpu_str = f" (GPU: {info.gpu_type})" if info.gpu_type else ""
            required = " [GPU Required]" if info.requires_gpu else ""

            print(f"\n{partition}{gpu_str}{required}")
            if info.accessible_accounts:
                for account in info.accessible_accounts:
                    print(f"  ✓ {account}")
            else:
                print("  ✗ No accessible accounts")

        print("\n" + "="*70)
        print("Account Access Summary")
        print("="*70)

        for account, partitions in sorted(self.account_partition_map.items()):
            print(f"\n{account}")
            print(f"  Can access: {', '.join(partitions)}")

        print("\n" + "="*70)

    def auto_select_partition(self,
                            gpu_type: Optional[str] = None,
                            account: Optional[str] = None,
                            prefer_public: bool = True) -> Optional[Tuple[str, str]]:
        """
        Automatically select best partition-account pair

        Args:
            gpu_type: Desired GPU type (e.g., 'H200', 'L40s')
            account: Specific account to use (if provided)
            prefer_public: Prefer public partitions over restricted ones

        Returns:
            Tuple of (partition, account) or None if no match
        """
        candidates = []

        for partition, info in self.partition_map.items():
            # Filter by GPU type if specified
            if gpu_type and info.gpu_type != gpu_type:
                continue

            # Filter by account if specified
            if account and account not in info.accessible_accounts:
                continue

            # Get best account for this partition
            best_account = account or self.get_account_for_partition(partition)

            if best_account:
                # Score partitions (higher is better)
                score = 0
                if prefer_public and 'public' in partition.lower():
                    score += 10
                if not info.requires_gpu:  # Flexible partitions
                    score += 5

                candidates.append((partition, best_account, score))

        if not candidates:
            return None

        # Return highest scored option
        candidates.sort(key=lambda x: x[2], reverse=True)
        partition, account, _ = candidates[0]

        return (partition, account)


# Convenience function for quick validation
def validate_job_config(partition: str, account: str, username: Optional[str] = None) -> bool:
    """
    Quick validation of partition-account combination

    Args:
        partition: Partition name
        account: Account name
        username: Optional username (defaults to current user)

    Returns:
        True if valid combination, False otherwise
    """
    validator = PartitionValidator(username)
    return validator.validate_partition_account(partition, account)
