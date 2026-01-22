# Partition Validator - Quick Start

ExpFlow now **automatically detects and selects** the correct partition-account combinations!

## Quick Commands

```bash
# Show your partition access map
expflow partitions

# Output (example):
# ======================================================================
# Partition Access Map
# ======================================================================
#
# h200_public (GPU: H200) [GPU Required]
#   ✓ torch_pr_68_general
#
# h200_tandon (GPU: H200) [GPU Required]
#   ✓ torch_pr_68_tandon_advanced
#
# l40s_public (GPU: L40s) [GPU Required]
#   ✓ torch_pr_68_general
#   ✓ torch_pr_68_tandon_advanced
```

## Use in Your Code

### Option 1: Automatic Selection (Recommended)

```python
from expflow import load_project_config, PartitionValidator

# Initialize
config = load_project_config()
validator = PartitionValidator()
validator.detect_partition_access()

# Auto-select best partition for H200
partition, account = validator.auto_select_partition(gpu_type="H200")

print(f"Use partition={partition}, account={account}")
# Output: Use partition=h200_public, account=torch_pr_68_general
```

### Option 2: Validate Before Submitting

```python
# Check if combination is valid
is_valid = validator.validate_partition_account(
    partition="h200_tandon",
    account="torch_pr_68_tandon_advanced"
)

if not is_valid:
    print("ERROR: Cannot access h200_tandon with this account!")
```

### Option 3: Get Account for Specific Partition

```python
# Get best account for a partition
account = validator.get_account_for_partition("h200_tandon")
print(f"Use account: {account}")
# Output: Use account: torch_pr_68_tandon_advanced
```

### Option 4: List Partitions for Your Account

```python
# What can I access with torch_pr_68_general?
partitions = validator.get_accessible_partitions("torch_pr_68_general")
print(f"Accessible: {partitions}")
# Output: Accessible: ['h200_public', 'l40s_public', 'rtx8000']
```

## Use in Custom Managers

```python
from expflow import BaseExperimentManager, PartitionValidator

class MyManager(BaseExperimentManager):
    def __init__(self, hpc_config):
        super().__init__(hpc_config)

        # Set up validator
        self.validator = PartitionValidator()
        self.validator.detect_partition_access()

    def create_smart_experiment(self, exp_id, gpu_type="H200", **kwargs):
        """Create experiment with automatic partition selection"""

        # Auto-select partition and account
        partition, account = self.validator.auto_select_partition(
            gpu_type=gpu_type
        )

        # Create experiment
        return self.create_experiment(
            exp_id=exp_id,
            partition=partition,
            account=account,
            **kwargs
        )
```

## Example: Partition-Aware Manager

See `examples/partition_aware_manager.py` for a complete example:

```bash
# Show your partition access
python examples/partition_aware_manager.py show-access

# Create experiment with auto-selection
python examples/partition_aware_manager.py new \
    --exp-id test001 \
    --gpu-type H200 \
    --num-gpus 4

# Output:
# Auto-selected:
#   Partition: h200_public
#   Account: torch_pr_68_general
# Experiment test001 created successfully!
```

## How It Works

1. **Detection:** Tests each partition-account combination using `sbatch --test-only`
2. **Validation:** Checks for GPU requirements (H200 partitions need `--gres=gpu:N`)
3. **Selection:** Chooses best account based on partition name matching
4. **Caching:** Stores results to avoid repeated tests

## Performance

- Initial detection: ~10-30 seconds (tests all combinations)
- Validation: <1 second (uses cached results)
- Auto-selection: <1 second (uses cached results)

## Tips

1. **Run once per session:** Detection results are cached in memory
2. **Use auto-select for new experiments:** Eliminates guesswork
3. **Validate before batch submissions:** Catch errors early
4. **Check access map periodically:** Your access may change

## Comparison

### Before (Manual)
```python
# User has to know which account works
manager.create_experiment(
    exp_id="exp001",
    partition="h200_tandon",  # Will this work?
    account="torch_pr_68_general",  # Is this right?
    num_gpus=4
)
# ❌ ERROR: partition 'h200_tandon' is not valid for this job
```

### After (Automatic)
```python
# ExpFlow figures it out
validator = PartitionValidator()
validator.detect_partition_access()

partition, account = validator.auto_select_partition(gpu_type="H200")

manager.create_experiment(
    exp_id="exp001",
    partition=partition,  # h200_public (auto-selected)
    account=account,  # torch_pr_68_general (auto-selected)
    num_gpus=4
)
# ✅ Works perfectly!
```

## Next Steps

1. Install latest version:
   ```bash
   pip install --upgrade --no-cache-dir git+https://github.com/hurryingauto3/expflow-hpc.git
   ```

2. Check your access:
   ```bash
   expflow partitions
   ```

3. Use in your manager:
   ```python
   from expflow import PartitionValidator
   validator = PartitionValidator()
   validator.detect_partition_access()
   ```

4. Enjoy automatic partition selection! 🎉
