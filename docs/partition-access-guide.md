# NYU Greene Partition Access Guide

## Understanding Partition Access

On NYU Greene HPC, **partition access is controlled by your SLURM account**. Not all accounts can access all partitions.

## Finding Your Partitions

### Check Your Accounts
```bash
sacctmgr show associations user=$USER format=Account%40 -n
```

### Test Partition Access
```bash
# Test if you can access a partition (replace values)
sbatch --test-only -p PARTITION -A ACCOUNT -N1 -t 1:00:00 --gres=gpu:1 --wrap="hostname"
```

**Success:**
```
sbatch: Job 12345 to start at 2026-01-22T12:01:02 using 1 processors...
```

**Access Denied:**
```
sbatch: error: *** Error partition 'PARTITION' is not valid for this job
```

## Common NYU Greene Configurations

### H200 GPU Partitions

**Partition: `h200_public`**
- Account: `torch_pr_68_general` (or other general accounts)
- Access: Broad access, public queue
- GPU: H200 (141GB)
- Notes: Shared queue, may have wait times

**Partition: `h200_tandon`**
- Account: `torch_pr_68_tandon_advanced` (Tandon-specific)
- Access: Restricted to Tandon accounts
- GPU: H200 (141GB)
- Notes: Tandon researchers only

**Partition: `h200_bpeher`**
- Account: Restricted (specific research groups)
- Access: Very limited
- GPU: H200 (141GB)
- Notes: Not accessible to most users

### L40s GPU Partitions

**Partition: `l40s_public`**
- Account: Most general accounts
- Access: Broad access
- GPU: L40s (48GB)
- Notes: Often good availability

## Important Rules

### 1. H200 Partitions Are GPU-Only
```bash
# This FAILS (no GPU requested)
srun -p h200_public -A account -N1 -t 1:00:00 /bin/bash

# This WORKS (GPU requested)
srun -p h200_public -A account -N1 -t 1:00:00 --gres=gpu:1 /bin/bash
```

### 2. Account-Partition Matching

Always use the correct account for each partition:

| Your Account | Can Access | Cannot Access |
|--------------|------------|---------------|
| `torch_pr_68_general` | `h200_public`, `l40s_public` | `h200_tandon`, `h200_bpeher` |
| `torch_pr_68_tandon_advanced` | `h200_tandon`, `l40s_public` | `h200_bpeher` |

### 3. Check Before Submitting

Always test with `--test-only` before submitting real jobs:

```bash
sbatch --test-only \
    -p h200_tandon \
    -A torch_pr_68_tandon_advanced \
    -N1 \
    -t 48:00:00 \
    --gres=gpu:4 \
    --wrap="python train.py"
```

## Using With ExpFlow

### Method 1: Specify in Template

```yaml
# experiment_templates/h200_baseline.yaml
partition: h200_tandon
account: torch_pr_68_tandon_advanced  # Must match partition access
num_gpus: 4
time_limit: "48:00:00"
```

### Method 2: Specify When Creating Experiment

```python
manager.create_experiment(
    exp_id="exp001",
    partition="h200_tandon",
    account="torch_pr_68_tandon_advanced",  # Override default
    num_gpus=4
)
```

### Method 3: Check Your Config

```bash
# Check what account is set as default
cat .hpc_config.yaml | grep default_account

# Update if needed
vim .hpc_config.yaml
```

## Troubleshooting

### Error: "partition is not valid for this job"

**Cause:** Account doesn't have access to partition

**Solution:**
1. Check which accounts you have: `sacctmgr show associations user=$USER format=Account -n`
2. Test partition access with different accounts
3. Use matching account-partition pairs

### Error: "CPU job setup is not valid"

**Cause:** H200 partitions require GPU allocation

**Solution:** Always include `--gres=gpu:N` for H200 partitions

```bash
# Wrong
#SBATCH --partition=h200_public

# Correct
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:1
```

### Jobs Stay Pending Forever

**Cause:** Wrong account or no partition access

**Solution:**
1. Check job reason: `squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"`
2. Look for "AssocMaxJobsLimit", "QOSMaxJobsPerUserLimit", or "InvalidAccount"
3. Cancel job: `scancel JOBID`
4. Fix account/partition and resubmit

## Best Practices

### 1. Create Account-Specific Templates

```bash
# For general account
expflow template h200_public_baseline
# Edit: set account=torch_pr_68_general, partition=h200_public

# For Tandon account
expflow template h200_tandon_baseline
# Edit: set account=torch_pr_68_tandon_advanced, partition=h200_tandon
```

### 2. Document Your Access

Keep track of which partitions you can use:

```bash
# Create a quick reference
cat > ~/my_partitions.txt << EOF
My SLURM Accounts and Partition Access:

torch_pr_68_general:
  - h200_public (H200, public queue)
  - l40s_public (L40s, public queue)

torch_pr_68_tandon_advanced:
  - h200_tandon (H200, Tandon only)
  - l40s_public (L40s, public queue)
EOF
```

### 3. Always Test First

Before running expensive experiments:

```bash
# Test partition access
sbatch --test-only -p PARTITION -A ACCOUNT -N1 --gres=gpu:1 -t 1:00:00 --wrap="hostname"

# If successful, submit real job
sbatch your_script.slurm
```

## Getting More Access

To request access to restricted partitions:

1. **Email NYU HPC team:** hpc@nyu.edu
2. **Include:**
   - Your NetID
   - Current SLURM accounts
   - Partition you need access to
   - Research justification
   - PI/advisor approval

Example:
```
Subject: Request H200 Tandon Partition Access

Hello,

I'm requesting access to the h200_tandon partition for my deep learning research.

NetID: ah7072
Current Account: torch_pr_68_general
Requested Partition: h200_tandon
PI: Professor Name

We need H200 GPUs for large language model training that requires 141GB GPU memory.

Thank you,
Your Name
```

## Reference Commands

```bash
# List all partitions
sinfo -o "%R %a"

# Show partition details
sinfo -p h200_public,h200_tandon,l40s_public --long

# Check your account associations
sacctmgr show associations user=$USER format=Account,Partition,QOS

# Test partition access
sbatch --test-only -p PARTITION -A ACCOUNT -N1 --gres=gpu:1 -t 1:00:00 --wrap="hostname"

# Check queue for specific partition
squeue -p h200_tandon -o "%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R"
```
