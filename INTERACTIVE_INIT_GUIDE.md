# Interactive Initialization Guide

ExpFlow now features a professional interactive setup experience!

## Quick Start

### Interactive Mode (Recommended)

```bash
expflow init -i my-research
```

**What you'll see:**

```
======================================================================
ExpFlow Interactive Setup
======================================================================

[1/4] Detecting HPC environment...
  Cluster: greene
  Username: ah7072
  Scratch: /scratch/ah7072

[2/4] Selecting SLURM account...

  Available accounts (2):
    1. torch_pr_68_general [RECOMMENDED: Broadest access]
    2. torch_pr_68_tandon_advanced

  Select account [1-2] (default: 1):
```

**Features:**
- ✅ Guided step-by-step setup
- ✅ Intelligent recommendations
- ✅ Real-time partition validation
- ✅ GPU categorization (H200, L40s, A100)
- ✅ Configuration preview before saving

### Quick Mode (No Prompts)

```bash
expflow init -q my-research
```

**Output:**
```
Quick setup for: my-research
  Account: torch_pr_68_general
  Partition: l40s_public

======================================================================
 Project ready!
======================================================================
```

**Features:**
- ⚡ Instant setup
- 🎯 Smart defaults
- 🔧 Edit `.hpc_config.yaml` later if needed

### Legacy Mode (Auto-detect Only)

```bash
expflow init my-research
```

Still works, but less intelligent than interactive/quick modes.

## Interactive Setup Flow

### Step 1: Environment Detection
Auto-detects:
- Cluster name (greene, etc.)
- Username
- Scratch directory

### Step 2: Account Selection

```
[2/4] Selecting SLURM account...

  Available accounts (2):
    1. torch_pr_68_general [RECOMMENDED: Broadest access]
    2. torch_pr_68_tandon_advanced

  Select account [1-2] (default: 1):
```

**Recommendations:**
- **`general` accounts** → Marked as RECOMMENDED (broadest access)
- **`public` accounts** → Noted as public access
- **Specific accounts** → No marker

**What to choose:**
- Default (press Enter) → Uses recommended account
- Or type number to select specific account

### Step 3: GPU/Partition Selection

```
[3/4] Selecting default GPU partition...
  Analyzing partition access (10-30 seconds)...

  Accessible partitions with account 'torch_pr_68_general':

  L40s GPUs:
    1. l40s_public (L40s) [RECOMMENDED: Best availability]

  H200 GPUs:
    2. h200_public (H200) [RECOMMENDED: Powerful & available]

  RTX8000 GPUs:
    3. rtx8000 (RTX8000) [Public access]

  Select partition [1-3] (default: 1):
```

**Categorized by GPU type:**
- **H200**: Most powerful, 141GB memory
- **L40s**: Great balance, 48GB memory, best availability
- **A100**: 40GB/80GB memory
- **RTX8000**: Older, 48GB memory

**Recommendations:**
- **`l40s_public`** → Usually best availability
- **`h200_public`** → Powerful but may have queue
- **`public` partitions** → Marked for broad access

**What to choose:**
- Default (press Enter) → Uses recommended partition
- Or type number for specific GPU type

### Step 4: Additional Preferences

```
[4/4] Additional settings...

  Default time limit for jobs:
    1. 6 hours
    2. 12 hours
    3. 24 hours
    4. 48 hours [RECOMMENDED]
    5. 72 hours
    6. Custom

  Select time limit [1-6] (default: 4):
```

**Time limit options:**
- **6h**: Quick experiments
- **12h**: Medium jobs
- **24h**: Day-long training
- **48h**: Recommended (2 days)
- **72h**: Long runs
- **Custom**: Enter your own (HH:MM:SS)

### Step 5: Confirmation

```
======================================================================
Configuration Summary
======================================================================
  Project: my-research
  Location: /scratch/ah7072/my-research
  Account: torch_pr_68_general
  Default GPU: l40s_public
  Time Limit: 48:00:00
======================================================================

Proceed with this configuration? [Y/n]:
```

**Options:**
- **Y / Enter** → Save and create project
- **n** → Cancel setup

## Comparison

| Mode | Speed | Customization | When to Use |
|------|-------|---------------|-------------|
| **Interactive** (`-i`) | ~30s | Full control | First time, want to choose |
| **Quick** (`-q`) | Instant | Smart defaults | Quick setup, trust defaults |
| **Legacy** | Instant | Auto-only | Backwards compatibility |

## Examples

### Example 1: First-Time User (Interactive)

```bash
$ expflow init -i my-project

[Goes through all steps, selects preferences]

✓ Project created with your preferences
```

### Example 2: Experienced User (Quick)

```bash
$ expflow init -q test-project

Quick setup for: test-project
  Account: torch_pr_68_general
  Partition: l40s_public

✓ Ready in <1 second
```

### Example 3: Custom Time Limit

```bash
$ expflow init -i long-training

[... select account and partition ...]

[4/4] Additional settings...
  Select time limit [1-6] (default: 4): 6

  Enter time limit (HH:MM:SS): 120:00:00

✓ Set to 5 days (120 hours)
```

## Tips

1. **Use interactive mode first time** - See all options
2. **Use quick mode for subsequent projects** - Fast with good defaults
3. **Partition access is validated** - Only shows what you can actually use
4. **Preferences saved in `.hpc_config.yaml`** - Edit anytime
5. **Run `expflow partitions`** - See full access map anytime

## What Gets Configured

The interactive init creates:

```yaml
# .hpc_config.yaml
username: ah7072
scratch_dir: /scratch/ah7072
project_name: my-research
project_root: /scratch/ah7072/my-research
default_account: torch_pr_68_general      # Your choice
default_partition: l40s_public             # Your choice
default_time_limit: "48:00:00"             # Your choice
cluster_name: greene
available_partitions:
  - l40s_public
  - h200_public
  - rtx8000
  # ... (only accessible ones)
```

## Modifying Later

Edit `.hpc_config.yaml` anytime:

```bash
cd /scratch/YOUR_ID/my-project
vim .hpc_config.yaml
```

Or reinitialize:

```bash
rm .hpc_config.yaml
expflow init -i my-project  # Start over
```

## Advanced: Programmatic Use

```python
from expflow import interactive_init, quick_init

# Interactive
config = interactive_init("my-project")

# Quick
config = quick_init("my-project")
```

## Troubleshooting

**"No partitions accessible"**
- Your account may lack permissions
- Contact HPC admin: hpc@nyu.edu

**"Setup takes too long"**
- Partition validation can take 10-30s
- Use quick mode (`-q`) for instant setup

**"Want different defaults"**
- Re-run with `-i` and select different options
- Or edit `.hpc_config.yaml` manually

## Next Steps After Init

```bash
# Go to project
cd /scratch/YOUR_ID/my-project

# Create experiment template
expflow template baseline

# Check resources
expflow resources --status

# View partition access
expflow partitions
```

---

**The interactive init makes ExpFlow feel like a professional tool, not a script!**
