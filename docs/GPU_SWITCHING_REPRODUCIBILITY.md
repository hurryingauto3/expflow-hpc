# GPU Switching and Reproducibility Guide

## Overview

When switching between GPU configurations (e.g., L40S×4 → H200×2), results can change due to various factors. This guide explains when results should match, when they diverge, and how to maintain reproducibility.

## TL;DR: When Results Should Match

Results should be **nearly identical** (within seed-to-seed variance) if ALL these are constant:

✅ **Same global batch size** (not per-GPU batch!)
✅ **Same LR schedule in steps** (not epochs)
✅ **Same precision mode** (BF16/FP16/FP32) + TF32 setting
✅ **Same data order** (DistributedSampler, num_workers)
✅ **Same code** (PyTorch/CUDA/cuDNN/NCCL versions)
✅ **Deterministic settings** enabled (if critical)

You'll still get slight differences due to floating-point reduction order, but metrics should be comparable.

## Common Causes of Divergence

### 1. Global Batch Size Changes ⚠️ MOST COMMON

**Problem:**
```python
# L40S×4 with batch_size=48/GPU
global_batch = 48 * 4 = 192

# H200×2 with batch_size=48/GPU (WRONG!)
global_batch = 48 * 2 = 96  # Different optimization dynamics!
```

**Fix:**
```python
# L40S×4
per_gpu_batch = 48
num_gpus = 4
global_batch = 192

# H200×2 (correct)
per_gpu_batch = 96  # Doubled to maintain global batch
num_gpus = 2
global_batch = 192  # Same!

# Or use gradient accumulation
per_gpu_batch = 48
grad_accum = 2
num_gpus = 2
global_batch = 48 * 2 * 2 = 192  # Same!
```

**Why it matters:**
- Global batch size affects gradient noise scale
- Larger batches → more stable gradients, may need higher LR
- Smaller batches → noisier gradients, better generalization sometimes
- Changing batch size often requires LR retuning

### 2. LR Schedule Tied to Epochs (Not Steps)

**Problem:**
```python
# L40S×4: 103,000 samples / (4*48) = 537 steps/epoch
# H200×2: 103,000 samples / (2*48) = 1,073 steps/epoch

# If warmup = 5 epochs:
# L40S×4: warmup = 5 * 537 = 2,685 steps
# H200×2: warmup = 5 * 1,073 = 5,365 steps  # DIFFERENT!
```

**Fix:**
```python
# Define schedule in steps, not epochs
warmup_steps = 2685  # Fixed
max_steps = 16110  # 30 epochs * 537 steps (from baseline)

# Calculate epochs based on world size
steps_per_epoch = len(dataloader)  # Varies with world_size
max_epochs = max_steps // steps_per_epoch
```

### 3. Precision / Math Kernel Changes

**Problem:**
```python
# L40S (Ada Lovelace)
# - TF32 default: ON
# - FP8 support: NO
# - Typical: BF16/FP16

# H200 (Hopper)
# - TF32 default: ON
# - FP8 support: YES
# - Typical: BF16/FP8

# If you enable FP8 on H200 but not L40S → different training!
```

**Fix:**
```python
# Explicitly set precision mode
torch.set_float32_matmul_precision('high')  # or 'highest', 'medium'

# Explicitly control TF32
torch.backends.cuda.matmul.allow_tf32 = True  # or False
torch.backends.cudnn.allow_tf32 = True  # or False

# Use same precision for both
trainer = Trainer(precision="bf16-mixed")  # or "16-mixed", "32"
```

### 4. DataLoader Nondeterminism

**Problem:**
```python
# Different num_workers → different augmentation RNG consumption
# Different world_size → different sample sharding

# L40S×4: 4 dataloaders, each with different RNG state
# H200×2: 2 dataloaders, RNG consumption pattern changes
```

**Fix:**
```python
# Use same num_workers
num_workers = 12  # Fixed across setups

# Set worker init function
def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

train_loader = DataLoader(
    dataset,
    batch_size=per_gpu_batch,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn,
    generator=torch.Generator().manual_seed(seed)
)

# Use DistributedSampler with same seed
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    seed=seed,
    shuffle=True
)
```

### 5. DDP Gradient Reduction Order

**Problem:**
```python
# Different world_size → different reduction tree
# L40S×4: 4-way reduction tree
# H200×2: 2-way reduction tree

# Floating-point sum is order-dependent:
# (a + b) + (c + d) ≠ (a + c) + (b + d) (in floating point)
```

**Fix:**
```
# Can't fully fix, but can minimize:
1. Use higher precision (BF16 > FP16)
2. Enable deterministic algorithms
3. Run multiple seeds to quantify variance
```

## Reproducibility Checklist

### Level 1: Comparable Results (Recommended)
Use this for most experiments. Results will be close enough to compare:

```python
# 1. Lock global batch size
global_batch = 192
per_gpu_batch = global_batch // world_size  # Or use grad_accum

# 2. Schedule by steps, not epochs
warmup_steps = 2685
total_steps = 16110

# 3. Set precision explicitly
torch.set_float32_matmul_precision('high')
trainer = Trainer(precision="bf16-mixed")

# 4. Use same num_workers
num_workers = 12
```

Expected variance: **0.1-0.5% in metrics** (e.g., PDM 0.8206 → 0.8198-0.8214)

### Level 2: High Reproducibility (For critical comparisons)
Add these on top of Level 1:

```python
# 5. Fix all seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# 6. Deterministic algorithms
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 7. Set CUBLAS workspace
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 8. Control TF32 explicitly
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

Expected variance: **0.01-0.1% in metrics**

**Warning:** Level 2 can slow training by 10-30%!

### Level 3: Bitwise Reproducibility (Rarely needed)
For exact numerical reproduction:

```python
# All of Level 2, plus:

# 9. Single-threaded ops
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 10. Disable async CUDA operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

Expected variance: **Bitwise identical** (but 50%+ slower!)

**Warning:** Usually not worth it. Use Level 2 instead.

## NYU Greene Specific: L40S vs H200

### Hardware Differences

| Feature | L40S (Ada Lovelace) | H200 (Hopper) |
|---------|-------------------|---------------|
| GPUs/node | 4 | 2 |
| Memory | 48GB | 141GB |
| FP8 support | No | Yes |
| Typical precision | BF16/FP16 | BF16/FP8 |
| TF32 default | ON | ON |

### Recommended Settings for Fair Comparison

```python
# config.yaml
experiment:
  # Lock these
  global_batch_size: 192
  warmup_steps: 2685  # Not epochs!
  total_steps: 16110
  precision: "bf16-mixed"  # Same for both
  tf32_enabled: true  # Same for both
  num_dataloader_workers: 12

  # Partition-specific (auto-calculated)
  l40s:
    partition: "l40s_public"
    num_gpus: 4
    per_gpu_batch: 48  # 192 / 4
    grad_accum: 1

  h200:
    partition: "h200_tandon"
    num_gpus: 2
    per_gpu_batch: 96  # 192 / 2
    grad_accum: 1
    # Or: per_gpu_batch=48, grad_accum=2
```

### Using the Resource Advisor

The experiment manager's resource advisor automatically handles this:

```bash
# Check current resources
python experiment_manager.py resources --exp-id b15

# Output shows:
# 1. H200 - 2 GPUs (1 node)
#    Batch Config: 96/GPU × 2 GPUs = 192 global  ✓ Matches original
#    ⚠ GPU type changed from L40S to H200.
#       Consider locking precision mode.
```

## Practical Workflow for GPU Switching

### Step 1: Baseline Run (L40S×4)
```bash
python experiment_manager.py new \
    --exp-id b15_baseline \
    --template ijepa_mlp \
    --description "Baseline on L40S" \
    --partition l40s_public \
    --num-gpus 4 \
    --batch-size 48

python experiment_manager.py submit b15_baseline
```

### Step 2: Equivalent Run (H200×2)
```bash
python experiment_manager.py new \
    --exp-id b15_h200 \
    --template ijepa_mlp \
    --description "Same config on H200" \
    --partition h200_tandon \
    --num-gpus 2 \
    --batch-size 96  # Maintain global batch = 192

python experiment_manager.py submit b15_h200
```

### Step 3: Verify Reproducibility
```python
# After both complete
python experiment_manager.py harvest b15_baseline
python experiment_manager.py harvest b15_h200

python experiment_manager.py export comparison.csv

# Check results
import pandas as pd
df = pd.read_csv("comparison.csv")
print(df[["exp_id", "pdm_score", "partition", "num_gpus", "batch_size"]])
```

Expected output:
```
exp_id         pdm_score  partition      num_gpus  batch_size
b15_baseline   0.8206     l40s_public    4         48
b15_h200       0.8203     h200_tandon    2         96
```

If scores differ by >0.5%, check:
1. Global batch size actually matches
2. LR schedule in steps, not epochs
3. Same precision mode
4. Same PyTorch/CUDA versions

### Step 4: Multi-Seed Validation (If Critical)
```bash
# Run 3 seeds for each setup
for seed in 42 123 456; do
    python experiment_manager.py new \
        --exp-id b15_l40s_seed${seed} \
        --template ijepa_mlp \
        --partition l40s_public \
        --seed ${seed}

    python experiment_manager.py new \
        --exp-id b15_h200_seed${seed} \
        --template ijepa_mlp \
        --partition h200_tandon \
        --num-gpus 2 \
        --batch-size 96 \
        --seed ${seed}
done

# Compare distributions
python analyze_variance.py --exp-prefix b15_l40s
python analyze_variance.py --exp-prefix b15_h200
```

## Summary

### When to Worry About GPU Switching

❌ **High Risk** - Results likely to change:
- Changing global batch size
- LR schedule in epochs (not steps)
- Switching precision (BF16→FP8)
- Different PyTorch versions

✅ **Low Risk** - Results should be close:
- Same global batch, precision, schedule
- Just different GPU count
- Same code versions

### Quick Decision Tree

```
Are you comparing two experiments?
├─ Yes → Use same GPU count if possible
│   └─ Not possible?
│       └─ Follow Level 2 checklist
└─ No → Use resource advisor to pick fastest available
```

### Best Practice

For **NavSim experiments**:
1. Pick one GPU config as "canonical" (e.g., L40S×4)
2. Run all comparisons on same config
3. Only switch GPUs for:
   - Throughput optimization
   - Memory constraints
   - Availability reasons
4. When switching, verify with 3 seeds
5. Document in experiment notes

The experiment manager tracks all this automatically in the metadata!
