# Testing ExpFlow Locally (Without HPC)

This guide shows how to test ExpFlow on your local machine without access to NYU Greene HPC.

## Quick Fix: Force Reinstall on HPC

If you're on HPC and the package didn't update:

```bash
pip uninstall expflow -y
pip install --no-cache-dir git+https://github.com/hurryingauto3/expflow-hpc.git
```

Or if you have the repo cloned:

```bash
cd ~/expflow-hpc
git pull
pip install --upgrade --force-reinstall --no-deps .
pip install pyyaml pandas  # Reinstall dependencies
```

## Local Testing Setup

### 1. Install for Local Development

```bash
# Clone the repo
git clone https://github.com/hurryingauto3/expflow-hpc.git
cd expflow-hpc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### 2. What Works Locally vs HPC

**Works Locally:**
- `expflow init <project>` - Creates project structure (uses local paths)
- `expflow template <name>` - Creates YAML templates
- `expflow info` - Shows environment (detects local instead of HPC)
- Custom managers: Creating experiments, generating scripts
- Metadata tracking, config management
- Result harvesting (if results exist)
- CSV export

**Requires HPC:**
- `expflow resources --status` - Queries SLURM queue (needs squeue, sinfo)
- Actual job submission - Submitting to SLURM (needs sbatch)
- SLURM account detection - Uses sacctmgr

### 3. Local Testing Commands

```bash
# Test environment detection (works locally)
expflow info

# Output will show:
# Cluster: unknown (not an HPC cluster)
# Username: your-local-username
# Scratch: /Users/yourname or /home/yourname
# (Won't detect SLURM accounts)

# Test project initialization (works locally)
expflow init test-project
# Creates: /Users/yourname/test-project/ (or Linux equivalent)

# Test template creation (works locally)
cd test-project
expflow template baseline

# Create experiment config (works locally)
python -m expflow.examples.simple new \
    --exp-id test001 \
    --template baseline \
    --description "Local test"

# Generate SLURM scripts (works locally, creates scripts)
python -m expflow.examples.simple submit test001 --dry-run
# This will show you the generated script without submitting
```

### 4. Mock SLURM for Testing

If you want to fully test locally, you can mock SLURM commands:

```bash
# Create mock SLURM commands in ~/bin/
mkdir -p ~/bin

# Mock squeue
cat > ~/bin/squeue << 'EOF'
#!/bin/bash
echo "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)"
echo "12345 l40s_public test user1 R 1:23:45 1 node001"
EOF

# Mock sinfo
cat > ~/bin/sinfo << 'EOF'
#!/bin/bash
echo "l40s_public"
echo "h200_tandon"
EOF

# Mock sacctmgr
cat > ~/bin/sacctmgr << 'EOF'
#!/bin/bash
if [[ "$*" == *"show user"* ]]; then
    echo "Account|"
    echo "test_account|"
fi
EOF

# Mock sbatch
cat > ~/bin/sbatch << 'EOF'
#!/bin/bash
echo "Submitted batch job 12346"
EOF

# Make executable
chmod +x ~/bin/{squeue,sinfo,sacctmgr,sbatch}

# Add to PATH
export PATH="$HOME/bin:$PATH"

# Now test
expflow resources --status  # Should work with mocked data
```

### 5. Unit Testing

Create a test file to verify imports work:

```python
# test_imports.py
try:
    from expflow import (
        BaseExperimentManager,
        BaseExperimentConfig,
        HPCConfig,
        load_project_config,
        initialize_project,
        ResourceAdvisor
    )
    print("All imports successful!")

    # Test environment detection
    from expflow.hpc_config import HPCEnvironment
    username = HPCEnvironment.get_username()
    scratch = HPCEnvironment.get_scratch_dir()
    cluster = HPCEnvironment.detect_cluster()

    print(f"Username: {username}")
    print(f"Scratch: {scratch}")
    print(f"Cluster: {cluster}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

Run with:
```bash
python test_imports.py
```

### 6. Testing Custom Managers Locally

You can develop and test custom managers completely locally:

```python
# my_local_test.py
from expflow import BaseExperimentManager, load_project_config, initialize_project
from pathlib import Path

# Initialize a test project
config = initialize_project("local-test")

class TestManager(BaseExperimentManager):
    def _generate_train_script(self, config):
        return f'''#!/bin/bash
#SBATCH --partition={config['partition']}
#SBATCH --gres=gpu:{config['num_gpus']}
echo "Training locally"
python train.py
'''

    def _generate_eval_script(self, config):
        return "#!/bin/bash\necho 'Eval script'"

    def harvest_results(self, exp_id):
        return {"test_metric": 0.95}

# Test it
manager = TestManager(config)

# Create experiment (works locally)
manager.create_experiment(
    exp_id="test001",
    description="Local test",
    partition="l40s_public",
    num_gpus=1
)

# List experiments (works locally)
exps = manager.list_experiments()
print(f"Created {len(exps)} experiments")

# Generate scripts (works locally - just creates files)
print("Generated scripts in:", config.project_root / "generated_scripts")
```

### 7. Debugging Import Issues

If you get `ModuleNotFoundError`:

```bash
# Check installation
pip show expflow

# Check installed location
python -c "import expflow; print(expflow.__file__)"

# Force clean reinstall
pip uninstall expflow -y
rm -rf build/ dist/ *.egg-info
pip install -e .

# Verify
python -c "from expflow import BaseExperimentManager; print('OK')"
```

### 8. Development Workflow

```bash
# 1. Make changes to code in src/expflow/
vim src/expflow/hpc_config.py

# 2. No need to reinstall (using -e mode)

# 3. Test immediately
python test_imports.py

# 4. Commit when working
git add -A
git commit -m "Fix: description"
git push

# 5. Users on HPC update with:
pip install --upgrade --force-reinstall git+https://github.com/hurryingauto3/expflow-hpc.git
```

## Common Local Testing Scenarios

### Test 1: Verify Package Structure
```bash
python -c "import expflow; print(dir(expflow))"
# Should show: BaseExperimentManager, HPCConfig, etc.
```

### Test 2: Create and Track Experiments
```bash
expflow init my-test
cd my-test
python -m expflow.examples.simple new --exp-id exp001
ls experiments/  # Should see exp001/
```

### Test 3: Generate Scripts Without Submitting
```bash
python -m expflow.examples.simple submit exp001 --dry-run
cat generated_scripts/exp001_train.slurm
```

### Test 4: Metadata Database
```bash
python -c "import json; print(json.dumps(json.load(open('metadata.json')), indent=2))"
```

## Limitations of Local Testing

1. **No SLURM**: Can't actually submit jobs or check queue
2. **No HPC paths**: Uses local paths instead of /scratch/
3. **No GPU detection**: Can't detect actual GPU availability
4. **No SLURM accounts**: Won't detect your actual HPC accounts

But you CAN:
- Test all Python code
- Verify imports work correctly
- Generate SLURM scripts
- Test experiment tracking
- Develop custom managers
- Test configuration system

## Tips

1. **Use `--dry-run`** for submit commands when testing locally
2. **Mock SLURM commands** if you need to test resource advisor
3. **Create test data** in local directories to test harvesting
4. **Use debugger** freely (doesn't work well on HPC)
5. **Iterate faster** - no need to SSH to HPC for every change

## When to Test on HPC

Test on HPC when you need to verify:
- SLURM job submission actually works
- Resource detection is accurate
- GPU allocation works correctly
- Queue analysis is correct
- Actual training runs complete

But develop and debug locally first!
