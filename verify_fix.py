#!/usr/bin/env python3
"""
Verify that the import fix is correct.
This checks the source code directly without installing.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Checking import statements in source files...")
print("=" * 60)

# Check hpcexp_core.py
core_file = src_path / "expflow" / "hpcexp_core.py"
with open(core_file) as f:
    content = f.read()

if "from .hpc_config import" in content:
    print("[OK] hpcexp_core.py uses relative import: 'from .hpc_config import'")
elif "from hpc_config import" in content:
    print("[FAIL] hpcexp_core.py still uses absolute import: 'from hpc_config import'")
    print("      This will cause ModuleNotFoundError when installed")
    sys.exit(1)
else:
    print("[WARN] Could not find hpc_config import in hpcexp_core.py")

# Check __init__.py
init_file = src_path / "expflow" / "__init__.py"
with open(init_file) as f:
    content = f.read()

# Verify all imports use relative imports
if "from ." in content:
    print("[OK] __init__.py uses relative imports")
else:
    print("[WARN] __init__.py may not be using relative imports")

# Check other files
resource_file = src_path / "expflow" / "resource_advisor.py"
if resource_file.exists():
    with open(resource_file) as f:
        content = f.read()

    if "from hpc_config import" in content or "from hpcexp_core import" in content:
        print("[FAIL] resource_advisor.py uses absolute imports")
        sys.exit(1)
    else:
        print("[OK] resource_advisor.py looks good")

print("=" * 60)
print("\nAll import checks passed!")
print("\nOn HPC, run:")
print("  pip uninstall expflow -y")
print("  pip install --no-cache-dir git+https://github.com/hurryingauto3/expflow-hpc.git")
