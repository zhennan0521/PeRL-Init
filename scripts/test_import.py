#!/usr/bin/env python3
"""Test which AReaL is being imported and whether LoRA checkpoint code exists."""
import sys
import os

print(f"Python: {sys.executable}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
print(f"sys.path[:5]: {sys.path[:5]}")
print()

import areal
print(f"areal imported from: {areal.__file__}")

# Check if LoRA checkpoint module exists (from PR #1015)
try:
    from areal.experimental.engine.archon_lora_checkpoint import save_lora_adapter
    print("save_lora_adapter: FOUND")
except ImportError as e:
    print(f"save_lora_adapter: MISSING ({e})")

# Check if archon_engine has LoRA-aware save
try:
    import inspect
    from areal.experimental.engine.archon_engine import ArchonEngine
    src = inspect.getsource(ArchonEngine.save)
    has_lora = "lora_config" in src
    print(f"ArchonEngine.save has lora_config check: {has_lora}")
except Exception as e:
    print(f"ArchonEngine check failed: {e}")
