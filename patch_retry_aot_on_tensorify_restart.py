from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/backends/common.py"
)

BACKUP = TARGET.with_suffix(
    ".py.backup_tensorify_restart_retry"
)

OLD = """    try:
        # NB: NOT cloned!
        with enable_aot_logging(), patch_config:
            cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
            counters["aot_autograd"]["ok"] += 1
            return disable(cg, reason="do not trace AOT-compiled graph")
    except TensorifyScalarRestartAnalysis:
        raise
    except Exception:
        counters["aot_autograd"]["not_ok"] += 1
        raise
"""

NEW = """    try:
        # NB: NOT cloned!
        with enable_aot_logging(), patch_config:
            cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
            counters["aot_autograd"]["ok"] += 1
            return disable(cg, reason="do not trace AOT-compiled graph")
    except TensorifyScalarRestartAnalysis:
        with enable_aot_logging(), patch_config:
            cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
            counters["aot_autograd"]["ok"] += 1
            return disable(cg, reason="do not trace AOT-compiled graph")
    except Exception:
        counters["aot_autograd"]["not_ok"] += 1
        raise
"""

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

source = TARGET.read_text()

if NEW in source:
    print("[INFO] Patch already installed")
    raise SystemExit(0)

if OLD not in source:
    raise RuntimeError(
        "Expected original AOT backend block not found. "
        "Nothing was modified."
    )

if not BACKUP.exists():
    BACKUP.write_text(source)
    print("[OK] Backup created:", BACKUP)
else:
    print("[OK] Backup already exists:", BACKUP)

patched = source.replace(OLD, NEW, 1)

ast.parse(patched)

TARGET.write_text(patched)

print("[OK] TensorifyScalarRestartAnalysis retry installed")
print("[OK] Syntax check passed")
print("[OK] Patch written successfully")


# Important

# Le Patch 4 ne fait pas simplement ignorer TensorifyScalarRestartAnalysis.

# Il fait :

# premier passage
#      ↓
# TensorifyScalarRestartAnalysis
#      ↓
# TensorifyState contient les spécialisations
#      ↓
# aot_module_simplified() relancé immédiatement
#      ↓
# compilation
#      ↓
# AOTAutogradCache.save()

# C'est précisément ce qui a fait disparaître le [0/0_1] externe et permis de conserver le même key AOT [0/0].