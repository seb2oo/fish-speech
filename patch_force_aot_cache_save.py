from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py"
)

BACKUP = TARGET.with_suffix(".py.backup_force_aot_cache_save")

OLD = """    cache_info = aot_config.cache_info
    if cache_info is not None:
        if hasattr(compiled_fw, "_fx_graph_cache_key"):
            time_taken_ns = time.time_ns() - cache_info.start_time_ns
"""

NEW = """    cache_info = aot_config.cache_info
    if cache_info is not None:
        time_taken_ns = time.time_ns() - cache_info.start_time_ns
"""

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

source = TARGET.read_text()

if OLD in source:
    if not BACKUP.exists():
        BACKUP.write_text(source)
        print("[OK] Backup created:", BACKUP)

    patched = source.replace(OLD, NEW, 1)

    ast.parse(patched)
    TARGET.write_text(patched)

    print("[OK] Removed _fx_graph_cache_key save gate")
    print("[OK] Syntax check passed")

elif NEW in source:
    print("[INFO] Patch already installed")

else:
    raise RuntimeError(
        "Expected AOT cache block not found. "
        "PyTorch source may have changed."
    )