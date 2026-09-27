from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py"
)

BACKUP = TARGET.with_suffix(".py.backup_enable_fx_cache_aot")

OLD = """            and (config.fx_graph_cache or fx_graph_remote_cache)
            and not aot_mode
            and backends_support_caching
"""

NEW = """            and (config.fx_graph_cache or fx_graph_remote_cache)
            and backends_support_caching
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

    print("[OK] Removed AOT FX-cache bypass")
    print("[OK] Syntax check passed")

elif NEW in source:
    print("[INFO] Patch already installed")

else:
    raise RuntimeError(
        "Expected use_cache block not found. "
        "PyTorch source may have changed."
    )
PY