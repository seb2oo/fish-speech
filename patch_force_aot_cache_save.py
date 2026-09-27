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
            guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
            entry = AOTAutogradCache.make_entry(
"""

NEW = """    cache_info = aot_config.cache_info
    if cache_info is not None:
        time_taken_ns = time.time_ns() - cache_info.start_time_ns
        guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
        entry = AOTAutogradCache.make_entry(
"""

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

source = TARGET.read_text()

if OLD not in source:
    raise RuntimeError(
        "Expected block not found. The PyTorch file was not modified."
    )

if not BACKUP.exists():
    BACKUP.write_text(source)
    print("[OK] Backup created:", BACKUP)
else:
    print("[OK] Backup already exists:", BACKUP)

patched = source.replace(OLD, NEW, 1)

ast.parse(patched)
TARGET.write_text(patched)

print("[OK] Removed _fx_graph_cache_key gate")
print("[OK] Syntax check passed")
print("[OK] Patch written successfully")