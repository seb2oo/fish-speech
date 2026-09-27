from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py"
)

BACKUP = TARGET.with_suffix(".py.backup_force_aot_cache_save")

OLD = """        if cache_info is not None:
            if hasattr(compiled_fw, "_fx_graph_cache_key"):
                compiled_fw = AOTAutogradCache.make_entry(
                    compiled_fw,
                    None,
                    cache_info,
                    updated_flat_args,
                )
                AOTAutogradCache.save(
                    cache_info.cache_key,
                    compiled_fw,
                    remote=should_use_remote_autograd_cache(),
                )
"""

NEW = """        if cache_info is not None:
            compiled_fw = AOTAutogradCache.make_entry(
                compiled_fw,
                None,
                cache_info,
                updated_flat_args,
            )
            AOTAutogradCache.save(
                cache_info.cache_key,
                compiled_fw,
                remote=should_use_remote_autograd_cache(),
            )
"""

if not TARGET.exists():
    raise FileNotFoundError(f"Target not found: {TARGET}")

source = TARGET.read_text()

if OLD in source:
    if not BACKUP.exists():
        BACKUP.write_text(source)
        print("[OK] Backup created:", BACKUP)

    patched = source.replace(OLD, NEW, 1)

    ast.parse(patched)
    TARGET.write_text(patched)

    print("[OK] Forced AOTAutograd cache save")
    print("[OK] Removed _fx_graph_cache_key gate")
    print("[OK] Syntax check passed")

elif NEW in source:
    print("[INFO] Patch already installed")

else:
    raise RuntimeError(
        "Expected AOT cache-save block not found. "
        "PyTorch source may differ from expected version."
    )