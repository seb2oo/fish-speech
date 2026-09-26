
from pathlib import Path
import shutil

FILE = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/"
    "jit_compile_runtime_wrappers.py"
)

BACKUP = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/"
    "jit_compile_runtime_wrappers.py.backup_aot_fw_debug"
)

OLD = """        compiled_fw = compiler(fw_module, updated_flat_args)
"""

NEW = """        compiled_fw = compiler(fw_module, updated_flat_args)

        print(
            "[AOT-FW-DEBUG]",
            "type=", type(compiled_fw).__name__,
            "fx_key=", getattr(compiled_fw, "_fx_graph_cache_key", None),
            "time=", getattr(compiled_fw, "_time_taken_ns", None),
            "boxed=", getattr(compiled_fw, "_boxed_call", None),
            flush=True,
        )
"""


def main():
    if not FILE.exists():
        raise FileNotFoundError(f"Source file not found: {FILE}")

    # Backup
    if BACKUP.exists():
        print(f"[BACKUP] Already exists: {BACKUP}")
    else:
        shutil.copy2(FILE, BACKUP)
        print(f"[BACKUP] Created: {BACKUP}")

    # Read source
    source = FILE.read_text()

    # Prevent accidental double patch
    if "[AOT-FW-DEBUG]" in source:
        raise RuntimeError(
            "Patch already appears to be applied. "
            "No modification performed."
        )

    # Make sure we modify exactly the expected location
    count = source.count(OLD)

    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 occurrence of target block, found {count}."
        )

    # Apply patch
    patched = source.replace(OLD, NEW, 1)

    FILE.write_text(patched)

    print(f"[PATCH] Successfully patched: {FILE}")


if __name__ == "__main__":
    main()

