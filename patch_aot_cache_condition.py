from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py"
)

backup = path.with_suffix(".py.backup_aot_cache_condition_debug")

text = path.read_text()

if not backup.exists():
    backup.write_text(text)
    print(f"Backup created: {backup}")

old = """    if cache_info is not None:
        if hasattr(compiled_fw, "_fx_graph_cache_key"):
"""

new = """    if cache_info is not None:
        print(
            "[AOT-CACHE-CONDITION]",
            "cache_info=",
            cache_info,
            "compiled_fw_type=",
            type(compiled_fw).__name__,
            "has_fx_key=",
            hasattr(compiled_fw, "_fx_graph_cache_key"),
            "fx_key=",
            getattr(compiled_fw, "_fx_graph_cache_key", None),
            flush=True,
        )

        if hasattr(compiled_fw, "_fx_graph_cache_key"):
"""

if "[AOT-CACHE-CONDITION]" not in text:
    if old not in text:
        raise RuntimeError("Target block not found")
    text = text.replace(old, new, 1)

path.write_text(text)

print("Patch applied successfully.")
print(f"File   : {path}")
print(f"Backup : {backup}")