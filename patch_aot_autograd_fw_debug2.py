from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/"
    "_aot_autograd/jit_compile_runtime_wrappers.py"
)

backup = path.with_suffix(".backup_aot_autograd_fw_debug2")

if not backup.exists():
    backup.write_bytes(path.read_bytes())
    print(f"Backup créé: {backup}")

text = path.read_text()

old = """            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)
"""

new = """            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)

            print(
                "[AOT-AUTOGRAD-FW-DEBUG]",
                "type=", type(compiled_fw_func).__name__,
                "fx_key=", getattr(compiled_fw_func, "_fx_graph_cache_key", None),
                "time=", getattr(compiled_fw_func, "_time_taken_ns", None),
                "boxed=", getattr(compiled_fw_func, "_boxed_call", None),
                flush=True,
            )
"""

if old not in text:
    raise RuntimeError("Bloc cible introuvable — aucun patch appliqué.")

path.write_text(text.replace(old, new, 1))

print("Patch appliqué.")