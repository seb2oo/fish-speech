from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/aot_autograd.py"
)

backup = path.with_suffix(".backup_choose_dispatcher_debug")

if not backup.exists():
    backup.write_bytes(path.read_bytes())
    print(f"Backup créé: {backup}")

text = path.read_text()

old = """        compiler_fn = choose_dispatcher(needs_autograd, aot_config)
"""

new = """        compiler_fn = choose_dispatcher(needs_autograd, aot_config)

        print(
            "[AOT-DISPATCH-DEBUG]",
            "needs_autograd=", needs_autograd,
            "pre_dispatch=", aot_config.pre_dispatch,
            "is_export=", aot_config.is_export,
            "dispatcher=", getattr(compiler_fn, "func", compiler_fn).__name__,
            flush=True,
        )
"""

if old not in text:
    raise RuntimeError("Bloc cible introuvable — aucun patch appliqué.")

path.write_text(text.replace(old, new, 1))

print("Patch appliqué.")