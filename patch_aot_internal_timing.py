from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/aot_autograd.py"
)

text = path.read_text()

backup = path.with_suffix(".py.backup_create_aot_internal_timing")

if not backup.exists():
    backup.write_text(text)
    print(f"Backup created: {backup}")

# On ajoute un timer au tout début de _create_aot_dispatcher_function.
old = """    # This is the main entry point.
    # TODO: Chillee argues that dynamo itself should pass in fake tensors to
"""

new = """    _aot_internal_t0 = time.perf_counter()

    print(
        "[AOT-INTERNAL-TIMER] START",
        flush=True,
    )

    # This is the main entry point.
    # TODO: Chillee argues that dynamo itself should pass in fake tensors to
"""

if "[AOT-INTERNAL-TIMER] START" not in text:
    if old not in text:
        raise RuntimeError("Start marker not found")

    text = text.replace(old, new, 1)

# Maintenant on instrumente les appels importants.
targets = [
    (
        "aot_dispatcher_pre_compile",
        """    fw_metadata = pre_compile(
""",
    ),
]

# Pour l'instant on ne modifie PAS les appels internes.
# On place simplement un timer juste avant le premier gros bloc
# après la construction des décompositions.

marker = """    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )
"""

replacement = """    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    print(
        "[AOT-INTERNAL-TIMER] after_setup",
        "elapsed=",
        round(time.perf_counter() - _aot_internal_t0, 3),
        flush=True,
    )
"""

if "[AOT-INTERNAL-TIMER] after_setup" not in text:
    if marker not in text:
        raise RuntimeError("python_dispatcher_mode marker not found")

    text = text.replace(marker, replacement, 1)

path.write_text(text)

print("Patch applied successfully.")
print(f"File   : {path}")
print(f"Backup : {backup}")