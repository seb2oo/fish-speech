from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/aot_autograd.py"
)

text = path.read_text()

old = """    with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
        return _create_aot_dispatcher_function(
            flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
        )
"""

new = """    _aot_dispatch_debug_t0 = time.perf_counter()

    print(
        "[AOT-DISPATCHER-TIMER] START",
        flush=True,
    )

    with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
        result = _create_aot_dispatcher_function(
            flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
        )

    print(
        "[AOT-DISPATCHER-TIMER] END",
        "total=",
        round(time.perf_counter() - _aot_dispatch_debug_t0, 3),
        flush=True,
    )

    return result
"""

if "[AOT-DISPATCHER-TIMER] START" in text:
    print("Patch already present.")
else:
    if old not in text:
        raise RuntimeError(
            "Exact create_aot_dispatcher_function block not found"
        )

    backup = path.with_suffix(".py.backup_create_aot_dispatcher_timing_v2")

    if not backup.exists():
        backup.write_text(text)
        print(f"Backup created: {backup}")

    text = text.replace(old, new, 1)
    path.write_text(text)

    print("Patch applied successfully.")

print(f"File: {path}")