from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/aot_autograd.py"
)

backup = path.with_suffix(".py.backup_create_aot_dispatcher_timing")

text = path.read_text()

if not backup.exists():
    backup.write_text(text)
    print(f"Backup created: {backup}")

old = """def create_aot_dispatcher_function(
    flat_fn,
    flat_args,
    aot_config,
    *,
    fake_mode,
    shape_env,
):
    with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
        return _create_aot_dispatcher_function(
            flat_fn,
            flat_args,
            aot_config,
            fake_mode=fake_mode,
            shape_env=shape_env,
        )
"""

new = """def create_aot_dispatcher_function(
    flat_fn,
    flat_args,
    aot_config,
    *,
    fake_mode,
    shape_env,
):
    _aot_dispatch_debug_t0 = time.perf_counter()

    print(
        "[AOT-DISPATCHER-TIMER] START",
        "nodes=",
        len(list(flat_fn.graph.nodes)) if hasattr(flat_fn, "graph") else "NA",
        "cache_key=",
        getattr(aot_config, "cache_key", None),
        flush=True,
    )

    with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
        result = _create_aot_dispatcher_function(
            flat_fn,
            flat_args,
            aot_config,
            fake_mode=fake_mode,
            shape_env=shape_env,
        )

    print(
        "[AOT-DISPATCHER-TIMER] END",
        "total=",
        round(time.perf_counter() - _aot_dispatch_debug_t0, 3),
        flush=True,
    )

    return result
"""

if "[AOT-DISPATCHER-TIMER] START" not in text:
    if old not in text:
        raise RuntimeError(
            "Target create_aot_dispatcher_function block not found"
        )

    text = text.replace(old, new, 1)

path.write_text(text)

print("Patch applied successfully.")
print(f"File   : {path}")
print(f"Backup : {backup}")