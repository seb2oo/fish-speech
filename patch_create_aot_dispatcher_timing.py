from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/aot_autograd.py"
)

text = path.read_text()

old = """    with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
        return _create_aot_dispatcher_function(
"""

new = """    _aot_dispatch_debug_t0 = time.perf_counter()

    print(
        "[AOT-DISPATCHER-TIMER] START",
        flush=True,
    )

    with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
        result = _create_aot_dispatcher_function(
"""

if "[AOT-DISPATCHER-TIMER] START" not in text:
    if old not in text:
        raise RuntimeError(
            "Target create_aot_dispatcher_function call not found"
        )

    text = text.replace(old, new, 1)

    old_return = """        )
    return result
"""

    new_return = """        )

    print(
        "[AOT-DISPATCHER-TIMER] END",
        "total=",
        round(time.perf_counter() - _aot_dispatch_debug_t0, 3),
        flush=True,
    )

    return result
"""

    # Attention : le premier ")\n    return result" après notre modification
    # doit être celui de create_aot_dispatcher_function.
    marker_pos = text.find(
        '    _aot_dispatch_debug_t0 = time.perf_counter()'
    )

    if marker_pos == -1:
        raise RuntimeError("Inserted timer marker not found")

    return_pos = text.find(
        "    return result",
        marker_pos,
    )

    if return_pos == -1:
        raise RuntimeError("return result not found")

    text = (
        text[:return_pos]
        + """    print(
        "[AOT-DISPATCHER-TIMER] END",
        "total=",
        round(time.perf_counter() - _aot_dispatch_debug_t0, 3),
        flush=True,
    )

"""
        + text[return_pos:]
    )

    path.write_text(text)
    print("Patch applied successfully.")

else:
    print("Patch already present.")

print(f"File: {path}")