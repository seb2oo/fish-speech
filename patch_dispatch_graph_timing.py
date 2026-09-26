from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py"
)

backup = path.with_suffix(".py.backup_dispatch_graph_timing")

text = path.read_text()

# Backup original
if not backup.exists():
    backup.write_text(text)
    print(f"Backup created: {backup}")

# Add time import
if "import time as _aot_debug_time" not in text:
    text = "import time as _aot_debug_time\n" + text

# ---------------------------------------------------------
# START TIMER
# ---------------------------------------------------------

needle = """    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
"""

replacement = """    _aot_t0 = _aot_debug_time.perf_counter()
    print("[AOT-GRAPH-TIMER] START", flush=True)

    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
"""

if "[AOT-GRAPH-TIMER] START" not in text:
    if needle not in text:
        raise RuntimeError("START insertion point not found")

    text = text.replace(needle, replacement, 1)


# ---------------------------------------------------------
# AFTER create_functionalized_fn
# ---------------------------------------------------------

needle = """    fn_to_trace, updated_flat_args = create_functionalized_fn(
        fn_to_trace,
        flat_args,
        meta=fw_metadata,
        aot_config=aot_config,
        trace_joint=False,
    )
"""

replacement = needle + """
    print(
        "[AOT-GRAPH-TIMER] after create_functionalized_fn",
        round(_aot_debug_time.perf_counter() - _aot_t0, 3),
        flush=True,
    )
"""

if "[AOT-GRAPH-TIMER] after create_functionalized_fn" not in text:
    if needle not in text:
        raise RuntimeError(
            "create_functionalized_fn insertion point not found"
        )

    text = text.replace(needle, replacement, 1)


# ---------------------------------------------------------
# AFTER aot_dispatch_subclass
# ---------------------------------------------------------

needle = """    ) = aot_dispatch_subclass(
        fn_to_trace,
        updated_flat_args,
        is_joint_structure=False,
        meta=fw_metadata,
        fw_only=flat_fn,
    )
"""

replacement = needle + """
    print(
        "[AOT-GRAPH-TIMER] after aot_dispatch_subclass",
        round(_aot_debug_time.perf_counter() - _aot_t0, 3),
        flush=True,
    )
"""

if "[AOT-GRAPH-TIMER] after aot_dispatch_subclass" not in text:
    if needle not in text:
        raise RuntimeError(
            "aot_dispatch_subclass insertion point not found"
        )

    text = text.replace(needle, replacement, 1)


# ---------------------------------------------------------
# AROUND _create_graph
# ---------------------------------------------------------

needle = """    fw_module = _create_graph(
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        aot_config=aot_config,
    )
"""

replacement = """    _aot_create_graph_t0 = _aot_debug_time.perf_counter()

    fw_module = _create_graph(
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        aot_config=aot_config,
    )

    print(
        "[AOT-GRAPH-TIMER] _create_graph",
        round(
            _aot_debug_time.perf_counter() - _aot_create_graph_t0,
            3,
        ),
        "total=",
        round(
            _aot_debug_time.perf_counter() - _aot_t0,
            3,
        ),
        "nodes=",
        len(list(fw_module.graph.nodes)),
        flush=True,
    )
"""

if "[AOT-GRAPH-TIMER] _create_graph" not in text:
    if needle not in text:
        raise RuntimeError(
            "_create_graph insertion point not found"
        )

    text = text.replace(needle, replacement, 1)


# ---------------------------------------------------------
# WRITE
# ---------------------------------------------------------

path.write_text(text)

print()
print("Patch applied successfully.")
print(f"File   : {path}")
print(f"Backup : {backup}")