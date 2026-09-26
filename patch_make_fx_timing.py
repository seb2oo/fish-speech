from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py"
)

backup = path.with_suffix(".py.backup_make_fx_timing")

text = path.read_text()

if not backup.exists():
    backup.write_text(text)
    print(f"Backup created: {backup}")

old = """        fx_g = make_fx(
            f,
            decomposition_table=aot_config.decompositions,
            record_module_stack=True,
            pre_dispatch=aot_config.pre_dispatch,
        )(*args)
"""

new = """        _make_fx_t0 = _aot_debug_time.perf_counter()

        fx_g = make_fx(
            f,
            decomposition_table=aot_config.decompositions,
            record_module_stack=True,
            pre_dispatch=aot_config.pre_dispatch,
        )(*args)

        print(
            "[AOT-MAKE-FX-TIMER]",
            "make_fx=",
            round(_aot_debug_time.perf_counter() - _make_fx_t0, 3),
            "nodes=",
            len(list(fx_g.graph.nodes)),
            flush=True,
        )
"""

if "[AOT-MAKE-FX-TIMER]" not in text:
    if old not in text:
        raise RuntimeError("make_fx block not found")

    text = text.replace(old, new, 1)

path.write_text(text)

print("Patch applied successfully.")
print(f"File   : {path}")
print(f"Backup : {backup}")