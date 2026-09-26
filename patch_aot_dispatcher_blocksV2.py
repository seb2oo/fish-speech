import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py"
)

BACKUP = TARGET.with_name(
    "jit_compile_runtime_wrappers.py.backup_mega_aot_base"
)

text = TARGET.read_text()

if "[MEGA-AOT-BASE]" in text:
    print("Patch already applied.")
    raise SystemExit(0)

BACKUP.write_text(text)

# ============================================================
# IMPORT TIME
# ============================================================

if "import time" not in text:
    # Find first import and insert before it.
    pos = text.find("import ")
    if pos == -1:
        print("ERROR: could not find import section")
        raise SystemExit(1)

    text = text[:pos] + "import time\n" + text[pos:]

# ============================================================
# START OF aot_dispatch_base
# ============================================================

needle = """    wrappers = _create_wrappers_for_dispatch(needs_autograd=False)
"""

replacement = """    _mega_t0 = time.perf_counter()

    print(
        "[MEGA-AOT-BASE] START",
        flush=True,
    )

    _mega_t = time.perf_counter()

    wrappers = _create_wrappers_for_dispatch(needs_autograd=False)

    print(
        "[MEGA-AOT-BASE] _create_wrappers_for_dispatch =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if needle not in text:
    print("ERROR: wrappers block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# pre_compile
# ============================================================

needle = """    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers, flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )
"""

replacement = """    _mega_t = time.perf_counter()

    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers, flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    print(
        "[MEGA-AOT-BASE] pre_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if needle not in text:
    print("ERROR: pre_compile block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# aot_dispatch_base_graph
# ============================================================

needle = """    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(  # type: ignore[misc]
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )
"""

replacement = """    _mega_t = time.perf_counter()

    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(  # type: ignore[misc]
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    print(
        "[MEGA-AOT-BASE] aot_dispatch_base_graph =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if needle not in text:
    print("ERROR: aot_dispatch_base_graph block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# print_readable
# ============================================================

needle = """        aot_forward_graph_str = fw_module.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            fast_sympy_print=True,
        )
"""

replacement = """        _mega_t = time.perf_counter()

        aot_forward_graph_str = fw_module.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            fast_sympy_print=True,
        )

        print(
            "[MEGA-AOT-BASE] print_readable =",
            round(time.perf_counter() - _mega_t, 3),
            "s",
            flush=True,
        )
"""

if needle not in text:
    print("ERROR: print_readable block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# FakifiedOutWrapper.pre_compile
# ============================================================

needle = """    ) = fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
"""

replacement = """    _mega_t = time.perf_counter()

    ) = fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
"""

# We cannot safely replace this syntax directly because of tuple assignment.
# Instead locate the exact preceding tuple and rewrite it.

old = """    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
"""

new = """    _mega_t = time.perf_counter()

    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )

    print(
        "[MEGA-AOT-BASE] fakified_out_wrapper.pre_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if old not in text:
    print("ERROR: fakified_out_wrapper block not found")
    raise SystemExit(1)

text = text.replace(old, new, 1)

# ============================================================
# FunctionalizedRngRuntimeWrapper.pre_compile
# ============================================================

old = """    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = functionalized_rng_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
"""

new = """    _mega_t = time.perf_counter()

    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = functionalized_rng_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )

    print(
        "[MEGA-AOT-BASE] functionalized_rng_wrapper.pre_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if old not in text:
    print("ERROR: functionalized_rng_wrapper block not found")
    raise SystemExit(1)

text = text.replace(old, new, 1)

# ============================================================
# BEFORE compiler
# ============================================================

needle = """        with TracingContext.report_output_strides() as fwd_output_strides:
"""

replacement = """        print(
            "[MEGA-AOT-BASE] BEFORE compiler",
            flush=True,
        )

        _mega_t = time.perf_counter()

        with TracingContext.report_output_strides() as fwd_output_strides:
"""

if needle not in text:
    print("ERROR: compiler context not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# tensorify_python_scalars
# ============================================================

needle = """                tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
            compiled_fw = compiler(fw_module, updated_flat_args)
"""

replacement = """                _mega_t_tensorify = time.perf_counter()

                tensorify_python_scalars(
                    fw_module,
                    fake_mode.shape_env,
                    fake_mode,
                )

                print(
                    "[MEGA-AOT-BASE] tensorify_python_scalars =",
                    round(time.perf_counter() - _mega_t_tensorify, 3),
                    "s",
                    flush=True,
                )

            _mega_t_compiler = time.perf_counter()

            compiled_fw = compiler(fw_module, updated_flat_args)

            print(
                "[MEGA-AOT-BASE] compiler(fw_module) =",
                round(time.perf_counter() - _mega_t_compiler, 3),
                "s",
                flush=True,
            )
"""

if needle not in text:
    print("ERROR: compiler call block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# AFTER compiler context
# ============================================================

needle = """        if fakified_out_wrapper.needs_post_compile:
            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)
"""

replacement = """        if fakified_out_wrapper.needs_post_compile:
            _mega_t = time.perf_counter()

            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

            print(
                "[MEGA-AOT-BASE] set_fwd_output_strides =",
                round(time.perf_counter() - _mega_t, 3),
                "s",
                flush=True,
            )

        print(
            "[MEGA-AOT-BASE] compiler context total =",
            round(time.perf_counter() - _mega_t, 3),
            "s",
            flush=True,
        )
"""

# Don't apply this one because _mega_t was reused by tensorify/compiler.
# Instead use a dedicated context timer.
replacement = """        if fakified_out_wrapper.needs_post_compile:
            _mega_t = time.perf_counter()

            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

            print(
                "[MEGA-AOT-BASE] set_fwd_output_strides =",
                round(time.perf_counter() - _mega_t, 3),
                "s",
                flush=True,
            )
"""

if needle not in text:
    print("ERROR: post compiler context block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# make_runtime_safe
# ============================================================

needle = """    make_runtime_safe(fw_metadata, maybe_subclass_meta)
"""

replacement = """    _mega_t = time.perf_counter()

    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    print(
        "[MEGA-AOT-BASE] make_runtime_safe =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if needle not in text:
    print("ERROR: make_runtime_safe block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# AOT CACHE SAVE
# ============================================================

needle = """            AOTAutogradCache.save(
                cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
            )
"""

replacement = """            _mega_t = time.perf_counter()

            AOTAutogradCache.save(
                cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
            )

            print(
                "[MEGA-AOT-BASE] AOTAutogradCache.save =",
                round(time.perf_counter() - _mega_t, 3),
                "s",
                flush=True,
            )
"""

if needle not in text:
    print("ERROR: AOTAutogradCache.save block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# POST COMPILE WRAPPERS
# ============================================================

# fakified_out_wrapper.post_compile

needle = """    compiled_fw = fakified_out_wrapper.post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )
"""

replacement = """    _mega_t = time.perf_counter()

    compiled_fw = fakified_out_wrapper.post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    print(
        "[MEGA-AOT-BASE] fakified_out_wrapper.post_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if needle not in text:
    print("ERROR: fakified post_compile block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# EffectTokensWrapper
# ============================================================

needle = """    compiled_fw = EffectTokensWrapper().post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )
"""

replacement = """    _mega_t = time.perf_counter()

    compiled_fw = EffectTokensWrapper().post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    print(
        "[MEGA-AOT-BASE] EffectTokensWrapper.post_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )
"""

if needle not in text:
    print("ERROR: EffectTokensWrapper block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# AOTDispatchSubclassWrapper
# ============================================================

needle = """    compiled_fw = AOTDispatchSubclassWrapper(
        trace_joint=False,
"""

replacement = """    _mega_t = time.perf_counter()

    compiled_fw = AOTDispatchSubclassWrapper(
        trace_joint=False,
"""

if needle not in text:
    print("ERROR: subclass wrapper start not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

needle = """    ).post_compile(
        compiled_fw,
        aot_config,  # not used
        runtime_metadata=fw_metadata,
    )

    if not getattr(compiled_fw, "_boxed_call", False):
"""

replacement = """    ).post_compile(
        compiled_fw,
        aot_config,  # not used
        runtime_metadata=fw_metadata,
    )

    print(
        "[MEGA-AOT-BASE] AOTDispatchSubclassWrapper.post_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )

    if not getattr(compiled_fw, "_boxed_call", False):
"""

if needle not in text:
    print("ERROR: subclass wrapper end not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# RuntimeWrapper
# ============================================================

needle = """    compiled_fn = RuntimeWrapper(
"""

replacement = """    _mega_t = time.perf_counter()

    compiled_fn = RuntimeWrapper(
"""

if needle not in text:
    print("ERROR: RuntimeWrapper start not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

needle = """    ).post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    compiled_fn = post_compile(
"""

replacement = """    ).post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    print(
        "[MEGA-AOT-BASE] RuntimeWrapper.post_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )

    compiled_fn = post_compile(
"""

if needle not in text:
    print("ERROR: RuntimeWrapper end not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ============================================================
# FINAL post_compile + TOTAL
# ============================================================

needle = """    compiled_fn = post_compile(
        wrappers, compiled_fn, aot_config, runtime_metadata=fw_metadata
    )
    return compiled_fn
"""

replacement = """    _mega_t = time.perf_counter()

    compiled_fn = post_compile(
        wrappers, compiled_fn, aot_config, runtime_metadata=fw_metadata
    )

    print(
        "[MEGA-AOT-BASE] final post_compile =",
        round(time.perf_counter() - _mega_t, 3),
        "s",
        flush=True,
    )

    print(
        "[MEGA-AOT-BASE] TOTAL =",
        round(time.perf_counter() - _mega_t0, 3),
        "s",
        flush=True,
    )

    return compiled_fn
"""

if needle not in text:
    print("ERROR: final post_compile block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

TARGET.write_text(text)

print("Mega AOT base diagnostic patch applied successfully.")
print("Target:", TARGET)
print("Backup:", BACKUP)