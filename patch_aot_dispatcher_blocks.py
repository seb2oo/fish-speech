import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/aot_autograd.py"
)

BACKUP = TARGET.with_name(
    "aot_autograd.py.backup_aot_dispatcher_blocks"
)

text = TARGET.read_text()

if "[AOT-BLOCK-TIMER]" in text:
    print("Patch already applied.")
    raise SystemExit(0)

BACKUP.write_text(text)

# ------------------------------------------------------------
# 1. import time
# ------------------------------------------------------------

if "import time\n" not in text:
    needle = "import functools\n"

    if needle not in text:
        print("ERROR: could not find import functools")
        raise SystemExit(1)

    text = text.replace(
        needle,
        needle + "import time\n",
        1,
    )

# ------------------------------------------------------------
# 2. BEFORE metadata collection
# ------------------------------------------------------------

needle = """                with dynamo_timed_ctx, ctx:
                    fw_metadata = run_functionalized_fw_and_collect_metadata(
"""

replacement = """                _block_t0 = time.perf_counter()
                print(
                    "[AOT-BLOCK-TIMER] BEFORE metadata collection",
                    flush=True,
                )

                with dynamo_timed_ctx, ctx:
                    fw_metadata = run_functionalized_fw_and_collect_metadata(
"""

if needle not in text:
    print("ERROR: metadata start block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ------------------------------------------------------------
# 3. AFTER metadata collection
# ------------------------------------------------------------

needle = """                    )(*_dup_fake_script_obj(fake_flat_args))

                req_subclass_dispatch = requires_subclass_dispatch(
"""

replacement = """                    )(*_dup_fake_script_obj(fake_flat_args))

                print(
                    "[AOT-BLOCK-TIMER] metadata collection =",
                    round(time.perf_counter() - _block_t0, 3),
                    "s",
                    flush=True,
                )

                _block_t1 = time.perf_counter()

                req_subclass_dispatch = requires_subclass_dispatch(
"""

if needle not in text:
    print("ERROR: metadata end block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ------------------------------------------------------------
# 4. BEFORE compiler_fn
# ------------------------------------------------------------

needle = """        compiler_fn = choose_dispatcher(needs_autograd, aot_config)

        compiled_fn, fw_metadata = compiler_fn(
"""

replacement = """        compiler_fn = choose_dispatcher(needs_autograd, aot_config)

        print(
            "[AOT-BLOCK-TIMER] metadata/checks after collection =",
            round(time.perf_counter() - _block_t1, 3),
            "s",
            flush=True,
        )

        print(
            "[AOT-BLOCK-TIMER] BEFORE compiler_fn",
            getattr(compiler_fn, "__name__", repr(compiler_fn)),
            flush=True,
        )

        _block_t2 = time.perf_counter()

        compiled_fn, fw_metadata = compiler_fn(
"""

if needle not in text:
    print("ERROR: compiler start block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

# ------------------------------------------------------------
# 5. AFTER compiler_fn
# ------------------------------------------------------------

needle = """        )
        return compiled_fn, fw_metadata
"""

replacement = """        )

        print(
            "[AOT-BLOCK-TIMER] compiler_fn total =",
            round(time.perf_counter() - _block_t2, 3),
            "s",
            flush=True,
        )

        return compiled_fn, fw_metadata
"""

if needle not in text:
    print("ERROR: compiler end block not found")
    raise SystemExit(1)

text = text.replace(needle, replacement, 1)

TARGET.write_text(text)

print("Patch applied successfully.")
print("Backup:", BACKUP)