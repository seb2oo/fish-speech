from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py"
)

BACKUP = TARGET.with_name("compile_fx.py.backup_mega_compiler_timing")


def patch():
    src = TARGET.read_text()

    if not BACKUP.exists():
        BACKUP.write_text(src)
        print(f"Backup created: {BACKUP}")

    # ---------------------------------------------------------
    # import time
    # ---------------------------------------------------------

    if "import time\n" not in src:
        marker = "import functools\n"

        if marker not in src:
            raise RuntimeError("import functools marker not found")

        src = src.replace(
            marker,
            marker + "import time\n",
            1,
        )

    # ---------------------------------------------------------
    # recursive_joint_graph_passes
    # ---------------------------------------------------------

    old = """                if is_inference:
                    # partition_fn won't be called
                    _recursive_joint_graph_passes(gm)
"""

    new = """                if is_inference:
                    # partition_fn won't be called
                    _mega_t0 = time.perf_counter()

                    print(
                        "[MEGA-COMPILER] BEFORE recursive_joint_graph_passes",
                        flush=True,
                    )

                    _recursive_joint_graph_passes(gm)

                    print(
                        "[MEGA-COMPILER] recursive_joint_graph_passes =",
                        round(time.perf_counter() - _mega_t0, 3),
                        "s",
                        flush=True,
                    )
"""

    if old not in src:
        raise RuntimeError(
            "recursive_joint_graph_passes block not found"
        )

    src = src.replace(old, new, 1)

    # ---------------------------------------------------------
    # inner_compile
    # ---------------------------------------------------------

    old = """                return inner_compile(
                    gm,
                    example_inputs,
                    static_input_idxs=get_static_input_idxs(fixed),
                    cudagraphs=cudagraphs,
                    graph_id=graph_id,
                    is_inference=is_inference,
                    boxed_forward_device_index=forward_device,
                )
"""

    new = """                _mega_inner_t0 = time.perf_counter()

                print(
                    "[MEGA-COMPILER] BEFORE inner_compile",
                    "nodes=",
                    len(list(gm.graph.nodes)),
                    flush=True,
                )

                _mega_result = inner_compile(
                    gm,
                    example_inputs,
                    static_input_idxs=get_static_input_idxs(fixed),
                    cudagraphs=cudagraphs,
                    graph_id=graph_id,
                    is_inference=is_inference,
                    boxed_forward_device_index=forward_device,
                )

                print(
                    "[MEGA-COMPILER] inner_compile =",
                    round(time.perf_counter() - _mega_inner_t0, 3),
                    "s",
                    flush=True,
                )

                return _mega_result
"""

    if old not in src:
        raise RuntimeError(
            "inner_compile block not found"
        )

    src = src.replace(old, new, 1)

    TARGET.write_text(src)

    print()
    print("========================================")
    print(" MEGA COMPILE FX PATCH APPLIED")
    print("========================================")
    print(f"Target : {TARGET}")
    print(f"Backup : {BACKUP}")
    print()


if __name__ == "__main__":
    patch()