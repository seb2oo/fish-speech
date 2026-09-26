# patch_trace_aot_tensor.py

from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/"
    "_functorch/_aot_autograd/autograd_cache.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_trace_aot_tensor"
)

MARKER = "[AOT-TRACE] torch.tensor node"


if not TARGET.exists():
    raise FileNotFoundError(f"Target not found: {TARGET}")


source = TARGET.read_text()


if MARKER in source:
    print("[INFO] Patch already installed")
    raise SystemExit(0)


if not BACKUP.exists():
    BACKUP.write_text(source)
    print(f"[OK] Backup created: {BACKUP}")
else:
    print(f"[OK] Backup already exists: {BACKUP}")


old = """            with sanitize_gm_for_cache(gm):
                compiled_fn = None
"""


new = """            with sanitize_gm_for_cache(gm):
                # ---------------------------------------------------------
                # Diagnostic: find torch.tensor() nodes in the GraphModule
                # that cause AOTAutogradCache to bypass.
                # ---------------------------------------------------------
                try:
                    nodes = list(gm.graph.nodes)

                    print(
                        f"[AOT-TRACE] GraphModule={type(gm).__name__} "
                        f"nodes={len(nodes)}",
                        flush=True,
                    )

                    for node in nodes:
                        if node.op != "call_function":
                            continue

                        target = node.target
                        target_name = getattr(target, "__name__", "")
                        target_repr = repr(target)

                        if (
                            target_name == "tensor"
                            or "torch.tensor" in target_repr
                        ):
                            print(
                                "[AOT-TRACE] torch.tensor node:",
                                flush=True,
                            )
                            print(
                                f"    name={node.name}",
                                flush=True,
                            )
                            print(
                                f"    target={target_repr}",
                                flush=True,
                            )
                            print(
                                f"    args={node.args}",
                                flush=True,
                            )
                            print(
                                f"    kwargs={node.kwargs}",
                                flush=True,
                            )
                            print(
                                f"    stack_trace="
                                f"{getattr(node, 'stack_trace', None)}",
                                flush=True,
                            )

                except Exception as trace_exc:
                    print(
                        "[AOT-TRACE] Diagnostic failed: "
                        f"{type(trace_exc).__name__}: {trace_exc}",
                        flush=True,
                    )

                compiled_fn = None
"""


if old not in source:
    raise RuntimeError(
        "Target block not found. "
        "The PyTorch file may have changed or the patch is already modified."
    )


patched_source = source.replace(old, new, 1)

# Syntax check before touching the installed file.
ast.parse(patched_source)

TARGET.write_text(patched_source)

print("[OK] AOT tensor diagnostic installed")
print("[OK] Syntax check passed")