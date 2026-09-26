from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/"
    "_functorch/_aot_autograd/autograd_cache.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_trace_aot_tensor_v2"
)

MARKER = "[AOT-TRACE] GraphModule="


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


needle = (
    "        gm = mod.gm "
    "if isinstance(mod, torch._dynamo.utils.GmWrapper) else mod\n"
)


if needle not in source:
    raise RuntimeError(
        "Could not find AOTAutogradCache.load() GraphModule line."
    )


injection = r'''
        # -------------------------------------------------------------
        # Diagnostic: locate torch.tensor() nodes that cause the
        # AOTAutograd cache to bypass.
        # -------------------------------------------------------------
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

'''


patched_source = source.replace(
    needle,
    needle + injection,
    1,
)


# Verify syntax before modifying the installed PyTorch file.
ast.parse(patched_source)

TARGET.write_text(patched_source)

print("[OK] AOT tensor diagnostic installed")
print("[OK] Syntax check passed")