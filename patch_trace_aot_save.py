from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/autograd_cache.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_trace_aot_save"
)

source = TARGET.read_text()

if not BACKUP.exists():
    BACKUP.write_text(source)
    print(f"[OK] Backup created: {BACKUP}")

tree = ast.parse(source)

found = False

for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef):
        continue

    if node.name != "load":
        continue

    # Inject just before the cache-save section by locating
    # the "if compiled_fn is None" block that follows dispatch_and_compile().
    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue

        # Look for:
        # if compiled_fn is None:
        #     ...
        #     compiled_fn = dispatch_and_compile()
        has_dispatch = False

        for sub in ast.walk(child):
            if (
                isinstance(sub, ast.Assign)
                and isinstance(sub.value, ast.Call)
                and isinstance(sub.value.func, ast.Name)
                and sub.value.func.id == "dispatch_and_compile"
            ):
                has_dispatch = True
                break

        if not has_dispatch:
            continue

        # Find the assignment and inject after it.
        for idx, stmt in enumerate(child.body):
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "dispatch_and_compile"
            ):
                diagnostic = ast.parse(
                    '''
print(
    "[AOT-SAVE-TRACE] after dispatch_and_compile "
    f"cache_key={cache_key} "
    f"compiled_fn={type(compiled_fn).__name__ if compiled_fn is not None else None} "
    f"cache_state={cache_state}",
    flush=True,
)
'''
                ).body[0]

                child.body.insert(idx + 1, diagnostic)
                found = True
                break

        if found:
            break

    if found:
        break

if not found:
    raise RuntimeError("Could not find dispatch_and_compile assignment")

new_source = ast.unparse(tree) + "\n"

TARGET.write_text(new_source)

compile(new_source, str(TARGET), "exec")

print("[OK] AOT save diagnostic installed")
print("[OK] Syntax check passed")