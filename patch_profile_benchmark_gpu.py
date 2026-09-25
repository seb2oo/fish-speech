from pathlib import Path
import ast

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/"
    "runtime/benchmarking.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_profile_benchmark_gpu"
)


def main():
    if not TARGET.exists():
        raise SystemExit(f"[ERROR] File not found: {TARGET}")

    text = TARGET.read_text()

    if "[PROFILE-BENCH-GPU]" in text:
        print("[INFO] benchmark_gpu profiling already installed")
        return

    if not BACKUP.exists():
        BACKUP.write_text(text)
        print(f"[OK] Backup created: {BACKUP}")
    else:
        print(f"[INFO] Backup already exists: {BACKUP}")

    tree = ast.parse(text)

    target_class = None
    target_function = None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "InductorBenchmarker":
            target_class = node
            break

    if target_class is None:
        raise SystemExit(
            "[ERROR] Class InductorBenchmarker not found"
        )

    for node in target_class.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "benchmark_gpu":
                target_function = node
                break

    if target_function is None:
        raise SystemExit(
            "[ERROR] InductorBenchmarker.benchmark_gpu() not found"
        )

    # We insert timing immediately after the function definition.
    # This measures the complete body by using a try/finally wrapper.
    lines = text.splitlines(keepends=True)

    def line_indent(line):
        return line[: len(line) - len(line.lstrip())]

    def_indent = line_indent(lines[target_function.lineno - 1])
    body_indent = def_indent + "    "
    wrapper_indent = body_indent + "    "

    # Insert after the def line.
    insert_at = target_function.lineno

    wrapper = [
        f"{body_indent}import time as _profile_bench_time\n",
        f"{body_indent}_profile_bench_t0 = _profile_bench_time.perf_counter()\n",
        f"{body_indent}print(\n",
        f'{body_indent}    f"[PROFILE-BENCH-GPU] START "\n',
        f'{body_indent}    f"function={{getattr(self, "name", "unknown")}}",\n',
        f"{body_indent}    flush=True,\n",
        f"{body_indent})\n",
        f"{body_indent}try:\n",
    ]

    # Indent the entire existing function body by 4 spaces.
    body_start = target_function.lineno
    body_end = target_function.end_lineno

    for idx in range(body_start, body_end):
        lines[idx] = "    " + lines[idx]

    # Add finally after the indented original body.
    finally_block = [
        f"{body_indent}finally:\n",
        f"{wrapper_indent}_profile_bench_dt = _profile_bench_time.perf_counter() - _profile_bench_t0\n",
        f"{wrapper_indent}print(\n",
        f'{wrapper_indent}    f"[PROFILE-BENCH-GPU] END "\n',
        f'{wrapper_indent}    f"function={{getattr(self, "name", "unknown")}} "\n',
        f'{wrapper_indent}    f"elapsed={{_profile_bench_dt:.6f}}s",\n',
        f"{wrapper_indent}    flush=True,\n",
        f"{wrapper_indent})\n",
    ]

    lines[insert_at:insert_at] = wrapper

    # target_function.end_lineno refers to the original file, so after
    # inserting the wrapper we need the new end position.
    new_end = body_end + len(wrapper)

    lines[new_end:new_end] = finally_block

    new_text = "".join(lines)

    try:
        ast.parse(new_text)
    except SyntaxError as e:
        raise SystemExit(
            f"[ERROR] Syntax check failed: {e}"
        )

    TARGET.write_text(new_text)

    print("[OK] benchmark_gpu profiling installed")
    print("[OK] Syntax check passed")
    print()
    print("Every InductorBenchmarker.benchmark_gpu() call will report")
    print("its complete elapsed time.")


if __name__ == "__main__":
    main()