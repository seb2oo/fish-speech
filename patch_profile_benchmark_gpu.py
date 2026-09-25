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

    # Find the exact decorator insertion point.
    # We put a small wrapper immediately before benchmark_gpu.
    insert_line = target_function.lineno - 1

    wrapper = """    def _profile_benchmark_gpu(self, *args, **kwargs):
        import time as _profile_bench_time

        _profile_bench_t0 = _profile_bench_time.perf_counter()

        print(
            f"[PROFILE-BENCH-GPU] START "
            f"function={getattr(self, 'name', 'unknown')}",
            flush=True,
        )

        try:
            return self._profile_benchmark_gpu_original(
                *args,
                **kwargs,
            )
        finally:
            _profile_bench_dt = (
                _profile_bench_time.perf_counter()
                - _profile_bench_t0
            )

            print(
                f"[PROFILE-BENCH-GPU] END "
                f"function={getattr(self, 'name', 'unknown')} "
                f"elapsed={_profile_bench_dt:.6f}s",
                flush=True,
            )

"""

    # Rename the original method.
    lines = text.splitlines(keepends=True)

    original_line = lines[insert_line]

    if "def benchmark_gpu(" not in original_line:
        raise SystemExit(
            "[ERROR] Unexpected benchmark_gpu definition line:\n"
            + original_line
        )

    lines[insert_line] = original_line.replace(
        "def benchmark_gpu(",
        "def _profile_benchmark_gpu_original(",
        1,
    )

    # Insert wrapper + alias after the original method.
    # We need the end of the class, not the end of benchmark_gpu,
    # because benchmark_gpu may contain nested structures.
    class_end_line = target_class.end_lineno

    alias = """    benchmark_gpu = _profile_benchmark_gpu

"""

    lines.insert(
        class_end_line,
        alias,
    )

    # Wrapper must appear before the alias, but after the original method.
    # Recompute the class end after the inserted alias.
    wrapper_insert_line = class_end_line

    lines.insert(
        wrapper_insert_line,
        wrapper,
    )

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
    print("Every benchmark_gpu() call will report its complete duration.")


if __name__ == "__main__":
    main()