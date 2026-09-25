
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

    marker = "def benchmark_gpu("
    pos = text.find(marker)

    if pos == -1:
        raise SystemExit(
            "[ERROR] benchmark_gpu() not found in benchmarking.py"
        )

    # Find the function body.
    line_start = text.rfind("\n", 0, pos) + 1

    # Find the first line after the function definition.
    body_start = text.find("\n", pos)
    if body_start == -1:
        raise SystemExit("[ERROR] Could not find benchmark_gpu body")

    body_start += 1

    # Determine indentation of the first body line.
    next_nonempty = body_start
    while next_nonempty < len(text):
        line_end = text.find("\n", next_nonempty)
        if line_end == -1:
            line_end = len(text)

        line = text[next_nonempty:line_end]

        if line.strip():
            indent = line[: len(line) - len(line.lstrip())]
            break

        next_nonempty = line_end + 1
    else:
        raise SystemExit("[ERROR] Could not determine function indentation")

    tracer = (
        f"{indent}import time as _profile_bench_time\n"
        f"{indent}_profile_bench_t0 = _profile_bench_time.perf_counter()\n"
        f"{indent}print(\n"
        f"{indent}    f\"[PROFILE-BENCH-GPU] START \"\n"
        f"{indent}    f\"function={{getattr(self, 'name', 'unknown')}}\",\n"
        f"{indent}    flush=True,\n"
        f"{indent})\n"
    )

    text = text[:body_start] + tracer + text[body_start:]

    # We need to measure the complete function, including its return.
    # Find benchmark_gpu AST and instrument every return statement.
    tree = ast.parse(text)

    benchmark_node = None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "benchmark_gpu":
                benchmark_node = node
                break

    if benchmark_node is None:
        raise SystemExit("[ERROR] benchmark_gpu AST node not found")

    # Find the first return in benchmark_gpu.
    return_nodes = [
        node
        for node in ast.walk(benchmark_node)
        if isinstance(node, ast.Return)
    ]

    if not return_nodes:
        raise SystemExit(
            "[ERROR] benchmark_gpu has no return statement. "
            "Patch needs manual adjustment."
        )

    # We don't rewrite the return expression itself.
    # Instead insert timing immediately before every return.
    lines = text.splitlines(keepends=True)

    inserts = []

    for node in return_nodes:
        # AST line numbers are 1-based.
        idx = node.lineno - 1

        original = lines[idx]
        indent_return = original[: len(original) - len(original.lstrip())]

        timing = (
            f"{indent_return}_profile_bench_dt = "
            f"_profile_bench_time.perf_counter() - _profile_bench_t0\n"
            f"{indent_return}print(\n"
            f"{indent_return}    f\"[PROFILE-BENCH-GPU] END \"\n"
            f"{indent_return}    f\"function={{getattr(self, 'name', 'unknown')}} \"\n"
            f"{indent_return}    f\"elapsed={{_profile_bench_dt:.6f}}s\",\n"
            f"{indent_return}    flush=True,\n"
            f"{indent_return})\n"
        )

        inserts.append((idx, timing))

    # Insert backwards so line indices remain valid.
    for idx, timing in sorted(inserts, reverse=True):
        lines.insert(idx, timing)

    text = "".join(lines)

    try:
        ast.parse(text)
    except SyntaxError as e:
        raise SystemExit(
            f"[ERROR] Syntax check failed: {e}"
        )

    TARGET.write_text(text)

    print("[OK] benchmark_gpu profiling installed")
    print("[OK] Syntax check passed")
    print()
    print("The patch measures every benchmark_gpu() call.")
    print("It prints START/END and elapsed time for each call.")


if __name__ == "__main__":
    main()