#!/usr/bin/env python3

from pathlib import Path
import py_compile
import sys


TRITON_HEURISTICS = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py"
)

COORDESC_TUNER = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/coordinate_descent_tuner.py"
)


def fail(message):
    print(f"\n[ERROR] {message}")
    sys.exit(1)


def replace_once(source, old, new, description):
    count = source.count(old)

    if count == 0:
        if new in source:
            print(f"[OK] Already patched: {description}")
            return source

        fail(
            f"Cannot find expected code for:\n"
            f"  {description}"
        )

    if count != 1:
        fail(
            f"Expected exactly 1 occurrence for:\n"
            f"  {description}\n"
            f"Found: {count}"
        )

    print(f"[PATCH] {description}")
    return source.replace(old, new, 1)


# ============================================================
# TRITON HEURISTICS
# ============================================================

print("=" * 70)
print("PATCH 1/2 — triton_heuristics.py")
print("=" * 70)

if not TRITON_HEURISTICS.exists():
    fail(f"File not found: {TRITON_HEURISTICS}")

s = TRITON_HEURISTICS.read_text()


# ------------------------------------------------------------
# 1. cached_configs condition
# ------------------------------------------------------------

s = replace_once(
    s,
    "if len(cached_configs) == 1 and len(configs) > 1:",
    "if len(cached_configs) == 1:",
    "allow cached config when only one config exists",
)


# ------------------------------------------------------------
# 2. found_by_coordesc
# ------------------------------------------------------------

old = """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    self.compile_results = [compile_result]
                    return"""

new = """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    compile_result.config.found_by_coordesc = True
                    self.compile_results = [compile_result]
                    return"""

s = replace_once(
    s,
    old,
    new,
    "mark cached config as found_by_coordesc=True",
)


# ------------------------------------------------------------
# 3. RECHECK diagnostic
#
# We locate the FUNCTION itself, not a generic marker.
# ------------------------------------------------------------

TRACE = (
    '        print(f"[RECHECK] name={self.fn.__name__} '
    'configs={len(configs)} cached={len(cached_configs)}", flush=True)'
)

if TRACE in s:
    print("[OK] Already patched: RECHECK diagnostic")
else:

    func_marker = "    def recheck_autotune_cache("

    start = s.find(func_marker)

    if start == -1:
        fail(
            "Cannot find recheck_autotune_cache() in "
            "triton_heuristics.py"
        )

    # Find the beginning of the next method.
    next_method = s.find("\n    def ", start + len(func_marker))

    if next_method == -1:
        function_block = s[start:]
    else:
        function_block = s[start:next_method]

    marker = "        self.autotune_cache_info = autotune_cache_info"

    if marker not in function_block:
        fail(
            "Found recheck_autotune_cache(), but could not find "
            "self.autotune_cache_info inside that function."
        )

    if function_block.count(marker) != 1:
        fail(
            "Unexpected number of autotune_cache_info markers "
            "inside recheck_autotune_cache()."
        )

    function_block = function_block.replace(
        marker,
        marker + "\n" + TRACE,
        1,
    )

    s = s[:start] + function_block + s[next_method:]
    print("[PATCH] Added RECHECK diagnostic inside recheck_autotune_cache()")


TRITON_HEURISTICS.write_text(s)


# ============================================================
# COORDINATE DESCENT TUNER
# ============================================================

print()
print("=" * 70)
print("PATCH 2/2 — coordinate_descent_tuner.py")
print("=" * 70)

if not COORDESC_TUNER.exists():
    fail(f"File not found: {COORDESC_TUNER}")

s = COORDESC_TUNER.read_text()


s = replace_once(
    s,
    """        found = self.lookup_in_cache(config)""",

    """        found = self.lookup_in_cache(config)
        print(f"[AUTO-TRACE-CALLFUNC] LOOKUP name={self.name}", flush=True)""",

    "trace coordinate descent cache lookup",
)


s = replace_once(
    s,
    """        if found is not None:
            log.debug""",

    """        if found is not None:
            print(f"[AUTO-TRACE-CALLFUNC] CACHED name={self.name} timing={found:.6f}", flush=True)
            log.debug""",

    "trace cached benchmark result",
)


s = replace_once(
    s,
    """        timing = func(config)""",

    """        print(f"[AUTO-TRACE-CALLFUNC] BENCHMARK name={self.name}", flush=True)
        timing = func(config)""",

    "trace actual benchmark",
)


s = replace_once(
    s,
    """        self.cache_benchmark_result(config, timing)""",

    """        print(f"[AUTO-TRACE-CALLFUNC] RESULT name={self.name} timing={timing:.6f}", flush=True)
        self.cache_benchmark_result(config, timing)""",

    "trace benchmark result",
)


COORDESC_TUNER.write_text(s)


# ============================================================
# STRUCTURAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("STRUCTURAL VERIFICATION")
print("=" * 70)

triton = TRITON_HEURISTICS.read_text()
coordesc = COORDESC_TUNER.read_text()


# ------------------------------------------------------------
# Required patches
# ------------------------------------------------------------

if "if len(cached_configs) == 1:" not in triton:
    fail("cached_configs patch missing")

if "compile_result.config.found_by_coordesc = True" not in triton:
    fail("found_by_coordesc=True patch missing")

if TRACE not in triton:
    fail("RECHECK diagnostic missing")


# ------------------------------------------------------------
# CRITICAL CHECK:
# RECHECK must be inside recheck_autotune_cache()
# and NOT inside __init__()
# ------------------------------------------------------------

recheck_start = triton.find(
    "    def recheck_autotune_cache("
)

trace_pos = triton.find(
    '[RECHECK] name={self.fn.__name__}'
)

if recheck_start == -1:
    fail("recheck_autotune_cache() not found")

if trace_pos == -1:
    fail("RECHECK diagnostic not found")

next_method = triton.find(
    "\n    def ",
    recheck_start + len("    def recheck_autotune_cache("),
)

if next_method == -1:
    next_method = len(triton)

if not (
    recheck_start < trace_pos < next_method
):
    fail(
        "RECHECK diagnostic is NOT inside "
        "recheck_autotune_cache()"
    )


# ------------------------------------------------------------
# Coordinate descent diagnostics
# ------------------------------------------------------------

for marker in [
    "[AUTO-TRACE-CALLFUNC] LOOKUP",
    "[AUTO-TRACE-CALLFUNC] CACHED",
    "[AUTO-TRACE-CALLFUNC] BENCHMARK",
    "[AUTO-TRACE-CALLFUNC] RESULT",
]:
    if marker not in coordesc:
        fail(f"Missing diagnostic: {marker}")


print("[VERIFY] cached_configs condition       OK")
print("[VERIFY] found_by_coordesc=True         OK")
print("[VERIFY] RECHECK diagnostic             OK")
print("[VERIFY] RECHECK location               OK")
print("[VERIFY] coordinate descent diagnostics OK")


# ============================================================
# PYTHON COMPILE CHECK
# ============================================================

print()
print("=" * 70)
print("PYTHON SYNTAX CHECK")
print("=" * 70)

try:
    py_compile.compile(
        str(TRITON_HEURISTICS),
        doraise=True,
    )

    py_compile.compile(
        str(COORDESC_TUNER),
        doraise=True,
    )

except Exception as e:
    fail(f"Python syntax check failed:\n{e}")


print("[CHECK] triton_heuristics.py        OK")
print("[CHECK] coordinate_descent_tuner.py OK")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PATCH OK")
print("=" * 70)