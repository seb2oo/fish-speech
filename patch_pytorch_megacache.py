#!/usr/bin/env python3

from pathlib import Path
import py_compile
import sys


# ============================================================
# CONFIGURATION
# ============================================================

TRITON_HEURISTICS = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py"
)

COORDESC_TUNER = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/coordinate_descent_tuner.py"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print(f"\n[ERROR] {message}")
    sys.exit(1)


def replace_exact(source, old, new, description):
    count = source.count(old)

    if count == 0:
        # Already patched?
        if new in source:
            print(f"[OK] Already patched: {description}")
            return source

        fail(
            f"Cannot find expected block for:\n"
            f"  {description}"
        )

    if count != 1:
        fail(
            f"Expected exactly 1 occurrence for:\n"
            f"  {description}\n"
            f"Found: {count}"
        )

    source = source.replace(old, new, 1)

    print(f"[PATCH] {description}")
    return source


# ============================================================
# PATCH triton_heuristics.py
# ============================================================

print("=" * 70)
print("PATCH 1/2 — triton_heuristics.py")
print("=" * 70)

if not TRITON_HEURISTICS.exists():
    fail(f"File not found: {TRITON_HEURISTICS}")

s = TRITON_HEURISTICS.read_text()


# ------------------------------------------------------------
# 1. Allow cached config even when only one config exists
# ------------------------------------------------------------

s = replace_exact(
    s,
    "if len(cached_configs) == 1 and len(configs) > 1:",
    "if len(cached_configs) == 1:",
    "allow cached config when only one config exists",
)


# ------------------------------------------------------------
# 2. Mark the cached config as already coordinate-descent tuned
# ------------------------------------------------------------

old = """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    self.compile_results = [compile_result]
                    return"""

new = """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    compile_result.config.found_by_coordesc = True
                    self.compile_results = [compile_result]
                    return"""

s = replace_exact(
    s,
    old,
    new,
    "mark cached config as found_by_coordesc=True",
)


# ------------------------------------------------------------
# 3. Add RECHECK diagnostic ONLY inside recheck_autotune_cache()
# ------------------------------------------------------------

old = """    def recheck_autotune_cache(
        self, reload_kernel_from_src: Callable[[], CachingAutotuner]
    ) -> None:
        assert self.is_statically_launchable()

        configs = [result.config for result in self.compile_results]

        (cached_configs, _, autotune_cache_info) = check_autotune_cache(
            configs, self.filename, self.inductor_meta
        )
        self.autotune_cache_info = autotune_cache_info
"""

new = """    def recheck_autotune_cache(
        self, reload_kernel_from_src: Callable[[], CachingAutotuner]
    ) -> None:
        assert self.is_statically_launchable()

        configs = [result.config for result in self.compile_results]

        (cached_configs, _, autotune_cache_info) = check_autotune_cache(
            configs, self.filename, self.inductor_meta
        )
        self.autotune_cache_info = autotune_cache_info
        print(f"[RECHECK] name={self.fn.__name__} configs={len(configs)} cached={len(cached_configs)}", flush=True)
"""

s = replace_exact(
    s,
    old,
    new,
    "add RECHECK diagnostic inside recheck_autotune_cache()",
)


TRITON_HEURISTICS.write_text(s)


# ============================================================
# PATCH coordinate_descent_tuner.py
# ============================================================

print()
print("=" * 70)
print("PATCH 2/2 — coordinate_descent_tuner.py")
print("=" * 70)

if not COORDESC_TUNER.exists():
    fail(f"File not found: {COORDESC_TUNER}")

s = COORDESC_TUNER.read_text()


# ------------------------------------------------------------
# 1. LOOKUP
# ------------------------------------------------------------

s = replace_exact(
    s,
    """        found = self.lookup_in_cache(config)""",

    """        found = self.lookup_in_cache(config)
        print(f"[AUTO-TRACE-CALLFUNC] LOOKUP name={self.name}", flush=True)""",

    "trace coordinate descent cache lookup",
)


# ------------------------------------------------------------
# 2. CACHED
# ------------------------------------------------------------

s = replace_exact(
    s,
    """        if found is not None:
            log.debug""",

    """        if found is not None:
            print(f"[AUTO-TRACE-CALLFUNC] CACHED name={self.name} timing={found:.6f}", flush=True)
            log.debug""",

    "trace cached benchmark result",
)


# ------------------------------------------------------------
# 3. BENCHMARK
# ------------------------------------------------------------

s = replace_exact(
    s,
    """        timing = func(config)""",

    """        print(f"[AUTO-TRACE-CALLFUNC] BENCHMARK name={self.name}", flush=True)
        timing = func(config)""",

    "trace actual benchmark",
)


# ------------------------------------------------------------
# 4. RESULT
# ------------------------------------------------------------

s = replace_exact(
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
# Verify desired patches
# ------------------------------------------------------------

if "if len(cached_configs) == 1:" not in triton:
    fail("cached_configs condition patch is missing")

if "compile_result.config.found_by_coordesc = True" not in triton:
    fail("found_by_coordesc=True patch is missing")

if (
    '[RECHECK] name={self.fn.__name__} '
    'configs={len(configs)} cached={len(cached_configs)}'
    not in triton
):
    fail("RECHECK diagnostic is missing")


# ------------------------------------------------------------
# CRITICAL: verify RECHECK is NOT inside __init__
# ------------------------------------------------------------

init_start = triton.find("    def __init__(")
recheck_start = triton.find("    def recheck_autotune_cache(")
trace_pos = triton.find("[RECHECK]")

if trace_pos == -1:
    fail("RECHECK diagnostic not found")

if init_start != -1 and recheck_start != -1:
    if init_start < trace_pos < recheck_start:
        fail(
            "RECHECK diagnostic was inserted inside __init__ "
            "instead of recheck_autotune_cache()"
        )


# ------------------------------------------------------------
# Verify coordinate descent traces
# ------------------------------------------------------------

required_coordesc = [
    '[AUTO-TRACE-CALLFUNC] LOOKUP',
    '[AUTO-TRACE-CALLFUNC] CACHED',
    '[AUTO-TRACE-CALLFUNC] BENCHMARK',
    '[AUTO-TRACE-CALLFUNC] RESULT',
]

for marker in required_coordesc:
    if marker not in coordesc:
        fail(f"Missing diagnostic: {marker}")


print("[VERIFY] cached_configs condition       OK")
print("[VERIFY] found_by_coordesc=True         OK")
print("[VERIFY] RECHECK diagnostic             OK")
print("[VERIFY] RECHECK location               OK")
print("[VERIFY] coordinate descent diagnostics OK")


# ============================================================
# PYTHON SYNTAX CHECK
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


print("[CHECK] triton_heuristics.py       OK")
print("[CHECK] coordinate_descent_tuner.py OK")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PATCH OK")
print("=" * 70)
print()
print("MegaCache patch installed successfully.")
print()
print("Expected flow:")
print("  FXGraph HIT")
print("      -> recheck_autotune_cache()")
print("      -> cached=1")
print("      -> found_by_coordesc=True")
print("      -> coordinate descent skipped")
print()