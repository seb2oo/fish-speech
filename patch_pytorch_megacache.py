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

def patch_file(path: Path, replacements):
    if not path.exists():
        raise RuntimeError(f"File not found: {path}")

    source = path.read_text()

    for old, new, description in replacements:
        if new in source:
            print(f"[OK] Already patched: {description}")
            continue

        if old not in source:
            raise RuntimeError(
                f"Cannot find expected code for: {description}\n"
                f"File: {path}"
            )

        source = source.replace(old, new, 1)
        print(f"[PATCH] Applied: {description}")

    path.write_text(source)


def verify_contains(path: Path, expected, description):
    source = path.read_text()

    if expected not in source:
        raise RuntimeError(
            f"Verification failed: {description}\n"
            f"File: {path}"
        )

    print(f"[VERIFY] OK: {description}")


def compile_check(path: Path):
    print(f"[CHECK] Compiling {path.name} ...")
    py_compile.compile(
        str(path),
        doraise=True,
    )
    print(f"[CHECK] Syntax OK: {path.name}")


# ============================================================
# PATCH 1
# triton_heuristics.py
# ============================================================

print("=" * 70)
print("PATCH 1/2 — triton_heuristics.py")
print("=" * 70)

patch_file(
    TRITON_HEURISTICS,
    [
        (
            "if len(cached_configs) == 1 and len(configs) > 1:",
            "if len(cached_configs) == 1:",
            "allow cached config when only one config exists",
        ),

        (
            """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    self.compile_results = [compile_result]
                    return""",

            """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    compile_result.config.found_by_coordesc = True
                    self.compile_results = [compile_result]
                    return""",

            "mark cached config as found_by_coordesc",
        ),

        (
            """        self.autotune_cache_info = autotune_cache_info""",

            """        self.autotune_cache_info = autotune_cache_info
        print(f"[RECHECK] name={self.fn.__name__} configs={len(configs)} cached={len(cached_configs)}", flush=True)""",

            "add RECHECK diagnostic",
        ),
    ],
)


# ============================================================
# PATCH 2
# coordinate_descent_tuner.py
# ============================================================

print()
print("=" * 70)
print("PATCH 2/2 — coordinate_descent_tuner.py")
print("=" * 70)

patch_file(
    COORDESC_TUNER,
    [
        (
            """        found = self.lookup_in_cache(config)""",

            """        found = self.lookup_in_cache(config)
        print(f"[AUTO-TRACE-CALLFUNC] LOOKUP name={self.name}", flush=True)""",

            "trace coordinate descent cache lookup",
        ),

        (
            """        if found is not None:
            log.debug""",

            """        if found is not None:
            print(f"[AUTO-TRACE-CALLFUNC] CACHED name={self.name} timing={found:.6f}", flush=True)
            log.debug""",

            "trace cached benchmark result",
        ),

        (
            """        timing = func(config)""",

            """        print(f"[AUTO-TRACE-CALLFUNC] BENCHMARK name={self.name}", flush=True)
        timing = func(config)""",

            "trace actual benchmark",
        ),

        (
            """        self.cache_benchmark_result(config, timing)""",

            """        print(f"[AUTO-TRACE-CALLFUNC] RESULT name={self.name} timing={timing:.6f}", flush=True)
        self.cache_benchmark_result(config, timing)""",

            "trace benchmark result",
        ),
    ],
)


# ============================================================
# VERIFICATION — EXACT PATCH CONTENT
# ============================================================

print()
print("=" * 70)
print("VERIFYING PATCHES")
print("=" * 70)

verify_contains(
    TRITON_HEURISTICS,
    "if len(cached_configs) == 1:",
    "cached_configs condition patched",
)

verify_contains(
    TRITON_HEURISTICS,
    "compile_result.config.found_by_coordesc = True",
    "cached config marked found_by_coordesc=True",
)

verify_contains(
    TRITON_HEURISTICS,
    '[RECHECK] name={self.fn.__name__} configs={len(configs)} cached={len(cached_configs)}',
    "RECHECK diagnostic installed",
)

verify_contains(
    COORDESC_TUNER,
    '[AUTO-TRACE-CALLFUNC] LOOKUP name={self.name}',
    "CALLFUNC LOOKUP diagnostic installed",
)

verify_contains(
    COORDESC_TUNER,
    '[AUTO-TRACE-CALLFUNC] CACHED name={self.name} timing={found:.6f}',
    "CALLFUNC CACHED diagnostic installed",
)

verify_contains(
    COORDESC_TUNER,
    '[AUTO-TRACE-CALLFUNC] BENCHMARK name={self.name}',
    "CALLFUNC BENCHMARK diagnostic installed",
)

verify_contains(
    COORDESC_TUNER,
    '[AUTO-TRACE-CALLFUNC] RESULT name={self.name} timing={timing:.6f}',
    "CALLFUNC RESULT diagnostic installed",
)


# ============================================================
# PYTHON SYNTAX CHECK
# ============================================================

print()
print("=" * 70)
print("PYTHON SYNTAX CHECK")
print("=" * 70)

try:
    compile_check(TRITON_HEURISTICS)
    compile_check(COORDESC_TUNER)
except Exception as e:
    print()
    print("[ERROR] Python syntax check failed")
    print(e)
    sys.exit(1)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("PATCH COMPLETE")
print("=" * 70)

print()
print("PyTorch/TorchInductor MegaCache patch is installed correctly.")
print()
print("Expected behavior:")
print("  FXGraph cache HIT")
print("       -> recheck_autotune_cache()")
print("       -> cached=1")
print("       -> found_by_coordesc=True")
print("       -> coordinate descent should be skipped")
print()
print("PATCH OK")