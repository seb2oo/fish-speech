
#!/usr/bin/env python3

from pathlib import Path
import py_compile
import shutil
import sys
from datetime import datetime


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
    print()
    print("=" * 70)
    print("[ERROR]")
    print("=" * 70)
    print(message)
    sys.exit(1)


def backup_file(path):
    """
    Create exactly one backup before modifying a file.

    If the backup already exists, do not overwrite it.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(
        f"{path.name}.backup_{timestamp}"
    )

    try:
        shutil.copy2(path, backup)
    except Exception as e:
        fail(
            f"Impossible de sauvegarder:\n"
            f"  {path}\n"
            f"Erreur: {e}"
        )

    print(f"[BACKUP] {path}")
    print(f"         -> {backup}")

    return backup


def replace_once(source, old, new, description):
    """
    Idempotent replacement.

    - If old exists exactly once: replace it.
    - If new already exists and old doesn't: consider patched.
    - Otherwise: fail.
    """
    old_count = source.count(old)
    new_count = source.count(new)

    if old_count == 1:
        print(f"[PATCH] {description}")
        return source.replace(old, new, 1)

    if old_count == 0 and new_count >= 1:
        print(f"[OK] Already patched: {description}")
        return source

    if old_count > 1:
        fail(
            f"Plusieurs occurrences inattendues pour:\n"
            f"  {description}\n"
            f"Occurrences: {old_count}"
        )

    fail(
        f"Impossible de trouver le code attendu pour:\n"
        f"  {description}"
    )


def get_function_block(source, function_marker):
    """
    Return:

        start
        end
        function_block

    for a top-level class method/function with 4-space indentation.
    """
    start = source.find(function_marker)

    if start == -1:
        return None

    next_method = source.find(
        "\n    def ",
        start + len(function_marker),
    )

    if next_method == -1:
        end = len(source)
    else:
        end = next_method

    return start, end, source[start:end]


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("PYTORCH MEGACACHE PATCH")
print("=" * 70)

print()
print("Checking files...")

if not TRITON_HEURISTICS.exists():
    fail(
        f"Fichier introuvable:\n"
        f"  {TRITON_HEURISTICS}"
    )

if not COORDESC_TUNER.exists():
    fail(
        f"Fichier introuvable:\n"
        f"  {COORDESC_TUNER}"
    )

print("[OK] triton_heuristics.py found")
print("[OK] coordinate_descent_tuner.py found")


# ============================================================
# READ ORIGINAL FILES
# ============================================================

triton_original = TRITON_HEURISTICS.read_text()
coordesc_original = COORDESC_TUNER.read_text()


# ============================================================
# BACKUPS
# ============================================================

print()
print("=" * 70)
print("BACKUPS")
print("=" * 70)

backup_file(TRITON_HEURISTICS)
backup_file(COORDESC_TUNER)


# ============================================================
# PATCH 1/2 — TRITON HEURISTICS
# ============================================================

print()
print("=" * 70)
print("PATCH 1/2 — triton_heuristics.py")
print("=" * 70)

s = triton_original


# ------------------------------------------------------------
# PATCH A
#
# Original:
#
#     if len(cached_configs) == 1 and len(configs) > 1:
#
# New:
#
#     if len(cached_configs) == 1:
#
# This allows a cached single configuration to be reused.
# ------------------------------------------------------------

s = replace_once(
    s,
    "if len(cached_configs) == 1 and len(configs) > 1:",
    "if len(cached_configs) == 1:",
    "allow cached config when only one config exists",
)


# ------------------------------------------------------------
# PATCH B
#
# When the cached configuration matches an already compiled
# compile_result, mark it as found_by_coordesc.
#
# This prevents the later coordinate-descent benchmark storm.
# ------------------------------------------------------------

old_matching_block = """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    self.compile_results = [compile_result]
                    return"""

new_matching_block = """                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    compile_result.config.found_by_coordesc = True
                    self.compile_results = [compile_result]
                    return"""

s = replace_once(
    s,
    old_matching_block,
    new_matching_block,
    "mark cached matching config as found_by_coordesc=True",
)


# ------------------------------------------------------------
# PATCH C
#
# Add RECHECK diagnostic INSIDE recheck_autotune_cache().
#
# IMPORTANT:
# We deliberately search inside the function block.
# We never perform a global replacement.
# ------------------------------------------------------------

recheck_marker = "    def recheck_autotune_cache("

result = get_function_block(
    s,
    recheck_marker,
)

if result is None:
    fail(
        "Impossible de trouver recheck_autotune_cache()."
    )

recheck_start, recheck_end, recheck_block = result

trace_line = (
    '        print(f"[RECHECK] name={self.fn.__name__} '
    'configs={len(configs)} cached={len(cached_configs)}", flush=True)'
)

autotune_info_marker = (
    "        self.autotune_cache_info = autotune_cache_info"
)


# Remove any existing RECHECK lines INSIDE the function first.
# This makes the operation idempotent and guarantees exactly one.
lines = recheck_block.splitlines()

clean_lines = [
    line
    for line in lines
    if "[RECHECK] name={self.fn.__name__}" not in line
]

clean_recheck_block = "\n".join(clean_lines)

if recheck_block.endswith("\n"):
    clean_recheck_block += "\n"


# The diagnostic must have exactly one insertion point.
count = clean_recheck_block.count(
    autotune_info_marker
)

if count != 1:
    fail(
        "Dans recheck_autotune_cache(), le marqueur\n"
        "  self.autotune_cache_info = autotune_cache_info\n"
        f"n'apparaît pas exactement une fois.\n"
        f"Occurrences: {count}"
    )


# Insert immediately after autotune_cache_info assignment.
clean_recheck_block = clean_recheck_block.replace(
    autotune_info_marker,
    autotune_info_marker + "\n" + trace_line,
    1,
)


s = (
    s[:recheck_start]
    + clean_recheck_block
    + s[recheck_end:]
)

print(
    "[PATCH] RECHECK diagnostic placed inside "
    "recheck_autotune_cache()"
)


# ============================================================
# WRITE TRITON HEURISTICS
# ============================================================

if s != triton_original:
    TRITON_HEURISTICS.write_text(s)
    print("[OK] triton_heuristics.py written")
else:
    print("[OK] triton_heuristics.py unchanged")


# ============================================================
# PATCH 2/2 — COORDINATE DESCENT TUNER
# ============================================================

print()
print("=" * 70)
print("PATCH 2/2 — coordinate_descent_tuner.py")
print("=" * 70)

s = coordesc_original


# ------------------------------------------------------------
# PATCH D
# ------------------------------------------------------------

s = replace_once(
    s,
    """        found = self.lookup_in_cache(config)""",
    """        found = self.lookup_in_cache(config)
        print(
            f"[AUTO-TRACE-CALLFUNC] LOOKUP name={self.name}",
            flush=True,
        )""",
    "trace coordinate descent cache lookup",
)


# ------------------------------------------------------------
# PATCH E
# ------------------------------------------------------------

s = replace_once(
    s,
    """        if found is not None:
            log.debug""",
    """        if found is not None:
            print(
                f"[AUTO-TRACE-CALLFUNC] CACHED "
                f"name={self.name} timing={found:.6f}",
                flush=True,
            )
            log.debug""",
    "trace cached benchmark result",
)


# ------------------------------------------------------------
# PATCH F
# ------------------------------------------------------------

s = replace_once(
    s,
    """        timing = func(config)""",
    """        print(
            f"[AUTO-TRACE-CALLFUNC] BENCHMARK name={self.name}",
            flush=True,
        )
        timing = func(config)""",
    "trace actual benchmark",
)


# ------------------------------------------------------------
# PATCH G
# ------------------------------------------------------------

s = replace_once(
    s,
    """        self.cache_benchmark_result(config, timing)""",
    """        print(
            f"[AUTO-TRACE-CALLFUNC] RESULT "
            f"name={self.name} timing={timing:.6f}",
            flush=True,
        )
        self.cache_benchmark_result(config, timing)""",
    "trace benchmark result",
)


# ============================================================
# WRITE COORDESC
# ============================================================

if s != coordesc_original:
    COORDESC_TUNER.write_text(s)
    print("[OK] coordinate_descent_tuner.py written")
else:
    print("[OK] coordinate_descent_tuner.py unchanged")


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
# Verify cached config condition
# ------------------------------------------------------------

if "if len(cached_configs) == 1:" not in triton:
    fail(
        "Patch cached_configs == 1 absente."
    )

print(
    "[VERIFY] cached_configs condition       OK"
)


# ------------------------------------------------------------
# Verify found_by_coordesc
# ------------------------------------------------------------

if (
    "compile_result.config.found_by_coordesc = True"
    not in triton
):
    fail(
        "Patch found_by_coordesc=True absente."
    )

print(
    "[VERIFY] found_by_coordesc=True         OK"
)


# ------------------------------------------------------------
# Verify recheck function
# ------------------------------------------------------------

result = get_function_block(
    triton,
    recheck_marker,
)

if result is None:
    fail(
        "recheck_autotune_cache() introuvable "
        "pendant la vérification."
    )

recheck_start, recheck_end, recheck_block = result


trace_count = recheck_block.count(
    '[RECHECK] name={self.fn.__name__}'
)

if trace_count != 1:
    fail(
        "Le diagnostic RECHECK doit apparaître "
        "exactement une fois dans recheck_autotune_cache().\n"
        f"Occurrences trouvées: {trace_count}"
    )

print(
    "[VERIFY] RECHECK diagnostic             OK"
)
print(
    "[VERIFY] RECHECK location               OK"
)


# ------------------------------------------------------------
# IMPORTANT:
# Make sure there is NO RECHECK diagnostic outside the function.
# ------------------------------------------------------------

outside_before = triton[:recheck_start]
outside_after = triton[recheck_end:]

outside = outside_before + outside_after

if "[RECHECK] name={self.fn.__name__}" in outside:
    fail(
        "Un ancien diagnostic RECHECK existe encore "
        "en dehors de recheck_autotune_cache()."
    )

print(
    "[VERIFY] no RECHECK diagnostic outside   OK"
)


# ------------------------------------------------------------
# Verify coordinate descent diagnostics
# ------------------------------------------------------------

required_coordesc_markers = [
    "[AUTO-TRACE-CALLFUNC] LOOKUP",
    "[AUTO-TRACE-CALLFUNC] CACHED",
    "[AUTO-TRACE-CALLFUNC] BENCHMARK",
    "[AUTO-TRACE-CALLFUNC] RESULT",
]

for marker in required_coordesc_markers:
    if marker not in coordesc:
        fail(
            f"Diagnostic manquant:\n"
            f"  {marker}"
        )

print(
    "[VERIFY] coordinate descent diagnostics OK"
)


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
    fail(
        f"Python syntax check failed:\n{e}"
    )

print(
    "[CHECK] triton_heuristics.py        OK"
)

print(
    "[CHECK] coordinate_descent_tuner.py OK"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PATCH OK")
print("=" * 70)
print()
print("Les deux fichiers PyTorch ont été patchés et vérifiés.")
print("Les backups ont été créés avant modification.")
print()

