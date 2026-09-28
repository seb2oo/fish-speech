from pathlib import Path
import ast

TARGET = Path(
    "/app/fish-speech/fish_speech/models/text2semantic/inference.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_profile_decode_call"
)

OLD = """        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(**decode_kwargs).clone()
"""

NEW = """        if i < 3:
            _decode_t0 = time.perf_counter()

        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(**decode_kwargs).clone()

        if i < 3:
            print(
                f"[PROFILE DECODE CALL] token {i}: "
                f"{time.perf_counter() - _decode_t0:.3f}s",
                flush=True,
            )
"""

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

source = TARGET.read_text()

if NEW in source:
    print("[INFO] Profiling patch already installed")
    raise SystemExit(0)

if OLD not in source:
    raise RuntimeError(
        "Expected decode_one_token block not found. "
        "Nothing was modified."
    )

if "import time" not in source:
    raise RuntimeError(
        "import time not found. Refusing to modify the file."
    )

if not BACKUP.exists():
    BACKUP.write_text(source)
    print("[OK] Backup created:", BACKUP)

patched = source.replace(OLD, NEW, 1)

ast.parse(patched)
TARGET.write_text(patched)

print("[OK] First 3 decode_one_token calls will be timed")
print("[OK] Syntax check passed")