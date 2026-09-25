from pathlib import Path
import shutil
import time
import py_compile


PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "torch/_inductor/runtime/triton_heuristics.py"
)

BACKUP = PATH.with_name(PATH.name + ".backup_profile_precompile")


# ---------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------

if not BACKUP.exists():
    shutil.copy2(PATH, BACKUP)
    print(f"[OK] Backup created: {BACKUP}")


# ---------------------------------------------------------------------
# Read source
# ---------------------------------------------------------------------

text = PATH.read_text()

start = text.find("    def precompile(\n")

if start < 0:
    raise RuntimeError("precompile() not found")

next_def = text.find("\n    def ", start + len("    def precompile(\n"))

if next_def < 0:
    raise RuntimeError("End of precompile() not found")

block = text[start:next_def]


# ---------------------------------------------------------------------
# Prevent double patch
# ---------------------------------------------------------------------

if "[PROFILE-PRECOMPILE]" in block:
    print("[OK] Profiling already installed")
else:

    old = """            self._precompile_worker()
            if static_triton_bundle_key is not None and self.is_statically_launchable():
                TritonBundler.put_static_autotuner(static_triton_bundle_key, self)
            self._make_launchers()
            self._dynamic_scale_rblock()
"""

    new = """            _t0 = time.perf_counter()
            self._precompile_worker()
            print(
                f"[PROFILE-PRECOMPILE] worker={time.perf_counter() - _t0:.6f}s "
                f"kernel={self.fn.__name__}",
                flush=True,
            )

            if static_triton_bundle_key is not None and self.is_statically_launchable():
                _t0 = time.perf_counter()
                TritonBundler.put_static_autotuner(
                    static_triton_bundle_key,
                    self,
                )
                print(
                    f"[PROFILE-PRECOMPILE] static_bundle="
                    f"{time.perf_counter() - _t0:.6f}s "
                    f"kernel={self.fn.__name__}",
                    flush=True,
                )

            _t0 = time.perf_counter()
            self._make_launchers()
            print(
                f"[PROFILE-PRECOMPILE] make_launchers="
                f"{time.perf_counter() - _t0:.6f}s "
                f"kernel={self.fn.__name__}",
                flush=True,
            )

            _t0 = time.perf_counter()
            self._dynamic_scale_rblock()
            print(
                f"[PROFILE-PRECOMPILE] dynamic_scale="
                f"{time.perf_counter() - _t0:.6f}s "
                f"kernel={self.fn.__name__}",
                flush=True,
            )
"""

    if old not in block:
        raise RuntimeError(
            "Expected precompile() block not found. "
            "The installed PyTorch source may have changed."
        )

    block = block.replace(old, new, 1)

    text = text[:start] + block + text[next_def:]

    # time is needed by the injected profiling code.
    if not any(
        line.strip() == "import time"
        for line in text.splitlines()
    ):
        text = "import time\n" + text

    PATH.write_text(text)

    print("[OK] Profiling installed")


# ---------------------------------------------------------------------
# Syntax check
# ---------------------------------------------------------------------

py_compile.compile(str(PATH), doraise=True)

print("[OK] Syntax check passed")
print("[PATCH OK]")