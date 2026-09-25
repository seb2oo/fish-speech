from pathlib import Path
import shutil
import re

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "torch/_inductor/runtime/triton_heuristics.py"
)

BACKUP = TARGET.with_name(
    TARGET.name + ".backup_profile_static_launcher"
)


def main():
    text = TARGET.read_text()

    if "[PROFILE-STATIC]" in text:
        print("[ERROR] Patch already installed.")
        return

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"[OK] Backup created: {BACKUP}")
    else:
        print(f"[OK] Backup already exists: {BACKUP}")

    # ------------------------------------------------------------
    # 1. Instrument _make_launchers()
    # ------------------------------------------------------------
    old = """        # load binary to the correct device
        with DeviceGuard(device_interface, self.triton_meta["device"]):
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())
            launchers = []
            exc = None
            for result in self.compile_results:
                try:
                    launchers.append(result.make_launcher())
"""

    new = """        # load binary to the correct device
        with DeviceGuard(device_interface, self.triton_meta["device"]):
            # PROFILE-STATIC: measure CUDA synchronization separately
            _profile_sync_t0 = time.perf_counter()
            device_interface.synchronize(device_interface.current_device())
            print(
                f"[PROFILE-STATIC] synchronize="
                f"{time.perf_counter() - _profile_sync_t0:.6f}s "
                f"kernel={self.fn.__name__}",
                flush=True,
            )

            launchers = []
            exc = None
            for result in self.compile_results:
                try:
                    _profile_launcher_t0 = time.perf_counter()
                    launcher = result.make_launcher()
                    print(
                        f"[PROFILE-STATIC] make_launcher="
                        f"{time.perf_counter() - _profile_launcher_t0:.6f}s "
                        f"kernel={self.fn.__name__} "
                        f"result={type(result).__name__}",
                        flush=True,
                    )
                    launchers.append(launcher)
"""

    if old not in text:
        raise RuntimeError(
            "Could not find _make_launchers() target block."
        )

    text = text.replace(old, new, 1)

    # ------------------------------------------------------------
    # 2. Instrument StaticTritonCompileResult.make_launcher()
    # ------------------------------------------------------------
    old = """        device = self.compile_meta.get("device", 0)
        if device is None:
            device = 0
        self.kernel.load_kernel(device)
        scope = {
            "runner": self.kernel.run,
        }
"""

    new = """        device = self.compile_meta.get("device", 0)
        if device is None:
            device = 0

        # PROFILE-STATIC: isolate the actual CUDA kernel loading time
        _profile_load_t0 = time.perf_counter()
        self.kernel.load_kernel(device)
        print(
            f"[PROFILE-STATIC] load_kernel="
            f"{time.perf_counter() - _profile_load_t0:.6f}s "
            f"kernel={self.kernel.name} "
            f"hash={self.kernel.hash}",
            flush=True,
        )

        scope = {
            "runner": self.kernel.run,
        }
"""

    if old not in text:
        raise RuntimeError(
            "Could not find StaticTritonCompileResult.make_launcher() "
            "load_kernel block."
        )

    text = text.replace(old, new, 1)

    # ------------------------------------------------------------
    # 3. Instrument reload_cubin_path()
    # ------------------------------------------------------------
    old = """        if not os.path.exists(cubin_location):
            if self.kernel.cubin_raw is not None:
                # We saved the raw cubin, so write it to he appropriate location
                self.kernel.reload_cubin_from_raw(cubin_location)
            else:
                raise RuntimeError(
                    "Cubin file saved by TritonBundler not found at %s", cubin_location
                )
        self.kernel.cubin_path = cubin_location
"""

    new = """        _profile_reload_t0 = time.perf_counter()

        if not os.path.exists(cubin_location):
            if self.kernel.cubin_raw is not None:
                # We saved the raw cubin, so write it to he appropriate location
                self.kernel.reload_cubin_from_raw(cubin_location)
            else:
                raise RuntimeError(
                    "Cubin file saved by TritonBundler not found at %s", cubin_location
                )

        self.kernel.cubin_path = cubin_location

        print(
            f"[PROFILE-STATIC] reload_cubin_path="
            f"{time.perf_counter() - _profile_reload_t0:.6f}s "
            f"kernel={self.kernel.name}",
            flush=True,
        )
"""

    if old not in text:
        raise RuntimeError(
            "Could not find reload_cubin_path() target block."
        )

    text = text.replace(old, new, 1)

    TARGET.write_text(text)

    # ------------------------------------------------------------
    # Syntax check
    # ------------------------------------------------------------
    import py_compile

    py_compile.compile(str(TARGET), doraise=True)

    print("[OK] Syntax check passed")
    print("[OK] Static launcher profiling installed")


if __name__ == "__main__":
    main()