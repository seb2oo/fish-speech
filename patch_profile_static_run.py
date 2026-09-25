from pathlib import Path
import ast


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/static_cuda_launcher.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_profile_static_run"
)


def main():
    if not TARGET.exists():
        raise SystemExit(f"[ERROR] File not found: {TARGET}")

    text = TARGET.read_text()

    # ------------------------------------------------------------
    # Already patched?
    # ------------------------------------------------------------
    if "[PROFILE-RUN]" in text:
        print("[INFO] Static run profiling already installed")
        return

    # ------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------
    if not BACKUP.exists():
        BACKUP.write_text(text)
        print(f"[OK] Backup created: {BACKUP}")
    else:
        print(f"[INFO] Backup already exists: {BACKUP}")

    # ------------------------------------------------------------
    # Exact target
    # ------------------------------------------------------------
    old = """        _StaticCudaLauncher._launch_kernel(
            self.function,
            grid_x,
            grid_y,
            grid_z,
            self.num_warps,
            self.shared,
            arg_tys,
            args,
            stream,
        )
"""

    new = """        import time

        _profile_t0 = time.perf_counter()

        _StaticCudaLauncher._launch_kernel(
            self.function,
            grid_x,
            grid_y,
            grid_z,
            self.num_warps,
            self.shared,
            arg_tys,
            args,
            stream,
        )

        _profile_dt = time.perf_counter() - _profile_t0

        # Only report unusually long kernel launches.
        # This keeps the log small even though thousands of kernels run.
        if _profile_dt > 1.0:
            print(
                f"[PROFILE-RUN] launch={_profile_dt:.6f}s "
                f"kernel={self.name} "
                f"grid=({grid_x},{grid_y},{grid_z}) "
                f"warps={self.num_warps}",
                flush=True,
            )
"""

    if old not in text:
        raise SystemExit(
            "[ERROR] Target _launch_kernel() block not found.\n"
            "The installed PyTorch file may differ from the expected version."
        )

    # ------------------------------------------------------------
    # Apply patch
    # ------------------------------------------------------------
    text = text.replace(old, new, 1)

    # ------------------------------------------------------------
    # Syntax check
    # ------------------------------------------------------------
    try:
        ast.parse(text)
    except SyntaxError as e:
        raise SystemExit(
            f"[ERROR] Syntax check failed: {e}"
        )

    TARGET.write_text(text)

    print("[OK] Static run profiling installed")
    print("[OK] Syntax check passed")
    print()
    print("The patch reports only _launch_kernel() calls taking > 1 second.")
    print("Use [PROFILE-RUN] lines to locate the long first-runtime operation.")


if __name__ == "__main__":
    main()