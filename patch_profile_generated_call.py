from pathlib import Path
import ast


TARGET = Path(
    "/app/torchinductor-cache/ib/"
    "cibkrtnu4g7f76uhlyblu6zyyvnybgaahrtfi2i33h5iijpz3odq.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_profile_call"
)


def main():
    if not TARGET.exists():
        raise SystemExit(f"[ERROR] Generated file not found: {TARGET}")

    text = TARGET.read_text()

    if "[PROFILE-CALL]" in text:
        print("[INFO] Generated call profiling already installed")
        return

    if not BACKUP.exists():
        BACKUP.write_text(text)
        print(f"[OK] Backup created: {BACKUP}")
    else:
        print(f"[INFO] Backup already exists: {BACKUP}")

    # ------------------------------------------------------------
    # Locate def call(args):
    # ------------------------------------------------------------
    marker = "def call(args):\n"

    if marker not in text:
        raise SystemExit("[ERROR] def call(args) not found")

    # We insert the tracer immediately after the huge argument
    # unpacking line. This avoids tracing module initialization.
    #
    # Find the beginning of the first executable line after:
    #
    #     arg0_1, ... = args
    #
    call_pos = text.index(marker)

    # Find the first line after "def call(args):"
    first_line_start = call_pos + len(marker)

    # The generated file has one enormous argument-unpacking line.
    unpack_end = text.find("\n", first_line_start)

    if unpack_end == -1:
        raise SystemExit("[ERROR] Could not locate args unpacking line")

    insert_pos = unpack_end + 1

    tracer = r'''    import time as _profile_time

    _profile_last_t = _profile_time.perf_counter()
    _profile_last_line = None

    def _profile_trace(frame, event, arg):
        nonlocal _profile_last_t, _profile_last_line

        if event != "line":
            return _profile_trace

        _profile_now = _profile_time.perf_counter()
        _profile_dt = _profile_now - _profile_last_t

        if _profile_dt > 0.5:
            print(
                f"[PROFILE-CALL] gap={_profile_dt:.6f}s "
                f"line={frame.f_lineno} "
                f"code={frame.f_code.co_name}",
                flush=True,
            )

        _profile_last_t = _profile_now
        _profile_last_line = frame.f_lineno

        return _profile_trace

    _profile_old_trace = __import__("sys").gettrace()
    __import__("sys").settrace(_profile_trace)
'''

    text = text[:insert_pos] + tracer + text[insert_pos:]

    # ------------------------------------------------------------
    # Restore tracing immediately before leaving call().
    #
    # We locate the end of call() by finding the next top-level
    # function definition after "def call(args):".
    # ------------------------------------------------------------
    lines = text.splitlines(keepends=True)

    call_line_index = None

    for i, line in enumerate(lines):
        if line.startswith("def call(args):"):
            call_line_index = i
            break

    if call_line_index is None:
        raise SystemExit("[ERROR] call() line not found after modification")

    end_call_index = None

    for i in range(call_line_index + 1, len(lines)):
        line = lines[i]

        if line.startswith("def ") or line.startswith("class "):
            end_call_index = i
            break

    if end_call_index is None:
        end_call_index = len(lines)

    # Insert restoration immediately before the next top-level
    # definition.
    restore = """
    __import__("sys").settrace(_profile_old_trace)
    print("[PROFILE-CALL] call() finished", flush=True)
"""

    lines.insert(end_call_index, restore)

    text = "".join(lines)

    # ------------------------------------------------------------
    # Syntax check
    # ------------------------------------------------------------
    try:
        ast.parse(text)
    except SyntaxError as e:
        raise SystemExit(f"[ERROR] Syntax check failed: {e}")

    TARGET.write_text(text)

    print("[OK] Generated call profiling installed")
    print("[OK] Syntax check passed")
    print()
    print("Only gaps > 0.5 seconds between Python lines are reported.")
    print()
    print("IMPORTANT:")
    print("This patch targets ONLY the generated FXGraph Python file.")
    print("It does not modify PyTorch or Triton runtime code.")


if __name__ == "__main__":
    main()