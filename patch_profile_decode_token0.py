from pathlib import Path
import ast

TARGET = Path(
    "/app/fish-speech/fish_speech/models/text2semantic/inference.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_profile_token0"
)

OLD = """        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(**decode_kwargs).clone()

"""

NEW = """        if i == 0:
            import torch.profiler as _torch_profiler
            _prof = _torch_profiler.profile(
                activities=[
                    _torch_profiler.ProfilerActivity.CPU,
                    _torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            )
            _prof.__enter__()

        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(**decode_kwargs).clone()

        if i == 0:
            _prof.__exit__(None, None, None)
            print(
                "[PROFILE] Top operations:",
                flush=True,
            )
            print(
                _prof.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=30,
                ),
                flush=True,
            )

"""

if not TARGET.exists():
    raise FileNotFoundError(TARGET)

source = TARGET.read_text()

if NEW in source:
    print("[INFO] Patch already installed")
    raise SystemExit(0)

if OLD not in source:
    raise RuntimeError(
        "Expected decode_one_token block not found."
    )

if not BACKUP.exists():
    BACKUP.write_text(source)
    print("[OK] Backup created:", BACKUP)

patched = source.replace(OLD, NEW, 1)

ast.parse(patched)
TARGET.write_text(patched)
