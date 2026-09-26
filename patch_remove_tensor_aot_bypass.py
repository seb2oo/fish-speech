# patch_remove_tensor_aot_bypass.py

from pathlib import Path
import ast

TARGET = Path(
    "/app/fish-speech/fish_speech/models/text2semantic/inference.py"
)

BACKUP = TARGET.with_suffix(
    TARGET.suffix + ".backup_remove_tensor_aot_bypass"
)

MARKER = "torch.full_like(temperature, RAS_HIGH_TEMP)"


if not TARGET.exists():
    raise FileNotFoundError(f"Target not found: {TARGET}")

source = TARGET.read_text()

if MARKER in source:
    print("[INFO] Patch already installed")
    raise SystemExit(0)

if not BACKUP.exists():
    BACKUP.write_text(source)
    print(f"[OK] Backup created: {BACKUP}")
else:
    print(f"[OK] Backup already exists: {BACKUP}")


replacements = [
    (
        """    high_temp = torch.tensor(
        RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
    )
""",
        """    high_temp = torch.full_like(
        temperature, RAS_HIGH_TEMP
    )
""",
    ),
    (
        """    high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
""",
        """    high_top_p = torch.full_like(
        top_p, RAS_HIGH_TOP_P
    )
""",
    ),
    (
        """    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
""",
        """    input_pos = torch.zeros(
        1, device=hidden_states.device, dtype=torch.long
    )
""",
    ),
    (
        """        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
""",
        """        input_pos = torch.full(
            (1,),
            codebook_idx,
            device=hidden_states.device,
            dtype=torch.long,
        )
""",
    ),
]

patched = source

for old, new in replacements:
    if old not in patched:
        raise RuntimeError(
            "Expected code block not found:\n\n" + old
        )

    patched = patched.replace(old, new, 1)


ast.parse(patched)

TARGET.write_text(patched)

print("[OK] torch.tensor() replacements installed")
print("[OK] Syntax check passed")
print("[OK] Backup:", BACKUP)