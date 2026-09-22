import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from fish_speech.models.text2semantic.inference import (
    init_model,
    load_codec_model,
    encode_audio,
    decode_to_audio,
    generate_long,
)


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = Path("/app/checkpoints/s2-pro")
CODEC_CHECKPOINT = CHECKPOINT / "codec.pth"

REFERENCE_AUDIO = Path("/app/fish-speech/input/fr.wav")

OUTPUT_DIR = Path("/app/fish-speech/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
PRECISION = torch.bfloat16

PROMPT_TEXT = (
    "Le rire d'un proche a le pouvoir d'effacer mes soucis. "
    "Il éclate comme une lumière claire et me remplit de joie. "
    "Dans ces instants, tout semble plus léger, "
    "et je retrouve confiance en l'avenir."
)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 80)
print("LOADING S2-PRO")
print("=" * 80)

t0 = time.perf_counter()

model, decode_one_token = init_model(
    checkpoint_path=str(CHECKPOINT),
    device=DEVICE,
    precision=PRECISION,
    compile=False,
)

torch.cuda.synchronize()

print(f"S2-Pro loaded in {time.perf_counter() - t0:.2f}s")


print()
print("=" * 80)
print("LOADING DAC")
print("=" * 80)

t0 = time.perf_counter()

codec = load_codec_model(
    codec_checkpoint_path=str(CODEC_CHECKPOINT),
    device=DEVICE,
    precision=PRECISION,
)

torch.cuda.synchronize()

print(f"DAC loaded in {time.perf_counter() - t0:.2f}s")


# ============================================================
# LOAD / ENCODE REFERENCE VOICE
# ============================================================

print()
print("=" * 80)
print("ENCODING REFERENCE VOICE")
print("=" * 80)

REFERENCE_CODES_PATH = OUTPUT_DIR / "reconstructed.npy"

if REFERENCE_CODES_PATH.exists():

    print(f"Using cached reference codes: {REFERENCE_CODES_PATH}")

    reference_codes = torch.from_numpy(
        np.load(REFERENCE_CODES_PATH)
    )

else:

    t0 = time.perf_counter()

    reference_codes = encode_audio(
        str(REFERENCE_AUDIO),
        codec,
        DEVICE,
    ).cpu()

    np.save(
        REFERENCE_CODES_PATH,
        reference_codes.numpy(),
    )

    print(
        f"Reference encoded in "
        f"{time.perf_counter() - t0:.2f}s"
    )


print(f"Reference codes shape: {reference_codes.shape}")


# ============================================================
# READY
# ============================================================

torch.cuda.synchronize()

print()
print("=" * 80)
print("READY")
print("=" * 80)

print(
    f"GPU memory allocated: "
    f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
)

print(
    f"GPU memory reserved: "
    f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
)


# ============================================================
# GENERATION FUNCTION
# ============================================================

print("DEBUG prompt_text:", type(PROMPT_TEXT), PROMPT_TEXT)
print("DEBUG reference_codes:", type(reference_codes), reference_codes.shape)
print("DEBUG prompt_tokens:", type([reference_codes]))
print("DEBUG prompt_tokens[0]:", type([reference_codes][0]))

def generate_voice(text: str, output_path: Path):

    print()
    print("=" * 80)
    print(f"GENERATING: {text}")
    print("=" * 80)

    t_total = time.perf_counter()

    responses = generate_long(
        model=model,
        device=DEVICE,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,
        top_p=0.9,
        top_k=30,
        repetition_penalty=1.1,
        temperature=1.0,
        compile=False,
        iterative_prompt=True,
        chunk_length=512,
        prompt_text=PROMPT_TEXT,
        prompt_tokens=[reference_codes],
    )

    generated_codes = None

    for response in responses:

        if response.action == "sample":

            generated_codes = response.codes

    if generated_codes is None:
        raise RuntimeError("No audio codes were generated")

    # Move codes to CPU before decoding.
    generated_codes_cpu = generated_codes.cpu()

    torch.cuda.synchronize()

    t_generation = time.perf_counter() - t_total

    print(
        f"Semantic generation: "
        f"{t_generation:.2f}s"
    )

    print(
        f"Generated codes shape: "
        f"{generated_codes_cpu.shape}"
    )


    # ========================================================
    # DAC DECODING
    # ========================================================

    t0 = time.perf_counter()

    audio = decode_to_audio(
        generated_codes_cpu.to(DEVICE),
        codec,
    )

    torch.cuda.synchronize()

    t_decode = time.perf_counter() - t0

    # ========================================================
    # SAVE WAV
    # ========================================================

    audio_cpu = audio.float().cpu()

    torchaudio.save(
        str(output_path),
        audio_cpu.unsqueeze(0),
        codec.sample_rate,
    )

    t_total = time.perf_counter() - t_total

    print(f"DAC decode: {t_decode:.2f}s")
    print(f"Total generation: {t_total:.2f}s")
    print(f"Saved: {output_path}")

    print(
        f"GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    print(
        f"GPU memory reserved: "
        f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
    )


# ============================================================
# TEST 1
# ============================================================

generate_voice(
    "Bonjour, ceci est le premier test de génération avec ma voix clonée.",
    OUTPUT_DIR / "persistent_test_1.wav",
)


# ============================================================
# TEST 2
# ============================================================

generate_voice(
    "Maintenant nous testons une deuxième génération sans recharger le modèle.",
    OUTPUT_DIR / "persistent_test_2.wav",
)


# ============================================================
# TEST 3
# ============================================================

generate_voice(
    "Si tout fonctionne correctement, le modèle reste chargé en mémoire entre les générations.",
    OUTPUT_DIR / "persistent_test_3.wav",
)