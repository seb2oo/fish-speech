import os
import subprocess
import time
import base64
import io

import numpy as np
import soundfile as sf
os.makedirs("/runpod-volume/torchinductor-cache", exist_ok=True)
import torch
import runpod

import sys
import contextlib


# ============================================================
# CONFIG
# ============================================================

REPO_PATH = "/app/fish-speech"

CHECKPOINT_PATH = "/app/checkpoints/s2-pro"

REFERENCE_WAV = "/app/fish-speech/input/fr.wav"
REFERENCE_TOKENS = "/app/fish-speech/output/reconstructed.npy"

DEVICE = "cuda"
PRECISION = torch.bfloat16

COMPILE = True

PROMPT_TEXT = (
    "Le rire d'un proche a le pouvoir d'effacer mes soucis. "
    "Il éclate comme une lumière claire et me remplit de joie. "
    "Dans ces instants, tout semble plus léger, "
    "et je retrouve confiance en l'avenir."
)

TEMPERATURE = 0.7
TOP_P = 0.7
TOP_K = 30
REPETITION_PENALTY = 1.1

CHUNK_LENGTH = 300


# ============================================================
# HELPERS
# ============================================================

def run_command(command):
    print(f"[COMMAND] {command}")

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_PATH + ":" + env.get("PYTHONPATH", "")

    subprocess.run(
        command,
        shell=True,
        check=True,
        env=env,
    )


# ============================================================
# CACHED MODEL
# ============================================================

from pathlib import Path

MODEL_ID = "fishaudio/s2-pro"

def find_cached_model(model_id):

    org, name = model_id.split("/", 1)

    cache_dir = Path(
        "/runpod-volume/huggingface-cache/hub"
    )

    model_dir = cache_dir / f"models--{org}--{name}"
    snapshots_dir = model_dir / "snapshots"

    if not snapshots_dir.exists():
        raise RuntimeError(
            f"HF cache not found: {snapshots_dir}"
        )

    snapshots = [
        p for p in snapshots_dir.iterdir()
        if p.is_dir()
    ]

    if not snapshots:
        raise RuntimeError(
            f"No cached snapshot found for {model_id}"
        )

    if len(snapshots) > 1:
        print(f"Found {len(snapshots)} cached snapshots.")

    return str(snapshots[0])


CHECKPOINT_PATH = find_cached_model(MODEL_ID)

print("S2-Pro cached model found:")
print(CHECKPOINT_PATH)

# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap():

    print("=" * 70)
    print("FISH SPEECH SERVERLESS BOOTSTRAP")
    print("=" * 70)

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    # no need anymore because we use cached model now
    # os.makedirs(
    #     "/app/checkpoints/s2-pro",
    #     exist_ok=True,
    # )


    # --------------------------------------------------------
    # CLONE FISH SPEECH
    # --------------------------------------------------------

    if not os.path.exists(
        os.path.join(REPO_PATH, ".git")
    ):

        print("Fish Speech repository not found.")

        if os.path.exists(REPO_PATH):
            print(
                f"Removing incomplete directory: {REPO_PATH}"
            )

            run_command(
                f"rm -rf '{REPO_PATH}'"
            )

        print("Cloning Fish Speech repository...")

        run_command(
            "git clone -b docker "
            "https://github.com/seb2oo/fish-speech.git "
            f"'{REPO_PATH}'"
        )

    else:

        print(
            "Fish Speech repository already exists."
        )

    
    # --------------------------------------------------------
    # CREATE RUNTIME OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        "/app/fish-speech/output",
        exist_ok=True,
    )


    # Make Fish Speech available to this Python process
    if REPO_PATH not in sys.path:
        sys.path.insert(0, REPO_PATH)

    # --------------------------------------------------------
    # DOWNLOAD CHECKPOINT
    # --------------------------------------------------------

    codec_path = os.path.join(
        CHECKPOINT_PATH,
        "codec.pth",
    )

    if not os.path.exists(codec_path):

        # print("Downloading S2-Pro checkpoint...")

        # run_command(
        #     "hf download fishaudio/s2-pro "
        #     "--local-dir /app/checkpoints/s2-pro"
        # )
        raise RuntimeError(
            f"codec.pth not found in cached S2-Pro model: {CHECKPOINT_PATH}"
        )

    else:

        print(
            "S2-Pro checkpoint already exists."
        )

    # --------------------------------------------------------
    # CREATE REFERENCE TOKENS
    # --------------------------------------------------------

    if not os.path.exists(REFERENCE_TOKENS):

        print(
            "Reference tokens not found."
        )

        if not os.path.exists(REFERENCE_WAV):
            raise FileNotFoundError(
                f"Reference WAV not found: {REFERENCE_WAV}"
            )

        print(
            "Encoding reference voice with DAC..."
        )

        run_command(
            "python3 -m fish_speech.models.dac.inference "
            f"-i '{REFERENCE_WAV}' "
            f"-o '{os.path.join('/app/fish-speech/output', 'reconstructed.wav')}' "
            f"--checkpoint-path '{os.path.join(CHECKPOINT_PATH, 'codec.pth')}' "
            "-d cuda"
        )

        # DAC inference creates reconstructed.npy next
        # to the output WAV.

        if not os.path.exists(REFERENCE_TOKENS):

            raise RuntimeError(
                "DAC finished but reconstructed.npy "
                "was not created."
            )

        print(
            "Reference tokens created."
        )

    else:

        print(
            "Reference tokens already exist. "
            "Skipping DAC encoding."
        )


# ============================================================
# STARTUP
# ============================================================

startup_start = time.perf_counter()

bootstrap()

print("=" * 70)
print("IMPORTING FISH SPEECH")
print("=" * 70)

from fish_speech.models.text2semantic.inference import (
    init_model,
    load_codec_model,
    generate_long,
    decode_to_audio,
)


# ============================================================
# LOAD S2-PRO
# ============================================================

print("=" * 70)
print("LOADING S2-PRO")
print("=" * 70)

t0 = time.perf_counter()

model, decode_one_token = init_model(
    CHECKPOINT_PATH,
    DEVICE,
    PRECISION,
    compile=COMPILE,
)

print(
    f"S2-Pro loaded in "
    f"{time.perf_counter() - t0:.2f}s"
)


# ============================================================
# LOAD DAC
# ============================================================

print("=" * 70)
print("LOADING DAC")
print("=" * 70)

t0 = time.perf_counter()

codec = load_codec_model(
    os.path.join(
        CHECKPOINT_PATH,
        "codec.pth",
    ),
    DEVICE,
    PRECISION,
)

print(
    f"DAC loaded in "
    f"{time.perf_counter() - t0:.2f}s"
)


# ============================================================
# LOAD REFERENCE TOKENS
# ============================================================

print("=" * 70)
print("LOADING REFERENCE VOICE")
print("=" * 70)

reference_codes = torch.from_numpy(
    np.load(REFERENCE_TOKENS)
)

print(
    f"Reference codes shape: "
    f"{reference_codes.shape}"
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()

    print(
        f"GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    print(
        f"GPU memory reserved: "
        f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
    )


# ============================================================
# WORKER READY
# ============================================================

startup_time = time.perf_counter() - startup_start

print("=" * 70)
print("WORKER READY")
print("=" * 70)

print(
    f"Total startup time: "
    f"{startup_time:.2f}s"
)

print(
    f"torch.compile: "
    f"{COMPILE}"
)

print("=" * 70)


# ============================================================
# GENERATE AUDIO
# ============================================================
# ============================================================
# GENERATE AUDIO
# ============================================================

def generate_audio(text):

    print("=" * 70)
    print("GENERATING AUDIO")
    print("=" * 70)

    print(
        f"Text: {text}"
    )

    generation_start = time.perf_counter()

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    # Fish Speech / TorchInductor can produce a huge amount
    # of internal output during torch.compile.
    #
    # We intentionally hide stdout/stderr here so that:
    #
    # - "Compiling function..."
    # - Triton warnings
    # - 32449-step progress bars
    # - internal TorchInductor messages
    #
    # do not pollute the RunPod logs.
    #
    # Our own logs remain visible outside this block.

    with open(os.devnull, "w") as devnull:

        with contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):

            generator = generate_long(
                model=model,
                device=DEVICE,
                decode_one_token=decode_one_token,

                text=text,

                num_samples=1,

                max_new_tokens=0,

                top_p=TOP_P,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,

                compile=COMPILE,

                iterative_prompt=True,
                chunk_length=CHUNK_LENGTH,

                prompt_text=[
                    PROMPT_TEXT
                ],

                prompt_tokens=[
                    reference_codes
                ],
            )

            generated_codes = []

            for response in generator:

                if response.action == "sample":

                    generated_codes.append(
                        response.codes
                    )

    if not generated_codes:

        raise RuntimeError(
            "Fish Speech generated no audio codes."
        )

    generated_codes = torch.cat(
        generated_codes,
        dim=1,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    num_tokens = generated_codes.shape[1]

    print(
        f"Generated tokens: {num_tokens}"
    )

    print(
        f"Generation time: "
        f"{generation_time:.2f}s"
    )

    if generation_time > 0:

        print(
            f"Generation speed: "
            f"{num_tokens / generation_time:.2f} tok/s"
        )


    # --------------------------------------------------------
    # DAC DECODE
    # --------------------------------------------------------

    decode_start = time.perf_counter()

    audio = decode_to_audio(
        generated_codes.to(DEVICE),
        codec,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    decode_time = (
        time.perf_counter()
        - decode_start
    )

    print(
        f"DAC decode time: "
        f"{decode_time:.2f}s"
    )


    # --------------------------------------------------------
    # WAV IN MEMORY
    # --------------------------------------------------------

    wav_buffer = io.BytesIO()

    audio_numpy = (
        audio
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    sf.write(
        wav_buffer,
        audio_numpy,
        codec.sample_rate,
        format="WAV",
    )

    wav_bytes = wav_buffer.getvalue()

    # --------------------------------------------------------
    # TORCHINDUCTOR CACHE DIAGNOSTIC
    # --------------------------------------------------------

    cache_dir = os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        "/runpod-volume/torchinductor-cache",
    )

    print("=" * 70)
    print("TORCHINDUCTOR CACHE")
    print("=" * 70)

    print(
        f"Cache directory: {cache_dir}"
    )

    if os.path.exists(cache_dir):

        total_size = 0
        file_count = 0

        for root, dirs, files in os.walk(cache_dir):

            for filename in files:

                filepath = os.path.join(
                    root,
                    filename,
                )

                try:
                    total_size += os.path.getsize(
                        filepath
                    )

                    file_count += 1

                except OSError:
                    pass

        print(
            f"Cache exists: YES"
        )

        print(
            f"Cache files: {file_count}"
        )

        print(
            f"Cache size: "
            f"{total_size / (1024 * 1024):.2f} MB"
        )

    else:

        print(
            "Cache exists: NO"
        )

    print("=" * 70)



    total_time = (
        generation_time
        + decode_time
    )

    duration = (
        len(audio_numpy)
        / codec.sample_rate
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        f"Audio duration: "
        f"{duration:.2f}s"
    )

    print(
        f"Total inference time: "
        f"{total_time:.2f}s"
    )

    print("=" * 70)

    return {
        "audio_base64": base64.b64encode(
            wav_bytes
        ).decode("utf-8"),

        "sample_rate": codec.sample_rate,

        "duration": duration,

        "num_tokens": num_tokens,

        "generation_time": generation_time,

        "decode_time": decode_time,

        "total_time": total_time,
    }


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):

    """
    Expected request:

    {
        "input": {
            "text": "Bonjour, ceci est un test."
        }
    }

    For this diagnostic test, the same worker performs
    3 consecutive generations.
    """

    job_input = job.get(
        "input",
        {},
    )

    text = job_input.get(
        "text"
    )

    if not text:

        raise ValueError(
            "Missing required input: 'text'"
        )


    # --------------------------------------------------------
    # TEST TEXTS
    # --------------------------------------------------------

    test_texts = [
        text,

        "Ceci est la deuxième génération effectuée "
        "par le même worker Fish Speech.",

        "Et ceci est la troisième génération. "
        "Le modèle devrait maintenant être complètement chaud.",
    ]


    # --------------------------------------------------------
    # RUN 3 GENERATIONS
    # --------------------------------------------------------

    print("=" * 70)
    print("RUNNING 3 CONSECUTIVE GENERATIONS")
    print("=" * 70)

    results = []

    for i, test_text in enumerate(test_texts, start=1):

        print()
        print("=" * 70)
        print(f"GENERATION {i}/3")
        print("=" * 70)

        result = generate_audio(test_text)

        results.append(result)

        print(
            f"Generation {i}/3 finished in "
            f"{result['total_time']:.2f}s"
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3-GENERATION TEST SUMMARY")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(
            f"Generation {i}: "
            f"{result['num_tokens']} tokens | "
            f"{result['generation_time']:.2f}s generation | "
            f"{result['generation_time'] / result['num_tokens']:.2f}s/token | "
            f"{result['num_tokens'] / result['generation_time']:.2f} tok/s | "
            f"{result['duration']:.2f}s audio"
        )

    print("=" * 70)


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    # Return the audio from the LAST generation.
    # The statistics of all 3 generations are returned
    # so we can compare warm-up vs steady-state performance.

    return {
        "audio_base64": results[-1]["audio_base64"],

        "sample_rate": results[-1]["sample_rate"],

        "duration": results[-1]["duration"],

        "num_tokens": results[-1]["num_tokens"],

        "generation_time": results[-1]["generation_time"],

        "decode_time": results[-1]["decode_time"],

        "total_time": results[-1]["total_time"],

        "test_generations": [
            {
                "generation": i + 1,
                "duration": result["duration"],
                "num_tokens": result["num_tokens"],
                "generation_time": result["generation_time"],
                "decode_time": result["decode_time"],
                "total_time": result["total_time"],
            }
            for i, result in enumerate(results)
        ],
    }


# ============================================================
# START RUNPOD SERVERLESS
# ============================================================

if __name__ == "__main__":

    runpod.serverless.start(
        {
            "handler": handler
        }
    )