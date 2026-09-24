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

# ------------------------------------------------------------
# EXTENDED CACHE WARM-UP
# ------------------------------------------------------------

RUN_EXTENDED_TESTS = True

# True = sauvegarde tous les WAV
# False = fait les générations mais ne conserve pas les WAV
SAVE_AUDIO = True

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
    compile=True,
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

print(
    "DEBUG prompt_text:",
    type(PROMPT_TEXT),
    PROMPT_TEXT
)

print(
    "DEBUG reference_codes:",
    type(reference_codes),
    reference_codes.shape
)

print(
    "DEBUG prompt_tokens:",
    type([reference_codes])
)

print(
    "DEBUG prompt_tokens[0]:",
    type([reference_codes][0])
)


generation_counter = 0


def generate_voice(text: str, output_path: Path):

    global generation_counter

    generation_counter += 1

    print()
    print("=" * 80)
    print(
        f"GENERATION #{generation_counter}"
    )
    print("=" * 80)

    print(f"Text: {text}")
    print(f"Characters: {len(text)}")

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
        prompt_text=[PROMPT_TEXT],
        prompt_tokens=[reference_codes],
    )

    generated_codes = None

    for response in responses:

        if response.action == "sample":

            generated_codes = response.codes

    if generated_codes is None:
        raise RuntimeError(
            "No audio codes were generated"
        )

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

    if SAVE_AUDIO:

        torchaudio.save(
            str(output_path),
            audio_cpu.unsqueeze(0),
            codec.sample_rate,
        )

        print(f"Saved: {output_path}")

    else:

        print("Audio saving disabled.")


    t_total = time.perf_counter() - t_total

    print(f"DAC decode: {t_decode:.2f}s")
    print(f"Total generation: {t_total:.2f}s")

    print(
        f"GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    print(
        f"GPU memory reserved: "
        f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
    )

    return {
        "generation": generation_counter,
        "characters": len(text),
        "semantic_time": t_generation,
        "decode_time": t_decode,
        "total_time": t_total,
        "codes_shape": tuple(generated_codes_cpu.shape),
    }


# ============================================================
# BASIC TESTS
# ============================================================

BASIC_TESTS = [

    (
        "Bonjour, ceci est le premier test de génération avec ma voix clonée.",
        "persistent_test_1.wav",
    ),

    (
        "Maintenant nous testons une deuxième génération sans recharger le modèle.",
        "persistent_test_2.wav",
    ),

    (
        "Si tout fonctionne correctement, le modèle reste chargé en mémoire entre les générations.",
        "persistent_test_3.wav",
    ),
]


# ============================================================
# EXTENDED TESTS
# ============================================================
#
# Ces textes sont volontairement de longueurs très différentes.
#
# L'objectif n'est PAS de tester la qualité audio.
#
# L'objectif est de faire rencontrer au modèle différentes
# longueurs de séquences afin de remplir progressivement
# le cache TorchInductor.
#
# ============================================================

EXTENDED_TESTS = [

    # --------------------------------------------------------
    # Très courts
    # --------------------------------------------------------

    (
        "Bonjour.",
        "extended_01_very_short.wav",
    ),

    (
        "Comment vas-tu aujourd'hui ?",
        "extended_02_short.wav",
    ),

    (
        "Ceci est un test.",
        "extended_03_short.wav",
    ),

    (
        "La machine fonctionne parfaitement.",
        "extended_04_short.wav",
    ),

    (
        "Il fait beau ce matin.",
        "extended_05_short.wav",
    ),

    # --------------------------------------------------------
    # Courts
    # --------------------------------------------------------

    (
        "Bonjour, je suis heureux de pouvoir parler avec toi aujourd'hui.",
        "extended_06.wav",
    ),

    (
        "Nous allons maintenant effectuer plusieurs tests de synthèse vocale.",
        "extended_07.wav",
    ),

    (
        "Cette phrase permet de tester une séquence légèrement plus longue.",
        "extended_08.wav",
    ),

    (
        "Le système génère maintenant de l'audio directement à partir du texte.",
        "extended_09.wav",
    ),

    (
        "Chaque nouvelle phrase peut avoir une longueur différente.",
        "extended_10.wav",
    ),

    # --------------------------------------------------------
    # Moyens
    # --------------------------------------------------------

    (
        "Aujourd'hui nous allons vérifier que le modèle reste stable "
        "pendant une longue série de générations successives.",
        "extended_11.wav",
    ),

    (
        "Cette expérience permet également de vérifier que les différents "
        "artefacts générés par TorchInductor sont correctement conservés.",
        "extended_12.wav",
    ),

    (
        "Nous utilisons plusieurs formulations différentes afin de produire "
        "des séquences de tailles variées et de tester davantage de chemins.",
        "extended_13.wav",
    ),

    (
        "Le système doit être capable de transformer ce texte en une voix "
        "naturelle tout en conservant les caractéristiques de la voix de référence.",
        "extended_14.wav",
    ),

    (
        "Lorsque le modèle est correctement initialisé, les générations suivantes "
        "deviennent beaucoup plus rapides que la première génération.",
        "extended_15.wav",
    ),

    # --------------------------------------------------------
    # Longs
    # --------------------------------------------------------

    (
        "Nous allons maintenant augmenter progressivement la longueur du texte "
        "afin de vérifier comment le système se comporte lorsque la séquence "
        "devient plus importante et nécessite davantage de calculs.",
        "extended_16.wav",
    ),

    (
        "Cette série de tests est volontairement composée de phrases différentes, "
        "car notre objectif principal est de remplir autant que possible le cache "
        "de compilation et de vérifier ensuite si ce cache peut être réutilisé "
        "par un nouveau processus.",
        "extended_17.wav",
    ),

    (
        "Dans une application réelle, les utilisateurs peuvent envoyer des textes "
        "très courts, des phrases normales, des paragraphes entiers ou même des "
        "documents beaucoup plus longs. Nous voulons donc tester plusieurs tailles "
        "afin de couvrir un ensemble aussi large que possible de séquences.",
        "extended_18.wav",
    ),

    (
        "Le moteur de synthèse vocale doit conserver une bonne vitesse de génération "
        "même lorsque les textes changent constamment. Cette génération supplémentaire "
        "permet de créer une nouvelle combinaison de dimensions et de vérifier si "
        "TorchInductor produit de nouveaux artefacts pour cette situation.",
        "extended_19.wav",
    ),

    (
        "Nous continuons cette expérience avec un texte encore plus long. Le but est "
        "simplement de multiplier les formes possibles rencontrées par le modèle. "
        "Nous ne cherchons pas ici à mesurer la qualité de la voix mais uniquement "
        "à provoquer suffisamment de calculs différents pour que le cache puisse "
        "accumuler les artefacts nécessaires à une utilisation future.",
        "extended_20.wav",
    ),

    # --------------------------------------------------------
    # Très longs
    # --------------------------------------------------------

    (
        "Voici maintenant une génération particulièrement longue destinée à tester "
        "le comportement du modèle sur une séquence beaucoup plus importante. "
        "Dans une utilisation réelle, ce type de texte pourrait correspondre à un "
        "paragraphe provenant d'un assistant conversationnel, d'un article, d'un "
        "livre ou d'une réponse détaillée. Nous voulons donc vérifier que le moteur "
        "peut traiter cette situation sans problème et que les opérations de "
        "compilation associées sont correctement enregistrées dans le cache.",
        "extended_21_very_long.wav",
    ),

    (
        "Cette expérience permet de simuler une utilisation intensive du système. "
        "Nous générons volontairement plusieurs phrases avec des longueurs différentes "
        "et avec une structure linguistique variée. Certaines phrases sont très courtes, "
        "d'autres beaucoup plus longues. Certaines utilisent des virgules, des points "
        "et des questions. L'objectif est de multiplier les possibilités de formes "
        "d'entrée afin de donner à TorchInductor un ensemble aussi large que possible "
        "d'artefacts compilés. Une fois cette phase terminée, nous pourrons arrêter le "
        "processus, le redémarrer complètement et mesurer précisément combien de temps "
        "est nécessaire pour la première génération avec ce cache déjà rempli.",
        "extended_22_very_long.wav",
    ),

    (
        "Nous arrivons maintenant à une génération encore plus longue. Le contenu de "
        "ce texte n'a aucune importance particulière car nous cherchons principalement "
        "à varier la longueur de la séquence et donc les formes de tenseurs rencontrées "
        "par le modèle. Dans une application de production, un assistant vocal pourrait "
        "recevoir des messages de quelques mots seulement, puis immédiatement après "
        "recevoir une réponse de plusieurs centaines de mots. Le système doit être capable "
        "de gérer ces différences efficacement. Cette série de générations constitue donc "
        "un test volontairement large du comportement du compilateur et de son mécanisme "
        "de cache. Nous allons continuer suffisamment longtemps pour déterminer si le "
        "nombre d'artefacts continue d'augmenter ou si le cache atteint progressivement "
        "un état stable.",
        "extended_23_very_long.wav",
    ),

]


# ============================================================
# RUN BASIC TESTS
# ============================================================

print()
print("=" * 80)
print("RUNNING BASIC TESTS")
print("=" * 80)

for text, filename in BASIC_TESTS:

    generate_voice(
        text,
        OUTPUT_DIR / filename,
    )


# ============================================================
# RUN EXTENDED TESTS
# ============================================================

if RUN_EXTENDED_TESTS:

    print()
    print("=" * 80)
    print("RUNNING EXTENDED CACHE WARM-UP")
    print("=" * 80)

    print(
        f"Number of extended generations: "
        f"{len(EXTENDED_TESTS)}"
    )

    for index, (text, filename) in enumerate(
        EXTENDED_TESTS,
        start=1,
    ):

        print()
        print(
            f"EXTENDED TEST "
            f"{index}/{len(EXTENDED_TESTS)}"
        )

        generate_voice(
            text,
            OUTPUT_DIR / filename,
        )

else:

    print()
    print("=" * 80)
    print("EXTENDED TESTS DISABLED")
    print("=" * 80)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("ALL TESTS FINISHED")
print("=" * 80)

print(
    f"Total generations: "
    f"{generation_counter}"
)

print(
    f"Extended tests enabled: "
    f"{RUN_EXTENDED_TESTS}"
)

print("=" * 80)