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

RUN_EXTENDED_TESTS = False
RUN_EXTENDED_TESTS_V2 = False

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
        compile=True,
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
# EXTENDED CACHE WARM-UP V2
# ============================================================


EXTENDED_TESTS_V2 = [

    # ========================================================
    # 01-10 : TRÈS COURTS / QUESTIONS
    # ========================================================

    (
        "Oui ?",
        "v2_01.wav",
    ),

    (
        "Pourquoi ?",
        "v2_02.wav",
    ),

    (
        "Comment ça va ?",
        "v2_03.wav",
    ),

    (
        "Tu es là ?",
        "v2_04.wav",
    ),

    (
        "Quelle heure est-il ?",
        "v2_05.wav",
    ),

    (
        "Tout fonctionne ?",
        "v2_06.wav",
    ),

    (
        "On commence ?",
        "v2_07.wav",
    ),

    (
        "Est-ce vraiment nécessaire ?",
        "v2_08.wav",
    ),

    (
        "Peux-tu m'aider ?",
        "v2_09.wav",
    ),

    (
        "Qu'est-ce que tu proposes ?",
        "v2_10.wav",
    ),


    # ========================================================
    # 11-20 : COURTS / QUESTIONS
    # ========================================================

    (
        "Pourquoi cette machine est-elle si rapide ?",
        "v2_11.wav",
    ),

    (
        "Comment fonctionne ce système exactement ?",
        "v2_12.wav",
    ),

    (
        "Est-ce que le modèle est correctement chargé ?",
        "v2_13.wav",
    ),

    (
        "Peux-tu répéter cette phrase, s'il te plaît ?",
        "v2_14.wav",
    ),

    (
        "Quelle solution devons-nous utiliser pour ce problème ?",
        "v2_15.wav",
    ),

    (
        "Pourquoi la première génération est-elle plus lente ?",
        "v2_16.wav",
    ),

    (
        "Combien de temps faut-il pour générer cette réponse ?",
        "v2_17.wav",
    ),

    (
        "Est-ce que cette voix ressemble à la voix de référence ?",
        "v2_18.wav",
    ),

    (
        "Peut-on utiliser ce système avec plusieurs langues ?",
        "v2_19.wav",
    ),

    (
        "Que se passe-t-il lorsque le texte devient beaucoup plus long ?",
        "v2_20.wav",
    ),


    # ========================================================
    # 21-30 : MOYENS / QUESTIONS
    # ========================================================

    (
        "Pourquoi le temps de génération change-t-il lorsque la longueur "
        "du texte augmente progressivement ?",
        "v2_21.wav",
    ),

    (
        "Comment pouvons-nous vérifier que les artefacts de compilation "
        "sont correctement réutilisés par un nouveau processus ?",
        "v2_22.wav",
    ),

    (
        "Est-ce que le modèle peut conserver les caractéristiques de la "
        "voix de référence lorsque nous changeons complètement le texte ?",
        "v2_23.wav",
    ),

    (
        "Quelle serait la meilleure manière de réduire le temps nécessaire "
        "à la première génération après le démarrage d'un nouveau worker ?",
        "v2_24.wav",
    ),

    (
        "Pourquoi certaines générations semblent-elles nécessiter davantage "
        "de compilation alors que les phrases précédentes étaient beaucoup plus rapides ?",
        "v2_25.wav",
    ),

    (
        "Peux-tu expliquer pourquoi un cache TorchInductor persistant peut "
        "réduire le temps de compilation lors des exécutions suivantes ?",
        "v2_26.wav",
    ),

    (
        "Que faudrait-il changer dans cette architecture pour pouvoir servir "
        "plusieurs utilisateurs simultanément sans augmenter fortement la latence ?",
        "v2_27.wav",
    ),

    (
        "Est-il possible d'obtenir une première génération rapide même après "
        "le redémarrage complet du conteneur ?",
        "v2_28.wav",
    ),

    (
        "Comment le système réagit-il lorsqu'un utilisateur envoie une phrase "
        "très courte immédiatement après une réponse extrêmement longue ?",
        "v2_29.wav",
    ),

    (
        "Pourquoi est-il important de tester plusieurs longueurs de séquences "
        "plutôt que de générer plusieurs fois exactement la même phrase ?",
        "v2_30.wav",
    ),


    # ========================================================
    # 31-40 : LONGS / QUESTIONS
    # ========================================================

    (
        "Si un utilisateur pose une question très longue contenant plusieurs "
        "phrases, plusieurs virgules et plusieurs informations différentes, "
        "est-ce que le modèle utilise exactement les mêmes artefacts de compilation "
        "que lorsqu'il reçoit une question beaucoup plus courte ?",
        "v2_31.wav",
    ),

    (
        "Comment pouvons-nous déterminer expérimentalement si une nouvelle longueur "
        "de séquence provoque réellement une nouvelle compilation, ou si TorchInductor "
        "peut simplement réutiliser un artefact déjà présent dans son cache ?",
        "v2_32.wav",
    ),

    (
        "Si nous utilisons une voix de référence différente mais exactement le même "
        "modèle, le même type de données et des textes de longueur comparable, "
        "est-ce que les compilations précédemment réalisées peuvent être réutilisées ?",
        "v2_33.wav",
    ),

    (
        "Pourquoi une architecture de production utilisant plusieurs GPU différents "
        "pourrait-elle rencontrer des temps de compilation différents alors que le "
        "code Python et le modèle utilisés sont exactement les mêmes ?",
        "v2_34.wav",
    ),

    (
        "Est-ce qu'une application conversationnelle réelle rencontrera suffisamment "
        "de longueurs de séquences différentes pour justifier un warm-up important "
        "avant de commencer à accepter les requêtes des utilisateurs ?",
        "v2_35.wav",
    ),

    (
        "Que se passerait-il si nous envoyions successivement des réponses très courtes, "
        "puis moyennes, puis longues, puis très longues, avant de redémarrer complètement "
        "le processus afin de mesurer l'effet réel du cache persistant ?",
        "v2_36.wav",
    ),

    (
        "Comment faudrait-il organiser le cache si plusieurs workers utilisent le même "
        "volume persistant mais ne possèdent pas nécessairement exactement le même GPU, "
        "la même version de CUDA ou la même version de PyTorch ?",
        "v2_37.wav",
    ),

    (
        "Pourquoi une phrase contenant beaucoup de ponctuation, des nombres, des symboles "
        "et des changements de structure peut-elle être intéressante pour tester le "
        "comportement du compilateur dans une application de synthèse vocale ?",
        "v2_38.wav",
    ),

    (
        "Si nous voulons construire un assistant vocal multilingue capable de répondre "
        "en français, en anglais, en espagnol et en allemand, quelles différences "
        "de longueur et de structure devons-nous prendre en compte pendant le warm-up ?",
        "v2_39.wav",
    ),

    (
        "Peut-on considérer que le cache est suffisamment complet lorsque plusieurs "
        "générations successives produisent très peu de nouveaux fichiers et que le "
        "temps de première génération après redémarrage commence lui aussi à se stabiliser ?",
        "v2_40.wav",
    ),


    # ========================================================
    # 41-48 : PONCTUATION / NOMBRES / STRUCTURES
    # ========================================================

    (
        "Test numéro 1 : 10, 20, 30, 40 et 50.",
        "v2_41.wav",
    ),

    (
        "Le résultat est-il de 98,5 %, 99 % ou 100 % ?",
        "v2_42.wav",
    ),

    (
        "Attention ! Est-ce vraiment terminé ? Oui !",
        "v2_43.wav",
    ),

    (
        "Voici une phrase avec des parenthèses (et quelques détails supplémentaires).",
        "v2_44.wav",
    ),

    (
        "Que signifie exactement cette expression : « intelligence artificielle » ?",
        "v2_45.wav",
    ),

    (
        "Première étape ; deuxième étape ; troisième étape : tout fonctionne.",
        "v2_46.wav",
    ),

    (
        "Un, deux, trois... Est-ce que tu m'entends encore ?",
        "v2_47.wav",
    ),

    (
        "Version 2.0 — test complet : CPU, GPU, CUDA 12.9, PyTorch 2.8.",
        "v2_48.wav",
    ),


    # ========================================================
    # 49-56 : ENGLISH
    # ========================================================

    (
        "Hello, how are you today?",
        "v2_49_en.wav",
    ),

    (
        "Can you explain how this system works?",
        "v2_50_en.wav",
    ),

    (
        "Why does the first generation take so much longer than the next ones?",
        "v2_51_en.wav",
    ),

    (
        "How can we make the first inference faster after restarting the container?",
        "v2_52_en.wav",
    ),

    (
        "What happens when the input sequence becomes much longer than expected?",
        "v2_53_en.wav",
    ),

    (
        "Could a persistent TorchInductor cache significantly reduce compilation time "
        "for a production text-to-speech service?",
        "v2_54_en.wav",
    ),

    (
        "If several workers share the same persistent cache but run on different GPU "
        "architectures, will all compiled artifacts remain reusable?",
        "v2_55_en.wav",
    ),

    (
        "How should we design the warm-up process if users can send short questions, "
        "long answers, paragraphs, and multilingual conversations?",
        "v2_56_en.wav",
    ),


    # ========================================================
    # 57-64 : ESPAÑOL
    # ========================================================

    (
        "Hola, ¿cómo estás hoy?",
        "v2_57_es.wav",
    ),

    (
        "¿Puedes explicarme cómo funciona este sistema?",
        "v2_58_es.wav",
    ),

    (
        "¿Por qué la primera generación tarda tanto tiempo?",
        "v2_59_es.wav",
    ),

    (
        "¿Cómo podemos reducir el tiempo de compilación después de reiniciar el contenedor?",
        "v2_60_es.wav",
    ),

    (
        "¿Qué ocurre cuando el texto de entrada es mucho más largo de lo habitual?",
        "v2_61_es.wav",
    ),

    (
        "¿Puede una caché persistente reducir significativamente el tiempo necesario "
        "para generar la primera respuesta de voz?",
        "v2_62_es.wav",
    ),

    (
        "Si varios trabajadores utilizan la misma caché persistente pero diferentes "
        "GPU, ¿los artefactos compilados pueden reutilizarse de la misma manera?",
        "v2_63_es.wav",
    ),

    (
        "¿Cómo deberíamos preparar el sistema si queremos responder rápidamente a "
        "preguntas cortas y también a respuestas largas en varios idiomas?",
        "v2_64_es.wav",
    ),


    # ========================================================
    # 65-72 : DEUTSCH
    # ========================================================

    (
        "Hallo, wie geht es dir heute?",
        "v2_65_de.wav",
    ),

    (
        "Kannst du erklären, wie dieses System funktioniert?",
        "v2_66_de.wav",
    ),

    (
        "Warum dauert die erste Generierung so lange?",
        "v2_67_de.wav",
    ),

    (
        "Wie können wir die Kompilierungszeit nach einem Neustart reduzieren?",
        "v2_68_de.wav",
    ),

    (
        "Was passiert, wenn der eingegebene Text viel länger als erwartet ist?",
        "v2_69_de.wav",
    ),

    (
        "Kann ein persistenter Cache die Zeit für die erste Sprachgenerierung "
        "deutlich reduzieren?",
        "v2_70_de.wav",
    ),

    (
        "Wenn mehrere Worker denselben persistenten Cache verwenden, aber auf "
        "unterschiedlichen GPUs laufen, können die kompilierten Artefakte weiterhin "
        "verwendet werden?",
        "v2_71_de.wav",
    ),

    (
        "Wie sollten wir den Warm-up-Prozess gestalten, wenn Benutzer kurze Fragen, "
        "lange Antworten und mehrsprachige Gespräche senden können?",
        "v2_72_de.wav",
    ),


    # ========================================================
    # 73-80 : TRÈS LONGS
    # ========================================================

    (
        "Voici un texte particulièrement long destiné à provoquer une séquence "
        "beaucoup plus importante que les précédentes. Dans une application réelle, "
        "un assistant vocal peut recevoir une demande contenant plusieurs paragraphes, "
        "des explications détaillées, des questions successives et de nombreuses "
        "informations contextuelles. Nous voulons donc également tester ce type de "
        "situation afin de déterminer si de nouveaux artefacts TorchInductor sont "
        "créés lorsque la longueur de la séquence augmente fortement.",
        "v2_73_very_long.wav",
    ),

    (
        "Imaginons maintenant une conversation complète entre un utilisateur et un "
        "assistant vocal. L'utilisateur commence par poser une question très courte. "
        "L'assistant répond ensuite avec quelques phrases. L'utilisateur demande alors "
        "davantage de détails et reçoit une réponse beaucoup plus longue. Enfin, il pose "
        "une dernière question contenant plusieurs informations différentes. Cette "
        "situation est intéressante pour notre test parce qu'elle reproduit davantage "
        "le comportement d'une véritable application conversationnelle et permet de "
        "faire varier progressivement la longueur des séquences traitées par le modèle.",
        "v2_74_very_long.wav",
    ),

    (
        "Nous allons terminer cette série avec une génération volontairement très longue. "
        "Le contenu exact du texte est moins important que sa structure et sa longueur. "
        "Nous cherchons principalement à rencontrer des séquences qui n'ont pas encore "
        "été utilisées pendant les générations précédentes. Si TorchInductor doit créer "
        "de nouveaux artefacts pour certaines de ces dimensions, ils seront enregistrés "
        "dans le cache persistant. Après cette génération, nous pourrons comparer la taille "
        "du cache avec les valeurs précédentes et déterminer si nous sommes réellement "
        "arrivés à un plateau ou si certaines formes de calcul restent encore à explorer.",
        "v2_75_very_long.wav",
    ),

    (
        "Dernière question de cette série : si nous avons déjà généré des centaines de "
        "séquences différentes, avec plusieurs longueurs, plusieurs langues, différentes "
        "ponctuations et plusieurs structures de texte, mais que la première génération "
        "après le redémarrage reste encore relativement lente, cela signifie-t-il que "
        "le cache est incomplet, ou bien qu'une partie du temps observé correspond à "
        "des opérations qui ne peuvent tout simplement pas être supprimées, comme "
        "l'initialisation du modèle, le chargement du codec, la préparation du GPU ou "
        "certaines compilations dépendantes de l'architecture matérielle ?",
        "v2_76_final.wav",
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
# RUN EXTENDED TESTS V2
# ============================================================

if RUN_EXTENDED_TESTS_V2:

    print()
    print("=" * 80)
    print("RUNNING EXTENDED CACHE WARM-UP V2")
    print("=" * 80)

    print(
        f"Number of V2 generations: "
        f"{len(EXTENDED_TESTS_V2)}"
    )

    for index, (text, filename) in enumerate(
        EXTENDED_TESTS_V2,
        start=1,
    ):

        print()
        print(
            f"EXTENDED V2 TEST "
            f"{index}/{len(EXTENDED_TESTS_V2)}"
        )

        generate_voice(
            text,
            OUTPUT_DIR / filename,
        )

else:

    print()
    print("=" * 80)
    print("EXTENDED V2 TESTS DISABLED")
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