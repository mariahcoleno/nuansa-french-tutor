"""
Word-by-word comparison for the read-aloud mode ("Lecture à voix haute").

Compares the sentence the learner was asked to read with the raw Whisper
transcription. A word that doesn't match means the app didn't recognize it
clearly; it is not a pronunciation score.

Since this checks what was said, not how it is spelled, words that sound the same
count as a match: accents are ignored ("a"/"à", "ou"/"où") and endings that
sound like /e/ are treated as one ending ("allé"/"aller"/"allez", "marché"/"marcher").
"""
import difflib
import re
import unicodedata

import numpy as np

# Outcomes of speech_status, with the message shown instead of a transcription
SPEECH = "speech"
NO_SPEECH = "no_speech"   # silence: nothing that sounds like speech
UNCLEAR = "unclear"       # sound that may be speech, but Whisper can't make out the words
STATUS_MESSAGES = {
    NO_SPEECH: "Aucune parole détectée. Rapprochez-vous du micro et réessayez.",
    UNCLEAR: "Parole peu claire, réessayez en articulant.",
}

# A recording whose loudest 50 ms frame is quieter than this is treated as silent.
# Normal speech peaks around -10 to -15 dBFS; Whisper still "hears" words in much
# quieter audio (it normalizes the volume), which is how it hallucinates on silence.
SILENCE_DBFS = -60
FRAME_SECONDS = 0.05

# A Whisper segment only counts as speech if it is probably speech AND Whisper is
# reasonably sure of the words. Whisper itself only drops a segment when both are bad,
# so hallucinations on near-silence ("Merci beaucoup.", "Això.") get through.
# Segments above NO_SPEECH_PROB are silence; probable speech below MIN_AVG_LOGPROB is unclear.
#
# How the thresholds were chosen (read-aloud settings: "small" model, temperature=0),
# as (no_speech_prob, avg_logprob):
#   - TTS of the 5 practice sentences, also played quietly (peak -30 to -50 dB):
#     0.02-0.35, -0.30 to -0.51 -> speech
#   - Human sample recordings, must stay speech:
#     input.wav  "Jsui-Alaire-A-Icole."             0.11, -1.03
#     input2.m4a "Allemands et Pomme sont équilés." 0.22, -1.12
#     input2.wav "Allemands et Pomme sont équilés." 0.23, -1.16  <- lowest kept
#   - Generated hallucinations, must not be shown:
#     muffled TTS (300 Hz low-pass) "Vous vous remerciez." 0.40, -1.22 <- caught only by
#                                                                       MIN_AVG_LOGPROB
#     reversed TTS "Il serait moïve..."                      0.53, -1.26 <- NO_SPEECH_PROB
#     quiet gated residue, noise at -50 dB: no words; silence, TTS at -55 dB or quieter:
#     caught by is_silent
# MIN_AVG_LOGPROB = -1.18 sits between input2.wav (-1.16) and the muffled clip (-1.22),
# looser than Whisper's own default of -1.0 so accented but real readings are kept.
# The margins are small (0.02 and 0.04) and based on few recordings: revisit with real
# learner recordings (the route logs both values for every check).
NO_SPEECH_PROB = 0.5
MIN_AVG_LOGPROB = -1.18

# Word endings that sound like /e/, written as "é" before accents are removed.
# Longest endings first so "ées" isn't read as "é" + "es". The lookbehind only
# rewrites an ending that follows at least one letter of the word.
E_SOUND_ENDING = re.compile(r"(?<=\w)(ées|ée|és|é|er|ez)$")

# Hyphens Whisper or a target sentence may use: ASCII, Unicode hyphen, non-breaking hyphen
HYPHENS = re.compile(r"[-‐‑]")


def normalize_word(word):
    """
    Lowercase a word, drop its punctuation and accents, keep inner apostrophes, and
    give endings that sound like /e/ one spelling ("Marché." -> "marche",
    "marcher" -> "marche", "L’école" -> "l'ecole", "allées" -> "alle").
    """
    word = unicodedata.normalize("NFC", word.lower().replace("’", "'"))
    word = "'".join(re.findall(r"\w+", word))
    word = E_SOUND_ENDING.sub("é", word)
    decomposed = unicodedata.normalize("NFD", word)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def is_silent(samples, sample_rate=16000):
    """True if the loudest frame of the audio (floats in -1..1) is below SILENCE_DBFS."""
    samples = np.asarray(samples, dtype=np.float32)
    frame = int(sample_rate * FRAME_SECONDS)
    if len(samples) < frame:
        return True
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    loudest = np.sqrt((frames ** 2).mean(axis=1)).max()
    return loudest < 10 ** (SILENCE_DBFS / 20)


def speech_status(text, segments):
    """
    Decide whether Whisper's result can be shown (segments are Whisper's result["segments"],
    each with a "no_speech_prob" and an "avg_logprob"):
    - SPEECH if at least one segment is probably speech and Whisper is sure of its words,
    - UNCLEAR if some segment is probably speech but Whisper is unsure of all of them,
    - NO_SPEECH if Whisper found no words or every segment is probably not speech.
    """
    if not any(normalize_word(w) for w in text.split()):
        return NO_SPEECH
    probable_speech = [s for s in segments if s["no_speech_prob"] <= NO_SPEECH_PROB]
    if not probable_speech:
        return NO_SPEECH
    if any(s["avg_logprob"] >= MIN_AVG_LOGPROB for s in probable_speech):
        return SPEECH
    return UNCLEAR


def word_parts(word):
    """
    Split a word on hyphens and normalize each part, dropping empty ones
    ("Peut-être" -> ["peut", "etre"], "Jsui-Alaire-A-Icole." -> ["jsui", "alaire", "a", "icole"]).
    Whisper sometimes joins separate words with hyphens, and may write "peut-être"
    with or without its hyphen, so words are compared part by part.
    """
    return [n for n in (normalize_word(p) for p in HYPHENS.split(word)) if n]


def compare_words(target, heard):
    """
    Align the target sentence with what Whisper heard, word by word.

    Uses difflib so a missing or extra word doesn't shift every later word out of place.
    Words are split on hyphens and compared with normalize_word, so case, punctuation,
    accents, /e/ endings and hyphens don't count as mismatches.
    Returns the target words (with their original spelling and punctuation, for display),
    each marked matched or not, plus the number of matched words out of the total.
    A hyphenated target word ("peut-être") is one word, matched only if all its parts are.
    """
    target_words = [w for w in target.split() if word_parts(w)]
    # Each target part remembers which displayed target word it belongs to
    target_parts, owners = [], []
    for index, word in enumerate(target_words):
        for part in word_parts(word):
            target_parts.append(part)
            owners.append(index)
    heard_parts = [part for word in heard.split() for part in word_parts(word)]

    part_matched = [False] * len(target_parts)
    matcher = difflib.SequenceMatcher(None, target_parts, heard_parts, autojunk=False)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                part_matched[i] = True

    matched = [True] * len(target_words)
    for owner, ok in zip(owners, part_matched):
        matched[owner] = matched[owner] and ok

    return {
        "words": [{"text": w, "matched": m} for w, m in zip(target_words, matched)],
        "matched": sum(matched),
        "total": len(target_words),
    }
