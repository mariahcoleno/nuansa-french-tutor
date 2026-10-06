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

NO_SPEECH_MESSAGE = "Aucune parole détectée. Rapprochez-vous du micro et réessayez."

# A recording whose loudest 50 ms frame is quieter than this is treated as silent.
# Normal speech peaks around -10 to -15 dBFS; Whisper still "hears" words in much
# quieter audio (it normalizes the volume), which is how it hallucinates on silence.
SILENCE_DBFS = -60
FRAME_SECONDS = 0.05

# Whisper's own default no-speech threshold. Whisper only drops a segment when it is
# also unsure of the words, so confident hallucinations get through; here a segment
# above this probability always counts as no speech.
NO_SPEECH_PROB = 0.6

# Word endings that sound like /e/, written as "é" before accents are removed.
# Longest endings first so "ées" isn't read as "é" + "es". The lookbehind only
# rewrites an ending that follows at least one letter of the word.
E_SOUND_ENDING = re.compile(r"(?<=\w)(ées|ée|és|é|er|ez)$")


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


def no_speech_detected(text, segments):
    """
    True if Whisper found no words, or every segment it returned is probably not speech
    (segments are Whisper's result["segments"], each with a "no_speech_prob").
    """
    if not any(normalize_word(w) for w in text.split()):
        return True
    return all(s["no_speech_prob"] > NO_SPEECH_PROB for s in segments)


def compare_words(target, heard):
    """
    Align the target sentence with what Whisper heard, word by word.

    Uses difflib so a missing or extra word doesn't shift every later word out of place.
    Words are compared with normalize_word, so case, punctuation, accents and /e/
    endings don't count as mismatches.
    Returns the target words (with their original spelling and punctuation, for display),
    each marked matched or not, plus the number of matched words out of the total.
    """
    target_words = [w for w in target.split() if normalize_word(w)]
    target_norm = [normalize_word(w) for w in target_words]
    heard_norm = [n for n in (normalize_word(w) for w in heard.split()) if n]

    matched = [False] * len(target_words)
    matcher = difflib.SequenceMatcher(None, target_norm, heard_norm, autojunk=False)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                matched[i] = True

    return {
        "words": [{"text": w, "matched": m} for w, m in zip(target_words, matched)],
        "matched": sum(matched),
        "total": len(target_words),
    }
