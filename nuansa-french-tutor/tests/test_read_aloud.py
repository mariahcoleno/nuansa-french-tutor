"""
Tests for the read-aloud word comparison: the target sentence is aligned with the
Whisper transcription word by word, ignoring case, punctuation, accents and
the different spellings of endings that sound like /e/ (-é, -ée, -er, -ez, ...).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from src.read_aloud import (MIN_AVG_LOGPROB, NO_SPEECH, SPEECH, STATUS_MESSAGES, UNCLEAR,
                            compare_words, is_silent, normalize_word, speech_status)


class TestReadAloud(unittest.TestCase):
    def unrecognized(self, result):
        return [w["text"] for w in result["words"] if not w["matched"]]

    def test_perfect_match(self):
        result = compare_words("Je vais au marché.", "Je vais au marché.")
        self.assertEqual((result["matched"], result["total"]), (4, 4))
        self.assertEqual(self.unrecognized(result), [])

    def test_missing_word_only_marks_that_word(self):
        # The words after the missing "au" stay recognized
        result = compare_words("Je vais au marché.", "Je vais marché")
        self.assertEqual(self.unrecognized(result), ["au"])
        self.assertEqual((result["matched"], result["total"]), (3, 4))

    def test_wrong_word(self):
        result = compare_words("Elle mange une pomme.", "Elle mange un pomme")
        self.assertEqual(self.unrecognized(result), ["une"])
        self.assertEqual((result["matched"], result["total"]), (3, 4))

    def test_punctuation_and_case_are_ignored(self):
        result = compare_words("Je vais au marché.", "je vais au marché")
        self.assertEqual(self.unrecognized(result), [])
        result = compare_words("C'est une belle fille.", "C’EST UNE BELLE FILLE !")
        self.assertEqual(self.unrecognized(result), [])

    # --- Words that sound the same match (pronunciation, not spelling) ---------

    def test_accents_are_ignored(self):
        result = compare_words("Je mange à l'école.", "Je mange a l'ecole")
        self.assertEqual(self.unrecognized(result), [])
        result = compare_words("Où vas-tu ?", "ou vas-tu")
        self.assertEqual(self.unrecognized(result), [])

    def test_alle_forms_match(self):
        for heard in ["Je suis allé chez ma mère", "Je suis allés chez ma mère",
                      "Je suis allées chez ma mère", "Je suis alle chez ma mere",
                      "Je suis aller chez ma mère", "Je suis allez chez ma mère"]:
            with self.subTest(heard=heard):
                result = compare_words("Je suis allée chez ma mère.", heard)
                self.assertEqual(self.unrecognized(result), [])

    def test_other_forms_of_aller_still_mismatch(self):
        # "allons" sounds different, so it stays red
        result = compare_words("Je suis allée chez ma mère.", "Je suis allons chez ma mère")
        self.assertEqual(self.unrecognized(result), ["allée"])

    def test_e_sound_endings_match(self):
        # The browser test: Whisper heard "marcher" for "marché"
        result = compare_words("Je vais au marché.", "J'ai véc à marcher.")
        self.assertEqual(self.unrecognized(result), ["Je", "vais", "au"])
        self.assertEqual((result["matched"], result["total"]), (1, 4))
        for heard in ["marché", "marchée", "marchés", "marchées", "marcher", "marchez"]:
            with self.subTest(heard=heard):
                result = compare_words("Je vais au marché.", f"Je vais au {heard}")
                self.assertEqual(self.unrecognized(result), [])

    def test_e_sound_endings_need_a_stem(self):
        # Different stems with the same ending don't match
        result = compare_words("Je vais au marché.", "Je vais au manger")
        self.assertEqual(self.unrecognized(result), ["marché."])

    # --- Hyphens ------------------------------------------------------------

    def test_words_whisper_joined_with_hyphens_are_compared_one_by_one(self):
        # Measured: Whisper heard input.wav as one hyphenated word
        result = compare_words("Je mange à l'école.", "Jsui-Alaire-A-Icole.")
        self.assertEqual(self.unrecognized(result), ["Je", "mange", "l'école."])
        self.assertEqual((result["matched"], result["total"]), (1, 4))
        result = compare_words("Je vais au marché.", "Je-vais au marché")
        self.assertEqual(self.unrecognized(result), [])

    def test_hyphenated_french_words(self):
        target = "Il va peut-être venir."
        # With or without the hyphen, peut-être is recognized and shown as one word
        for heard in ["Il va peut-être venir.", "il va peut être venir"]:
            with self.subTest(heard=heard):
                result = compare_words(target, heard)
                self.assertEqual(self.unrecognized(result), [])
                self.assertEqual((result["matched"], result["total"]), (4, 4))
        # Only part of it heard: the whole word is red, and the score counts it once
        result = compare_words(target, "Il va peut venir")
        self.assertEqual([w["text"] for w in result["words"]], ["Il", "va", "peut-être", "venir."])
        self.assertEqual(self.unrecognized(result), ["peut-être"])
        self.assertEqual((result["matched"], result["total"]), (3, 4))
        # Inverted questions and other hyphens (Unicode hyphen U+2010)
        result = compare_words("Où vas-tu ?", "ou vas‐tu")
        self.assertEqual(self.unrecognized(result), [])

    def test_display_keeps_target_spelling(self):
        result = compare_words("Je mange à l'école.", "je mange a l'ecole")
        self.assertEqual([w["text"] for w in result["words"]], ["Je", "mange", "à", "l'école."])

    def test_extra_word_does_not_lower_score(self):
        result = compare_words("Je vais au marché.", "Euh je vais au marché")
        self.assertEqual((result["matched"], result["total"]), (4, 4))

    def test_empty_transcription(self):
        result = compare_words("Je vais au marché.", "")
        self.assertEqual((result["matched"], result["total"]), (0, 4))

    def test_target_words_keep_their_display_form(self):
        result = compare_words("Je vais au marché.", "je vais au marché")
        self.assertEqual([w["text"] for w in result["words"]], ["Je", "vais", "au", "marché."])

    # --- Silent recordings ---------------------------------------------------

    def tone(self, dbfs, seconds=2):
        t = np.arange(16000 * seconds) / 16000
        return np.sin(2 * np.pi * 220 * t) * 10 ** (dbfs / 20) * np.sqrt(2)  # RMS = dbfs

    def test_silent_and_near_silent_recordings(self):
        self.assertTrue(is_silent(np.zeros(16000 * 2)))
        self.assertTrue(is_silent(self.tone(-70)))
        self.assertTrue(is_silent(np.array([])))

    def test_speech_level_audio_is_not_silent(self):
        self.assertFalse(is_silent(self.tone(-15)))
        # Mostly silence with one short loud part (a word in a long recording)
        audio = np.zeros(16000 * 5)
        audio[16000:16000 + 3200] = self.tone(-20)[:3200]
        self.assertFalse(is_silent(audio))

    def segment(self, no_speech_prob, avg_logprob):
        return {"no_speech_prob": no_speech_prob, "avg_logprob": avg_logprob}

    def test_no_words_is_no_speech(self):
        self.assertEqual(speech_status("", []), NO_SPEECH)
        self.assertEqual(speech_status(" ... ", [self.segment(0.1, -0.3)]), NO_SPEECH)

    # Values below are the measurements listed next to the thresholds in src/read_aloud.py

    def test_high_no_speech_prob_is_no_speech(self):
        # The browser test: Whisper invented "meteorite" from near-silent audio
        # (assumed values; the recording itself wasn't measured)
        self.assertEqual(speech_status("Météorite.", [self.segment(0.85, -0.6)]), NO_SPEECH)
        # Generated: reversed TTS
        self.assertEqual(speech_status("Il serait moïve...", [self.segment(0.53, -1.26)]), NO_SPEECH)
        self.assertEqual(speech_status("Merci.", [self.segment(0.7, -0.5), self.segment(0.9, -0.5)]),
                         NO_SPEECH)

    def test_unsure_words_are_unclear(self):
        # Generated: muffled TTS, caught only by MIN_AVG_LOGPROB
        self.assertEqual(speech_status("Vous vous remerciez.", [self.segment(0.40, -1.22)]), UNCLEAR)
        self.assertEqual(speech_status("Euh merci", [self.segment(0.3, -2.0)]), UNCLEAR)
        # A non-speech segment next to an unsure speech segment is still unclear
        self.assertEqual(speech_status("Euh merci", [self.segment(0.9, -0.5), self.segment(0.3, -1.5)]),
                         UNCLEAR)

    def test_logprob_cutoff_between_human_samples_and_muffled_clip(self):
        self.assertTrue(-1.22 < MIN_AVG_LOGPROB < -1.16)

    def test_status_messages(self):
        self.assertEqual(STATUS_MESSAGES[NO_SPEECH],
                         "Aucune parole détectée. Rapprochez-vous du micro et réessayez.")
        self.assertEqual(STATUS_MESSAGES[UNCLEAR], "Parole peu claire, réessayez en articulant.")

    def test_speech_detected(self):
        # TTS, clear and quiet (peak -50 dB)
        self.assertEqual(speech_status("Je vais au marché.", [self.segment(0.03, -0.44)]), SPEECH)
        self.assertEqual(speech_status("Je vais au marché.", [self.segment(0.35, -0.51)]), SPEECH)
        # Human sample recordings, accented but real readings
        self.assertEqual(speech_status("Jsui-Alaire-A-Icole.", [self.segment(0.11, -1.03)]), SPEECH)
        self.assertEqual(speech_status("Allemands et Pomme sont équilés.", [self.segment(0.22, -1.12)]),
                         SPEECH)
        self.assertEqual(speech_status("Allemands et Pomme sont équilés.", [self.segment(0.23, -1.16)]),
                         SPEECH)
        # One confident speech segment is enough
        self.assertEqual(speech_status("Je vais au marché.",
                                       [self.segment(0.9, -2.0), self.segment(0.1, -0.4)]), SPEECH)

    def test_read_aloud_transcription_settings(self):
        from src.analyze import FrenchAnalyzer

        class FakeModel:
            def transcribe(self, audio, **options):
                self.options = options
                return {"text": " Je vais au marché. ", "segments": []}

        analyzer = FrenchAnalyzer.__new__(FrenchAnalyzer)  # skips loading the real models
        analyzer.read_aloud_whisper_model = FakeModel()
        analyzer.transcribe_read_aloud(np.zeros(16000))
        options = analyzer.read_aloud_whisper_model.options
        # Forced French, so near-silence isn't "detected" as another language ("Això.")
        self.assertEqual(options["language"], "fr")
        self.assertEqual(options["task"], "transcribe")
        # One deterministic pass, no random retries at higher temperatures
        self.assertEqual(options["temperature"], 0)
        # The target sentence is never given as a hint
        self.assertNotIn("initial_prompt", options)

    def test_normalize_word(self):
        self.assertEqual(normalize_word("Marché."), "marche")
        self.assertEqual(normalize_word("L’école"), "l'ecole")
        self.assertEqual(normalize_word("Allées"), "alle")
        self.assertEqual(normalize_word("marcher"), "marche")
        self.assertEqual(normalize_word("Chez"), "che")
        self.assertEqual(normalize_word("!"), "")


if __name__ == "__main__":
    unittest.main()
