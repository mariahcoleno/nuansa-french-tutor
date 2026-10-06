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
from src.read_aloud import compare_words, is_silent, no_speech_detected, normalize_word


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

    def test_no_speech_detected(self):
        # The browser test: echo cancellation removed the sound and Whisper invented a word
        self.assertTrue(no_speech_detected("Météorite.", [{"no_speech_prob": 0.85}]))
        self.assertTrue(no_speech_detected("", []))
        self.assertTrue(no_speech_detected(" ... ", [{"no_speech_prob": 0.1}]))
        self.assertTrue(no_speech_detected("Merci.", [{"no_speech_prob": 0.7}, {"no_speech_prob": 0.9}]))

    def test_speech_detected(self):
        self.assertFalse(no_speech_detected("Je vais au marché.", [{"no_speech_prob": 0.03}]))
        # One segment of speech is enough
        self.assertFalse(no_speech_detected("Je vais au marché.",
                                            [{"no_speech_prob": 0.9}, {"no_speech_prob": 0.1}]))

    def test_normalize_word(self):
        self.assertEqual(normalize_word("Marché."), "marche")
        self.assertEqual(normalize_word("L’école"), "l'ecole")
        self.assertEqual(normalize_word("Allées"), "alle")
        self.assertEqual(normalize_word("marcher"), "marche")
        self.assertEqual(normalize_word("Chez"), "che")
        self.assertEqual(normalize_word("!"), "")


if __name__ == "__main__":
    unittest.main()
