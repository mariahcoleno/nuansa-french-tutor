"""
Tests for the custom regex rules in FrenchAnalyzer: contractions, feminine
speaker agreement, and the custom-rule errors shown in the error table.
Correct sentences must come back unchanged.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import enchant
import language_tool_python
from src.analyze import FrenchAnalyzer, LANGUAGETOOL_VERSION, words_changed


class TestCustomRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Skip __init__ so the tests don't load Whisper;
        # analyze_text and apply_corrections only need these two tools.
        cls.analyzer = FrenchAnalyzer.__new__(FrenchAnalyzer)
        cls.analyzer.grammar_tool = language_tool_python.LanguageTool(
            'fr', language_tool_download_version=LANGUAGETOOL_VERSION)
        cls.analyzer.d = enchant.Dict("fr_FR")

    @classmethod
    def tearDownClass(cls):
        cls.analyzer.grammar_tool.close()

    def correct(self, text, gender="masculine"):
        matches = self.analyzer.grammar_tool.check(text)
        return self.analyzer.apply_corrections(text, matches, gender)

    def assertUnchanged(self, text, gender="masculine"):
        self.assertEqual(self.correct(text, gender), text)

    # --- Contractions -------------------------------------------------------

    def test_contractions_are_corrected(self):
        self.assertEqual(self.correct("Je vais à le marché."), "Je vais au marché.")
        self.assertEqual(self.correct("Je parle de le film."), "Je parle du film.")
        self.assertEqual(self.correct("Je vais à les magasins."), "Je vais aux magasins.")
        self.assertEqual(self.correct("Je parle de les enfants."), "Je parle des enfants.")

    def test_object_pronouns_are_unchanged(self):
        for text in ["Je vais commencer à le faire.",
                     "Il vient de le dire.",
                     "Je commence à les comprendre.",
                     "J'ai peur de les perdre."]:
            with self.subTest(text=text):
                self.assertUnchanged(text)

    # --- Feminine speaker agreement -------------------------------------------

    def test_feminine_correct_sentences_are_unchanged(self):
        for text in ["Je suis à Paris.",
                     "Je suis en retard.",
                     "Je suis très contente.",
                     "Je suis heureuse.",
                     "Je suis allée au marché.",
                     "Je suis Marie."]:
            with self.subTest(text=text):
                self.assertUnchanged(text, "feminine")

    def test_masculine_sentences_are_unchanged(self):
        for text in ["Je suis content.", "Je suis à Paris."]:
            with self.subTest(text=text):
                self.assertUnchanged(text, "masculine")

    def test_feminine_agreement_is_applied(self):
        self.assertEqual(self.correct("Je suis content.", "feminine"), "Je suis contente.")
        self.assertEqual(self.correct("Je suis heureux.", "feminine"), "Je suis heureuse.")
        self.assertEqual(self.correct("Je suis fatigué.", "feminine"), "Je suis fatiguée.")
        self.assertEqual(self.correct("Je suis aller au marché.", "feminine"), "Je suis allée au marché.")

    # --- Other custom rules ---------------------------------------------------

    def test_c_est_with_un_une_noun(self):
        self.assertEqual(self.correct("Il est une belle fille."), "C'est une belle fille.")
        self.assertEqual(self.correct("Elle est un bon garçon."), "C'est un bon garçon.")

    def test_mignon_only_for_les_chats(self):
        self.assertEqual(self.correct("Les chats sont mignonnes."), "Les chats sont mignons.")
        self.assertUnchanged("Les filles sont mignonnes.")
        self.assertUnchanged("Les chats sont mignons.")

    # --- Error table (analyze_text) -------------------------------------------

    def test_a_ecole_appears_in_error_table(self):
        result = self.analyzer.analyze_text("Je mange à école.")
        self.assertEqual(result["final_text"], "Je mange à l'école.")
        self.assertEqual(len(result["custom_errors"]), 1)
        error = result["custom_errors"][0]
        self.assertEqual(error["suggestions"], ["à l'école"])
        self.assertIn("voyelle", error["message"])
        self.assertNotIn("span", error)

    def test_c_est_appears_in_error_table(self):
        result = self.analyzer.analyze_text("Il est une belle fille.")
        self.assertEqual(result["final_text"], "C'est une belle fille.")
        self.assertEqual(len(result["custom_errors"]), 1)
        error = result["custom_errors"][0]
        self.assertEqual(error["error"], "Il est une ... fille")
        self.assertEqual(error["suggestions"], ["C'est une ... fille"])
        self.assertIn("c'est", error["message"])

    def test_cantine_message(self):
        result = self.analyzer.analyze_text("Nous mangeons dans la cantine.")
        self.assertEqual(result["final_text"], "Nous mangeons à la cantine.")
        messages = [e["message"] for e in result["custom_errors"]]
        self.assertEqual(len(messages), 1)
        self.assertIn("plus courant", messages[0])

    def test_custom_rules_replace_overlapping_languagetool_errors(self):
        result = self.analyzer.analyze_text("Je suis aller chez mon mère.", speaker_gender="feminine")
        self.assertEqual(result["final_text"], "Je suis allée chez ma mère.")
        self.assertEqual(result["grammar_errors"], [])
        by_error = {e["error"]: e["suggestions"] for e in result["custom_errors"]}
        self.assertEqual(by_error, {"Je suis aller": ["Je suis allée"], "mon mère": ["ma mère"]})

    def test_contraction_reported_once(self):
        result = self.analyzer.analyze_text("Je vais à le marché.")
        self.assertEqual(result["final_text"], "Je vais au marché.")
        self.assertEqual(result["custom_errors"], [])
        self.assertEqual(len(result["grammar_errors"]), 1)
        self.assertEqual(result["grammar_errors"][0].replacements[0], "au")

    def test_correct_sentences_report_no_errors(self):
        for text, gender in [("Je vais commencer à le faire.", "masculine"),
                             ("Je suis à Paris.", "feminine"),
                             ("Les filles sont mignonnes.", "masculine")]:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text, speaker_gender=gender)
                self.assertEqual(result["final_text"], text)
                self.assertEqual(result["grammar_errors"], [])
                self.assertEqual(result["custom_errors"], [])
                self.assertEqual(result["spelling_errors"], [])


    def test_feminine_adjective_appears_in_error_table(self):
        result = self.analyzer.analyze_text("Je suis content", speaker_gender="feminine")
        self.assertEqual(result["final_text"], "Je suis contente.")
        self.assertEqual(result["grammar_errors"], [])
        self.assertEqual(len(result["custom_errors"]), 1)
        error = result["custom_errors"][0]
        self.assertEqual(error["error"], "content")
        self.assertEqual(error["suggestions"], ["contente"])
        self.assertIn("Accord au féminin", error["message"])
        self.assertIn("content → contente", error["message"])

    def test_feminine_participle_appears_in_error_table(self):
        result = self.analyzer.analyze_text("Je suis fatigué", speaker_gender="feminine")
        self.assertEqual(result["final_text"], "Je suis fatiguée.")
        self.assertEqual(result["grammar_errors"], [])
        # fatigué is also in the adjective list; it must only be reported once
        self.assertEqual(len(result["custom_errors"]), 1)
        error = result["custom_errors"][0]
        self.assertEqual(error["error"], "fatigué")
        self.assertEqual(error["suggestions"], ["fatiguée"])
        self.assertIn("participe passé", error["message"])
        self.assertIn("fatigué → fatiguée", error["message"])

    def test_feminine_alle_reported_once(self):
        # Both the aller rule and the -é participle rule match "je suis allé"
        result = self.analyzer.analyze_text("Je suis allé au marché.", speaker_gender="feminine")
        self.assertEqual(result["final_text"], "Je suis allée au marché.")
        self.assertEqual([e["suggestions"] for e in result["custom_errors"]], [["Je suis allée"]])

    def test_masculine_adjective_has_no_error_row(self):
        result = self.analyzer.analyze_text("Je suis content", speaker_gender="masculine")
        self.assertEqual(result["custom_errors"], [])

    def test_missing_final_period_is_not_an_error(self):
        # LanguageTool's POINTS_2 flags the missing period, but the app adds it itself
        text = "Je suis content"
        result = self.analyzer.analyze_text(text, speaker_gender="masculine")
        self.assertEqual(result["final_text"], "Je suis content.")
        self.assertEqual(result["grammar_errors"], [])
        self.assertEqual(result["custom_errors"], [])
        self.assertEqual(result["spelling_errors"], [])
        self.assertFalse(words_changed(text, result["final_text"]))

    # --- "Grammaire/Genre" fallback row (words_changed) -------------------------

    def test_correct_sentence_has_no_word_changes(self):
        # apply_corrections capitalizes and adds a final period; that alone isn't a correction
        text = "je suis à paris"
        final_text = self.analyzer.analyze_text(text, speaker_gender="feminine")["final_text"]
        self.assertEqual(final_text, "Je suis à paris.")
        self.assertFalse(words_changed(text, final_text))

    def test_word_correction_is_detected(self):
        # A real word change counts, unlike the added period and capital letter
        text = "je suis fatigué"
        result = self.analyzer.analyze_text(text, speaker_gender="feminine")
        self.assertEqual(result["final_text"], "Je suis fatiguée.")
        self.assertTrue(words_changed(text, result["final_text"]))

    # --- Spelling (dictionary check) -------------------------------------------

    def test_missing_accent_is_fixed(self):
        result = self.analyzer.analyze_text("Je vais à l'ecole.")
        self.assertEqual(result["final_text"], "Je vais à l'école.")
        self.assertEqual(len(result["spelling_errors"]), 1)
        error = result["spelling_errors"][0]
        self.assertEqual(error["error"], "ecole")
        self.assertEqual(error["suggestions"], ["école"])
        self.assertEqual(result["grammar_errors"], [])

    def test_capitalized_missing_accent_is_fixed(self):
        result = self.analyzer.analyze_text("Tres bien.")
        self.assertEqual(result["final_text"], "Très bien.")
        self.assertEqual(len(result["spelling_errors"]), 1)
        error = result["spelling_errors"][0]
        self.assertEqual(error["error"], "Tres")
        self.assertEqual(error["suggestions"], ["Très"])
        self.assertEqual(error["message"], "Accent manquant ou incorrect.")

    def test_mid_sentence_name_is_unchanged(self):
        # LanguageTool's spelling rule flags "Mariah" too; neither check may report it
        text = "Je m'appelle Mariah."
        result = self.analyzer.analyze_text(text)
        self.assertEqual(result["final_text"], text)
        self.assertEqual(result["spelling_errors"], [])
        self.assertEqual(result["grammar_errors"], [])
        self.assertEqual(result["custom_errors"], [])

    def test_sentence_initial_name_is_reported_but_unchanged(self):
        # "Mariah" may be a name, so it gets a row but isn't corrected; "Paris" is a known word
        text = "Mariah aime Paris."
        result = self.analyzer.analyze_text(text)
        self.assertEqual(result["final_text"], text)
        self.assertEqual([e["error"] for e in result["spelling_errors"]], ["Mariah"])
        self.assertEqual(result["spelling_errors"][0]["message"],
                         "Mot non reconnu — s'il s'agit d'un nom propre, ignorez cette remarque.")
        self.assertEqual(result["grammar_errors"], [])
        self.assertEqual(result["custom_errors"], [])

    def test_misspelled_word_uses_top_suggestion_with_row(self):
        result = self.analyzer.analyze_text("Le foteuil est grand.")
        self.assertEqual(result["final_text"], "Le fauteuil est grand.")
        self.assertEqual(len(result["spelling_errors"]), 1)
        error = result["spelling_errors"][0]
        self.assertEqual(error["error"], "foteuil")
        self.assertIn("fauteuil", error["suggestions"])
        self.assertLessEqual(len(error["suggestions"]), 3)
        self.assertEqual(error["message"], "Mot non reconnu : vérifiez l'orthographe.")
        self.assertEqual(result["grammar_errors"], [])

    def test_repeated_misspelling_reported_once(self):
        result = self.analyzer.analyze_text("Le foteuil et le foteuil.")
        self.assertEqual(result["final_text"], "Le fauteuil et le fauteuil.")
        self.assertEqual([e["error"] for e in result["spelling_errors"]], ["foteuil"])


if __name__ == '__main__':
    unittest.main()
