"""
Tests for French grammar checking functionality using LanguageTool and PyEnchant.
Validates detection of common French grammar errors like past participle
and gender agreement mistakes.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import language_tool_python
from src.analyze import FrenchAnalyzer

class TestLanguageTool(unittest.TestCase):
    def setUp(self):
        self.tool = language_tool_python.LanguageTool('fr')
        self.analyzer = FrenchAnalyzer()

    def test_grammar_check(self):
        """Test detection of French grammar errors in sample text."""
        text = "Je suis aller chez mon mère"
        matches = self.tool.check(text)
        
        # Expected errors
        expected_errors = [
            {"error": "aller", "suggestions": ["allé", "allés", "allée", "allées"]},
            {"error": "mon mère", "suggestions": ["ma mère"]}
        ]
        
        # Verify number of errors
        self.assertEqual(len(matches), 2, "Should find exactly 2 grammar errors")
        
        # Verify each error
        for match, expected in zip(matches, expected_errors):
            error_text = text[match.offset: match.offset + match.errorLength]
            self.assertEqual(error_text, expected["error"], f"Expected error '{expected['error']}'")
            self.assertTrue(set(expected["suggestions"]).issubset(set(match.replacements)),
                           f"Expected suggestions {expected['suggestions']} in {match.replacements}")

    def test_dictionary_check(self):
            """Test detection of spelling errors missed by LanguageTool."""
            text = "Le foteuil est pour mon chomage"
            # This calls your function and gets the result
            result = self.analyzer.analyze_text(text)
            
            # We convert everything to a string to see if the misspelled words are mentioned
            result_str = str(result)
            
            self.assertIn("foteuil", result_str, "The dictionary should catch 'foteuil'")
            self.assertIn("chomage", result_str, "The dictionary should catch 'chomage'")
            
            print("Test Passed: Dictionary caught spelling errors successfully.")


    def tearDown(self):
        self.tool.close()

if __name__ == '__main__':
    unittest.main()
