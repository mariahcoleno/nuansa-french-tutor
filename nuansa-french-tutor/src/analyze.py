import language_tool_python
import enchant
import whisper
import torch
import os
import re
import uuid
import time
from gtts import gTTS

# Pinned LanguageTool release. The library's default ('latest' snapshot) re-downloads
# ~260 MB on every startup because it looks for a hardcoded 6.7-SNAPSHOT folder name
# that newer snapshots don't unzip to. A release downloads once to a matching folder.
LANGUAGETOOL_VERSION = "6.6"

# LanguageTool rules for à/de + le/les contractions, applied to the corrected text
CONTRACTION_RULES = {"A_LE", "DE_LE"}

# LanguageTool rules never shown in the error table, because apply_corrections already fixes them
# POINTS_2: missing period at the end of the sentence
IGNORED_RULES = {"POINTS_2"}

# Self-descriptive adjectives a feminine speaker should agree after "je suis"
FEMININE_ADJECTIVES = {
    "content": "contente",
    "heureux": "heureuse",
    "prêt": "prête",
    "grand": "grande",
    "petit": "petite",
    "français": "française",
    "américain": "américaine",
    "fatigué": "fatiguée",
    "désolé": "désolée",
}

C_EST_MESSAGE = "Devant un nom précédé de « un/une », on emploie « c'est » plutôt que « il/elle est »."


def words_changed(original, corrected):
    """
    True if a correction changed a word, ignoring the final punctuation,
    capitalization and spacing that apply_corrections always normalizes.
    """
    def normalize(text):
        return " ".join(text.strip().rstrip(".!?").split()).lower()
    return normalize(original) != normalize(corrected)

class FrenchAnalyzer:
    """
    A comprehensive French language analyzer that provides grammar checking,
    speech recognition, and audio feedback generation.
    """

    def __init__(self):
        """
        Initialize the French analyzer with all necessary models and tools.
        """
        self.grammar_tool = language_tool_python.LanguageTool('fr', language_tool_download_version=LANGUAGETOOL_VERSION)
        self.grammar_tool.enabledCategories = 'GRAMMAR,TYPOGRAPHY,STYLE'

        self.d = enchant.Dict("fr_FR") 
        self.whisper_model = whisper.load_model("base")

    def apply_corrections(self, text, matches, speaker_gender="masculine"):
        """
        Apply grammar corrections to French text with gender-aware adjustments.
        speaker_gender refers to the gender of the person speaking, not objects in the sentence.
        """
        # 1. Handle contractions with prepositions (à le -> au, de les -> des, ...)
        # LanguageTool's A_LE / DE_LE rules use part-of-speech tagging, so they skip
        # object pronouns ("commencer à le faire"). Splice right-to-left so earlier
        # offsets stay valid.
        corrected = text
        contraction_matches = [m for m in matches if m.ruleId in CONTRACTION_RULES and m.replacements]
        for m in sorted(contraction_matches, key=lambda m: m.offset, reverse=True):
            corrected = corrected[:m.offset] + m.replacements[0] + corrected[m.offset + m.errorLength:]
        corrected = corrected.strip()
        print(f"After contraction corrections: '{corrected}', Speaker gender: {speaker_gender}")

        # Apply specific corrections based on common French errors

        # 2. Handle 'à l'école' correction (elision)
        corrected = re.sub(r'\b[aà]\s+école\b', "à l'école", corrected, flags=re.IGNORECASE)
        print(f"After 'à l'école' correction: '{corrected}'")

        # 3. Plural noun corrections
        corrected = re.sub(r'\bles chat\b', 'les chats', corrected, flags=re.IGNORECASE)
        print(f"After plural noun correction: '{corrected}'")

        # 4. Plural adjective agreement (chats are masculine, so "mignons")
        # Limited to "les chats" so correct feminine plurals ("Les filles sont mignonnes") stay unchanged
        corrected = re.sub(r'\bles chats sont mignon(ne)?s?\b', 'les chats sont mignons', corrected, flags=re.IGNORECASE)
        print(f"After plural adjective correction: '{corrected}'")

        # 5. Verb conjugation corrections
        corrected = re.sub(r'\bnous mange\b', 'nous mangeons', corrected, flags=re.IGNORECASE)
        print(f"After verb conjugation correction: '{corrected}'")

        # 6. Preposition corrections
        corrected = re.sub(r'\bdans la cantine\b', 'à la cantine', corrected, flags=re.IGNORECASE)
        print(f"After preposition correction: '{corrected}'")

        # 7. Noun gender corrections (determiners) - these are about the nouns themselves
        corrected = re.sub(r'\bun pomme\b', 'une pomme', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\bmon mère\b', 'ma mère', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\bchez mon mère\b', 'chez ma mère', corrected, flags=re.IGNORECASE)
        print(f"After noun gender corrections: '{corrected}'")

        # 8. Speaker-specific corrections (only for self-reference)
        # These corrections should only apply when the speaker is talking about themselves
        if speaker_gender.lower() == "feminine":
            # For feminine speakers talking about themselves
            # "Je suis" + past participle ending in -é (fatigué -> fatiguée)
            corrected = re.sub(r'\b(je suis )(\w+é)\b', r'\1\2e', corrected, flags=re.IGNORECASE)
            print(f"After feminine past participle agreement: '{corrected}'")

            # Self-descriptive adjectives: only known adjectives directly after "je suis",
            # so "Je suis à Paris" or "Je suis en retard" are left alone
            corrected = re.sub(
                r'\b(je suis )(\w+)\b',
                lambda m: m.group(1) + FEMININE_ADJECTIVES.get(m.group(2).lower(), m.group(2)),
                corrected,
                flags=re.IGNORECASE,
            )
            print(f"After feminine adjective agreement for speaker: '{corrected}'")

        # 9. il/elle est + un/une + noun -> c'est
        # "Il est une belle fille" -> "C'est une belle fille" (regardless of speaker gender)
        if re.search(r'\bil est une\b.*\bfille\b', corrected, flags=re.IGNORECASE):
            corrected = re.sub(r'\bil est une\b', "c'est une", corrected, flags=re.IGNORECASE)
            print(f"After c'est correction (il est une/fille): '{corrected}'")

        if re.search(r'\belle est un\b.*\bgarçon\b', corrected, flags=re.IGNORECASE):
            corrected = re.sub(r'\belle est un\b', "c'est un", corrected, flags=re.IGNORECASE)
            print(f"After c'est correction (elle est un/garçon): '{corrected}'")

        # 10. Past participle agreement with "aller" (only for speaker self-reference)
        if re.search(r'\bje suis\s+aller?\b', corrected, flags=re.IGNORECASE):
            if speaker_gender.lower() == "feminine":
                corrected = re.sub(r'\bje suis aller?\b', 'je suis allée', corrected, flags=re.IGNORECASE)
            else:
                corrected = re.sub(r'\bje suis aller?\b', 'je suis allé', corrected, flags=re.IGNORECASE)
            print(f"After past participle correction for speaker: '{corrected}'")

        # Clean up extra spaces globally BEFORE final capitalization
        corrected = re.sub(r'\s+', ' ', corrected).strip()

        # Ensure sentence ends with period, exclamation mark, or question mark
        # Add a period only if it doesn't end with a common sentence-ending punctuation
        if corrected and not corrected.endswith(('.', '!', '?')):
            corrected += '.'

        # --- FINAL CAPITALIZATION LOGIC FOR THE MAIN CORRECTED TEXT ---
        # Capitalize the very first letter of the string if it's not empty
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]

        # Capitalize after sentence-ending punctuation (., !, ?)
        # This regex ensures capitalization only after these specific characters,
        # followed by optional whitespace, then a lowercase letter.
        # It uses a lambda function to make the matched letter uppercase.
        corrected = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), corrected)


        print(f"Final corrected text: '{corrected}'")
        return corrected

    def analyze_text(self, text, speaker_gender="masculine"):
        """
        Analyze French text for grammar errors and provide corrections.
        """
        # 1. Dictionary Check & Auto-Fix (Fixing spelling/accents FIRST)
        words = re.findall(r'\b\w+\b', text)
        corrected_text = text 
        spelling_errors = [] # Rename to keep it distinct
        
        for word in words:
            if not self.d.check(word):
                suggestions = self.d.suggest(word)
                if suggestions:
                    best_suggestion = suggestions[0]
                    # Replace misspelled word in the text we send to the grammar tool
                    corrected_text = re.sub(rf'\b{word}\b', best_suggestion, corrected_text)
                    
                    spelling_errors.append({
                        "error": word,
                        "suggestions": suggestions[:3],
                        "message": f"Le mot '{word}' n'est pas reconnu."
                    })

        # 2. Run Grammar Tool on the ALREADY SPELLED-CHECKED text
        # This prevents the '9 errors' issue because the spelling is now clean
        all_matches = [m for m in self.grammar_tool.check(corrected_text) if m.ruleId not in IGNORED_RULES]

        # 3. Detect custom-rule errors (with French explanations) on the same text
        custom_errors = self._detect_custom_errors(corrected_text, speaker_gender)

        # Drop LanguageTool matches already covered by a custom rule (e.g. "aller", "mon mère"),
        # since the custom rules know the speaker's gender
        custom_spans = [error.pop("span") for error in custom_errors]
        grammar_matches = [
            m for m in all_matches
            if not any(m.offset < end and start < m.offset + m.errorLength for start, end in custom_spans)
        ]

        # 4. Apply custom gender/grammar rules
        final_text = self.apply_corrections(corrected_text, all_matches, speaker_gender)

        # IMPORTANT: We return the grammar_matches object so the UI can draw dropdowns
        # and the spelling_errors so the user sees the accent fixes.
        return {
            "final_text": final_text,
            "grammar_errors": grammar_matches,
            "spelling_errors": spelling_errors,
            "custom_errors": custom_errors
        }

    def _detect_custom_errors(self, text, speaker_gender="masculine"):
        """
        Find errors handled by the custom rules in apply_corrections and explain them in French.
        Each error carries a "span" (start, end) into text so overlapping LanguageTool matches can be dropped.
        """
        errors = []
        sentence_start = len(text) - len(text.lstrip())

        # Function to capitalize a string if the given condition is true
        def capitalize_if(text_to_capitalize, condition):
            return text_to_capitalize.capitalize() if condition else text_to_capitalize

        def add_error(match, found_error, suggestion, message, group=0):
            # Capitalize both error and suggestion when the match starts the sentence
            should_capitalize = match.start(group) == sentence_start
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if(suggestion, should_capitalize)],
                "message": message,
                "span": match.span(group)
            })

        # 1. Élision avec à + école
        match = re.search(r'\b[aà]\s+école\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, match.group(), "à l'école",
                      "Utiliser 'à l'' devant les mots commençant par une voyelle.")

        # 2. Accord des noms au pluriel
        match = re.search(r'\bles chat\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, "les chat", "les chats",
                      "Accord au pluriel : 'chat' doit devenir 'chats' avec 'les'.")

        # 3. Accord des adjectifs au pluriel (limité à "les chats")
        match = re.search(r'\bles chats? sont mignon(ne)?s?\b', text, flags=re.IGNORECASE)
        if match and not match.group().lower().endswith('mignons'):
            add_error(match, match.group(), "les chats sont mignons",
                      "Accord de l'adjectif : le masculin pluriel utilise 'mignons'.")

        # 4. Conjugaison des verbes
        match = re.search(r'\bnous mange\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, "nous mange", "nous mangeons",
                      "Conjugaison : 'mange' doit être 'mangeons' avec 'nous'.")

        # 5. Préposition
        match = re.search(r'\bdans la cantine\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, "dans la cantine", "à la cantine",
                      "Préposition : « à la cantine » est plus courant que « dans la cantine ».")

        # 6. Genre des noms - déterminants
        match = re.search(r'\bun pomme\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, "un pomme", "une pomme",
                      "Accord de genre : 'pomme' est féminin, utiliser 'une'.")

        match = re.search(r'\bmon mère\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, "mon mère", "ma mère",
                      "Accord de genre : 'mère' est féminin, utiliser 'ma'.")

        # 7. Il/elle est + un/une + nom -> c'est
        match = re.search(r'\bil est une\b.*\bfille\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, "il est une ... fille", "c'est une ... fille", C_EST_MESSAGE)

        match = re.search(r'\belle est un\b.*\bgarçon\b', text, flags=re.IGNORECASE)
        if match:
            add_error(match, "elle est un ... garçon", "c'est un ... garçon", C_EST_MESSAGE)

        # 8. Accord du participe passé pour l'auto-référence du locuteur
        # Group 1: The entire "je suis [participle]" phrase (e.g., "je suis aller")
        # Group 2: Just the participle (e.g., "aller")
        match_aller = re.search(r'\b(je suis\s+(aller|allé|allée|allés|allées))\b', text, flags=re.IGNORECASE)
        if match_aller:
            found_error_phrase = match_aller.group(1)
            current_participle = match_aller.group(2).lower()

            if speaker_gender.lower() == "feminine":
                # If current participle is not 'allée' (or 'allee' transcribed), then it's an error for feminine speaker
                if current_participle not in ["allée", "allee"]:
                    add_error(match_aller, found_error_phrase, "je suis allée",
                              "Accord du participe passé : utiliser 'allée' pour une locutrice avec être.")
            else: # Masculine or default (treat as masculine)
                # If current participle is not 'allé' (or 'alle' for common typo) then it's an error for masculine speaker
                if current_participle not in ["allé", "alle"]:
                    add_error(match_aller, found_error_phrase, "je suis allé",
                              "Accord du participe passé : utiliser 'allé' pour un locuteur masculin avec être.")

        # 9. Accord au féminin pour la locutrice (same rules as apply_corrections)
        if speaker_gender.lower() == "feminine":
            def overlaps_existing(match):
                start, end = match.span(2)
                return any(start < e_end and e_start < end for e_start, e_end in (e["span"] for e in errors))

            # Participe passé en -é après "je suis" (fatigué -> fatiguée)
            for match in re.finditer(r'\b(je suis )(\w+é)\b', text, flags=re.IGNORECASE):
                if not overlaps_existing(match):
                    word = match.group(2)
                    add_error(match, word, word + "e",
                              f"Accord au féminin : la locutrice est une femme, donc le participe passé "
                              f"s'accorde ({word} → {word}e).", group=2)

            # Adjectifs connus après "je suis" (content -> contente)
            for match in re.finditer(r'\b(je suis )(\w+)\b', text, flags=re.IGNORECASE):
                word = match.group(2)
                feminine = FEMININE_ADJECTIVES.get(word.lower())
                if feminine and not overlaps_existing(match):
                    add_error(match, word, feminine,
                              f"Accord au féminin : la locutrice est une femme, donc l'adjectif "
                              f"s'accorde ({word} → {feminine}).", group=2)

        print(f"Found {len(errors)} custom-rule errors")
        return errors

    def analyze_speech(self, audio_file, speaker_gender="masculine"):
        """
        Analyze French speech audio for pronunciation and grammar errors.
        speaker_gender refers to the gender of the person speaking.
        """
        # Force a more "stable" transcription
        result = self.whisper_model.transcribe(
            audio_file, 
            language='fr', 
            task='transcribe',
            fp16=False  # This also silences that 'FP16 not supported' warning!
        )
        
        # Clean up the text
        text = result["text"].replace(',', '').strip().lower()
        
        # Log the raw output so you can see if the hallucinations persist
        print(f"Raw Whisper Output: {text}")


       # 1. Expanded Whisper Cleanup
        whisper_cleanup = {
            "alair": "aller",
            "ecolay": "école",
            "j suis": "je suis",
            "j'suis": "je suis",
            "t es": "tu es",
            "c est": "c'est",
            "collemange": "elle mange",  
            "en pompe": "une pomme",     
            "articlex": "à l'école"       
        }

        pronunciation_corrections = []

        # Apply cleanup
        for error_word, fix in whisper_cleanup.items():
            if error_word in text.lower():
                # We record the fix for the pronunciation feedback
                pronunciation_corrections.append({"error": error_word, "corrected": fix})
                text = text.lower().replace(error_word, fix)

        # KEEP THESE PRINTS - They are your audit trail!
        print(f"Transcription for grammar analysis: {text}")
        print(f"Pronunciation corrections: {pronunciation_corrections}")

        # 2. Final Logic Check for 'un pomme'
        if "un pomme" in text:
            text = text.replace("un pomme", "une pomme")
            print("Fixed masculine/feminine article for 'pomme'.")

        # 1. Catch the dictionary result (instead of unpacking variables)
        analysis_results = self.analyze_text(text, speaker_gender=speaker_gender)

        # 2. Extract the specific items your code needs
        errors = analysis_results["grammar_errors"]
        corrected_text = analysis_results["final_text"]

        # LanguageTool Match objects use .ruleId or .message instead of brackets
        print(f"Errors in order: {[e.ruleId for e in errors]}")

        # --- STEP 1: CONVERT OBJECTS TO DICTIONARIES ---
        # This prevents the "Match object is not subscriptable" error
        # 1. ADD THE CUSTOM-RULE ERRORS FIRST (The ones LanguageTool misses)
        formatted_errors = list(analysis_results["custom_errors"])

        # 2. PROCESS LANGUAGETOOL ERRORS (And override gender)
        for error in errors:
            original_word = error.context[error.offset : error.offset + error.errorLength]
            
            # Use your logic to override the default masculine suggestion
            if original_word.lower() == "aller" and speaker_gender == "feminine":
                suggestion = ["allée"]
                message = "Accord féminin requis ('allée') car le locuteur est une femme."
            else:
                suggestion = error.replacements
                message = error.message

            formatted_errors.append({
                'error': original_word,
                'suggestions': suggestion,
                'message': message
            })
        
        # Now the table will show both your manual school fix AND the gender fix!
        errors = formatted_errors

        # --- STEP 2: BUILD FEEDBACK TEXT ---
        feedback_parts = []
        
        if pronunciation_corrections:
            feedback_parts.append("Corrections de prononciation :")
            for correction in pronunciation_corrections:
                feedback_parts.append(f"Vous avez prononcé {correction['error']} mais cela a été corrigé en {correction['corrected']}.")

        # Now error['suggestions'] works because we converted it in Step 1!
        if errors and any(error['suggestions'] for error in errors):
            feedback_parts.append("Corrections grammaticales :")
            for error in errors:
                if error['suggestions']:
                    feedback_parts.append(f"Changer {error['error']} en {error['suggestions'][0]}.")

        feedback_text = " ".join(feedback_parts) if feedback_parts else "Aucune erreur trouvée."
        print(f"Feedback text: {feedback_text}")


        audio_path = self.generate_feedback_audio(feedback_text) if feedback_text.strip() else None

        return {
            "transcription": text,
            "errors": errors,
            "corrected_text": corrected_text,
            "audio_path": audio_path,
            "pronunciation_corrections": pronunciation_corrections
        }

    def generate_feedback_audio(self, text, filename=None):
        """
        Generate audio feedback using Google Text-to-Speech.
        """
        if not filename:
            filename = f"static/correction_{uuid.uuid4()}.mp3"

        try:
            if not text.strip():
                print("No feedback text to generate audio.")
                return None

            static_dir = os.path.dirname(filename)
            os.makedirs(static_dir, exist_ok=True)
            if not os.access(static_dir, os.W_OK):
                os.chmod(static_dir, 0o755)

            tts_text = text.replace("à l'", "a l").replace("à l", "a l")
            print(f"TTS text: {tts_text}")

            tts = gTTS(tts_text, lang='fr', slow=False)
            tts.save(filename)

            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"Audio saved to {filename} (Size: {file_size} bytes)")

                if file_size == 0:
                    print(f"Audio file {filename} is empty, not serving.")
                    os.unlink(filename)
                    return None

                return "/static/" + os.path.basename(filename) + "?t=" + str(time.time())

            print(f"Audio file {filename} not created.")
            return None

        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
