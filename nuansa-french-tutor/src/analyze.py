import language_tool_python
import enchant
import whisper
import torch
import librosa
import numpy as np
from transformers import pipeline
import os
from sklearn.svm import SVC
import pickle
import shap
import re
import uuid
import time
from gtts import gTTS

class FrenchAnalyzer:
    """
    A comprehensive French language analyzer that provides grammar checking,
    speech recognition, accent classification, and audio feedback generation.
    """

    def __init__(self):
        """
        Initialize the French analyzer with all necessary models and tools.
        """
        self.grammar_tool = language_tool_python.LanguageTool('fr')
        self.grammar_tool.enabledCategories = 'GRAMMAR,TYPOGRAPHY,STYLE'

        self.d = enchant.Dict("fr_FR") 
        self.whisper_model = whisper.load_model("base")

        self.classifier = None
        self.shap_explainer = None

        if os.path.exists("src/accent_classifier.pkl"):
            with open("src/accent_classifier.pkl", "rb") as f:
                self.classifier = pickle.load(f)

        self.feedback_generator = pipeline("text-generation", model="distilgpt2")


    def apply_corrections(self, text, matches, speaker_gender="masculine"):
        """
        Apply grammar corrections to French text with gender-aware adjustments.
        speaker_gender refers to the gender of the person speaking, not objects in the sentence.
        """
        corrected = text.strip()
        print(f"Initial text for correction: '{corrected}', Speaker gender: {speaker_gender}")

        # Apply specific corrections based on common French errors

        # 1. Handle contractions with prepositions
        # à + le = au
        corrected = re.sub(r'\bà\s+le\b', 'au', corrected, flags=re.IGNORECASE)
        print(f"After 'à le -> au' correction: '{corrected}'")

        # à + les = aux
        corrected = re.sub(r'\bà\s+les\b', 'aux', corrected, flags=re.IGNORECASE)
        print(f"After 'à les -> aux' correction: '{corrected}'")

        # de + le = du
        corrected = re.sub(r'\bde\s+le\b', 'du', corrected, flags=re.IGNORECASE)
        print(f"After 'de le -> du' correction: '{corrected}'")

        # de + les = des
        corrected = re.sub(r'\bde\s+les\b', 'des', corrected, flags=re.IGNORECASE)
        print(f"After 'de les -> des' correction: '{corrected}'")

        # 2. Handle 'à l'école' correction (elision)
        corrected = re.sub(r'\b[aà]\s+école\b', "à l'école", corrected, flags=re.IGNORECASE)
        print(f"After 'à l'école' correction: '{corrected}'")

        # 3. Plural noun corrections
        corrected = re.sub(r'\bles chat\b', 'les chats', corrected, flags=re.IGNORECASE)
        print(f"After plural noun correction: '{corrected}'")

        # 4. Plural adjective agreement (chats are masculine, so "mignons")
        corrected = re.sub(r'\bsont mignon(ne)?s?\b', 'sont mignons', corrected, flags=re.IGNORECASE)
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
            # "Je suis" + past participle agreements
            if re.search(r'\bje suis\s+\w*é\b', corrected, flags=re.IGNORECASE):
                # Make past participles agree with feminine speaker
                corrected = re.sub(r'\bje suis ([^aeiou\s]*[^e])é\b', r'je suis \1ée', corrected, flags=re.IGNORECASE)
                print(f"After feminine past participle agreement: '{corrected}'")

            # Self-descriptive adjectives
            corrected = re.sub(r'\bje suis ([^aeiou\s]*[^e])\b', r'je suis \1e', corrected, flags=re.IGNORECASE)
            print(f"After feminine adjective agreement for speaker: '{corrected}'")

        # 9. Fix obvious gender disagreements in sentences (semantic errors)
        # "Il est une belle fille" -> "Elle est une belle fille" (regardless of speaker gender)
        # This is about fixing the logic of the sentence, not the speaker's gender
        if re.search(r'\bil est une\b.*\bfille\b', corrected, flags=re.IGNORECASE):
            corrected = re.sub(r'\bil est une\b', 'Elle est une', corrected, flags=re.IGNORECASE)
            print(f"After fixing semantic gender disagreement (il/fille): '{corrected}'")

        if re.search(r'\belle est un\b.*\bgarçon\b', corrected, flags=re.IGNORECASE):
            corrected = re.sub(r'\belle est un\b', 'Il est un', corrected, flags=re.IGNORECASE)
            print(f"After fixing semantic gender disagreement (elle/garçon): '{corrected}'")

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
        speaker_gender refers to the gender of the person speaking.
        """
        matches = self.grammar_tool.check(text)
        errors = []
    
        # 1. Dictionary Check & Auto-Fix
        words = re.findall(r'\b\w+\b', text)
        corrected_text = text # Start with the original text
        
        for word in words:
            if not self.d.check(word):
                suggestions = self.d.suggest(word)
                if suggestions:
                    best_suggestion = suggestions[0]
                    # Automatically replace the misspelled word in our working text
                    corrected_text = re.sub(rf'\b{word}\b', best_suggestion, corrected_text)
                    
                    errors.append({
                        "error": word,
                        "suggestions": suggestions[:3],
                        "message": f"Le mot '{word}' n'est pas reconnu. Suggestion : {best_suggestion}"
                    })

        # 2. Now pass the partially corrected text to the grammar tool
        # (This helps the grammar tool not get confused by misspellings)
        matches = self.grammar_tool.check(corrected_text)
        
        # 3. Apply the rest of your custom grammar rules
        final_text = self.apply_corrections(corrected_text, matches, speaker_gender)
        
        return final_text, errors


        # --- NEW DICTIONARY CHECK END ---

        print(f"Original text: '{text}', Speaker gender: {speaker_gender}")
        # ... (rest of your existing regex rules)


        print(f"Original text: '{text}', Speaker gender: {speaker_gender}")
        print(f"LanguageTool found {len(matches)} matches")

        # Capture the original text as it was passed to analyze_text
        original_input_text = text

        # Helper function to determine if a phrase starts the original input text
        def is_phrase_at_sentence_start(phrase, full_text_context):
            return full_text_context.strip().lower().startswith(phrase.strip().lower())

        # Function to capitalize a string if the given condition is true
        def capitalize_if(text_to_capitalize, condition):
            return text_to_capitalize.capitalize() if condition else text_to_capitalize

        # Vérifier les erreurs spécifiques que nous traitons et les ajouter à la liste d'erreurs

        # 1. Erreurs de contraction
        if re.search(r'\bà\s+le\b', text, flags=re.IGNORECASE):
            found_error = re.search(r'\bà\s+le\b', text, flags=re.IGNORECASE).group()
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("au", should_capitalize)],
                "message": "Contraction obligatoire : 'à le' devient 'au'."
            })

        if re.search(r'\bà\s+les\b', text, flags=re.IGNORECASE):
            found_error = re.search(r'\bà\s+les\b', text, flags=re.IGNORECASE).group()
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("aux", should_capitalize)],
                "message": "Contraction obligatoire : 'à les' devient 'aux'."
            })

        if re.search(r'\bde\s+le\b', text, flags=re.IGNORECASE):
            found_error = re.search(r'\bde\s+le\b', text, flags=re.IGNORECASE).group()
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("du", should_capitalize)],
                "message": "Contraction obligatoire : 'de le' devient 'du'."
            })

        if re.search(r'\bde\s+les\b', text, flags=re.IGNORECASE):
            found_error = re.search(r'\bde\s+les\b', text, flags=re.IGNORECASE).group()
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("des", should_capitalize)],
                "message": "Contraction obligatoire : 'de les' devient 'des'."
            })

        # 2. Élision avec à + école
        if re.search(r'\b[aà]\s+école\b', text, flags=re.IGNORECASE):
            found_error = re.search(r'\b[aà]\s+école\b', text, flags=re.IGNORECASE).group()
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("à l'école", should_capitalize)],
                "message": "Utiliser 'à l'' devant les mots commençant par une voyelle."
            })

        # 3. Accord des noms au pluriel
        if re.search(r'\bles chat\b', text, flags=re.IGNORECASE):
            found_error = "les chat"
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("les chats", should_capitalize)],
                "message": "Accord au pluriel : 'chat' doit devenir 'chats' avec 'les'."
            })

        # 4. Accord des adjectifs au pluriel
        if re.search(r'\bsont mignon(ne)?s?\b', text, flags=re.IGNORECASE):
            match = re.search(r'\bsont mignon(ne)?s?\b', text, flags=re.IGNORECASE)
            if match and 'mignons' not in match.group().lower():
                found_error = match.group()
                should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
                errors.append({
                    "error": capitalize_if(found_error, should_capitalize),
                    "suggestions": [capitalize_if("sont mignons", should_capitalize)],
                    "message": "Accord de l'adjectif : le masculin pluriel utilise 'mignons'."
                })

        # 5. Conjugaison des verbes
        if re.search(r'\bnous mange\b', text, flags=re.IGNORECASE):
            found_error = "nous mange"
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("nous mangeons", should_capitalize)],
                "message": "Conjugaison : 'mange' doit être 'mangeons' avec 'nous'."
            })

        # 6. Préposition
        if re.search(r'\bdans la cantine\b', text, flags=re.IGNORECASE):
            found_error = "dans la cantine"
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("à la cantine", should_capitalize)],
                "message": "Préposition : utiliser 'à' et non 'dans' avec 'la cantine'."
            })

        # 7. Genre des noms - déterminants
        if re.search(r'\bun pomme\b', text, flags=re.IGNORECASE):
            found_error = "un pomme"
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("une pomme", should_capitalize)],
                "message": "Accord de genre : 'pomme' est féminin, utiliser 'une'."
            })

        if re.search(r'\bmon mère\b', text, flags=re.IGNORECASE):
            found_error = "mon mère"
            should_capitalize = is_phrase_at_sentence_start(found_error, original_input_text)
            errors.append({
                "error": capitalize_if(found_error, should_capitalize),
                "suggestions": [capitalize_if("ma mère", should_capitalize)],
                "message": "Accord de genre : 'mère' est féminin, utiliser 'ma'."
            })

        # 8. Erreurs sémantiques de genre (phrases illogiques)
        if re.search(r'\bil est une\b.*\bfille\b', text, flags=re.IGNORECASE):
            found_error = "il est une ... fille" # Placeholder, actual capture depends on full match
            # For semantic errors spanning larger text, checking exact start is tricky
            # Instead, we can assume if the input starts with 'il est une' AND matches 'fille',
            # it might be the start of the sentence for capitalization.
            should_capitalize = is_phrase_at_sentence_start("il est une", original_input_text)
            errors.append({
                "error": capitalize_if("il est une ... fille", should_capitalize),
                "suggestions": [capitalize_if("elle est une ... fille", should_capitalize)],
                "message": "Erreur sémantique : utiliser 'elle' pour parler d'une fille."
            })

        if re.search(r'\belle est un\b.*\bgarçon\b', text, flags=re.IGNORECASE):
            found_error = "elle est un ... garçon" # Placeholder
            should_capitalize = is_phrase_at_sentence_start("elle est un", original_input_text)
            errors.append({
                "error": capitalize_if("elle est un ... garçon", should_capitalize),
                "suggestions": [capitalize_if("il est un ... garçon", should_capitalize)],
                "message": "Erreur sémantique : utiliser 'il' pour parler d'un garçon."
            })

        # 9. Accord du participe passé pour l'auto-référence du locuteur
        # Capture the exact "je suis [form of aller]" if it exists
        # Group 1: The entire "je suis [participle]" phrase (e.g., "je suis aller")
        # Group 2: Just the participle (e.g., "aller")
        match_aller = re.search(r'\b(je suis\s+(aller|allé|allée|allés|allées))\b', text, flags=re.IGNORECASE)

        if match_aller:
            found_error_phrase = match_aller.group(1) # e.g., "je suis aller", "je suis allé"
            current_participle = match_aller.group(2) # e.g., "aller", "allé"

            suggestion_text = ""
            error_message = ""
            add_error = False

            # This flag determines if both the 'error' and 'suggestion' should be capitalized.
            should_capitalize = is_phrase_at_sentence_start(found_error_phrase, original_input_text)

            if speaker_gender.lower() == "feminine":
                # If current participle is not 'allée' (or 'allee' transcribed), then it's an error for feminine speaker
                if current_participle.lower() not in ["allée", "allee"]:
                    suggestion_text = "je suis allée"
                    error_message = "Accord du participe passé : utiliser 'allée' pour une locutrice avec être."
                    add_error = True
            else: # Masculine or default (treat as masculine)
                # If current participle is not 'allé' (or 'alle' for common typo) then it's an error for masculine speaker
                if current_participle.lower() not in ["allé", "alle"]:
                    suggestion_text = "je suis allé"
                    error_message = "Accord du participe passé : utiliser 'allé' pour un locuteur masculin avec être."
                    add_error = True

            if add_error:
                errors.append({
                    "error": capitalize_if(found_error_phrase, should_capitalize),
                    "suggestions": [capitalize_if(suggestion_text, should_capitalize)],
                    "message": error_message
                })

        # Apply corrections (this will now also handle final capitalization)
        corrected_text = self.apply_corrections(text, matches, speaker_gender=speaker_gender)

        print(f"Found {len(errors)} total errors")
        print(f"Corrected text: '{corrected_text}'")

        return errors, corrected_text

    def analyze_speech(self, audio_file, speaker_gender="masculine"):
        """
        Analyze French speech audio for pronunciation and grammar errors.
        speaker_gender refers to the gender of the person speaking.
        """
        audio, sr = librosa.load(audio_file, sr=16000)

        result = self.whisper_model.transcribe(audio_file, language='fr')
        # Ensure it's lowercase for consistent processing by analyze_text and regex rules
        text = result["text"].replace(',', '').strip().lower()


        pronunciation_corrections = []
        if 'alair' in text:
            text = text.replace('alair', 'aller')
            pronunciation_corrections.append({"error": "alair", "corrected": "aller"})
        if 'ecolay' in text:
            text = text.replace('ecolay', 'école')
            pronunciation_corrections.append({"error": "ecolay", "corrected": "école"})


        print(f"Transcription for grammar analysis: {text}")
        print(f"Pronunciation corrections: {pronunciation_corrections}")

        # This call will now return a correctly capitalized sentence
        errors, corrected_text = self.analyze_text(text, speaker_gender=speaker_gender)
        print(f"Errors in order: {[error['error'] for error in errors]}")

        feedback_parts = []
        if pronunciation_corrections:
            feedback_parts.append("Corrections de prononciation :")
            for correction in pronunciation_corrections:
                feedback_parts.append(f"Vous avez prononcé {correction['error']} mais cela a été corrigé en {correction['corrected']}.")

        if errors and any(error['suggestions'] for error in errors):
            feedback_parts.append("Corrections grammaticales :")
            feedback_parts += [f"Changer {error['error']} en {error['suggestions'][0]}"
                               for error in errors if error['suggestions']]

        feedback_text = " ".join(feedback_parts) if feedback_parts else "Aucune erreur trouvée."
        print(f"Feedback text: {feedback_text}")

        audio_path = self.generate_feedback_audio(feedback_text) if feedback_text.strip() else None

        features = self.extract_features(audio, sr)

        if self.classifier:
            accent = self.classifier.predict([features])[0]
            shap_values = self.shap_explainer.shap_values([features]) if self.shap_explainer else None
        else:
            accent = "Unknown"
            shap_values = None

        return {
            "transcription": text,
            "errors": errors,
            "corrected_text": corrected_text,
            "accent": accent,
            "shap_values": shap_values,
            "audio_path": audio_path,
            "pronunciation_corrections": pronunciation_corrections
        }

    def extract_features(self, audio, sr):
        """
        Extract MFCC features from audio for accent classification.
        """
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)

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

























