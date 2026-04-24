import sys
import os
import uuid
import time
import tempfile

# Add parent directory to path for importing custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify, send_file
from src.analyze import FrenchAnalyzer
from gtts import gTTS

# Initialize Flask app with custom static folder path
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
print(f"Static folder: {app.static_folder}")

# Initialize French language analyzer
analyzer = FrenchAnalyzer()

# Configure upload directory for audio files
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Sample sentences with common French grammar errors for testing/demo
PRELOADED_SENTENCES = [
    {
        "text": "Je vais à le marché.",
        "description": "Erreur de contraction - devrait être 'au marché'"
    },
    {
        "text": "Je suis aller chez mon mère.",
        "description": "Accord du participe passé et genre - devrait être 'allé' et 'ma mère'"
    },
    {
        "text": "Elle mange un pomme.",
        "description": "Accord de genre - devrait être 'une pomme'"
    },
    {
        "text": "Je mange à école.",
        "description": "Contraction manquante - devrait être 'à l'école'"
    },
    {
        "text": "Il est une belle fille.",
        "description": "Accord pronom-adjectif - devrait être 'Elle est une belle fille'"
    }
]

# French interface text
FRENCH_INTERFACE = {
    "page_title": "Analyseur de Français - Correction Grammaticale et Prononciation",
    "main_heading": "Analyseur de Langue Française",
    "description": "Améliorez votre français avec notre outil d'analyse grammaticale et de prononciation",
    "text_analysis_tab": "Analyse de Texte",
    "audio_analysis_tab": "Analyse Audio",
    "sample_sentences": "Phrases d'exemple :",
    "gender_label": "Genre du locuteur :",
    "gender_masculine": "Masculin",
    "gender_feminine": "Féminin",
    "text_input_placeholder": "Entrez votre texte français ici...",
    "analyze_button": "Analyser",
    "record_button": "Enregistrer",
    "upload_button": "Télécharger un fichier audio",
    "results_heading": "Résultats de l'analyse :",
    "transcription_label": "Transcription :",
    "errors_label": "Erreurs détectées :",
    "corrected_text_label": "Texte corrigé :",
    "accent_label": "Accent détecté :",
    "pronunciation_corrections_label": "Corrections de prononciation :",
    "no_errors": "Aucune erreur détectée. Excellent travail !",
    "demo_popup": "Cette application fournit des corrections grammaticales, d'accent et de prononciation pour vous aider à améliorer votre français.",
    "error_no_audio": "Aucun fichier audio fourni",
    "error_no_text": "Aucun texte fourni",
    "file_not_found": "Fichier non trouvé"
}

@app.route('/')
def index():
    """
    Renders the main page with preloaded sample sentences and French interface.
    """
    return render_template('index.html',
                            sentences=PRELOADED_SENTENCES,
                            interface=FRENCH_INTERFACE)

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    """
    Analyzes uploaded audio file for pronunciation and grammar errors.

    Expected form data:
    - audio: Audio file (.wav)
    - gender: 'masculine' or 'feminine' for grammar agreement
    - recruiter_mode: 'true' for demo mode with popup

    Returns JSON with transcription, errors, corrections, and accent analysis.
    """
    recruiter_mode = request.form.get('recruiter_mode') == 'true'
    gender = request.form.get('gender', 'masculine')

    # Convert gender to French and capitalize for display
    display_gender = ""
    if gender.lower() == "feminine":
        display_gender = FRENCH_INTERFACE["gender_feminine"] # Uses "Féminin"
    elif gender.lower() == "masculine":
        display_gender = FRENCH_INTERFACE["gender_masculine"] # Uses "Masculin"
    else:
        display_gender = "Non spécifié" # Fallback, though your UI should prevent this.

    if 'audio' not in request.files:
        return jsonify({"error": FRENCH_INTERFACE["error_no_audio"]}), 400

    audio = request.files['audio']
    if audio.filename == '' or not audio.filename.endswith('.wav'):
        return jsonify({"error": "Seul le format .wav est accepté"}), 400

    # Save audio file temporarily
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{uuid.uuid4()}.wav')
    audio.save(filename)

    try:
        # Analyze audio using French analyzer
        result = analyzer.analyze_speech(filename, speaker_gender=gender)

        response = {
            "transcription": result["transcription"],
            "errors": result["errors"],
            "corrected_text": result["corrected_text"],
            "accent": result["accent"],
            "audio": result.get("audio_path"),
            "pronunciation_corrections": result.get("pronunciation_corrections", []),
            "recruiter_mode": recruiter_mode,
            "interface": FRENCH_INTERFACE,
            "display_gender": display_gender # Added for UI display
        }

        if recruiter_mode:
            response["popup"] = FRENCH_INTERFACE["demo_popup"]

        return jsonify(response)

    finally:
        # Clean up temporary file
        if os.path.exists(filename):
            os.remove(filename)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """
    Analyzes submitted text for French grammar errors.

    Expected JSON data:
    - text: French text to analyze
    - gender: 'masculine' or 'feminine' for grammar agreement
    - recruiter_mode: 'true' for demo mode with popup

    Returns JSON with original text, detected errors, and corrections.
    """
    data = request.get_json()
    recruiter_mode = data.get('recruiter_mode', False)
    gender = data.get('gender', 'masculine')
    text = data.get('text', '')

    # Convert gender to French and capitalize for display
    display_gender = ""
    if gender.lower() == "feminine":
        display_gender = FRENCH_INTERFACE["gender_feminine"] # Uses "Féminin"
    elif gender.lower() == "masculine":
        display_gender = FRENCH_INTERFACE["gender_masculine"] # Uses "Masculin"
    else:
        display_gender = "Non spécifié" # Fallback

    if not text:
            return jsonify({"error": FRENCH_INTERFACE["error_no_text"]}), 400

    # UPDATE THIS SECTION:
    analysis_results = analyzer.analyze_text(text, speaker_gender=gender)
        
    # Extract the data from the dictionary we built in analyze.py
    corrected_text = analysis_results["final_text"]
    grammar_matches = analysis_results["grammar_errors"]
    spelling_errors = analysis_results["spelling_errors"]

    # We use spelling_errors + grammar_matches for the 'errors' list the UI expects
    # This keeps your UI working without needing to rewrite the frontend!
    # 1. Initialize the error list with spelling fixes
    all_errors = spelling_errors 

    # 2. Format the grammar matches so the UI Table can read them
    for m in grammar_matches:
        all_errors.append({
            "error": text[m.offset:m.offset + m.errorLength], # This fills the 'Original' column
            "suggestions": m.replacements[:3],               # This fills the 'Suggestion' column
            "message": m.message,
            "context": m.context
        })

    # 3. Handle custom grammar fixes (like 'allée' or 'ma mère') 
    # if they aren't already in grammar_matches
    if corrected_text != text and not grammar_matches and not spelling_errors:
        all_errors.append({
            "error": "Grammaire/Genre",
            "suggestions": [corrected_text],
            "message": "Correction d'accord personnalisée."
        })

    result = {
        "transcription": text,
        "errors": all_errors,
        "corrected_text": corrected_text,
        "accent": "N/A", 
        "audio": None, 
        "pronunciation_corrections": [] 
    }

    response = {
        "transcription": result["transcription"],
        "errors": result["errors"],
        "corrected_text": result["corrected_text"],
        "accent": result["accent"],
        "audio": result["audio"],
        "pronunciation_corrections": result["pronunciation_corrections"],
        "recruiter_mode": recruiter_mode,
        "interface": FRENCH_INTERFACE,
        "display_gender": display_gender # Added for UI display
    }

    if recruiter_mode:
        response["popup"] = FRENCH_INTERFACE["demo_popup"]

    return jsonify(response)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Generates audio from text using gTTS.

    Expected JSON data:
    - text: Text to convert to speech
    - lang: Language code (e.g., 'fr')
    - gender: 'masculine' or 'feminine' (currently unused by gTTS)

    Returns audio file as response.
    """
    data = request.get_json()
    text = data.get('text', '')
    lang = data.get('lang', 'fr')

    if not text:
        return jsonify({"error": "Aucun texte fourni pour la synthèse vocale"}), 400

    try:
        # Generate temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_file.name)

            # Send file and clean up
            response = send_file(temp_file.name, mimetype='audio/mpeg')

        # Schedule cleanup after response is sent
        # Note: This os.remove might execute before send_file fully streams the file.
        # For production, consider using a background task queue or Flask's after_request hook
        # to ensure the file is served before deletion. For a small demo, this might be acceptable.
        os.remove(temp_file.name)
        return response

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération audio : {str(e)}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    Serves static files (CSS, JS, images) with debugging information.
    """
    static_path = os.path.join(app.static_folder, filename)
    print(f"Serving file: {static_path}, Exists: {os.path.exists(static_path)}")

    if os.path.exists(static_path):
        return send_file(static_path)
    return FRENCH_INTERFACE["file_not_found"], 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
