## Nuansa — AI-Powered French Tutor | End-to-End AI Product Development Case Study

[🎥 Watch the demo here](https://drive.google.com/file/d/1Pg62pcAmxF_dRQyI7ZhqZzax-O6a576N/view?usp=sharing)

Nuansa is an independent AI product-development initiative focused on building practical AI applications. Its first product, an AI-powered French Tutor, demonstrates end-to-end AI product leadership, including speech recognition, NLP, and text-to-speech integration.

As the project lead, I defined the product vision and requirements, directed the AI system architecture, defined the testing approach, and coordinated AI-assisted development workflows to deliver a working AI application that runs locally.

The French Tutor gives explainable feedback on written and spoken French by combining speech recognition (Whisper), spelling and grammar checking (pyenchant, LanguageTool, and custom rules), and text-to-speech (gTTS).

### Features
- Gender-Aware Grammar Feedback: Corrects grammar based on the speaker's gender (e.g., "je suis allé" vs. "je suis allée", "content" vs. "contente") and explains each correction in French.
- Rule-Based Grammar Engine: Combines a French spell check (pyenchant), LanguageTool (via language_tool_python), and custom regex rules for common learner errors such as contractions ("à le" → "au"), noun gender ("mon mère" → "ma mère"), and accent marks.
- Speech Input:
  - Speech-to-Text (STT): Transcribes spoken French with OpenAI Whisper, then grammar-checks the transcription.
  - Transcription Cleanup: A fixed list of fixes for common Whisper mistranscriptions (e.g., "alair" → "aller", "ecolay" → "école") is applied before grammar checking. This is not a pronunciation score; the app does not measure how a word was pronounced.
- Text-to-Speech (TTS): Reads the corrected sentence aloud in French using gTTS (Google Text-to-Speech).
- Error Table: Shows each error, a suggested fix, and an explanation in French.
- Read-Aloud Practice ("Lecture à voix haute"):
  - Pick a practice sentence (the corrected versions of the sample sentences, with "allé"/"allée" following the selected gender) and click "🔊 Écouter la phrase" to hear it with gTTS.
  - Record yourself in the browser with the microphone, or upload a `.wav` file. Browser recordings (`.webm`, `.ogg`, `.mp4`) are converted to `.wav` with FFmpeg.
  - Whisper transcribes the recording with the larger "small" model (the grammar analysis keeps the faster "base" model), **without** the transcription cleanup list and without giving Whisper the target sentence as a hint, so mistranscriptions are not hidden. The transcription is then aligned with the target sentence word by word, using `difflib` so one missing word doesn't mark every following word wrong.
  - Because this checks what was said, not spelling, words that sound the same count as a match: case, punctuation and accents are ignored ("a"/"à", "ou"/"où", "ecole"/"école"), and word endings that sound like /e/ (-é, -ée, -és, -ées, -er, -ez) match each other ("allé"/"aller"/"allez", "marché"/"marcher").
  - The target sentence is shown in its correct spelling, with recognized words in green and unrecognized or missing words in red, plus a score such as "4/5 mots reconnus". Click a red word to hear it.
  - Silent or near-silent recordings (for example when the browser's echo cancellation removes the sound) are detected from the recording's loudness and Whisper's no-speech probability. Instead of a transcription, the app shows « Aucune parole détectée. Rapprochez-vous du micro et réessayez. », since Whisper tends to invent words from near-silence.
  - Red words are words the app didn't recognize clearly. This is not a precise pronunciation score: Whisper can mishear a well-pronounced word.

### Future Ideas
- Real pronunciation scoring: compare the learner's audio with a reference pronunciation (e.g., phoneme-level alignment) to point out mispronounced sounds, instead of only fixing known mistranscriptions.

### Technical Leadership
As project lead, I was responsible for:
- Defining the product vision and user experience.
- Designing the overall AI system architecture.
- Establishing product requirements and technical priorities.
- Defining what correct output looks like for grammar corrections and their French explanations, and turning it into automated tests.
- Coordinating AI-assisted engineering workflows using Claude, Gemini, ChatGPT, and Grok to accelerate prototyping, debugging, documentation, and iterative product development.
- Iteratively validating functionality through automated tests and manual testing with text and audio input.
- Integrating speech recognition, grammar checking, and text-to-speech into a single workflow.

## Testing
The grammar and spelling logic is covered by automated unit tests in `nuansa-french-tutor/tests/`:
- `test_custom_rules.py`: Checks the custom rules: contractions, feminine speaker agreement, "c'est" vs. "il/elle est", and the French explanations shown in the error table. It also checks that **correct sentences stay unchanged** (e.g., "Je suis à Paris.", "Les filles sont mignonnes.", "Je vais commencer à le faire.") and produce no error rows, so the rules don't over-correct.
- `test_language_tool.py`: Checks that LanguageTool and the French dictionary catch grammar and spelling errors.
- `test_read_aloud.py`: Checks the read-aloud word comparison: a perfect match, a missing word, a wrong word, extra words, case/punctuation differences, words that sound the same (accents, /e/ endings such as "marché"/"marcher"), and the detection of silent recordings.

Speech transcription and text-to-speech are not covered by automated tests; they were checked by hand with the sample audio files. See "Run the Tests" below.

### Screenshots
#### Main Interface (Initial State)
This screenshot shows the main interface of Nuansa’s AI-powered French Tutor in its initial state, providing the core layout for user interaction. It includes the header, example errors, an empty text input field, audio upload options, and gender selection. 

![GUI MainInterfaceInitial](screenshots/main_interface_initial.png)

#### Main Interface (Populated with Input)
This screenshot shows the main interface of Nuansa's AI-driven French Tutor with user input, illustrating the app's readiness for analysis.

![GUI MainInterfacePopulated](screenshots/main_interface_populated.png)

#### Feedback Interface (Results Display) 
After the user clicks the "Analyser" button, this section displays and dynamically extends the main interface to present the analysis results. It provides comprehensive feedback for both text input and uploaded audio files.

In this section:
- The "Transcription du texte français : " field shows the original input text (or, for audio, the Whisper transcription after known mistranscriptions have been fixed).
- For **text input**, the app applies spelling (including accent marks) and grammar corrections.
- For **audio input**, the app first transcribes the speech with Whisper (e.g., "Je suis aller à école" from `input.wav`), fixes known Whisper mistranscriptions, and then applies the same spelling and grammar corrections as for text.
- In both cases, "🔊 Écouter la correction" reads the corrected sentence aloud with gTTS.

This example shows text analysis of the sentence "Je suis aller chez mon mère" demonstrating the system's multi-layered error detection:
- Grammar Corrections: Detects errors such as incorrect past participle agreement ("aller" should be "allée" for feminine gender) and incorrect determiner agreement ("mon mère" should be "ma mère"), providing the corrected sentence.
- French Explanations: Provides detailed explanations in French for each grammatical correction.

![GUI TextInputResults](screenshots/text_input_results.png)

## System Architecture
```
Speech Input (.wav)              Text Input
        │                             │
        ▼                             │
OpenAI Whisper (transcription)        │
        │                             │
        ▼                             │
Fixes for common Whisper              │
mistranscriptions                     │
        │                             │
        └──────────────┬──────────────┘
                       ▼
          Spell Check (pyenchant, fr_FR)
                       │
                       ▼
          Grammar Engine
          (LanguageTool + custom regex rules,
           gender-aware)
                       │
                       ▼
          Flask Web Interface
          (corrected text + error table
           with French explanations)
                       │
                       ▼
          Text-to-Speech (gTTS), on request:
          reads the corrected sentence aloud
```

### Files
- `nuansa-french-tutor/app/main.py`: Runs the Flask application, with routes for the homepage, text analysis (`/analyze_text`), audio analysis (`/analyze_audio`), read-aloud practice (`/read_aloud`), text-to-speech (`/tts`), and static files.
- `nuansa-french-tutor/app/templates/index.html`: Provides the user interface with input fields for text or audio, buttons to trigger analysis, and a section to display feedback results.
- `nuansa-french-tutor/app/static/images/french-girl-icon.png`: French tutor image displayed in the application.
- `nuansa-french-tutor/app/static/audio/input.wav`, `input2.wav`, `input2.m4a`: Sample audio files containing example input.
- `nuansa-french-tutor/src/analyze.py`: Transcribes audio with Whisper, fixes common Whisper mistranscriptions, checks spelling with pyenchant and grammar with LanguageTool plus custom rules, and generates audio with gTTS.
- `nuansa-french-tutor/src/read_aloud.py`: Compares the read-aloud target sentence with the Whisper transcription word by word.
- `nuansa-french-tutor/tests/test_read_aloud.py`: Contains unit tests for the read-aloud word comparison.
- `nuansa-french-tutor/tests/test_language_tool.py`: Contains unit tests for grammar-checking functionality using language_tool_python.
- `nuansa-french-tutor/tests/test_custom_rules.py`: Contains unit tests for the custom regex rules (contractions, feminine speaker agreement, and the custom-rule errors shown in the error table).
- `requirements.txt`: Lists the Python dependencies required to run the application.

### Requirements
- Python 3.10 (required for compatibility with specific library versions, e.g., Whisper, as some libraries may have issues with the system default Python 3.13). Check your version with: `python3 --version`.
- Java 17 (required by language_tool_python to run the LanguageTool grammar engine).
- FFmpeg (required by OpenAI Whisper for audio transcription).
- An internet connection (required for gTTS to generate audio and for LanguageTool to download its local grammar engine on first run).
- flask==3.0.3
- language-tool-python==2.9.3
- torch>=2.4.0
- gTTS==2.5.3
- openai-whisper==20240930
- numpy==1.26.4
- pyenchant==3.2.2
  
### Setup and Usage
#### Option 1: From GitHub (First Time Setup)
- **Note**: Start in your Documents/Projects directory: `cd ~/Documents/Projects/` 
1. Clone the repository: `git clone https://github.com/mariahcoleno/nuansa-french-tutor.git`
2. Navigate to the **repository root** (the outer `nuansa-french-tutor` folder): `cd nuansa-french-tutor/`
3. Create a virtual environment: `python3.10 -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate`
   On Windows: `venv\Scripts\activate`
5. (Optional) Upgrade tools: pip install `--upgrade pip setuptools wheel` 
6. Install dependencies: `pip install -r requirements.txt`
   - If requirements.txt is missing, install manually: 
     ```
     pip install flask==3.0.3 language-tool-python==2.9.3 torch>=2.4.0 gTTS==2.5.3 openai-whisper==20240930 numpy==1.26.4 pyenchant==3.2.2
     ```
7. Navigate to the **application source code folder** (the inner `nuansa-french-tutor` folder): `cd nuansa-french-tutor/`
8. Proceed to "Run the App" below.

#### Option 2: Local Setup (Existing Repository)
1. Navigate to your **local repository root** (the outer `nuansa-french-tutor` folder): `cd ~/Documents/Projects/nuansa-french-tutor/`
2. Setup and activate a virtual environment:
   - If an existing virtual environment is present: `source venv/bin/activate`
   - If creating a new virtual environment:
     ```
     python3.10 -m venv venv
     source venv/bin/activate
     ```
   On Windows: `venv\Scripts\activate`
3. (Optional) Upgrade tools: `pip install --upgrade pip setuptools wheel` 
4. Install dependencies (if not already installed): `pip install -r requirements.txt` 
   - If requirements.txt is missing, use the manual install command above.
5. Navigate to the **application source code folder** (the inner `nuansa-french-tutor` folder): `cd nuansa-french-tutor/`
6. Proceed to "Run the App" below.

### System Dependencies
#### Java 17
LanguageTool requires Java 17 to run its local grammar-checking server.
1. On macOS with Homebrew: `brew install openjdk@17`
2. Because `openjdk@17` is keg-only, add it to your PATH:
   ```
   echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```
3. Verify the installation: `java -version`. The output should show Java 17.

#### FFmpeg
OpenAI Whisper requires FFmpeg to process uploaded audio files.
1. On macOS with Homebrew: `brew install ffmpeg`
2. Verify the installation: `ffmpeg -version`

### Run the App
After completing either setup option, make sure you are inside the **inner application source folder**:
```
nuansa-french-tutor/
└── nuansa-french-tutor/
    ├── app/
    ├── src/
    └── tests/
```
The terminal prompt should end with `nuansa-french-tutor %` with the virtual environment activated.
1. Start the Flask application: `python3 -m app.main`
   - For development, turn on the Flask debugger with `FLASK_DEBUG=1 python3 -m app.main` (it is off by default because it can run code from the browser).
2. The first startup may take several minutes. LanguageTool may download its grammar engine the first time the application is run. This download may be approximately 259 MB. Whisper also downloads its "base" (about 140 MB) and "small" (about 460 MB) models on first run.
3. Wait until the terminal displays (by default, the debugger is off):
   ```
    * Serving Flask app 'main'
    * Debug mode: off
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    * Running on http://127.0.0.1:5001
   Press CTRL+C to quit
   ```
   With `FLASK_DEBUG=1`, it shows `* Debug mode: on` instead, plus "Debugger is active!" and a "Debugger PIN".
4. Once the server is running, open a web browser and navigate to: `http://127.0.0.1:5001`. The server listens on 127.0.0.1 only, so the app is only reachable from this computer.
5. Use the interface:
   - Enter text, or select an example from "Testez un texte français incorrect", and click "🔍 Analyser".
   - Or click "Télécharger un fichier audio", choose a `.wav` file, and click "🔍 Analyser".
   - Choose the speaker's gender ("Genre du locuteur") so agreement is corrected for the person speaking.
   - View the transcription, the corrected text, and the error table with errors, suggestions, and French explanations.
   - Click "🔊 Écouter la correction" to hear the corrected sentence.
6. When finished, return to the terminal running Flask and press `Ctrl+C`. This stops the Flask development server.

### Run the Tests
From the **inner application source folder** (the same folder used to run the app), with the virtual environment activated:
```
python3 -m unittest discover tests
```
- The tests need Java 17 (for LanguageTool) and the French pyenchant dictionary, like the app itself.
- `test_language_tool.py` loads the full analyzer, including both Whisper models, so the first run can take a few minutes.
- All tests should finish with `OK`. The tests print debug output (such as "After contraction corrections: ...") while they run; this is expected.

### Sample Data
- Je vais à le marché.
- Je suis aller chez mon mère.
- Elle mange un pomme. 
- `input.wav`: "Je suis aller a école."

### Project Structure
- `nuansa-french-tutor/` (top-level Git repository folder)
  - `.gitignore` 
  - `README.md`
  - `requirements.txt`
  - `screenshots/` (images used in the README) 
  - `venv/` (local Virtual environment)
  - `nuansa-french-tutor/` (main Python application source code folder)
    - `app/`
      - `static/`
        - `audio/` (sample and generated audio files)
          - `input.wav`
          - `input2.wav`
          - `input2.m4a`
        - `images/`
          - `french-girl-icon.png`
      - `templates/`
        - `index.html`
      - `uploads/` (temporary user-uploaded audio files; not versioned)
      - `main.py`
    - `src/`
      - `__init__.py`
      - `analyze.py` 
      - `read_aloud.py`
    - `tests/`
      - `test_custom_rules.py`
      - `test_language_tool.py`
      - `test_read_aloud.py`

### Additional Notes
- The app runs on port 5001 to avoid common port conflicts. Access it at `http://127.0.0.1:5001` after starting the server.
- The first time the application runs, `language_tool_python` downloads the LanguageTool grammar engine to the local cache. This may take several minutes.
- The nuansa-french-tutor/app/static/audio/ directory contains sample and generated audio files used by the application.
- If port 5001 is already in use, check with: `lsof -i :5001`. You can run the application on a different port by modifying the port setting in `nuansa-french-tutor/app/main.py` (for example, change `port=5001` to `port=5002`) and then access the application at `http://127.0.0.1:5002`.

### License
- All rights reserved. Contact colenomariah92@gmail.com for licensing inquiries.

### Development Notes
- Modern AI engineering tools (Claude, Gemini, ChatGPT, and Grok) were used to accelerate prototyping, debugging, documentation, and iterative refinement during development.
- As project lead, I directed the overall system architecture, product requirements, testing approach, product design decisions, and technical direction.
- Documentation graphics and interface illustrations were created using ChatGPT and refined with GIMP.
 
