## Nuansa: AI-Powered French Tutor — End-to-End AI Product Development Case Study

[🎥 Watch the demo here](https://drive.google.com/file/d/1Pg62pcAmxF_dRQyI7ZhqZzax-O6a576N/view?usp=sharing)

Nuansa is an AI-powered language learning application demonstrating end-to-end AI product leadership, including multimodal AI integration, speech AI pipelines, NLP workflows, and systematic AI evaluation.

As the project lead, I defined the product vision, established product requirements, directed the AI system architecture, designed evaluation strategies, and coordinated AI-assisted development workflows to deliver a production-ready AI application.

The application provides real-time personalized feedback on grammar, pronunciation, and speech by combining NLP, speech recognition, text-to-speech, and AI evaluation workflows to deliver explainable language feedback.

### Features
- Real-time Personalized Feedback: Provides instant, gender-aware grammar corrections (e.g., past participle agreement) and pronunciation analysis.
- Rule-Based Grammar Engine: Utilizes a custom system built on language_tool_python and tailored regex patterns for highly specific corrections, including proper diacritical mark usage.
- Dynamic Audio Integration:
  - Speech-to-Text (STT): Integrates OpenAI Whisper for accurate transcription of spoken French, with custom rules to correct phonetic misspellings.
  - Text-to-Speech (TTS): Generates natural-sounding, custom audio feedback in French using gTTS (Google Text-to-Speech).
- Intuitive User Interface: A simple web interface with an interactive error table that provides explainability for corrections, showing the error, a suggested fix, and the error type.

### Technical Leadership
As project lead, I was responsible for:
- Defining the product vision and user experience.
- Designing the overall AI system architecture.
- Establishing product requirements and technical priorities.
- Designing evaluation criteria for transcription accuracy, grammar quality, pronunciation quality, and user explainability.
- Coordinating AI-assisted engineering workflows using Claude, Gemini, ChatGPT, and Grok to accelerate prototyping, debugging, documentation, and iterative product development.
- Iteratively validating functionality through testing and prompt refinement.
- Integrating multiple AI services into a single production workflow.

## AI Evaluation
Nuansa incorporates multiple evaluation layers throughout the AI pipeline:
- Speech transcription accuracy
- Grammar correction quality
- Pronunciation analysis consistency
- Human-readable explanations for every correction
- End-to-end workflow validation
- Functional testing across text and speech inputs

### Screenshots
#### Main Interface (Initial State)
This screenshot displays Nuansa's AI-driven French Tutor's main interface in its initial state, providing the core layout for user interaction. It includes the header, example errors, an empty text input field, audio upload options, and gender selection. 

![GUI MainInterfaceInitial](screenshots/main_interface_initial.png)

#### Main Interface (Populated with Input)
This screenshot displays Nuansa's AI-driven French Tutor's main interface with user input, illustrating the app's readiness for analysis.

![GUI MainInterfacePopulated](screenshots/main_interface_populated.png)

#### Feedback Interface (Results Display) 
After the user clicks the "Analyser" button, this section displays and dynamically extends the main interface to present the analysis results. It provides comprehensive feedback for both text input and uploaded audio files.

In this section:
- The "Transcription du texte français : " field shows the original input text (or transcribed audio after phonetic misspellings have been corrected).
- For **text input**, the app applies grammar corrections and diacritical mark corrections. It then generates custom audio feedback of the corrected sentence with integrated playback controls.
- For **audio input**, the app first performs Speech Recognition and Transcription (e.g., converting "Je suis aller à école" from `input.wav` to text). It then applies grammar corrections, diacritical mark corrections, and pronunciation corrections based on the original audio. Finally, it generates custom audio feedback of the corrected sentence with integrated playback controls.

This example shows text analysis of the sentence "Je suis aller chez mon mère" demonstrating the system's multi-layered error detection:
- Grammar Corrections: Detects errors such as incorrect past participle agreement ("aller" should be "allée" for feminine gender) and incorrect determiner agreement ("mon mère" should be "ma mère"), providing the corrected sentence.
- French Explanations: Provides detailed explanations in French for each grammatical correction.

![GUI TextInputResults](screenshots/text_input_results.png)

## System Architecture
```
Speech Input
        │
        ▼
OpenAI Whisper
        │
        ▼
Speech Transcription
        │
        ▼
Grammar Engine
(language-tool-python + regex)
        │
        ▼
Pronunciation Analysis
        │
        ▼
Text-to-Speech (gTTS)
        │
        ▼
Flask Web Interface
```

### Files
- `nuansa-french-tutor/app/main.py`: Manages the Flask application, handling routes for the homepage and analysis requests while serving static files such as audio feedback and the French tutor image.
- `nuansa-french-tutor/app/templates/index.html`: Provides the user interface with input fields for text or audio, buttons to trigger analysis, and a section to display feedback results.
- `nuansa-french-tutor/app/static/images/french-girl-icon.png`: French tutor image displayed in the application.
- `nuansa-french-tutor/app/uploads/input.wav`: Sample audio file containing example input.
- `nuansa-french-tutor/src/analyze.py`: Processes audio or text input using Whisper for transcription and language_tool_python for grammar checks, generating personalized audio feedback with gTTS.
- `nuansa-french-tutor/tests/test_language_tool.py`: Contains unit tests for grammar-checking functionality using language_tool_python.
- `requirements.txt`: Lists the Python dependencies required to run the application.

### Requirements
- Python 3.10 (required for compatibility with specific library versions, e.g., Whisper, as some libraries may have issues with the system default Python 3.13). Check your version with: `python3 --version`.
- Java 17 (required by language_tool_python to run the LanguageTool grammar engine).
- FFmpeg (required by OpenAI Whisper for audio transcription).
- An internet connection (required for gTTS to generate audio and for LanguageTool to download its local grammar engine on first run).
- flask==3.0.3
- language-tool-python==2.9.3
- torch>=2.4.0
- transformers==4.44.2
- gTTS==2.5.3
- openai-whisper==20240930
- shap==0.46.0
- librosa==0.10.2
- scikit-learn==1.5.1
- numpy==1.26.4
- plotly==5.24.0
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
     pip install flask==3.0.3 language-tool-python==2.9.3 torch>=2.4.0 transformers==4.44.2 gTTS==2.5.3 openai-whisper==20240930 shap==0.46.0 librosa==0.10.2 scikit-learn==1.5.1 numpy==1.26.4 plotly==5.24.0 pyenchant==3.2.2
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

#### Run the App
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
2. The first startup may take several minutes. LanguageTool may download its grammar engine the first time the application is run. This download may be approximately 259 MB.
3. Wait until the terminal displays:
   * Debugger is active!
   * Debugger PIN: ...
   * Running on http://127.0.0.1:5001
4. Once the server is running, open a web browser and navigate to: `http://127.0.0.1:5001`. If macOS asks whether Python is allowed to find devices on the local network, allow it.
5. Use the interface:
   - Enter text or select an example from the dropdown and click "Analyze Text".
   - Upload a `.wav` file by clicking "Choose File", selecting the file, clicking "Open", and then clicking "Analyze Speech".
   - View the transcription, grammar and pronunciation corrections, and the error table with errors, suggestions, and explanations.
   - Listen to the generated French audio feedback.
6. When finished, return to the terminal running Flask and press `Ctrl+C`. This stops the Flask development server.

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
        - `audio/` (generated audio feedback files)
        - `images/`
          - `french-girl-icon.png`
      - `templates/`
        - `index.html`
      - `uploads/`
        - `input.wav`
      - `main.py`
    - `src/`
      - `__init__.py`
      - `analyze.py` 
    - `tests/`
      - `test_language_tool.py` 

### Additional Notes
- The app runs on port 5001 to avoid common port conflicts. Access it at `http://127.0.0.1:5001` after starting the server.
- The first time the application runs, `language_tool_python` downloads the LanguageTool grammar engine to the local cache. This may take several minutes.
- The `nuansa-french-tutor/app/uploads/` directory stores audio files used by the application. Uploaded files do not need to be versioned.
- If port 5001 is already in use, check with: `lsof -i :5001`. You can run the application on a different port by modifying the port setting in `nuansa-french-tutor/app/main.py` (for example, change `port=5001` to `port=5002`) and then access the application at `http://127.0.0.1:5002`.

### License
- All rights reserved. Contact colenomariah92@gmail.com for licensing inquiries.

### Development Notes
- Modern AI engineering tools (Claude, Gemini, ChatGPT, and Grok) were used to accelerate prototyping, debugging, documentation, and iterative refinement during development.
- As project lead, I directed the overall system architecture, product requirements, AI evaluation strategy, product design decisions, and technical direction.
- Documentation graphics and interface illustrations were created using ChatGPT and refined with GIMP.
 
