# FlowNLP

A modular desktop AI dictation system that transforms natural speech into
application-ready written communication using Whisper ASR + multi-stage
NLP post-processing.

---

## Overview

```
Microphone → Full-session capture → VAD → Whisper → NLP Pipeline → Context Detector → Formatter → Active App
```

FlowNLP is not simply speech-to-text. After ASR, a full NLP pipeline:

- Removes filler words and repetitions
- Corrects grammar and spelling (LanguageTool)
- Restores punctuation
- Extracts named entities (spaCy / BERT)
- Classifies intent (TF-IDF → DistilBERT)
- Generates sentence embeddings (Sentence Transformers)
- Formats output for the detected context (email, chat, notes, search, code)

---

## Project Structure

```
FlowNLP/
├── main.py                 # Entry point
├── requirements.txt
│
├── app/
│   ├── controller.py       # Central wiring of all subsystems
│   ├── hotkey.py           # Global hotkey manager (pynput)
│   └── states.py           # AppState enum
│
├── audio/
│   ├── recorder.py         # Microphone capture (sounddevice)
│   ├── vad.py              # Silero VAD wrapper
│   └── audio_utils.py      # PCM helpers, constants
│
├── asr/
│   ├── base.py             # Abstract ASREngine interface
│   └── whisper_engine.py   # faster-whisper / openai-whisper
│
├── nlp/
│   ├── pipeline.py         # Orchestrates all NLP stages
│   ├── normalizer.py       # Text normalisation
│   ├── tokenizer.py        # spaCy / NLTK tokenisation + POS + lemma
│   ├── filler_remover.py   # Filler word detection and removal
│   ├── repetition_detector.py  # N-gram deduplication
│   ├── grammar.py          # LanguageTool grammar + spelling
│   ├── punctuation.py      # Punctuation restoration
│   ├── ner.py              # Named entity recognition
│   ├── intent.py           # Intent classification (TF-IDF + SVM / BERT)
│   └── embeddings.py       # Sentence Transformers embedding engine
│
├── context/
│   ├── detector.py         # Context detection (intent + active app)
│   └── application.py      # Active window name (macOS / Windows / Linux)
│
├── formatter/
│   ├── base.py             # FormatterBase interface
│   ├── email.py            # Professional email formatting
│   ├── chat.py             # Conversational chat formatting
│   ├── notes.py            # Structured note formatting
│   ├── search.py           # Concise search query formatting
│   └── code.py             # Code / comment formatting
│
├── output/
│   ├── injector.py         # Text injection into active application
│   └── clipboard.py        # Clipboard read/write + paste shortcut
│
├── ui/
│   ├── main_window.py      # PySide6 main window
│   └── settings.py         # Settings dialog
│
├── storage/
│   ├── database.py         # SQLite persistence layer
│   └── dictionary.py       # Personal dictionary helper
│
├── evaluation/
│   ├── wer.py              # WER / CER via jiwer
│   ├── metrics.py          # Classification + semantic metrics
│   └── benchmark.py        # Four-level experiment runner
│
├── tests/                  # pytest test suite
├── models/                 # Saved ML models (intent classifier, etc.)
└── data/                   # Evaluation datasets
```

---

## Installation

### 1. Clone

```bash
git clone <repo-url>
cd FlowNLP
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run

```bash
python main.py
```

On first launch, FlowNLP downloads its local speech and grammar models. The
grammar-correction model (`vennify/t5-base-grammar-correction`) runs locally and
is licensed for non-commercial use (CC BY-NC-SA 4.0).

---

## Usage

1. Launch FlowNLP — the main window appears.
2. Press **Ctrl+Shift+Space** anywhere on your desktop to start dictation.
3. Speak naturally.
4. Press the hotkey again (or click **Stop**) to end the full recording.
5. FlowNLP transcribes and processes the completed speech, then inserts the formatted text
    into the active application (or placed on your clipboard if injection fails).

### Output Modes

| Mode    | Best for                  |
|---------|---------------------------|
| Auto    | Detects context from app  |
| Email   | Professional tone         |
| Chat    | Conversational            |
| Notes   | Bullet-point structure    |
| Search  | Concise query             |
| Code    | Code editors / comments   |

---

## Evaluation

Run the benchmark against an evaluation dataset:

```bash
python -m evaluation.benchmark data/eval_samples.json --output results.json
```

Each sample in the JSON file should be:

```json
{
    "audio_path": "data/sample_001.wav",
    "reference":  "Send the report to Arun tomorrow.",
    "intent":     "REQUEST"
}
```

---

## Technology Stack

| Layer            | Library                   |
|------------------|---------------------------|
| ASR              | faster-whisper / openai-whisper |
| VAD              | Silero VAD                |
| NLP (classical)  | spaCy, NLTK               |
| Grammar          | LanguageTool              |
| Embeddings       | Sentence Transformers     |
| Transformers     | Hugging Face Transformers |
| ML baseline      | scikit-learn              |
| GUI              | PySide6                   |
| Text injection   | PyAutoGUI, pyperclip      |
| Hotkeys          | pynput                    |
| Storage          | SQLite (stdlib)           |
| Evaluation       | jiwer                     |

---

## Development Phases

| Phase | Goal                          |
|-------|-------------------------------|
| 1     | ASR prototype (Whisper)       |
| 2     | NLP cleanup pipeline          |
| 3     | NER + intent + embeddings     |
| 4     | Context-aware formatting      |
| 5     | System integration + UI       |
| 6     | Evaluation experiments        |
| 7     | Optimisation + packaging      |

---

## License

Academic / educational project — all rights reserved.
