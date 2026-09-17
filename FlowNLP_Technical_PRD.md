# FlowNLP — Technical Product Requirements Document

**Version:** 1.0  
**Project Type:** NLP / AI Desktop Application  
**Primary Platform:** Desktop (Windows/macOS)  
**Primary Language:** Python  
**Status:** Proposed MVP

---

## 1. Technical Overview

FlowNLP is a system-wide AI dictation application that captures natural speech, converts it into text using Automatic Speech Recognition (ASR), processes the transcript through a multi-stage NLP pipeline, understands user intent and context, and inserts the final text into the active desktop application.

The main technical contribution is **NLP post-processing and contextual understanding**, rather than developing a speech-recognition model from scratch.

### Core Pipeline

```text
Microphone
    ↓
Audio Capture
    ↓
Voice Activity Detection
    ↓
Whisper / faster-whisper
    ↓
Raw Transcript
    ↓
NLP Preprocessing
    ↓
Speech Cleanup
    ↓
Grammar / Spelling / Punctuation
    ↓
NER + Intent Classification
    ↓
Semantic Understanding
    ↓
Context Detection
    ↓
Output Formatting
    ↓
Text Injection
    ↓
Active Application
```

---

## 2. Technical Objectives

The system must:

1. Capture microphone audio reliably.
2. Detect speech segments and silence.
3. Convert speech into text using Whisper.
4. Normalize and preprocess the transcript.
5. Remove filler words and repetitions.
6. Correct spelling and grammar.
7. Restore punctuation and sentence boundaries.
8. Detect named entities.
9. Classify user intent.
10. Generate sentence embeddings.
11. Perform semantic similarity and context analysis.
12. Produce context-specific output.
13. Insert generated text into the active application.
14. Provide measurable improvement over raw ASR.
15. Support modular replacement of individual NLP components.

---

## 3. System Architecture

```text
┌──────────────────────────────────────────────────────┐
│                    FLOWNLP CLIENT                    │
│                                                      │
│  Global Hotkey                                       │
│        ↓                                             │
│  Audio Capture                                       │
│        ↓                                             │
│  Silero VAD                                          │
│        ↓                                             │
│  Whisper / faster-whisper                             │
│        ↓                                             │
│  Raw Transcript                                      │
│        ↓                                             │
│  ┌───────────────────────────────────────────────┐   │
│  │              NLP ENGINE                       │   │
│  │                                               │   │
│  │ Tokenization                                 │   │
│  │ Normalization                                │   │
│  │ POS / Lemmatization                          │   │
│  │ Filler Removal                               │   │
│  │ Repetition Detection                         │   │
│  │ Grammar / Spelling                           │   │
│  │ Punctuation                                  │   │
│  │ NER                                          │   │
│  │ Intent Classification                        │   │
│  │ Sentence Embeddings                          │   │
│  │ Semantic Similarity                           │   │
│  └──────────────────────┬────────────────────────┘   │
│                         ↓                            │
│                  Context Engine                      │
│                         ↓                            │
│                  Output Formatter                    │
│                         ↓                            │
│                  Text Injection                      │
└─────────────────────────┼────────────────────────────┘
                          ↓
                 Active Application
```

---

## 4. Technology Stack

| Layer | Technology | Responsibility |
|---|---|---|
| Programming | Python 3.11+ | Core application |
| Audio | sounddevice / PyAudio | Microphone capture |
| VAD | Silero VAD | Speech/silence detection |
| ASR | Whisper / faster-whisper | Speech-to-text |
| NLP | NLTK | Classical NLP operations |
| NLP | spaCy | Tokenization, POS, NER, sentence processing |
| ML | scikit-learn | Classical ML and intent baselines |
| Deep Learning | PyTorch | Neural model execution |
| Transformers | Hugging Face Transformers | Advanced NLP |
| Embeddings | Sentence Transformers | Semantic representations |
| Grammar | LanguageTool | Grammar and spelling correction |
| GUI | PySide6 | Desktop interface |
| Automation | PyAutoGUI / keyboard | Text injection |
| Storage | SQLite | Settings and user dictionary |
| Testing | pytest | Unit/integration testing |
| Evaluation | jiwer | WER/CER calculation |
| Version Control | Git/GitHub | Source control |
| Packaging | PyInstaller | Desktop executable |

---

## 5. Global Hotkey Manager

### Responsibility

Activate and deactivate dictation from anywhere on the desktop.

### Requirements

- Register a global shortcut.
- Detect key press/release.
- Start recording.
- Stop recording.
- Prevent duplicate recording sessions.

### Interface

```python
start_recording()
stop_recording()
is_recording()
```

---

## 6. Audio Capture Module

### Responsibility

Capture microphone audio and provide it to the VAD/ASR pipeline.

### Requirements

- Configurable microphone.
- Audio buffering.
- 16 kHz audio sampling preferred for ASR.
- Mono audio.
- Graceful microphone error handling.

### Suggested Libraries

```text
sounddevice
PyAudio
numpy
```

### Output

```text
PCM/WAV audio buffer
```

---

## 7. Voice Activity Detection

### Technology

**Silero VAD**

### Purpose

Detect actual speech segments.

```text
Audio
 ↓
VAD
 ├── Silence → Ignore
 └── Speech → Process
```

### Benefits

- Reduces unnecessary processing.
- Detects speech boundaries.
- Improves responsiveness.
- Helps segment longer dictation.

---

## 8. Automatic Speech Recognition

### Technology

**Whisper / faster-whisper**

### Input

```text
Audio waveform
```

### Output

```json
{
  "text": "uh hey can you send the report tomorrow",
  "language": "en",
  "segments": []
}
```

### Requirements

- Configurable model size.
- CPU/GPU execution.
- Transcript and optional timestamps.
- Raw transcript preserved for evaluation.

### Model Selection

Development options:

```text
Whisper base
Whisper small
Whisper medium
faster-whisper
```

Final selection should be based on accuracy, memory usage and latency benchmarks.

---

## 9. NLP Processing Engine

The NLP engine is the core academic component.

### Processing Order

```text
Raw Transcript
      ↓
Normalization
      ↓
Tokenization
      ↓
Sentence Segmentation
      ↓
Filler Detection
      ↓
Repetition Detection
      ↓
Grammar / Spelling
      ↓
Punctuation
      ↓
NER
      ↓
Intent
      ↓
Semantic Analysis
```

---

## 10. Text Normalization

### Technologies

- NLTK
- spaCy
- Python regex

### Operations

- Remove unnecessary whitespace.
- Normalize quotation marks.
- Normalize repeated punctuation.
- Standardize text representation.
- Handle common ASR artifacts.
- Preserve named entities.

### Example

```text
"hello    can you   send this"
```

↓

```text
"hello can you send this"
```

---

## 11. Tokenization

The transcript is divided into tokens.

### Example

```text
"Send the report to Arun."
```

↓

```text
["Send", "the", "report", "to", "Arun", "."]
```

### Technology

**spaCy / NLTK**

---

## 12. POS Tagging

Part-of-Speech tagging identifies the grammatical role of words.

### Example

```text
Arun    → PROPN
send    → VERB
report  → NOUN
the     → DET
```

### Use

- Sentence analysis
- Grammar processing
- Phrase detection
- NLP experiments

---

## 13. Lemmatization

Converts words into their base form.

### Example

```text
running → run
reports → report
```

### Technology

**spaCy**

Lemmatization is mainly used for analysis and downstream NLP tasks; final user-facing text retains appropriate word forms.

---

## 14. Filler Word Detection

Natural speech contains conversational fillers.

### Examples

```text
um
uh
like
basically
you know
actually
```

### Approach

Use a hybrid approach:

```text
Rule-based detection
        +
Context-aware classification
```

### Example

```text
Input:
"Um basically I wanted to like send the report."

Output:
"I wanted to send the report."
```

The system must avoid removing words when they carry meaningful context.

---

## 15. Repetition and False-Start Detection

The system detects:

- Repeated words.
- Repeated phrases.
- Partial repetitions.
- False starts.
- Self-corrections.

### Example

```text
"I want to send the send the report."

                ↓

"I want to send the report."
```

### Possible Techniques

```text
N-gram comparison
      +
Edit distance
      +
Context validation
```

---

## 16. Grammar and Spelling Correction

### Technologies

Primary:

```text
LanguageTool
```

Advanced:

```text
Transformer-based correction model
```

### Pipeline

```text
Clean Transcript
      ↓
Grammar Detector
      ↓
Candidate Corrections
      ↓
Context Validation
      ↓
Corrected Text
```

### Example

```text
"I has completed the project."

          ↓

"I have completed the project."
```

---

## 17. Punctuation Restoration

The system reconstructs punctuation missing from speech.

### Input

```text
can you send the report tomorrow
```

### Output

```text
Can you send the report tomorrow?
```

### Potential Implementation

- Transformer punctuation model.
- Sentence boundary detection.
- Rule-based baseline.

---

## 18. Named Entity Recognition

### Technologies

- spaCy NER
- BERT-based NER

### Entity Types

```text
PERSON
ORG
LOCATION
DATE
TIME
PRODUCT
MONEY
EVENT
```

### Example

```text
"Send the report to Arun in Chennai tomorrow."

Arun       → PERSON
Chennai    → LOCATION
tomorrow   → DATE
```

NER results should be available to downstream correction and formatting modules.

---

## 19. Intent Classification

The intent classifier determines what the user is trying to accomplish.

### Initial Classes

```text
GENERAL_TEXT
CHAT
EMAIL
NOTE
SEARCH
QUESTION
REQUEST
CODE
REMINDER
```

### Classical Baseline

```text
TF-IDF
+
Logistic Regression / SVM
```

### Advanced Model

```text
BERT / DistilBERT
```

### Output

```json
{
  "intent": "EMAIL",
  "confidence": 0.94
}
```

---

## 20. TF-IDF Baseline

TF-IDF converts text into numerical features.

### Purpose

Used as a classical NLP baseline for:

- Intent classification
- Keyword analysis
- Similarity
- Comparison against transformer approaches

### Pipeline

```text
Text
 ↓
Tokenization
 ↓
TF-IDF Vectorization
 ↓
ML Classifier
 ↓
Intent
```

---

## 21. Sentence Embeddings

### Technology

**Sentence Transformers**

Sentences are converted into vectors representing semantic meaning.

### Example

```text
"Send the report to Arun."

"Forward the report to Arun."
```

These should have semantically similar vector representations.

### Use Cases

- Semantic similarity.
- Context detection.
- Intent support.
- Repetition detection.
- Evaluation.

---

## 22. Context Detection

The system determines the appropriate output context.

### Context Sources

```text
User-selected mode
        +
Active application
        +
Detected intent
        +
Semantic content
```

### Possible Contexts

```text
EMAIL
CHAT
NOTES
SEARCH
CODE
GENERAL
```

---

## 23. Output Formatting Engine

The formatter transforms processed content into application-ready text.

### Architecture

```text
Intent + Context + Clean Text
             ↓
      Formatting Engine
             ↓
     Context-specific Text
```

### Formatting Modes

```text
EMAIL  → Professional
CHAT   → Conversational
NOTE   → Structured
SEARCH → Concise query
CODE   → Programming-oriented
```

---

## 24. Text Injection Layer

The final text must be inserted into the currently active application.

### Technologies

- PyAutoGUI
- Keyboard automation
- Clipboard APIs
- OS-specific APIs

### Flow

```text
Final Text
    ↓
Clipboard
    ↓
Paste
    ↓
Active Application
```

A clipboard fallback should be available if automatic insertion fails.

---

## 25. Desktop UI

### Technology

**PySide6**

### UI States

```text
IDLE
RECORDING
PROCESSING
COMPLETED
ERROR
```

### Example

```text
┌──────────────────────────────────┐
│              FlowNLP             │
│                                  │
│          ● Listening...          │
│                                  │
│  "Can you send the report..."    │
│                                  │
│  Mode: [Auto ▼]                  │
│                                  │
│  [Stop]             [Settings]  │
└──────────────────────────────────┘
```

---

## 26. Storage Layer

### Technology

**SQLite**

### Store

- User settings.
- Preferred microphone.
- Global shortcut.
- Preferred output mode.
- Personal dictionary.
- Optional transcript metadata.

### Do Not Store by Default

Raw microphone recordings should not be stored unless explicitly enabled by the user.

---

## 27. Personal Dictionary

Users can define words that are important to them.

Examples:

```text
PyTorch
TensorFlow
FlowNLP
OpenCV
Project-specific names
Technical terminology
```

### Purpose

Improve:

- ASR correction.
- Spelling correction.
- Entity preservation.

---

## 28. Module Interfaces

The system should use modular interfaces.

### ASR

```python
class ASREngine:
    def transcribe(self, audio) -> dict:
        pass
```

### NLP

```python
class NLPEngine:
    def process(self, text, context=None) -> dict:
        pass
```

### Formatter

```python
class Formatter:
    def format(self, processed_text, context) -> str:
        pass
```

### Text Injector

```python
class TextInjector:
    def insert(self, text) -> bool:
        pass
```

This architecture allows individual models to be replaced without rewriting the entire system.

---

## 29. Internal Data Model

A processed request can follow this structure:

```json
{
  "raw_text": "uh send the report to Arun tomorrow",
  "clean_text": "Send the report to Arun tomorrow.",
  "language": "en",
  "entities": [
    {
      "text": "Arun",
      "type": "PERSON"
    },
    {
      "text": "tomorrow",
      "type": "DATE"
    }
  ],
  "intent": {
    "label": "REQUEST",
    "confidence": 0.91
  },
  "context": "CHAT",
  "formatted_text": "Hey Arun, can you send the report tomorrow?"
}
```

---

## 30. Error Handling

### Microphone Failure

```text
"No microphone detected."
```

### ASR Failure

```text
"Unable to transcribe audio."
```

### NLP Failure

Fallback to:

```text
Raw / lightly cleaned transcript
```

### Text Injection Failure

Display:

```text
"Unable to insert text. Copy the generated text manually."
```

The system should never silently lose the user's transcript.

---

## 31. Performance Requirements

Target MVP requirements:

- Hotkey response: < 500 ms.
- Recording start: near-instant.
- Short transcription: a few seconds depending on hardware/model.
- NLP processing: preferably < 1–2 seconds after ASR.
- UI must remain responsive during inference.

Performance should be benchmarked separately for:

```text
CPU
GPU
Model size
Audio duration
NLP pipeline
```

---

## 32. Privacy Requirements

### Preferred Architecture

```text
Microphone
    ↓
Local ASR
    ↓
Local NLP
    ↓
Local Output
```

### Requirements

- Do not permanently store audio by default.
- Do not expose API keys.
- Allow transcript history to be disabled.
- Clearly identify cloud processing.
- Prefer local inference where feasible.

---

## 33. Dataset Strategy

Different components can use different datasets.

| Task | Dataset |
|---|---|
| ASR | Common Voice / public speech corpora |
| NER | CoNLL-2003 |
| Intent | CLINC-style dataset / custom dataset |
| Grammar Correction | BEA-2019 |
| Semantic Similarity | STS Benchmark |
| Custom Evaluation | Team-recorded speech dataset |

### Custom Dataset

Target:

**100–500 samples**

Include:

- Fillers.
- Repetitions.
- False starts.
- Names.
- Dates.
- Technical terms.
- Different speaking styles.
- Different sentence lengths.
- Optional English/Tamil/Tanglish samples.

---

## 34. Evaluation Architecture

The primary experiment compares raw ASR against FlowNLP.

```text
                    Same Audio
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
          Whisper              Whisper
              ↓                   ↓
        Raw Transcript      NLP Pipeline
              ↓                   ↓
          Baseline          FlowNLP Output
              │                   │
              └─────────┬─────────┘
                        ↓
                    Evaluation
```

---

## 35. Evaluation Metrics

### ASR

**Word Error Rate (WER)**

Measures word-level transcription errors.

**Character Error Rate (CER)**

Measures character-level errors.

### Intent Classification

```text
Accuracy
Precision
Recall
F1
```

### NER

```text
Entity Precision
Entity Recall
Entity F1
```

### Semantic Preservation

Use sentence similarity to ensure corrections do not change the user's intended meaning.

---

## 36. Human Evaluation

Human evaluators can compare:

```text
Raw Whisper
      VS
FlowNLP
```

Rate each on a 1–5 scale:

```text
Readability
Grammar
Punctuation
Meaning Preservation
Naturalness
Usefulness
```

Also measure:

**Manual editing effort**

The key question is:

> How much editing does the user need after Whisper compared with FlowNLP?

---

## 37. Academic Experiment

Implement progressively stronger approaches.

### Level 1 — Raw ASR

```text
Speech → Whisper → Raw Text
```

### Level 2 — Classical NLP

```text
Speech
 ↓
Whisper
 ↓
Rules + NLTK/spaCy + TF-IDF
 ↓
Classical ML
 ↓
Processed Text
```

### Level 3 — Embedding/ML

```text
Speech
 ↓
Whisper
 ↓
Sentence Embeddings
 ↓
Classification / Similarity
 ↓
Processed Text
```

### Level 4 — Transformer NLP

```text
Speech
 ↓
Whisper
 ↓
Transformer Models
 ↓
Context-aware Processing
 ↓
Formatted Text
```

---

## 38. Technical Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ASR mistakes | High | Stronger model and correction pipeline |
| High latency | High | Model benchmarking and GPU acceleration |
| Incorrect filler removal | Medium | Context-aware rules/classifier |
| Grammar correction changes meaning | High | Semantic validation + entity protection |
| Incorrect NER | Medium | Compare spaCy and transformer models |
| Text injection failure | Medium | Clipboard fallback |
| Cross-platform issues | High | Start with one desktop OS |
| Large model memory usage | High | Configurable model sizes |
| Dataset limitations | Medium | Public + custom evaluation data |

---

## 39. MVP Requirements

### Input

- Global shortcut.
- Microphone capture.
- VAD.

### ASR

- Whisper/faster-whisper.

### NLP

- Tokenization.
- Normalization.
- Filler removal.
- Repetition removal.
- Punctuation.
- Grammar/spelling correction.
- NER.
- Basic intent classification.

### Semantic Layer

- Sentence embeddings.
- Basic context detection.

### Output

- Chat.
- Email.
- Notes.

### Integration

- Clipboard.
- Text insertion into active application.

### Evaluation

- Raw vs processed comparison.
- WER/CER.
- Intent F1.
- NER F1.
- Human evaluation.

---

## 40. Future Technical Scope

### Multilingual Processing

```text
English
Tamil
Tanglish
Hinglish
```

### Streaming ASR

```text
Speech
 ↓
Partial Transcript
 ↓
NLP
 ↓
Live Text
```

### Application-Aware Context

```text
VS Code → Code
Gmail → Email
Slack → Chat
Browser → Search
Word → Document
```

### Advanced Voice Commands

```text
"New paragraph"
"Make this professional"
"Summarize this"
"Convert to bullets"
"Make this concise"
```

---

## 41. Recommended Software Structure

```text
FlowNLP/
│
├── main.py
│
├── app/
│   ├── controller.py
│   ├── hotkey.py
│   └── states.py
│
├── audio/
│   ├── recorder.py
│   ├── vad.py
│   └── audio_utils.py
│
├── asr/
│   ├── base.py
│   └── whisper_engine.py
│
├── nlp/
│   ├── pipeline.py
│   ├── tokenizer.py
│   ├── normalizer.py
│   ├── filler_remover.py
│   ├── repetition_detector.py
│   ├── punctuation.py
│   ├── grammar.py
│   ├── ner.py
│   ├── intent.py
│   └── embeddings.py
│
├── context/
│   ├── detector.py
│   └── application.py
│
├── formatter/
│   ├── base.py
│   ├── email.py
│   ├── chat.py
│   ├── notes.py
│   ├── search.py
│   └── code.py
│
├── output/
│   ├── clipboard.py
│   └── injector.py
│
├── ui/
│   ├── main_window.py
│   └── settings.py
│
├── storage/
│   ├── database.py
│   └── dictionary.py
│
├── evaluation/
│   ├── wer.py
│   ├── metrics.py
│   └── benchmark.py
│
├── tests/
├── models/
├── requirements.txt
└── README.md
```

---

## 42. Team Technical Division

| Module | Responsibility |
|---|---|
| Audio & ASR | Audio capture, VAD, Whisper integration |
| NLP Cleanup | Tokenization, normalization, fillers, repetitions, punctuation, correction |
| NLP Understanding | NER, intent, embeddings, semantic similarity |
| Application Layer | UI, global shortcut, active-app detection, text injection |
| Shared | Dataset preparation, evaluation, testing and integration |

---

## 43. Development Phases

### Phase 1 — ASR Prototype

```text
Microphone
 ↓
Whisper
 ↓
Raw Transcript
```

### Phase 2 — NLP Cleanup

```text
Whisper
 ↓
Normalization
 ↓
Filler Removal
 ↓
Repetition Removal
 ↓
Grammar
 ↓
Punctuation
```

### Phase 3 — NLP Understanding

```text
NER
+
Intent
+
Embeddings
+
Semantic Similarity
```

### Phase 4 — Context Formatting

```text
Email
Chat
Notes
Search
Code
```

### Phase 5 — System Integration

```text
Global Shortcut
+
Active Application
+
Text Injection
```

### Phase 6 — Evaluation

```text
Raw Whisper
      VS
FlowNLP
```

### Phase 7 — Final Product

- Optimization
- Testing
- Packaging
- Documentation
- Demonstration

---

## 44. Technical Success Criteria

The MVP is successful when:

- Audio can be captured reliably.
- Whisper produces usable transcripts.
- NLP processing improves readability.
- Fillers and repetitions are reduced.
- Grammar and punctuation improve.
- Named entities are preserved.
- Intent classification achieves acceptable validation performance.
- Semantic meaning is preserved.
- Context-specific formatting works.
- Text can be inserted into a supported active application.
- The complete pipeline runs end-to-end.
- Evaluation demonstrates measurable improvement over raw ASR.

---

## 45. Final Technical Definition

> **FlowNLP is a modular desktop NLP system that combines Whisper-based automatic speech recognition with classical NLP, machine learning, transformer models, semantic embeddings and context-aware text formatting to transform natural speech into application-ready written communication.**

The technical contribution is therefore not simply:

```text
Speech → Text
```

but:

```text
Speech
  ↓
ASR
  ↓
Raw Text
  ↓
Linguistic Processing
  ↓
Error Correction
  ↓
Entity & Intent Understanding
  ↓
Semantic Context
  ↓
Application-Aware Formatting
  ↓
Usable Written Communication
```
