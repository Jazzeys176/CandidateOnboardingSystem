# Automated Candidate Onboarding Validation System 

An intelligent system that automates the verification and validation of candidate information during HR onboarding by extracting data from multiple sources, building a consolidated knowledge base, and generating compliance reports with risk assessments.

---

## Overview

This system streamlines the HR onboarding process by:

- **Extracting data** from multiple sources: resumes (PDF), ID cards (images), onboarding forms (JSON), and HR call transcripts (TXT)
- **Building a Golden Record** — a consolidated, conflict-resolved knowledge base from multimodal data using LLM-powered extraction
- **Validating extracted data** against form submissions using semantic similarity (ONNX-optimized embeddings)
- **Generating compliance reports** with risk assessments (HIGH/MEDIUM/LOW) in both Markdown and PDF formats
- **Providing an interactive web UI** via Streamlit for easy document upload and processing

---

## Problem Statement

During the HR onboarding process, organizations face several challenges:

1. **Manual Data Entry Errors**: HR personnel manually transcribe information from resumes, ID cards, and call recordings, leading to typos and inconsistencies
2. **Data Integrity Issues**: Candidate-submitted forms may contain discrepancies compared to verified documents (resumes, government IDs)
3. **Time-Consuming Verification**: Cross-referencing multiple documents (resume, ID card, transcripts, forms) is labor-intensive
4. **Compliance Risk**: Missing or incorrect candidate information can lead to regulatory and legal issues
5. **Lack of Audit Trail**: No systematic way to track what was verified and from which source

**Solution**: This system automates the entire verification pipeline — extracting structured data from unstructured documents, resolving conflicts using source-priority hierarchies, validating semantic consistency, and generating actionable compliance reports.

---

## Technologies, Tools & Models

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **LLM API** | Groq (Llama-3.3-70B-Versatile) | Data extraction, conflict resolution, report summarization |
| **NLP/Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Semantic similarity scoring for validation |
| **ML Inference** | ONNX Runtime (via Hugging Face Optimum) | CPU-optimized embedding inference |
| **OCR Engine** | EasyOCR | Text extraction from PDFs and ID card images |
| **PDF Processing** | pdf2image, ReportLab | PDF reading and professional report generation |
| **Web Framework** | Streamlit | Interactive 4-page dashboard UI |
| **Deep Learning** | PyTorch | Tensor operations for embeddings |

### Key Dependencies

```
groq                 # LLM API client
python-dotenv        # Environment variable management
easyocr              # Optical character recognition
pdf2image            # PDF to image conversion
Pillow               # Image processing
numpy                # Numerical operations
torch                # PyTorch for embeddings
transformers         # Hugging Face transformers
optimum[onnxruntime] # ONNX optimization
reportlab            # PDF report generation
streamlit            # Web UI framework
```

### Hardware Target

- **Optimized for CPU-only deployment** (Intel i5-7300U / similar)
- EasyOCR runs in CPU mode (`gpu=False`)
- ONNX models provide lightweight, fast inference without GPU

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Poppler (for pdf2image)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd onboarding_system
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install groq python-dotenv easyocr pdf2image Pillow numpy torch transformers optimum[onnxruntime] reportlab streamlit
```

### Step 4: Install Poppler (for PDF processing)

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH.

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from [Groq Console](https://console.groq.com/).

### Step 6: Run the Application

**Web UI (Recommended):**
```bash
streamlit run main_v3.py
```

**CLI (Basic):**
```bash
python main.py      # Resume + Transcript only
python main_v2.py   # Full multimodal pipeline
```

---

## Project Structure

```
onboarding_system/
├── .env                          # Environment variables (GROQ_API_KEY)
├── .gitignore                    # Git ignore rules for sensitive files
├── README.md                     # Project documentation
├── main.py                       # Phase 1 entry point: basic extraction pipeline
├── main_v2.py                    # Phase 2 entry point: OCR + multimodal integration
├── main_v3.py                    # Phase 3 entry point: Streamlit web UI dashboard
├── src/                          # Core application modules
│   ├── extractor.py              # Data extraction and golden record generation via LLM
│   ├── validator.py              # Semantic validation engine with ONNX embeddings
│   ├── reporter.py               # Report generation (Markdown & PDF) with risk assessment
│   ├── ocr_engine.py             # Document OCR for resumes (PDF) and ID cards (images)
│   └── kb_builder.py             # Legacy knowledge base builder module
├── inputs/                       # Input data directory
│   ├── onboarding_form.json      # Sample user-submitted onboarding form data
│   ├── transcript.txt            # Sample HR call recording transcript
│   ├── Resume.pdf                # Sample candidate resume document
│   ├── id_card.jpeg              # Sample ID card scan (Aadhar/PAN)
│   └── validation_report.pdf     # Generated output report
└── venv/                         # Python virtual environment
```

### File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Basic CLI pipeline that processes resume and transcript to generate validation report |
| `main_v2.py` | Enhanced CLI with OCR support for PDFs and ID cards, integrates all data sources |
| `main_v3.py` | Full Streamlit web application with 4-page dashboard for interactive processing |
| `src/extractor.py` | Extracts ID fields via regex, combines multimodal data into golden record JSON using Groq LLM |
| `src/validator.py` | Semantic validator using ONNX-optimized embeddings, includes LLM fallback and temporal checks |
| `src/reporter.py` | Generates executive summary via LLM, creates PDF reports with risk badges |
| `src/ocr_engine.py` | Handles OCR extraction from resume PDFs (pdf2image + EasyOCR) and ID card images |
| `src/kb_builder.py` | Legacy module for golden record building (superseded by extractor.py) |

---

## What All Are Implemented

- Basic Implementation and working of the system.
- Takes in HR call recording transcript as txt file, resume as pdf file and ID card as image file.
- Streamlit Dashboard for interactive processing. 
- Can run properly on basic CPU machine.
---

## Future Scope

- Adding feature where you can upload multiple resumes and transcripts and get the final report.
- Adding the feature where you can directly upload HR call recording in form of .mp3, m4a, .aac, .webM and it should convert that into transcripts.
- Productionize it with high-quality cloud based LLM Models

---

## Application Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI (main_v3.py)                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │  Dashboard   │ │  Onboarding  │ │     HR       │ │  Final  │ │
│  │   (Stats)    │ │    Form      │ │ Transcripts  │ │ Output  │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   Resume.pdf          id_card.jpeg       transcript.txt
         │                   │                   │
         └─────── OCR Engine ┴───────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Golden Record   │
                    │ Generation      │
                    │ (Groq LLM)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Semantic        │
                    │ Validation      │
                    │ (ONNX Embeddings│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Risk Assessment │
                    │ & Report Gen    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ PDF Report      │
                    │ Download        │
                    └─────────────────┘
```

---

## Risk Classification

| Risk Level | Criteria |
|------------|----------|
| **HIGH** | More than 2 incorrect fields detected |
| **MEDIUM** | 1+ incorrect fields OR more than 3 ambiguous fields |
| **LOW** | All validations passed with high confidence |

---

## License

This project is developed for HR automation and compliance purposes.

---

## Contributing

Contributions are welcome! Please ensure all changes maintain CPU-only compatibility and follow the existing code structure.
