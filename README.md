# RAG-Based Offline PDF Assistant

An intelligent desktop application that enables conversational question-answering on PDF documents using Retrieval-Augmented Generation (RAG). The system works entirely offline, ensuring data privacy and security.

## Features

- **Multi-format PDF Support**: Handles both text-based and scanned PDFs
- **OCR Capabilities**: Extracts text from scanned documents and embedded images
- **Semantic Search**: Uses FAISS for efficient vector similarity search
- **Local LLM Integration**: Powered by Mistral via Ollama for offline operation
- **Interactive GUI**: ChatGPT-style interface built with Tkinter
- **Conversation History**: Maintains context across multiple questions
- **Audit Logging**: Automatically logs all Q&A interactions with timestamps

## Technology Stack

- **Python 3.10+**
- **PDF Processing**: `pdfplumber`, `PyMuPDF (fitz)`
- **OCR**: `pytesseract` with image preprocessing
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Vector Search**: `FAISS` (Facebook AI Similarity Search)
- **LLM**: `Mistral` via `Ollama` (local hosting)
- **GUI**: `Tkinter`
- **HTTP Client**: `requests`

## Prerequisites

- Python 3.8 or higher
- Minimum 8 GB RAM (16 GB recommended for large PDFs)
- [Ollama](https://ollama.ai/) installed and running locally
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed

### Installing Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```
3. Ensure Ollama is running (default: `http://localhost:11434`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. Install required dependencies:
   ```bash
   pip install pymupdf pdfplumber pytesseract faiss-cpu sentence-transformers requests pillow
   ```

3. Ensure Tesseract OCR is installed:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

## Usage

### Option 1: Using the Jupyter Notebook

1. Open `v7.ipynb` in Jupyter Notebook
2. Run all cells to load the functions
3. Execute:
   ```python
   run_qa_interface("./path/to/your/document.pdf")
   ```
4. Type your questions in the prompt. Type `exit` to quit.

### Option 2: Using the GUI Application

1. Run the GUI application:
   ```bash
   python gui_app.py
   ```
2. Click "Upload PDF" to load your document
3. Wait for processing to complete
4. Start asking questions in the chat interface

## Project Structure

```
RAG/
├── code.ipynb                    # Jupyter notebook with core implementation
├── pdf_qa_core.py              # Core PDF processing and RAG logic
├── gui_app.py                  # Tkinter-based GUI application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── logs/                       # Generated log files
    ├── *_qa_log.txt           # Q&A interaction logs
    └── *_chat_memory.json     # Chat history files
```

## How It Works

1. **PDF Extraction**: Extracts text, tables, and images from PDF documents
2. **OCR Processing**: Applies image preprocessing and OCR to scanned content
3. **Embedding Generation**: Converts extracted text into semantic embeddings using SentenceTransformer
4. **Vector Indexing**: Stores embeddings in FAISS for fast similarity search
5. **Query Processing**: 
   - Converts user question to embedding
   - Retrieves top-k most relevant document chunks
   - Sends context + question to local LLM (Mistral via Ollama)
6. **Response Generation**: LLM generates answer based on retrieved context
7. **Logging**: Saves Q&A pairs with timestamps for audit trail

## Example

```python
# Load and process PDF
index, tables = run_pipeline("document.pdf")

# Ask questions
question = "What is the main topic of this document?"
context = index.retrieve_context(question)
answer = query_ollama_http(question, context)
print(answer)
```

## Use Cases

- Technical documentation Q&A
- Legal document analysis
- Research paper summarization
- Compliance document review
- Project report analysis
- Engineering specification queries

## Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (can be changed in `PDFIndex` class)
- **LLM Model**: `mistral` (change in `query_ollama_http` function)
- **Top-K Retrieval**: Default is 4 chunks (adjustable in `retrieve_context`)
- **Ollama Endpoint**: `http://localhost:11434/api/generate`

## Future Enhancements

- [ ] Support for charts and figures analysis
- [ ] Voice command interface
- [ ] Multi-document querying
- [ ] Advanced table extraction and formatting
- [ ] Document summarization features
- [ ] Export conversations to various formats
- [ ] Custom model fine-tuning support

