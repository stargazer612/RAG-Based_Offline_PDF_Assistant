"""
pdf_qa_core.py
--------------
Core RAG logic for the Offline PDF Assistant.

Pipeline:
    PDF -> text (pdfplumber) + tables (pdfplumber) + images (PyMuPDF -> OCR)
        -> chunking -> embeddings (SentenceTransformers) -> FAISS index
        -> retrieval -> local LLM (Mistral via Ollama)

Everything runs locally. No document content leaves the machine.
"""

import os
import io
import json
from datetime import datetime

import fitz                      # PyMuPDF
import pdfplumber
import pytesseract
import requests
import faiss
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 300             # seconds

CHUNK_SIZE = 200                 # words per chunk
CHUNK_OVERLAP = 40               # words repeated between neighbouring chunks
TOP_K = 4                        # chunks retrieved per question

# Windows: pytesseract cannot always find tesseract.exe on PATH, so check the
# usual install locations and set it explicitly when found.
for _candidate in (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
):
    if os.path.exists(_candidate):
        pytesseract.pytesseract.tesseract_cmd = _candidate
        break


# --------------------------------------------------------------------------
# Step 1: extract text, tables and images from the PDF
# --------------------------------------------------------------------------

def extract_pdf_content(pdf_path):
    """Return (text_blocks, tables, images), each item tagged with its page.

    Two libraries are used deliberately:
      - pdfplumber reads text in layout order and extracts tables.
      - PyMuPDF makes pulling embedded image bytes straightforward.
    """
    text_blocks, tables, images = [], [], []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_blocks.append({"page": i + 1, "text": text})

            for tbl in page.extract_tables():
                tables.append({"page": i + 1, "table": tbl})

    pdf_doc = fitz.open(pdf_path)
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        for img in page.get_images(full=True):
            xref = img[0]                       # object id of the image in the PDF
            base_image = pdf_doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            images.append({"page": page_num + 1, "image": image})
    pdf_doc.close()

    return text_blocks, tables, images


# --------------------------------------------------------------------------
# Step 2: OCR the extracted images
# --------------------------------------------------------------------------

def preprocess_image_for_ocr(img):
    """Clean an image so Tesseract can read it more reliably."""
    img = img.convert("L")                                   # grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)            # separate ink from paper
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)   # upscale
    return img.filter(ImageFilter.SHARPEN)                   # crisp edges


def ocr_images(images, lang="eng"):
    """Run OCR on each image and keep the non-empty results."""
    blocks = []
    for obj in images:
        try:
            text = pytesseract.image_to_string(
                preprocess_image_for_ocr(obj["image"]), lang=lang
            )
        except Exception as exc:                # a single bad image must not stop the run
            print(f"[OCR skipped on page {obj['page']}] {exc}")
            continue
        if text.strip():
            blocks.append({
                "page": obj["page"],
                "text": f"[OCR from Image Page {obj['page']}]\n{text.strip()}",
            })
    return blocks


# --------------------------------------------------------------------------
# Step 3: tables and chunking
# --------------------------------------------------------------------------

def tables_to_blocks(tables):
    """Flatten each table into text so it can be embedded and retrieved."""
    blocks = []
    for t in tables:
        rows = []
        for row in t["table"]:
            cells = [str(c).strip() if c is not None else "" for c in row]
            if not any(cells):          # skip rows that are entirely empty
                continue
            rows.append(" | ".join(cells))
        text = "\n".join(rows)
        if text.strip():
            blocks.append({
                "page": t["page"],
                "text": f"[Table from Page {t['page']}]\n{text}",
            })
    return blocks


def chunk_blocks(blocks, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long blocks into overlapping word windows.

    The embedding model only reads the first ~256 tokens of any input, so a
    whole page as one block leaves most of a dense page unsearchable. The
    overlap keeps sentences that straddle a boundary intact in one chunk.
    """
    chunks = []
    step = max(size - overlap, 1)
    for b in blocks:
        words = b["text"].split()
        if not words:
            continue
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + size])
            if piece.strip():
                chunks.append({"page": b["page"], "text": piece})
            if start + size >= len(words):
                break
    return chunks


# --------------------------------------------------------------------------
# Step 4: embeddings + FAISS index
# --------------------------------------------------------------------------

class PDFIndex:
    """Semantic search over the extracted blocks.

    all-MiniLM-L6-v2 produces 384-dimensional vectors that are already
    normalised, so ranking by L2 distance gives the same order as cosine
    similarity.
    """

    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.pages = []
        self.index = None

    def add_documents(self, docs):
        if not docs:
            raise ValueError("No text could be extracted from this PDF.")

        self.texts = [d["text"] for d in docs]
        self.pages = [d.get("page") for d in docs]

        embeddings = self.model.encode(
            self.texts, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_context(self, question, top_k=TOP_K):
        """Return the most relevant chunks, joined into one context string."""
        if self.index is None:
            raise RuntimeError("No document indexed yet.")

        q_vec = self.model.encode(
            [question], convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

        # Asking for more neighbours than exist makes FAISS return -1 padding,
        # which would silently wrap around to the last chunk.
        k = min(top_k, self.index.ntotal)
        _scores, indices = self.index.search(q_vec, k)

        parts = []
        for i in indices[0]:
            if i < 0:
                continue
            page = self.pages[i]
            parts.append(f"[Page {page}]\n{self.texts[i]}")
        return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Step 5: local LLM via Ollama
# --------------------------------------------------------------------------

def build_prompt(question, context, history=None):
    """Prompt the model with the retrieved context and recent conversation.

    The instruction to stay inside the context is what keeps the model from
    inventing an answer when retrieval misses.
    """
    history_text = ""
    if history:
        turns = []
        for q, a in history[-3:]:
            turns.append(f"User: {q}\nAssistant: {a}")
        history_text = "Recent conversation:\n" + "\n".join(turns) + "\n\n"

    return (
        "You are a careful assistant answering questions about a document.\n"
        "Answer using only the context below. If the answer is not in the "
        "context, say you could not find it in the document. Cite the page "
        "number when you use a fact.\n\n"
        f"{history_text}"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def query_ollama_http(question, context, history=None,
                      model=OLLAMA_MODEL, url=OLLAMA_URL):
    """Send the prompt to Ollama and return the answer text."""
    prompt = build_prompt(question, context, history)
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            return f"[Ollama error] {data['error']}"
        return data.get("response", "").strip() or "[Empty response from model]"
    except requests.exceptions.ConnectionError:
        return ("[Error] Cannot reach Ollama on port 11434. "
                "Start Ollama and try again.")
    except requests.exceptions.Timeout:
        return "[Error] Ollama did not respond in time."
    except requests.exceptions.RequestException as exc:
        return f"[Error] Request to Ollama failed: {exc}"


# --------------------------------------------------------------------------
# Step 6: orchestration
# --------------------------------------------------------------------------

def run_pipeline(pdf_path, progress=print):
    """Build a searchable index for one PDF. Returns (index, stats)."""
    progress("Extracting text, tables and images...")
    text_blocks, tables, images = extract_pdf_content(pdf_path)

    progress(f"Running OCR on {len(images)} image(s)...")
    ocr_blocks = ocr_images(images)

    table_blocks = tables_to_blocks(tables)
    all_blocks = chunk_blocks(text_blocks + ocr_blocks + table_blocks)

    stats = {
        "pages_with_text": len(text_blocks),
        "ocr_blocks": len(ocr_blocks),
        "tables": len(table_blocks),
        "chunks": len(all_blocks),
    }
    progress(f"Indexing {stats['chunks']} chunks "
             f"({stats['pages_with_text']} text pages, "
             f"{stats['ocr_blocks']} OCR, {stats['tables']} tables)...")

    index = PDFIndex()
    index.add_documents(all_blocks)

    progress("Ready.")
    return index, stats


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_DIR = "logs"


def _log_paths(pdf_path):
    os.makedirs(LOG_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return (os.path.join(LOG_DIR, f"{base}_qa_log.txt"),
            os.path.join(LOG_DIR, f"{base}_chat_memory.json"))


def log_interaction(pdf_path, question, answer):
    """Append a timestamped Q&A entry to the audit log."""
    log_file, _ = _log_paths(pdf_path)
    serial = 1
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            serial = f.read().count("\nQuestion: ") + 1
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{serial}. {stamp}\nQuestion: {question}\nAnswer: {answer}\n\n")


def save_chat_memory(pdf_path, history):
    """Persist the conversation so it survives a restart."""
    _, mem_file = _log_paths(pdf_path)
    with open(mem_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_chat_memory(pdf_path):
    _, mem_file = _log_paths(pdf_path)
    if os.path.exists(mem_file):
        try:
            with open(mem_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


# --------------------------------------------------------------------------
# Console interface (the notebook path)
# --------------------------------------------------------------------------

def run_qa_interface(pdf_path):
    index, _stats = run_pipeline(pdf_path)
    history = []
    print("\nPDF loaded. Ask your questions. Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "exit":
            print("Goodbye.")
            break
        if not question:
            continue

        context = index.retrieve_context(question)
        answer = query_ollama_http(question, context, history)
        print(f"\nAnswer: {answer}\n")

        history.append((question, answer))
        log_interaction(pdf_path, question, answer)
        save_chat_memory(pdf_path, history)


def summarize_pdf(pdf_path):
    index, _stats = run_pipeline(pdf_path)
    context = index.retrieve_context("summary of this document", top_k=6)
    summary = query_ollama_http("Summarise this document in a short paragraph.",
                                context)
    print("\nSummary:\n", summary)
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_qa_interface(sys.argv[1])
    else:
        print("Usage: python pdf_qa_core.py <path-to-pdf>")
