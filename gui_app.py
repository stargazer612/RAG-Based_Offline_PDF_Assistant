"""
gui_app.py
----------
Tkinter desktop interface for the Offline PDF Assistant.

    python gui_app.py

Indexing and model calls run on background threads, so the window stays
responsive while a PDF is processed or Mistral is generating an answer.
Worker threads never touch widgets directly; they put messages on a queue
that the main thread drains on a timer, which is the safe pattern for
Tkinter.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

import pdf_qa_core as core


class PDFAssistantApp:
    APP_TITLE = "Offline PDF Assistant"

    def __init__(self, root):
        self.root = root
        self.root.title(self.APP_TITLE)
        self.root.geometry("900x650")
        self.root.minsize(640, 480)

        self.index = None            # PDFIndex once a PDF is loaded
        self.pdf_path = None
        self.history = []            # [(question, answer), ...]
        self.busy = False
        self.events = queue.Queue()  # worker threads -> UI

        self._build_widgets()
        self.root.after(100, self._drain_events)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_widgets(self):
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill=tk.X)

        self.upload_btn = ttk.Button(toolbar, text="Upload PDF",
                                     command=self.on_upload)
        self.upload_btn.pack(side=tk.LEFT)

        self.clear_btn = ttk.Button(toolbar, text="Clear chat",
                                    command=self.on_clear)
        self.clear_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.doc_label = ttk.Label(toolbar, text="No document loaded")
        self.doc_label.pack(side=tk.LEFT, padx=16)

        self.chat = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state=tk.DISABLED,
            font=("Segoe UI", 10), padx=12, pady=12,
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=10)

        self.chat.tag_config("user", foreground="#1a4f8b",
                             font=("Segoe UI", 10, "bold"), spacing1=8)
        self.chat.tag_config("bot", foreground="#123", spacing1=4, spacing3=8)
        self.chat.tag_config("system", foreground="#777",
                             font=("Segoe UI", 9, "italic"), spacing1=4)
        self.chat.tag_config("error", foreground="#a11",
                             font=("Segoe UI", 9), spacing1=4)

        entry_row = ttk.Frame(self.root, padding=(10, 8))
        entry_row.pack(fill=tk.X)

        self.entry = ttk.Entry(entry_row, font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.entry.bind("<Return>", lambda _e: self.on_send())

        self.send_btn = ttk.Button(entry_row, text="Send", command=self.on_send)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

        status_row = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        status_row.pack(fill=tk.X)

        self.status = ttk.Label(status_row, text="Upload a PDF to begin.")
        self.status.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(status_row, mode="indeterminate",
                                        length=140)
        self.progress.pack(side=tk.RIGHT)

        self._set_input_enabled(False)
        self._append("Upload a PDF to start asking questions about it.\n",
                     "system")

    # ------------------------------------------------------------------
    # UI helpers (main thread only)
    # ------------------------------------------------------------------

    def _append(self, text, tag):
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _set_input_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.entry.configure(state=state)
        self.send_btn.configure(state=state)

    def _set_busy(self, busy, message=""):
        self.busy = busy
        self.status.configure(text=message)
        self.upload_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self._set_input_enabled(not busy and self.index is not None)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if not busy and self.index is not None:
            self.entry.focus_set()

    # ------------------------------------------------------------------
    # Loading a PDF
    # ------------------------------------------------------------------

    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Select a PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        self.pdf_path = path
        self.index = None
        self.history = []
        name = os.path.basename(path)
        self.doc_label.configure(text=name)
        self._append(f"\nLoading {name}...\n", "system")
        self._set_busy(True, "Processing document...")

        threading.Thread(target=self._load_worker, args=(path,),
                         daemon=True).start()

    def _load_worker(self, path):
        try:
            index, stats = core.run_pipeline(
                path, progress=lambda m: self.events.put(("status", m))
            )
            self.events.put(("loaded", (index, stats)))
        except Exception as exc:
            self.events.put(("load_failed", str(exc)))

    # ------------------------------------------------------------------
    # Asking a question
    # ------------------------------------------------------------------

    def on_send(self):
        if self.busy or self.index is None:
            return
        question = self.entry.get().strip()
        if not question:
            return

        self.entry.delete(0, tk.END)
        self._append(f"\nYou: {question}\n", "user")
        self._set_busy(True, "Thinking...")

        threading.Thread(target=self._answer_worker, args=(question,),
                         daemon=True).start()

    def _answer_worker(self, question):
        try:
            context = self.index.retrieve_context(question)
            answer = core.query_ollama_http(question, context, self.history)
            self.events.put(("answer", (question, answer)))
        except Exception as exc:
            self.events.put(("answer_failed", str(exc)))

    def on_clear(self):
        self.history = []
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self._append("Chat cleared. The document is still loaded.\n", "system")

    # ------------------------------------------------------------------
    # Event pump: worker messages handled on the main thread
    # ------------------------------------------------------------------

    def _drain_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()

                if kind == "status":
                    self.status.configure(text=payload)

                elif kind == "loaded":
                    self.index, stats = payload
                    self.history = []
                    self._append(
                        f"Indexed {stats['chunks']} chunks from "
                        f"{stats['pages_with_text']} text page(s), "
                        f"{stats['ocr_blocks']} OCR block(s) and "
                        f"{stats['tables']} table(s). Ask a question below.\n",
                        "system",
                    )
                    self._set_busy(False, "Ready.")

                elif kind == "load_failed":
                    self._append(f"Could not read this PDF: {payload}\n",
                                 "error")
                    self._set_busy(False, "Upload a PDF to begin.")

                elif kind == "answer":
                    question, answer = payload
                    self._append(f"Assistant: {answer}\n", "bot")
                    self.history.append((question, answer))
                    try:
                        core.log_interaction(self.pdf_path, question, answer)
                        core.save_chat_memory(self.pdf_path, self.history)
                    except OSError as exc:
                        self._append(f"(Could not write log: {exc})\n", "error")
                    self._set_busy(False, "Ready.")

                elif kind == "answer_failed":
                    self._append(f"Something went wrong: {payload}\n", "error")
                    self._set_busy(False, "Ready.")

        except queue.Empty:
            pass

        self.root.after(100, self._drain_events)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")      # Windows; falls back below
    except tk.TclError:
        pass
    PDFAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
