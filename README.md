# Local AI - Elite Edition 🚀

Local AI - Elite Edition is a 100% offline, privacy-first desktop artificial intelligence application built with **Python**, **PySide6 (Qt)**, and **GPT4All**. It brings multi-modal capabilities—including standard chatting, advanced code generation, document-based Retrieval-Augmented Generation (RAG), and image vision—straight to your local machine's CPU. No API keys, no subscriptions, and zero data leaving your device.

## ✨ Key Features

- **Dynamic Dual-Brain Architecture:**
  - **Chat Mode (Llama 3.2 3B):** Lightweight, blazing-fast response engine optimized for everyday tasks, text manipulation, and rapid assistance.
  - **Pro Mode (Qwen 2.5 Coder 7B):** Heavy-duty, lazy-loaded engineering core optimized for multi-class scripting, strict software architecture design, and granular code debugging.
- **Advanced RAG Engine:** Upload `.txt`, `.pdf`, or `.docx` files to automatically generate a local context vector database powered by Meta's **FAISS**, enabling sub-second contextual querying of huge materials.
- **Visual Intelligence:** Dynamic local visual analysis powered by the **Moondream2** vision model.
- **Rich User Experience:** Features a fully asynchronous, multi-threaded `QThread` processing layer that prevents UI freezing, an elegant native dark-mode stylesheet, local markdown visualization, and automated chat session logging.

## 🛠️ Tech Stack

- **GUI Framework:** PySide6 (Qt for Python)
- **Inference Backends:** GPT4All, HuggingFace Transformers, PyTorch
- **Vector Indexing:** FAISS (Facebook AI Similarity Search)
- **Web Scraping Components:** DuckDuckGo Search API (`duckduckgo_search`)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10 or newer installed on your machine.

### 2. Clone the Repository
```bash
git clone [https://github.com/swagatambordoloi/Local-Ai.git](https://github.com/swagatambordoloi/Local-Ai.git)
cd Local-A
