import sys
import os
import json
import uuid
from datetime import datetime
import time
import PyPDF2
import docx
from PIL import Image
import gc
import re
import subprocess

import torch
import multiprocessing

# Force PyTorch to use all physical CPU cores
torch.set_num_threads(multiprocessing.cpu_count())

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QScrollArea, QFrame, QFileDialog, QMessageBox, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QCursor, QColor

# --- RAG LIBRARIES ---
import faiss
import numpy as np
from gpt4all import GPT4All, Embed4All
from duckduckgo_search import DDGS
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.utils 

# --- HOTFIX FOR MOONDREAM ---
if not hasattr(transformers.utils, 'is_flash_attn_greater_or_equal_2_10'):
    transformers.utils.is_flash_attn_greater_or_equal_2_10 = lambda: False

# ==========================================
# 1. GLOBAL VARIABLES & STORAGE
# ==========================================
CHAT_MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
PRO_MODEL_FILE = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" 
HISTORY_FILE = "chat_history.json"

# Persistent Database Files
DB_INDEX_FILE = "project_docs.faiss"
DB_CHUNKS_FILE = "project_chunks.json"

QWEN_SYSTEM_PROMPT = """You are an elite Senior Full-Stack Engineer. Provide highly optimized, production-ready code.
- Prioritize memory efficiency and error handling.
- Wrap all standard code in markdown blocks.
- If the user asks you to TEST or RUN code, write a python script and wrap it exactly inside <EXECUTE> and </EXECUTE> tags.
"""

if not os.path.exists(CHAT_MODEL_FILE):
    print(f"CRITICAL ERROR: Cannot find '{CHAT_MODEL_FILE}'.")
    sys.exit(1)

llm_chat = None
llm_pro = None
embedder = None
vision_model = None
vision_tokenizer = None

def web_search(topic):
    try:
        results = DDGS().text(topic, max_results=3)
        if not results: return "No internet results found."
        return "\n".join([f"- {res['body']}" for res in results])
    except Exception as e:
        return f"Internet search failed: {str(e)}"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}

def save_history(history_data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4)

# ==========================================
# 2. STARTUP SEQUENCE & SPLASH SCREEN
# ==========================================
class StartupWorker(QThread):
    progress = Signal(str)
    finished = Signal()

    def run(self):
        global llm_chat, embedder
        self.progress.emit("Allocating memory...")
        time.sleep(0.5) 
        self.progress.emit("Loading Llama 3B Text Engine...")
        llm_chat = GPT4All(CHAT_MODEL_FILE, model_path=".", allow_download=False, device="cpu")
        self.progress.emit("Loading Vector Embedding Engine...")
        embedder = Embed4All()
        self.progress.emit("Initializing Local AI...")
        time.sleep(0.5)
        self.finished.emit()

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(400, 250)

        self.frame = QFrame(self)
        self.frame.setFixedSize(400, 250)
        self.frame.setStyleSheet("QFrame { background-color: #1C1C1E; border-radius: 15px; border: 1px solid #38383A; }")

        layout = QVBoxLayout(self.frame)
        layout.setAlignment(Qt.AlignCenter)

        self.logo = QLabel("Local AI")
        self.logo.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.logo.setStyleSheet("color: white; border: none;")
        self.logo.setAlignment(Qt.AlignCenter)

        self.spinner = QLabel("⠋")
        self.spinner.setFont(QFont("Segoe UI", 24))
        self.spinner.setStyleSheet("color: #0A84FF; border: none;")
        self.spinner.setAlignment(Qt.AlignCenter)

        self.status = QLabel("Waking up agent...")
        self.status.setFont(QFont("Segoe UI", 11))
        self.status.setStyleSheet("color: #8E8E93; border: none;")
        self.status.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.logo)
        layout.addWidget(self.spinner)
        layout.addWidget(self.status)

        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_frame = 0
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_spinner)
        self.anim_timer.start(80) 

    def update_spinner(self):
        self.current_frame = (self.current_frame + 1) % len(self.spinner_frames)
        self.spinner.setText(self.spinner_frames[self.current_frame])

    def update_status(self, text):
        self.status.setText(text)

# ==========================================
# 3. WORKER THREADS (RAG & Chat)
# ==========================================
class IndexWorker(QThread):
    finished = Signal(object, object) 
    progress = Signal(str)

    def __init__(self, raw_text):
        super().__init__()
        self.raw_text = raw_text

    def run(self):
        try:
            self.progress.emit("System")
            self.progress.emit("Splitting document into chunks...")
            
            words = self.raw_text.split()
            chunks = []
            chunk_size = 400
            overlap = 50
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk: chunks.append(chunk)

            self.progress.emit("System")
            self.progress.emit(f"Building Vector Database ({len(chunks)} chunks)...")
            embeddings = [embedder.embed(chunk) for chunk in chunks]
            embeddings_np = np.array(embeddings).astype('float32')

            dimension = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)

            # Save RAG state to disk
            faiss.write_index(index, DB_INDEX_FILE)
            with open(DB_CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(chunks, f)

            self.finished.emit(chunks, index)
        except Exception as e:
            self.progress.emit("System")
            self.progress.emit(f"Indexing Error: {str(e)}")
            self.finished.emit([], None)


class AIWorker(QThread):
    new_message = Signal(str, str)
    vision_activated = Signal()
    pro_activated = Signal() 
    finished = Signal()

    def __init__(self, query, filetype, filepath, document_chunks, faiss_index, app_mode, history_buffer):
        super().__init__()
        self.query = query
        self.filetype = filetype
        self.filepath = filepath
        self.document_chunks = document_chunks
        self.faiss_index = faiss_index
        self.app_mode = app_mode 
        self.history_buffer = history_buffer

    def run(self):
        global llm_chat, llm_pro, vision_model, vision_tokenizer
        query_lower = self.query.lower()

        try:
            # MEMORY MANAGER: Hot Swap logic
            if self.app_mode == "Pro Mode":
                if llm_chat is not None:
                    self.new_message.emit("System", "🧹 Clearing Llama 3B from memory...")
                    del llm_chat
                    llm_chat = None
                    gc.collect() 

                if llm_pro is None:
                    self.new_message.emit("System", "⚙️ Waking up Qwen 2.5 Coder 7B... Allocating 4.7GB...")
                    if not os.path.exists(PRO_MODEL_FILE):
                        self.new_message.emit("System", f"CRITICAL ERROR: Cannot find '{PRO_MODEL_FILE}'. Falling back...")
                        return
                    else:
                        llm_pro = GPT4All(PRO_MODEL_FILE, model_path=".", allow_download=False, device="cpu")
                        self.pro_activated.emit()
                active_llm = llm_pro
                system_prompt = QWEN_SYSTEM_PROMPT

            else: 
                if llm_pro is not None:
                    self.new_message.emit("System", "🧹 Clearing Qwen 7B from memory...")
                    del llm_pro
                    llm_pro = None
                    gc.collect()

                if llm_chat is None:
                    self.new_message.emit("System", "⚙️ Waking up Llama 3B Text Engine...")
                    llm_chat = GPT4All(CHAT_MODEL_FILE, model_path=".", allow_download=False, device="cpu")
                active_llm = llm_chat
                system_prompt = "You are a helpful AI assistant."

            # --- ROUTE 1: VISION (Moondream doesn't use standard chat history well) ---
            if self.filetype == 'image':
                if vision_model is None:
                    self.new_message.emit("System", "⚙️ Waking up Vision Engine...")
                    VISION_MODEL_ID = "vikhyatk/moondream2"
                    VISION_REVISION = "2024-08-26"
                    vision_model = AutoModelForCausalLM.from_pretrained(VISION_MODEL_ID, trust_remote_code=True, revision=VISION_REVISION)
                    vision_tokenizer = AutoTokenizer.from_pretrained(VISION_MODEL_ID, revision=VISION_REVISION)
                    self.vision_activated.emit() 
                
                image = Image.open(self.filepath)
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                enc_image = vision_model.encode_image(image)
                response = vision_model.answer_question(enc_image, self.query, vision_tokenizer)
                self.new_message.emit("Agent", f"👁️ {response}")

            # --- ROUTE 2: RAG RETRIEVAL ---
            elif self.filetype == 'document' and self.faiss_index is not None:
                query_embedding = np.array([embedder.embed(self.query)]).astype('float32')
                k = min(3, len(self.document_chunks))
                distances, indices = self.faiss_index.search(query_embedding, k)
                
                relevant_chunks = [self.document_chunks[i] for i in indices[0] if i >= 0]
                context = "\n\n...[excerpt]...\n\n".join(relevant_chunks)

                with active_llm.chat_session(system_prompt=system_prompt):
                    for msg in self.history_buffer:
                        if msg["sender"] in ["You", "Agent"]:
                            role = "user" if msg["sender"] == "You" else "assistant"
                            active_llm.current_chat_session.append({"role": role, "content": msg["text"]})
                            
                    context_query = f"Use the retrieved excerpts to answer the user.\nDOCUMENT EXCERPTS:\n{context}\n\nUser Question: {self.query}"
                    response = active_llm.generate(context_query, max_tokens=1500, temp=0.2).strip()
                
            # --- ROUTE 3: WEB SEARCH ---
            elif "search" in query_lower or "look up" in query_lower:
                search_term = self.query.replace("search for", "").replace("search", "").replace("look up", "").strip()
                self.new_message.emit("System", f"🔍 Scraping web for: '{search_term}'...")
                
                raw_results = web_search(search_term)
                with active_llm.chat_session(system_prompt=system_prompt):
                    for msg in self.history_buffer:
                        if msg["sender"] in ["You", "Agent"]:
                            role = "user" if msg["sender"] == "You" else "assistant"
                            active_llm.current_chat_session.append({"role": role, "content": msg["text"]})
                            
                    context_query = f"Use these web results to answer the user.\nRESULTS:\n{raw_results}\n\nQuestion: {self.query}"
                    response = active_llm.generate(context_query, max_tokens=1000, temp=0.3).strip()

            # --- ROUTE 4: STANDARD CHAT (WITH CONTEXT) ---
            else:
                with active_llm.chat_session(system_prompt=system_prompt):
                    # Inject Sliding Window Context
                    for msg in self.history_buffer:
                        if msg["sender"] in ["You", "Agent"]:
                            role = "user" if msg["sender"] == "You" else "assistant"
                            active_llm.current_chat_session.append({"role": role, "content": msg["text"]})
                            
                    response = active_llm.generate(self.query, max_tokens=2048, temp=0.6).strip()
                    if not response: response = "I processed that, but I'm not sure how to respond."

            if self.filetype != 'image':
                self.new_message.emit("Agent", response)

            # --- AGENTIC CODE EXECUTION ---
            if "<EXECUTE>" in response and "</EXECUTE>" in response:
                code_match = re.search(r"<EXECUTE>(.*?)</EXECUTE>", response, re.DOTALL)
                if code_match:
                    raw_code = code_match.group(1).strip()
                    clean_code = raw_code.replace("```python", "").replace("```", "").strip()
                    
                    with open("temp_run.py", "w", encoding="utf-8") as f:
                        f.write(clean_code)
                        
                    self.new_message.emit("System", "⚙️ Autonomously executing generated script...")
                    
                    try:
                        result = subprocess.run([sys.executable, "temp_run.py"], capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            self.new_message.emit("System", f"🖥️ **Execution Success:**\n```text\n{result.stdout}\n```")
                        else:
                            self.new_message.emit("System", f"⚠️ **Execution Failed:**\n```text\n{result.stderr}\n```")
                    except subprocess.TimeoutExpired:
                        self.new_message.emit("System", "⚠️ **Execution Failed:** Script timed out after 10 seconds.")
                    except Exception as e:
                        self.new_message.emit("System", f"⚠️ **System Error:** {str(e)}")

        except Exception as e:
            self.new_message.emit("System", f"Error: {str(e)}")

        self.finished.emit()

# ==========================================
# 4. CUSTOM UI WIDGETS
# ==========================================
class ChatBubble(QFrame):
    def __init__(self, sender, text):
        super().__init__()
        if sender == "You":
            bg_color, text_color, align, margins = "#0A84FF", "white", Qt.AlignRight, (50, 0)
        elif sender == "System":
            bg_color, text_color, align, margins = "transparent", "#8E8E93", Qt.AlignCenter, (0, 0)
        else:
            bg_color, text_color, align, margins = "#2C2C2E", "#E5E5EA", Qt.AlignLeft, (0, 50)

        layout = QHBoxLayout()
        layout.setContentsMargins(margins[0], 5, margins[1], 5)
        
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setFont(QFont("Segoe UI", 11))
        self.label.setTextFormat(Qt.MarkdownText) 
        
        self.label.setStyleSheet(f"""
            QLabel {{ background-color: {bg_color}; color: {text_color}; border-radius: 12px; padding: 12px; }}
            code {{ background-color: #1C1C1E; color: #FF9500; padding: 2px 4px; border-radius: 4px; }}
        """)

        if align == Qt.AlignRight:
            layout.addStretch()
            layout.addWidget(self.label)
        elif align == Qt.AlignLeft:
            layout.addWidget(self.label)
            layout.addStretch()
        else:
            layout.addStretch()
            layout.addWidget(self.label)
            layout.addStretch()
        self.setLayout(layout)

# ==========================================
# 5. MAIN WINDOW APP
# ==========================================
class LocalAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local AI - Elite Edition")
        self.resize(1100, 800)
        
        self.attached_filepath = None
        self.attached_filetype = None 
        self.document_chunks = []
        self.faiss_index = None

        self.chat_history = load_history()
        self.current_session_id = None

        self.thinking_timer = QTimer(self)
        self.thinking_timer.timeout.connect(self.animate_thinking)
        self.thinking_bubble = None
        self.thinking_step = 0

        self.setStyleSheet("""
            QMainWindow { background-color: #1C1C1E; color: white; }
            QMessageBox { background-color: #1C1C1E; color: white; }
            QMessageBox QPushButton { background-color: #2C2C2E; color: white; border-radius: 5px; padding: 5px 15px; min-width: 60px; }
            QMessageBox QPushButton:hover { background-color: #3A3A3C; }
            QScrollBar:vertical { border: none; background: #1C1C1E; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #48484A; min-height: 20px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QComboBox { background-color: #2C2C2E; color: white; border-radius: 8px; padding: 5px 10px; border: 1px solid #38383A; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #2C2C2E; color: white; border: 1px solid #38383A; selection-background-color: #0A84FF; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background-color: #000000; border-right: 1px solid #38383A;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)

        logo = QLabel("Local AI")
        logo.setFont(QFont("Segoe UI", 20, QFont.Bold))
        logo.setStyleSheet("color: white; border: none;")
        
        self.btn_new_chat = QPushButton("➕ New Chat")
        self.btn_new_chat.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_new_chat.setFixedHeight(40)
        self.btn_new_chat.setStyleSheet("QPushButton { background-color: #2C2C2E; color: white; border-radius: 8px; font-weight: bold; font-size: 14px; margin-top: 10px; } QPushButton:hover { background-color: #3A3A3C; }")
        self.btn_new_chat.clicked.connect(self.start_new_session)

        self.btn_load_db = QPushButton("📂 Load RAG Database")
        self.btn_load_db.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_load_db.setFixedHeight(35)
        self.btn_load_db.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid #38383A; color: white; border-radius: 8px; font-size: 13px; margin-top: 5px; } QPushButton:hover { background-color: #2C2C2E; }")
        self.btn_load_db.clicked.connect(self.load_database)

        self.lbl_llama = QLabel("🟢 Llama 3B (Active)")
        self.lbl_llama.setStyleSheet("color: white; border: none; font-size: 12px; margin-top: 20px;")
        
        self.lbl_pro = QLabel("⚪ Qwen 7B (Standby)")
        self.lbl_pro.setStyleSheet("color: #8E8E93; border: none; font-size: 12px; margin-top: 5px;")
        
        self.lbl_rag = QLabel("⚪ RAG Database (Empty)")
        self.lbl_rag.setStyleSheet("color: #8E8E93; border: none; font-size: 12px; margin-top: 5px;")
        
        self.lbl_vision = QLabel("⚪ Vision Engine (Standby)")
        self.lbl_vision.setStyleSheet("color: #8E8E93; border: none; font-size: 12px; margin-top: 5px; margin-bottom: 20px;")

        lbl_history = QLabel("Recent Chats")
        lbl_history.setStyleSheet("color: #8E8E93; border: none; font-size: 12px; font-weight: bold; margin-bottom: 5px;")
        
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFrameShape(QFrame.NoFrame)
        self.history_scroll.setStyleSheet("background-color: transparent;")
        
        self.history_container = QWidget()
        self.history_container.setStyleSheet("background-color: transparent;")
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setAlignment(Qt.AlignTop)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(5)
        self.history_scroll.setWidget(self.history_container)

        sidebar_layout.addWidget(logo)
        sidebar_layout.addWidget(self.btn_new_chat)
        sidebar_layout.addWidget(self.btn_load_db)
        sidebar_layout.addWidget(self.lbl_llama)
        sidebar_layout.addWidget(self.lbl_pro)
        sidebar_layout.addWidget(self.lbl_rag)
        sidebar_layout.addWidget(self.lbl_vision)
        sidebar_layout.addWidget(lbl_history)
        sidebar_layout.addWidget(self.history_scroll)

        chat_container = QWidget()
        chat_container.setStyleSheet("background-color: #1C1C1E;")
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_widget)

        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #1C1C1E; border-top: 1px solid #38383A;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(20, 15, 20, 15)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Chat Mode", "Pro Mode"]) 
        self.mode_selector.setCursor(QCursor(Qt.PointingHandCursor))
        self.mode_selector.setFixedSize(110, 45)
        self.mode_selector.setFont(QFont("Segoe UI", 12))

        self.btn_attach = QPushButton("📎 Attach")
        self.btn_attach.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_attach.setFixedSize(80, 45)
        self.btn_attach.setStyleSheet("QPushButton { background-color: #2C2C2E; color: white; border-radius: 8px; font-weight: bold; font-size: 14px; } QPushButton:hover { background-color: #3A3A3C; }")
        self.btn_attach.clicked.connect(self.upload_file)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Message your agent, execute code, or attach files...")
        self.input_field.setFixedHeight(45)
        self.input_field.setStyleSheet("QLineEdit { background-color: #2C2C2E; color: white; border-radius: 20px; padding: 0 15px; font-size: 14px; border: none; }")
        self.input_field.returnPressed.connect(self.start_thinking)

        self.btn_send = QPushButton("Send ➔")
        self.btn_send.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_send.setFixedSize(80, 45)
        self.btn_send.setStyleSheet("QPushButton { background-color: #0A84FF; color: white; border-radius: 22px; font-weight: bold; font-size: 14px; } QPushButton:hover { background-color: #007AFF; } QPushButton:disabled { background-color: #3A3A3C; color: #8E8E93; }")
        self.btn_send.clicked.connect(self.start_thinking)

        input_layout.addWidget(self.mode_selector)
        input_layout.addWidget(self.btn_attach)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.btn_send)

        chat_layout.addWidget(self.scroll_area)
        chat_layout.addWidget(input_frame)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(chat_container)

        self.refresh_history_sidebar()
        self.start_new_session()

    def load_database(self):
        if os.path.exists(DB_INDEX_FILE) and os.path.exists(DB_CHUNKS_FILE):
            try:
                self.faiss_index = faiss.read_index(DB_INDEX_FILE)
                with open(DB_CHUNKS_FILE, "r", encoding="utf-8") as f:
                    self.document_chunks = json.load(f)
                
                self.attached_filetype = 'document'
                self.lbl_rag.setText("🟢 RAG Database (Active)")
                self.lbl_rag.setStyleSheet("color: white; border: none; font-size: 12px; margin-top: 5px;")
                self.add_message("System", f"✅ Successfully loaded {len(self.document_chunks)} context chunks from disk.", save=False)
            except Exception as e:
                self.add_message("System", f"⚠️ Error loading database: {str(e)}", save=False)
        else:
            self.add_message("System", "⚠️ No saved database found. Attach a document first.", save=False)

    def start_new_session(self):
        self.current_session_id = str(uuid.uuid4())
        self.chat_history[self.current_session_id] = {
            "title": "New Chat", "timestamp": datetime.now().isoformat(), "messages": []
        }
        self.clear_chat_ui()
        self.add_message("System", "Local AI ready. Select **Pro Mode** to wake up Qwen 7B for elite coding. Tell me to `test logic` to run local python scripts.", save=False)
        self.refresh_history_sidebar()

    def load_session(self, session_id):
        if session_id in self.chat_history:
            self.current_session_id = session_id
            self.clear_chat_ui()
            for msg in self.chat_history[session_id]["messages"]:
                self.add_message(msg["sender"], msg["text"], save=False)

    def delete_session(self, session_id):
        if session_id in self.chat_history:
            reply = QMessageBox.question(self, "Confirm Delete", f"Delete chat:\n\n'{self.chat_history[session_id]['title']}'?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.chat_history[session_id]
                save_history(self.chat_history)
                if self.current_session_id == session_id: self.start_new_session()
                else: self.refresh_history_sidebar()

    def clear_chat_ui(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()

    def refresh_history_sidebar(self):
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
                
        sorted_sessions = sorted(self.chat_history.items(), key=lambda x: x[1]['timestamp'], reverse=True)
        for sess_id, data in sorted_sessions:
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)

            btn_load = QPushButton(data["title"])
            btn_load.setCursor(QCursor(Qt.PointingHandCursor))
            btn_load.setFixedHeight(35)
            btn_load.setStyleSheet("QPushButton { background-color: transparent; color: #E5E5EA; text-align: left; padding-left: 10px; border-radius: 5px; font-size: 13px; } QPushButton:hover { background-color: #2C2C2E; }")
            btn_load.clicked.connect(lambda checked=False, sid=sess_id: self.load_session(sid))
            
            btn_del = QPushButton("🗑️")
            btn_del.setCursor(QCursor(Qt.PointingHandCursor))
            btn_del.setFixedSize(30, 35)
            btn_del.setStyleSheet("QPushButton { background-color: transparent; border-radius: 5px; font-size: 14px; } QPushButton:hover { background-color: #FF3B30; }")
            btn_del.clicked.connect(lambda checked=False, sid=sess_id: self.delete_session(sid))

            item_layout.addWidget(btn_load)
            item_layout.addWidget(btn_del)
            self.history_layout.addWidget(item_widget)

    def upload_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select a File", "", "Documents & Images (*.txt *.pdf *.docx *.jpg *.jpeg *.png)")
        if not filepath: return

        ext = filepath.lower().split('.')[-1]
        filename = os.path.basename(filepath)
        
        if ext in ['txt', 'pdf', 'docx']:
            self.attached_filetype = 'document'
            self.add_message("System", f"📎 Reading Document: {filename}")
            raw_text = self.extract_text(filepath)
            
            self.btn_send.setEnabled(False)
            self.btn_attach.setEnabled(False)
            
            self.index_worker = IndexWorker(raw_text)
            self.index_worker.progress.connect(lambda sender, msg: self.add_message(sender, msg, save=False))
            self.index_worker.finished.connect(self.finish_indexing)
            self.index_worker.start()

        elif ext in ['jpg', 'jpeg', 'png']:
            self.attached_filetype = 'image'
            self.attached_filepath = filepath
            self.add_message("System", f"🖼️ Attached Image: {filename}")

    def finish_indexing(self, chunks, index):
        self.document_chunks = chunks
        self.faiss_index = index
        self.lbl_rag.setText("🟢 RAG Database (Active)")
        self.lbl_rag.setStyleSheet("color: white; border: none; font-size: 12px; margin-top: 5px;")
        self.add_message("System", "✅ Database built & saved! You can now ask questions about the document.", save=False)
        self.btn_send.setEnabled(True)
        self.btn_attach.setEnabled(True)

    def extract_text(self, filepath):
        ext = filepath.lower().split('.')[-1]
        try:
            if ext == "txt":
                with open(filepath, "r", encoding="utf-8") as f: return f.read()
            elif ext == "pdf":
                reader = PyPDF2.PdfReader(filepath)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif ext == "docx":
                doc = docx.Document(filepath)
                return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e: return f"Error reading document: {str(e)}"

    def animate_thinking(self):
        if self.thinking_bubble:
            self.thinking_step = (self.thinking_step + 1) % 4
            self.thinking_bubble.label.setText(f"Thinking {'•' * self.thinking_step}")

    def remove_thinking_bubble(self):
        if self.thinking_bubble:
            self.thinking_timer.stop()
            self.scroll_layout.removeWidget(self.thinking_bubble)
            self.thinking_bubble.deleteLater()
            self.thinking_bubble = None

    def add_message(self, sender, text, save=True):
        self.remove_thinking_bubble()
        
        bubble = ChatBubble(sender, text)
        self.scroll_layout.addWidget(bubble)
        
        if save and self.current_session_id:
            if sender == "You" and len(self.chat_history[self.current_session_id]["messages"]) == 0:
                self.chat_history[self.current_session_id]["title"] = text[:25] + ("..." if len(text) > 25 else "")
                self.refresh_history_sidebar()

            self.chat_history[self.current_session_id]["messages"].append({"sender": sender, "text": text})
            self.chat_history[self.current_session_id]["timestamp"] = datetime.now().isoformat()
            save_history(self.chat_history)
        
        if not self.btn_send.isEnabled() and sender != "Agent":
            self.thinking_bubble = ChatBubble("Agent", "Thinking ")
            self.scroll_layout.addWidget(self.thinking_bubble)
            self.thinking_timer.start(400) 
            
        QTimer.singleShot(50, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_thinking(self):
        query = self.input_field.text().strip()
        if not query: return

        # --- CONTEXT INJECTION: Grab the last 6 messages ---
        history_buffer = []
        if self.current_session_id and self.current_session_id in self.chat_history:
            history_buffer = self.chat_history[self.current_session_id]["messages"][-6:]

        self.add_message("You", query)
        self.input_field.clear()
        
        selected_mode = self.mode_selector.currentText()
        
        self.btn_send.setEnabled(False)
        self.btn_send.setText("•••")
        self.btn_attach.setEnabled(False)

        self.thinking_bubble = ChatBubble("Agent", "Thinking ")
        self.scroll_layout.addWidget(self.thinking_bubble)
        self.thinking_timer.start(400)
        QTimer.singleShot(50, self.scroll_to_bottom)

        # Pass the history_buffer into the AI Worker
        self.worker = AIWorker(query, self.attached_filetype, self.attached_filepath, self.document_chunks, self.faiss_index, selected_mode, history_buffer)
        
        self.worker.new_message.connect(lambda s, m: self.add_message(s, m))
        self.worker.vision_activated.connect(self.update_vision_sidebar)
        self.worker.pro_activated.connect(self.update_pro_sidebar)
        self.worker.finished.connect(self.finish_thinking)
        self.worker.start()

    def update_vision_sidebar(self):
        self.lbl_vision.setText("🟢 Vision Engine (Active)")
        self.lbl_vision.setStyleSheet("color: white; border: none; font-size: 12px; margin-top: 5px; margin-bottom: 20px;")
        
    def update_pro_sidebar(self):
        self.lbl_pro.setText("🟢 Qwen 7B (Active)")
        self.lbl_pro.setStyleSheet("color: white; border: none; font-size: 12px; margin-top: 5px;")
        self.lbl_llama.setText("⚪ Llama 3B (Standby)")
        self.lbl_llama.setStyleSheet("color: #8E8E93; border: none; font-size: 12px; margin-top: 20px;")

    def finish_thinking(self):
        self.remove_thinking_bubble() 
        self.btn_send.setEnabled(True)
        self.btn_send.setText("Send ➔")
        self.btn_attach.setEnabled(True)
        
        if self.mode_selector.currentText() == "Chat Mode":
            self.lbl_llama.setText("🟢 Llama 3B (Active)")
            self.lbl_llama.setStyleSheet("color: white; border: none; font-size: 12px; margin-top: 20px;")
            self.lbl_pro.setText("⚪ Qwen 7B (Standby)")
            self.lbl_pro.setStyleSheet("color: #8E8E93; border: none; font-size: 12px; margin-top: 5px;")

        if self.attached_filetype == 'image':
            self.attached_filetype = None
            self.attached_filepath = None

# ==========================================
# 6. APP EXECUTION & BOOTSTRAP
# ==========================================
def main():
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    splash = SplashScreen()
    splash.show()

    main_window = LocalAIApp()

    startup = StartupWorker()
    startup.progress.connect(splash.update_status)
    
    def on_startup_finished():
        splash.close()
        main_window.show()

    startup.finished.connect(on_startup_finished)
    startup.start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()