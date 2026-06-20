import os
import uuid
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

# 1. TABLEAU BLANC (Variable d'environnement)
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
print(f"🔗 config.py -> Connexion à Ollama sur : {OLLAMA_BASE_URL}")
# =========================
# SESSION
# =========================
SESSION_ID = str(uuid.uuid4())
USER_ID = "med-rag-user"


# =========================
# LLM (OLLAMA)
# =========================
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="phi3:latest",
    base_url=OLLAMA_BASE_URL ,
    temperature=0.2
)


# =========================
# VECTOR STORE (LOAD EXISTANT)
# =========================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "/app/chroma_db"

# IMPORTANT: mêmes embeddings que vector.py
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url=OLLAMA_BASE_URL
)

vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})