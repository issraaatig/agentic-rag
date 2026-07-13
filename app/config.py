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
    model="qwen2.5:1.5b",
    base_url=OLLAMA_BASE_URL ,
    temperature=0.2
)


# =========================
# VECTOR STORE (LOAD EXISTANT)
# =========================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from pathlib import Path

# config.py est dans app/, donc la racine du projet est le parent de app/
BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db_fresh")

print(f"📂 Chroma persist dir résolu : {PERSIST_DIR}")
print(f"📂 Ce dossier existe : {os.path.exists(PERSIST_DIR)}")
# IMPORTANT: mêmes embeddings que vector.py
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url=OLLAMA_BASE_URL
)

vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings ,
    collection_name="medquad_cv"

)

print("Nombre de documents dans Chroma :")

try:
    print(vector_store._collection.count())
except Exception as e:
    print(e)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


try:
    test_docs = retriever.invoke("anemia")
    print("🧪 TEST RETRIEVER:", len(test_docs))
except Exception as e:
    print("RETRIEVER ERROR:", e)