import os
import uuid

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
    temperature=0.2
)


# =========================
# VECTOR STORE (LOAD EXISTANT)
# =========================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "./chroma_db_fresh"

# IMPORTANT: mêmes embeddings que vector.py
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})