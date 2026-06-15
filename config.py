from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import uuid

SESSION_ID = str(uuid.uuid4())
USER_ID = "med-rag-user"

# LLM
llm = OllamaLLM(
    model="phi3:latest",
    temperature=0.2
)

# Embeddings (CHOISIR UNE SEULE FOIS ET NE PLUS CHANGER)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

PERSIST_DIR = "./chroma_db_fresh"

vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})