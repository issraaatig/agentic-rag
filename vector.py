import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# =========================
# PATH STABLE
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db_fresh")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("medquad_cvd_final.csv")

print("DATAFRAME SIZE:", len(df))

documents = []
for idx, row in df.iterrows():
    text_content = f"Focus: {row['focus']}\nQuestion: {row['question']}\nAnswer: {row['answer']}"
    
    documents.append(Document(
        page_content=text_content,
        metadata={"focus": row["focus"], "source": "medquad_cvd_final.csv"}
    ))

# =========================
# CHUNKING
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75
)

chunks = text_splitter.split_documents(documents)

print("TOTAL CHUNKS:", len(chunks))

# =========================
# EMBEDDINGS
# =========================
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# =========================
# VECTOR STORE (CREATE)
# =========================
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("CHROMA COUNT:", vector_store._collection.count())

# =========================
# RETRIEVER
# =========================
retriever = vector_store.as_retriever(search_kwargs={"k": 5})