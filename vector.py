import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
df = pd.read_csv("medquad_cvd_final.csv")

documents = []
for idx, row in df.iterrows():
    text_content = f"Focus: {row['focus']}\nQuestion: {row['question']}\nAnswer: {row['answer']}"
    doc = Document(
        page_content=text_content,
        metadata={"focus": row["focus"], "source": "medquad_cvd_final.csv"}
    )
    documents.append(doc)

# ==========================================
# 2. TEXT CHUNKING (500 / 75)
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       
    chunk_overlap=75,     
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# ==========================================
# 3. VECTORIZATION & LOCAL STORAGE (CHROMA)
# ==========================================
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_fresh"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
