import os
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ==========================================
# 1. TABLEAU BLANC (Variable d'environnement)
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
print(f"🔗 vector.py -> Connexion à Ollama sur : {OLLAMA_BASE_URL}")

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
print("📖 Lecture du CSV...")
df = pd.read_csv("/app/data/medquad_cvd_final.csv")
print(f"✅ CSV chargé : {len(df)} lignes")

print("📝 Création des documents...")
documents = []
for idx, row in df.iterrows():
    text_content = f"Focus: {row['focus']}\nQuestion: {row['question']}\nAnswer: {row['answer']}"
    doc = Document(
        page_content=text_content,
        metadata={"focus": row["focus"], "source": "medquad_cvd_final.csv"}
    )
    documents.append(doc)
print(f"✅ {len(documents)} documents créés")

# ==========================================
# 3. TEXT CHUNKING (500 / 75)
# ==========================================
print("🔪 Découpage en chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       
    chunk_overlap=75,     
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks générés")

# ==========================================
# 4. VECTORIZATION & LOCAL STORAGE (BATCH MODE)
# ==========================================
print("🧠 Génération des embeddings avec mxbai...")
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url=OLLAMA_BASE_URL
)

# ==== CRÉATION DU CLIENT CHROMA (LA PARTIE QUI MANQUAIT) ====
print("💾 Création du client Chroma...")
vector_store = Chroma(
    persist_directory="/app/chroma_db",
    embedding_function=embeddings
)

# ==== TRAITEMENT PAR LOTS DE 50 ====
BATCH_SIZE = 50
total = len(chunks)
print(f"💾 Sauvegarde dans Chroma en lots de {BATCH_SIZE}...")

for i in range(0, total, BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    print(f"   📦 Lot {i//BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} chunks)...")
    try:
        vector_store.add_documents(batch)
    except Exception as e:
        print(f"   ❌ Erreur sur le lot {i//BATCH_SIZE + 1}: {e}")
        raise
    # Petit délai pour ne pas saturer Ollama
    time.sleep(0.5)

print("✅ Base vectorielle créée avec succès dans /app/chroma_db")

# ==========================================
# 5. RETRIEVER (Optionnel)
# ==========================================
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
print("🔍 Retriever configuré avec k=5")