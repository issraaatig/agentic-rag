from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    persist_directory="./chroma_db_fresh",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

query = "What is diabetes?"

docs = retriever.invoke(query)

print("DOCS:", len(docs))

for i, d in enumerate(docs):
    print("\n--- DOC", i+1, "---")
    print(d.page_content[:200])