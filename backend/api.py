"""
Minimal FastAPI bridge for the Agentic RAG frontend.

Place this file inside your `agentic-rag/` project (next to main.py) and run:
    pip install fastapi uvicorn --break-system-packages
    uvicorn api:app --reload --port 8000

It does NOT reimplement the RAG pipeline — it only exposes your existing
main.py logic over HTTP so the chat UI (index.html) can call it.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Import your existing RAG pipeline -------------------------------
# Adjust this import to match whatever function/class main.py exposes,
# e.g. `from main import rag_pipeline` or `from main import AgenticRAG`.


from app.main import get_answer_fast  # <-- replace with your actual entry point

app = FastAPI(title="Agentic RAG API")

# Allow the frontend (served from a different origin/port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this to your frontend's origin in production
    allow_methods=["POST"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str

# inclut a
@app.post("/chat", response_model=Answer)
def chat(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = get_answer_fast(query.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")
    return {"answer": answer}

# inclut answer + metrics + docs
from app.main import process_question

@app.post("/chat/debug")
def chat_debug(query: Query):
    result = process_question(query.question)
    return result   



@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/")
def root():
    return {"status": "API is running"}