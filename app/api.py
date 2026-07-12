"""
FastAPI bridge for the Agentic RAG frontend.

Ce fichier vit désormais DANS app/ (à côté de main.py, config.py, etc.),
donc l'import est relatif au package courant.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Import depuis le même package (app/) -----------------------------
from .main import get_answer_fast  # main.py est dans le même dossier app/

app = FastAPI(title="Agentic RAG API")

# Autorise le frontend (servi depuis une autre origine/port) à appeler cette API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # à resserrer à l'URL exacte du frontend en production
    allow_methods=["POST"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.post("/chat", response_model=Answer)
def chat(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = get_answer_fast(query.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")
    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "API is running"}