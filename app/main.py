import os
import pandas as pd
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import traceback

# Imports modules perso (faire import config seulement si j'exucute dans terminal )
from app import config
from app import prompts
from app import metrics


# =========================
# PROMPT
# =========================
PROMPT_CONFIG = PromptTemplate(
    template=prompts.ICDF_TEMPLATE,
    input_variables=["context", "question"]
)


# =========================
# RAG CHAIN (LCEL)
# =========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": config.retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | PROMPT_CONFIG
    | config.llm
    | StrOutputParser()
)

# =========================
# FAST PATH (pour l'API web — pas de métriques)
# =========================
def get_answer_fast(query: str) -> str:
    """
    Utilisée par le site web via /chat.
    Ne fait QUE la génération de réponse (comme le terminal),
    sans les 4 appels de juges LLM ni BERTScore — qui sont réservés
    à l'évaluation hors-ligne (process_question / run_interaction_loop).
    """
    try:
        final_answer = ""
        for chunk in rag_chain.stream(query):
            final_answer += chunk
        return final_answer
        print("\n" + "=" * 60)
        print(f"QUESTION : {query}")
        print(f"RÉPONSE  : {final_answer}")
        print("=" * 60 + "\n")
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}"




# =========================
# LOOP
# =========================
def process_question(query: str):
    try:
        print("🔍 Analyzing & Evaluating with LLM Judges...")

        start_perf = time.time()
        ttft = None
        final_answer = ""

        # =========================
        # STREAMING RAG (identique)
        # =========================
        for chunk in rag_chain.stream(query):
            if ttft is None:
                ttft = time.time() - start_perf
            final_answer += chunk

        end_perf = time.time()
        total_duration = end_perf - start_perf

        tps = (
            len(final_answer.split()) / total_duration
            if total_duration > 0 else 0
        )

        # =========================
        # RETRIEVED DOCS
        # =========================
        docs = config.retriever.invoke(query)
        context_combined = "\n".join([d.page_content for d in docs])

        # =========================
        # DEBUG (comme ton run loop)
        # =========================
        print("\n========== DEBUG RETRIEVAL ==========")
        print(f"Retrieved docs: {len(docs)}")

        for i, doc in enumerate(docs):
            print(f"\n--- DOC {i+1} ---")
            print(doc.page_content[:300])

        print("====================================\n")

        # =========================
        # METRICS (IDENTIQUE)
        # =========================
        context_precision = metrics.get_judge_score(
            config.llm,
            prompts.CONTEXT_PRECISION_PROMPT.format(
                question=query,
                context=context_combined
            )
        )

        context_recall = metrics.get_judge_score(
            config.llm,
            prompts.CONTEXT_RECALL_PROMPT.format(
                question=query,
                context=context_combined
            )
        )

        faith_score = metrics.get_judge_score(
            config.llm,
            prompts.FAITHFULNESS_PROMPT.format(
                answer=final_answer,
                context=context_combined
            )
        )

        answer_relevance = metrics.get_judge_score(
            config.llm,
            prompts.ANSWER_RELEVANCE_PROMPT.format(
                question=query,
                answer=final_answer
            )
        )

        lexical_f1 = metrics.calculate_lexical_f1(
            context_combined,
            final_answer
        )

        bert_f1 = metrics.calculate_bert_score(
            context_combined,
            final_answer
        )

        # =========================
        # RETURN (IMPORTANT)
        # =========================
        return {
            "answer": final_answer,
            "metrics": {
                "ttft": ttft,
                "tps": tps,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faith_score,
                "answer_relevance": answer_relevance,
                "bert_f1": bert_f1,
                "lexical_f1": lexical_f1
            },
            "docs": [d.page_content for d in docs]
        }

        print("\n===== CONTEXT =====")

        for i, doc in enumerate(docs):
            print(f"\nDOC {i+1}")
            print(doc.page_content[:500])

        print("====================")
    
        

    except Exception as e:
        traceback.print_exc()
        return {
            "answer": f"Error: {str(e)}",
            "metrics": {},
            "docs": []
        }




def run_interaction_loop():

    print(f"\n🚀 BIO-RAG ADVANCED STARTED (DATASET: MEDQUAD CVD)")
    print(f"📍 Session: {config.SESSION_ID}")

    while True:

        query = input("\n🤔 QUESTION : ")

        if query.lower() in ["quitter", "exit", "stop"]:
            break

        result = process_question(query)

        print("\n" + "=" * 60)
        print(f"RESPONSE :\n{result['answer']}")
        print("-" * 60)
        print(f"⏱️ PERF : TTFT: {result['metrics']['ttft']:.3f}s | TPS: {result['metrics']['tps']:.1f}")
        print(f"📊 RETRIEVAL : Context Precision: {result['metrics']['context_precision']:.2f} | Context Recall: {result['metrics']['context_recall']:.2f}")
        print(f"📊 GENERATION : Faithfulness: {result['metrics']['faithfulness']:.2f} | Answer Relevance: {result['metrics']['answer_relevance']:.2f}")
        print(f"📊 ALIGNMENT : BERT-F1: {result['metrics']['bert_f1']:.3f} | Lexical-F1: {result['metrics']['lexical_f1']:.3f}")
        print("=" * 60)


def rag_pipeline(query: str):
    return process_question(query)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_interaction_loop()