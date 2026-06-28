import os
import pandas as pd
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import traceback

# Imports modules perso
import config
import prompts 
import metrics


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
# LOOP
# =========================
def run_interaction_loop():

    print(f"\n🚀 BIO-RAG ADVANCED STARTED (DATASET: MEDQUAD CVD)")
    print(f"📍 Session: {config.SESSION_ID}")

    while True:

        query = input("\n🤔 QUESTION : ")
        if query.lower() in ["quitter", "exit", "stop"]:
            break

        try:
            print("🔍 Analyzing & Evaluating with LLM Judges...")

            start_perf = time.time()
            ttft = None
            final_answer = ""

            # =========================
            # STREAMING FIXED
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
            # RETRIEVED DOCS (FIX)
            # =========================
            docs = config.retriever.invoke(query)
            context_combined = "\n".join([d.page_content for d in docs])

            # =========================
            # METRICS
            # =========================
            print("\n========== DEBUG RETRIEVAL ==========")
            print(f"Retrieved docs: {len(docs)}")

            for i, doc in enumerate(docs):
                print(f"\n--- DOC {i+1} ---")
                print(doc.page_content[:300])

            print("====================================\n")
            
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

            

            # =========================
            # OUTPUT
            # =========================
            print("\n" + "=" * 60)
            print(f"RESPONSE :\n{final_answer}")
            print("-" * 60)
            print(f"⏱️ PERF : TTFT: {ttft:.3f}s | TPS: {tps:.1f} tok/s")
            print(f"📊 RETRIEVAL : Context Precision: {context_precision:.2f} | Context Recall: {context_recall:.2f}")
            print(f"📊 GENERATION : Faithfulness: {faith_score:.2f} | Answer Relevance: {answer_relevance:.2f}")
            print(f"📊 ALIGNMENT : Lexical-F1: {lexical_f1:.3f}")
            print("=" * 60)

        except Exception as e:
            traceback.print_exc()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_interaction_loop()