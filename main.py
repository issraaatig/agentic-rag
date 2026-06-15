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
from pydantic import BaseModel, Field
from typing import List, Literal
import json
import re 

import config
import prompts 
import metrics


# =========================
# ROUTER SCHEMA
# =========================
class QueryRoute(BaseModel):
    complexity: Literal["SIMPLE", "COMPLEX"]
    rewritten_queries: List[str]


router_prompt = PromptTemplate.from_template(prompts.ROUTER_TEMPLATE)
router_chain = router_prompt | config.llm | StrOutputParser()


# =========================
# RRF
# =========================
def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content

            if doc_str not in fused_scores:
                fused_scores[doc_str] = (doc, 0.0)

            previous_score = fused_scores[doc_str][1]
            fused_scores[doc_str] = (
                doc,
                previous_score + 1.0 / (rank + k)
            )

    reranked_results = [
        doc for doc, score in sorted(
            fused_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
    ]
    return reranked_results


# =========================
# PROMPT
# =========================
PROMPT_CONFIG = PromptTemplate(
    template=prompts.ICDF_TEMPLATE,
    input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_input(inputs):
    context = inputs["context"]
    question = inputs["question"]

    return {
        "context": context,
        "question": str(question)
    }


rag_chain = (
    build_rag_input
    | PROMPT_CONFIG
    | config.llm
    | StrOutputParser()
)


# =========================

# SAFE RETRIEVER (IMPORTANT FIX)
# =========================
def safe_retrieve(query: str):
    return config.retriever.get_relevant_documents(str(query))



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
            print("🔍 Analyzing & Evaluating...")

            start_perf = time.time()
            ttft = None

            # =========================
            # ROUTER
            # =========================
            router_output = router_chain.invoke({"question": query})

            cleaned = re.sub(r"```json|```", "", router_output).strip()

            try:
                route_result = json.loads(cleaned)
            except:
                print("⚠️ Router JSON error:", router_output)
                route_result = {
                    "complexity": "SIMPLE",
                    "rewritten_queries": [query]
                }

            is_complex = route_result["complexity"] == "COMPLEX"
            queries_to_run = route_result["rewritten_queries"]

            print(f"👁️ Complexity: {route_result['complexity']}")

            # =========================
            # RETRIEVAL
            # =========================
            if is_complex:
                print("🔄 Multi-query retrieval...")

                all_results = [
                    safe_retrieve(str(q))
                    for q in queries_to_run
                ]

                docs = reciprocal_rank_fusion(all_results)[:5]

            else:
                print("⚡ Simple retrieval...")
                docs = config.retriever.invoke(query["question"]) if isinstance(query, dict) else config.retriever.invoke(query)

            context_combined = "\n".join(d.page_content for d in docs)

            # =========================
            # GENERATION (FIX IMPORTANT)
            # =========================
            print("✍️ Generating...")

            final_answer = ""

            for chunk in rag_chain.stream({
                "context": context_combined,
                "question": str(query)
            }):
                if ttft is None:
                    ttft = time.time() - start_perf
                final_answer += chunk

            total_duration = time.time() - start_perf

            # =========================
            # METRICS
            # =========================
            print("\n========== RETRIEVED DOCS ==========")
            print(f"Docs: {len(docs)}")

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
            # OUTPUT
            # =========================
            print("\n" + "="*60)
            print(f"STRATEGY: {'COMPLEX' if is_complex else 'SIMPLE'}")
            print(f"\nRESPONSE:\n{final_answer}")
            print("-"*60)

            print(f"TTFT: {ttft:.3f}s")
            print(f"Time: {total_duration:.3f}s")

            print(f"Context Precision: {context_precision:.2f}")
            print(f"Context Recall: {context_recall:.2f}")
            print(f"Faithfulness: {faith_score:.2f}")
            print(f"Answer Relevance: {answer_relevance:.2f}")
            print(f"BERT-F1: {bert_f1:.3f}")
            print(f"Lexical-F1: {lexical_f1:.3f}")

            print("="*60)

        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    run_interaction_loop()