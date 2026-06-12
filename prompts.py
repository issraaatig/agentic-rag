# --- SYSTEM/USER PROMPT (ICDF - ZERO-SHOT CoT) ---
ICDF_TEMPLATE = """
<instructions>
[ROLE & MISSION]
You are an Expert Medical & Cardiology Agent. Your mission is to perform a step-by-step clinical analysis based strictly on the provided documents.

[RULES]
1. Think step by step before providing the final answer.
2. Use ONLY the provided context. If unsure, say "Information not available".
3. Maintain professional medical terminology in English. Match the exact vocabulary from the context.
</instructions>

<context_data>
[MEDICAL CONTEXT - RAG]
{context}
</context_data>

<user_input>
[QUESTION]
{question}
</user_input>

<format_specifications>
[ZERO-SHOT CoT STRUCTURE]
Follow this exact output format:
1. Logical Reasoning : (Detailed step-by-step thinking process using the context)
2. Clinical Analysis : (Final structured answer with bullet points extraction)
3. Confidence Score : (Score from 0 to 100%)

"DISCLAIMER: Academic use only. Consult a professional specialist."
</format_specifications>

[CLINICAL EXECUTIVE SUMMARY]:
Let's think step by step:
"""

# --- EVALUATION PROMPTS (LLM-AS-A-JUDGE - RAGAS TRIAD) ---
FAITHFULNESS_PROMPT = """
<task>
Compare the AI RESPONSE with the RETRIEVED CONTEXT. 
Score 1 if all facts are supported, 0 if any hallucination is found.
Output ONLY a single float between 0 and 1. No other text.
</task>

<data>
AI RESPONSE: {answer}
CONTEXT: {context}
</data>
"""

ANSWER_RELEVANCE_PROMPT = """
<task>
Evaluate the relevance of the AI RESPONSE to the USER QUESTION.
Score 1 if the response directly addresses the question, provides clear information, and contains no fluff.
Score 0 if the response is completely off-topic.
Output ONLY a single float between 0 and 1. No other text.
</task>

<data>
USER QUESTION: {question}
AI RESPONSE: {answer}
</data>
"""

CONTEXT_PRECISION_PROMPT = """
<task>
Analyze if the retrieved context is highly relevant to answer the user question.
Output ONLY a single float between 0 and 1 representing the precision score. No other text.
</task>

<data>
QUESTION: {question}
RETRIEVED CONTEXT: {context}
</data>
"""

CONTEXT_RECALL_PROMPT = """
<task>
Evaluate how much of the information required to answer the question
is present in the retrieved context.

Scoring:

1.0 = all required information is present

0.75 = most information is present

0.50 = some information is present

0.25 = little information is present

0.0 = no useful information

Output ONLY a single number between 0 and 1.
No explanation.
No text.
</task>

<data>
QUESTION: {question}
RETRIEVED CONTEXT: {context}
</data>
"""