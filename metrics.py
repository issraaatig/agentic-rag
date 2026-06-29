#from bert_score import score as bert_score_func
import re 
def get_judge_score(llm, prompt_text):
    """
    Utilise le LLM comme juge et retourne un score float.
    """

    try:
        response = llm.invoke(prompt_text)
        print(response)
        text = str(response)

        print("\n===== RAW JUDGE OUTPUT =====")
        print(text)
        print("============================\n")

        match = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)


        if not match:
            return 0.0
    
        score = float(match.group(1))

        
        return score


    except Exception as e:
        print(f"Judge Error: {e}")
        return 0.0


def calculate_lexical_f1(reference_text, candidate_text):
    """
    Calcule le score F1 lexical.
    """

    ref_tokens = re.findall(r"\w+", reference_text.lower())

    cand_tokens = re.findall(r"\w+", candidate_text.lower())

    if not ref_tokens or not cand_tokens:
        return 0.0

    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)

    intersection = ref_set.intersection(cand_set)

    if not intersection:
        return 0.0

    precision = len(intersection) / len(cand_set)
    recall = len(intersection) / len(ref_set)

    return 2 * (precision * recall) / (precision + recall)


def calculate_bert_score(reference_text, candidate_text):
    return 0.0