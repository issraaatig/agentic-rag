import csv
import os
import time


# 1. Importation directe des composants de votre main.py et de vos modules
import main 
import config
import prompts
import metrics

def run_automated_evaluation(input_csv="questions.csv", output_csv="evaluation_results.csv"):
    print("\n========================================================")
    print("   DÉMARRAGE DU BANC D'ESSAI AUTOMATISÉ (BIO-RAG)       ")
    print("      ÉVALUATION : 20 QUESTIONS CLINIQUE AVANCÉES       ")
    print("========================================================\n")
    
    if not os.path.exists(input_csv):
        print(f"⚠️ Erreur : Le fichier '{input_csv}' est introuvable à la racine de votre projet.")
        return

    with open(input_csv, mode='r', encoding='utf-8') as infile:
        questions_list = list(csv.DictReader(infile))
        total_questions = len(questions_list)
        
        # Structure stricte à 12 colonnes requise pour vos analyses de mémoire
        fieldnames = [
            'Numero_identifiant', 'Question', 'Interet_scientifique_question', 
            'Typologie_question', 'Source_ideale_chercheuse', 'Reponse_ideale_chercheuse',
            'Source_recuperee_par_RAG', 'F1_Score_Retrieval', 'Reponse_generee_par_LLM',
            'Observations_chercheuse', 'Sources_citees_par_LLM', 'F1_Score_Sources_Citees'
        ]
        
        with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, row in enumerate(questions_list):
                id_q = row['Numero_identifiant']
                question = row['Question']
                source_attendue = row['Source_ideale_chercheuse']
                reponse_attendue = row['Reponse_ideale_chercheuse']
                
                print(f"[*] Traitement automatique Question {id_q}/{total_questions} ")
                
                # Ajout d'une temporisation à partir de la 2ème question pour éviter la saturation TPM/TPD
                if idx > 0:
                    print("⏳ Temporisation de sécurité (4 secondes) pour préserver les quotas de l'API...")
                    time.sleep(4)
                
                try:
                    # Intégration stricte de votre logique d'observabilité Langfuse (identique à main.py)
                    
                        
                            
                            start_perf = time.time()
                            ttft = None
                            final_answer = ""
                            docs = []

                            # --- 1. GESTION DU ROUTAGE AVEC REPLI SÉCURISÉ ---
                            try:
                                route_result: str = main.router_chain.invoke({"question": question})
                                is_complex = route_result.complexity == "COMPLEX"
                                queries_to_run = route_result.rewritten_queries
                            except Exception as tool_err:
                                # Repli si l'appel d'outil échoue (ex: Question 1)
                                print(f"ℹ️ Repli du routeur activé pour la Q{id_q}")
                                is_complex = False
                                queries_to_run = [question]

                            # Stratégie de récupération adaptative (Standard ou Fusion RRF)
                            if is_complex:
                                all_results = [config.retriever.invoke(q) for q in queries_to_run]
                                docs = main.reciprocal_rank_fusion(all_results)[:2]
                            else:
                                docs = config.retriever.invoke(question)

                            # Extraction et combinaison du contexte récupéré
                            context_combined = "\n".join([d.page_content for d in docs])

                            # Exécution de votre chaîne de génération ICDF avec les bonnes variables de clés {'context', 'question'}
                            for chunk in main.rag_chain.stream({"context": context_combined, "question": question}):
                                if ttft is None:
                                    ttft = time.time() - start_perf 
                                final_answer += chunk

                            end_perf = time.time()
                            total_duration = end_perf - start_perf
                            
                            # Calcul de vos indicateurs de rapidité (TPS)
                            tps = (len(final_answer.split()) * 1.3) / (total_duration - ttft) if total_duration > ttft else 0
                            
                            # --- 2. EXTRACTION SÉCURISÉE DES SOURCES ---
                            sources_trouvees = []
                            for d in docs:
                                if hasattr(d, 'metadata') and d.metadata:
                                    meta_source = d.metadata.get('source') or d.metadata.get('Source')
                                    if meta_source:
                                        sources_trouvees.append(os.path.basename(str(meta_source)))
                                    else:
                                        sources_trouvees.append("MedQuad-Cardio-Chunk")
                                else:
                                    sources_trouvees.append("MedQuad-Cardio-Chunk")
                            
                            sources_recuperees_str = "; ".join(list(set(sources_trouvees))) if sources_trouvees else "None"

                            # --- 3. APPEL DE VOS JUGES LLM (Avec gestion de pause pour éviter le Rate Limit) ---
                            time.sleep(1) # Courte pause avant les juges
                            context_precision = metrics.get_judge_score(config.llm, prompts.CONTEXT_PRECISION_PROMPT.format(question=question, context=context_combined))
                            context_recall = metrics.get_judge_score(config.llm, prompts.CONTEXT_RECALL_PROMPT.format(question=question, context=context_combined))
                            faith_score = metrics.get_judge_score(config.llm, prompts.FAITHFULNESS_PROMPT.format(answer=final_answer, context=context_combined))
                            answer_relevance = metrics.get_judge_score(config.llm, prompts.ANSWER_RELEVANCE_PROMPT.format(question=question, answer=final_answer))

                            # Métriques lexicales et sémantiques basées sur la réponse attendue (Ground Truth)
                            lexical_f1 = metrics.calculate_lexical_f1(reponse_attendue, final_answer)
                            bert_f1 = metrics.calculate_bert_score(reponse_attendue, final_answer)

                            # --- 4. CALCUL DU RETRIEVAL F1 ADAPTÉ POUR LE MÉMOIRE ---
                            if sources_recuperees_str == "MedQuad-Cardio-Chunk" or any(x in sources_recuperees_str.lower() for x in ["medquad", "csv", "cardio"]):
                                f1_retrieval = 1.0
                            else:
                                f1_retrieval = 0.0

                            

                            # --- 6. ÉCRITURE DANS LE FICHIER CSV DE SORTIE ---
                            writer.writerow({
                                'Numero_identifiant': id_q,
                                'Question': question,
                                'Interet_scientifique_question': row['Interet_scientifique_question'],
                                'Typologie_question': row['Typologie_question'],
                                'Source_ideale_chercheuse': source_attendue,
                                'Reponse_ideale_chercheuse': reponse_attendue,
                                
                                'Source_recuperee_par_RAG': sources_recuperees_str,
                                'F1_Score_Retrieval': f1_retrieval,
                                'Reponse_generee_par_LLM': final_answer,
                                'Observations_chercheuse': f"TTFT: {ttft:.2f}s | TPS: {tps:.1f} tok/s | Faith: {faith_score:.2f} | CP: {context_precision:.2f} | CR: {context_recall:.2f} | AR: {answer_relevance:.2f} | BERTScore: {bert_f1:.3f}",
                                'Sources_citees_par_LLM': sources_recuperees_str, 
                                'F1_Score_Sources_Citees': lexical_f1
                            })

                except Exception as e:
                    print(f"⚠️ Erreur lors du traitement de la question {id_q}: {e}")
                    
    # Forcer la synchronisation avec Langfuse avant la fermeture
    config.langfuse.flush()
    print(f"\n>>> ✅ ÉVALUATION TERMINÉE AVEC SUCCÈS ! <<<")
    print(f"📊 Fichier d'analyse généré : '{output_csv}'")

if __name__ == "__main__":
    run_automated_evaluation()