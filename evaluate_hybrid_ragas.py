# %% 1️⃣ Imports
import os
import json
import traceback
import pandas as pd
import numpy as np
from datasets import Dataset
from dotenv import load_dotenv

# Désactiver les avertissements OpenMP sur Windows pour FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# --- IMPORTS PROJET ---
from utils.vector_store import VectorStoreManager
from utils.sql_tools import get_nba_sql_tool
from utils.schemas import RAGQuery, NBAResponseValidation

# %% 2️⃣ INITIALISATION DES COMPOSANTS
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FILE_PATH = "eval_dataset.json"

try:
    llm = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model="mistral-small-latest", temperature=0)
    embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
    vsm = VectorStoreManager()
    nba_sql_agent = get_nba_sql_tool() 
except Exception as e:
    print(f"❌ Erreur initialisation : {e}")
    raise

# %% 3️⃣ CHARGEMENT DES DONNÉES D'ÉVALUATION
if os.path.exists(FILE_PATH):
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data_loaded = json.load(f)
    print(f"✅ Dataset chargé : {len(data_loaded)} questions.")
else:
    print(f"⚠️ {FILE_PATH} non trouvé. Utilisation d'un échantillon de test.")
    data_loaded = [
        {"question": "Quel est le record de points de Luka Doncic cette saison ?", "ground_truths": ["73 points"]},
        {"question": "Quelles sont les notes du rapport sur les Lakers ?", "ground_truths": ["Défense solide mais manque de banc."]}
    ]

# %% 4️⃣ LOGIQUE DE ROUTAGE
def doit_utiliser_sql(query: str) -> bool:
    keywords = ['pts', 'points', 'moyenne', 'classement', 'stat', '%', 'rebond', 'joueur', 'score', 'équipe', 'précision']
    return any(word in query.lower() for word in keywords)

SYSTEM_PROMPT_HYBRIDE = """Tu es NBA Analyst AI. Réponds en utilisant UNIQUEMENT le contexte fourni.
Si les deux contextes sont présents, fusionne-les intelligemment.

CONTEXTE SQL (Données précises) :
{sql_context}

CONTEXTE RAG (Analyses et rapports) :
{rag_context}

QUESTION : {question}
RÉPONSE :"""

# %% 5️⃣ PIPELINE DE GÉNÉRATION ET VALIDATION
questions, ground_truths, answers, retrieved_contexts = [], [], [], []



for item in data_loaded:
    q_raw = item.get("question", "")
    gt = item.get("ground_truths", [""])[0]
    
    # --- A. Validation Entrée via Pydantic ---
    try:
        query_obj = RAGQuery(query_text=q_raw)
        q = query_obj.query_text
    except Exception as e:
        print(f"⚠️ Requête ignorée (format invalide) : {e}")
        continue

    # --- B. Récupération Contextes ---
    sql_ctx = "Aucune donnée statistique trouvée."
    if doit_utiliser_sql(q):
        try:
            res_sql = nba_sql_agent(q)
            if res_sql and "n'ai pas identifié" not in res_sql:
                sql_ctx = res_sql
        except: pass

    rag_ctx = "Aucun rapport textuel trouvé."
    try:
        # CORRECTION : Accès via .content (IndexedChunk Pydantic)
        search_results = vsm.search(q, k=2)
        if search_results:
            rag_ctx = "\n---\n".join([res.content for res in search_results])
    except Exception as e:
        print(f"❌ Erreur VectorStore : {e}")

    # --- C. Génération LLM ---
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_HYBRIDE.format(
            sql_context=sql_ctx, rag_context=rag_ctx, question=q
        )),
        HumanMessage(content=q)
    ]

    try:
        ans = llm.invoke(messages).content
        # Validation de la sortie (optionnel mais recommandé)
        NBAResponseValidation(answer=ans, contains_stats=any(c.isdigit() for c in ans))
    except Exception as e:
        ans = f"Désolé, je ne peux pas traiter cette demande. (Erreur: {str(e)})"

    questions.append(q)
    ground_truths.append(gt)
    answers.append(ans)
    retrieved_contexts.append([f"SQL: {sql_ctx}", f"RAG: {rag_ctx}"])

# %% 6️⃣ ÉVALUATION RAGAS ET SYNTHÈSE
print("\n⚖️ Lancement de l'évaluation RAGAS...")
try:
    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )

    # Conversion en DataFrame
    df = results.to_pandas()
    
    # --- CORRECTIF : Réinsertion de la colonne question si manquante ---
    if 'question' not in df.columns:
        df.insert(0, 'question', questions)
    # ------------------------------------------------------------------

    # Calcul des moyennes (uniquement sur les colonnes numériques)
    avg_faith = df['faithfulness'].mean()
    avg_relevancy = df['answer_relevancy'].mean()

    print("\n" + "="*80)
    print(f"📊 RÉSULTATS DE L'ANALYSE HYBRIDE")
    print("="*80)
    
    # Affichage propre
    cols_to_show = ['question', 'faithfulness', 'answer_relevancy']
    print(df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    print("-" * 80)
    print(f"MOYENNE GÉNÉRALE -> Faithfulness: {avg_faith:.4f} | Relevancy: {avg_relevancy:.4f}")
    print("="*80)

    # Sauvegarde
    df.to_csv("RAGAS_HYBRID_Model_Validated.csv", index=False)
    print("\n✅ Rapport sauvegardé sous 'evaluation_report.csv'")

except Exception as e:
    print(f"❌ Erreur lors de l'évaluation RAGAS : {e}")
    traceback.print_exc()
# %%
