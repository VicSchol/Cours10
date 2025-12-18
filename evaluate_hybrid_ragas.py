import os
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Imports LangChain / Ragas
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# --- IMPORTS DE VOTRE PROJET ---
from utils.vector_store import VectorStoreManager
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K

load_dotenv()

# ----------------- CONFIG & INIT ----------------- #
FILE_PATH = "eval_dataset.json"

# Initialisation du gestionnaire (vérifiez que votre classe supporte l'hybride)
v_manager = VectorStoreManager()

mistral_llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=0
)
mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)

# Prompt NBA avec placeholder pour le contexte hybride
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un expert en basketball. 
Utilise les informations suivantes (issues de recherches sémantiques et textuelles) pour répondre.
Si l'information n'est pas dans le contexte, dis-le poliment.

CONTEXTE :
{context_str}

QUESTION : {question}
RÉPONSE DE L'ANALYSTE :"""

# ----------------- PRÉPARATION DES DONNÉES ----------------- #
try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data_loaded = json.load(f)
except Exception as e:
    print(f"❌ Erreur chargement JSON: {e}")
    raise

questions, ground_truths, answers, retrieved_contexts = [], [], [], []

print(f"🚀 Évaluation du RAG HYBRIDE sur {len(data_loaded)} questions...")



for item in data_loaded:
    q = item["question"]
    gt = item["ground_truths"][0] if item["ground_truths"] else ""
    
    # --- LOGIQUE HYBRIDE ---
    # On simule ici la récupération hybride : 
    # Si votre v_manager a une méthode dédiée : search_hybrid(q, k=SEARCH_K)
    # Sinon, on utilise la recherche vectorielle standard
    search_results = v_manager.search(q, k=SEARCH_K) 
    
    contexts = [res['text'] for res in search_results]
    context_combined = "\n\n".join(contexts)
    
    # --- GÉNÉRATION ---
    prompt_final = SYSTEM_PROMPT.format(context_str=context_combined, question=q)
    try:
        ans = mistral_llm.invoke(prompt_final).content
    except Exception as e:
        print(f"Erreur API sur la question '{q}': {e}")
        ans = "Erreur de génération."

    questions.append(q)
    ground_truths.append(gt)
    answers.append(ans)
    retrieved_contexts.append(contexts)

# ----------------- CALCUL RAGAS ----------------- #
evaluation_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": retrieved_contexts,
    "ground_truth": ground_truths
})

print("\n⚖️ Lancement des métriques Ragas...")
results = evaluate(
    dataset=evaluation_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=mistral_llm,
    embeddings=mistral_embeddings
)

# ----------------- EXPORT & ANALYSE ----------------- #
df = results.to_pandas()

# Sauvegarde pour analyse ultérieure
df.to_csv("evaluation_hybride_results.csv", index=False)

print("\n" + "="*40)
print("📊 RÉSULTATS MOYENS (HYBRIDE)")
print(f"Fidélité (Faithfulness): {results['faithfulness']:.4f}")
print(f"Pertinence (Relevancy):  {results['answer_relevancy']:.4f}")
print(f"Précision Contexte:     {results['context_precision']:.4f}")
print(f"Rappel Contexte:        {results['context_recall']:.4f}")
print("="*40)