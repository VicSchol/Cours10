import os
import json
import traceback
import pandas as pd
from datasets import Dataset
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv

load_dotenv()
# ----------------- CONFIG ----------------- #
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FILE_PATH = "eval_dataset.json"

# ----------------- CHARGEMENT DU JSON ----------------- #
try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data_loaded = json.load(f)
    print(f"✅ Fichier {FILE_PATH} chargé ({len(data_loaded)} questions trouvées).")
except Exception as e:
    print(f"❌ Erreur lors du chargement du fichier JSON : {e}")
    raise

# ----------------- INITIALISATION ----------------- #
try:
    mistral_llm = ChatMistralAI(
        mistral_api_key=MISTRAL_API_KEY,
        model="mistral-small-latest",
        temperature=0.1
    )
    mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
except Exception as e:
    print(f"❌ Erreur initialisation LLM : {e}")
    raise

# ----------------- PRÉPARATION DES DONNÉES ----------------- #
questions = []
ground_truths = []
answers = []
retrieved_contexts = []

print("🚀 Génération des réponses via Mistral...")

for item in data_loaded:
    q = item["question"]
    # Ragas attend une seule string pour ground_truth, on prend le premier élément de la liste
    gt = item["ground_truths"][0] if item["ground_truths"] else ""
    # On garde la liste pour les contextes
    ctx_list = item["contexts"]
    
    # Appel au LLM
    try:
        ans = mistral_llm.invoke(q).content
    except Exception:
        ans = ""

    questions.append(q)
    ground_truths.append(gt)
    answers.append(ans)
    retrieved_contexts.append(ctx_list)

# Création du Dataset Ragas
evaluation_data = {
    "question": questions,
    "answer": answers,
    "contexts": retrieved_contexts,
    "ground_truth": ground_truths
}
evaluation_dataset = Dataset.from_dict(evaluation_data)

# ----------------- ÉVALUATION ----------------- #
# Note: Ces métriques comparent Answer vs Context (Faithfulness) 

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

try:
    print("\n⚖️ Lancement de l'évaluation Ragas...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=mistral_llm,
        embeddings=mistral_embeddings
    )
    
    # 1. On transforme les scores en DataFrame
    scores_df = results.to_pandas()
    
    # 2. On récupère les questions et données d'origine
    inputs_df = evaluation_dataset.to_pandas()
    
    # 3. FUSION : On s'assure d'avoir les questions ET les scores dans le même tableau
    # On concatène horizontalement (axis=1)
    # On drop les colonnes redondantes dans scores_df s'il y en a
    results_df = pd.concat([inputs_df, scores_df.drop(columns=['question', 'contexts', 'answer', 'ground_truth'], errors='ignore')], axis=1)

    # ----------------- AFFICHAGE SÉCURISÉ ----------------- #
    print("\n" + "="*80)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*80)
    
    # On définit les colonnes à afficher en vérifiant qu'elles existent
    actual_metrics = [m.name for m in metrics]
    cols_to_show = ['question'] + [c for c in actual_metrics if c in results_df.columns]
    
    print(results_df[cols_to_show])

    print("\n" + "="*80)
    print("SCORES MOYENS GLOBAUX")
    print("="*80)
    # Moyenne uniquement sur les colonnes numériques de métriques
    print(results_df[actual_metrics].mean())

except Exception as e:
    print(f"❌ Erreur lors de l'évaluation : {e}")
    traceback.print_exc()
