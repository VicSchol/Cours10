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

# Initialisation du moteur de recherche (le même que dans MistralChat.py)
v_manager = VectorStoreManager()

# Initialisation des modèles pour l'évaluateur Ragas
mistral_llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-small-latest", # Recommandé pour l'évaluation
    temperature=0
)
mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)

# Le prompt système utilisé dans votre App
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert...
{context_str}
QUESTION: {question}"""

# ----------------- PRÉPARATION DES DONNÉES ----------------- #
try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data_loaded = json.load(f)
except Exception as e:
    print(f"Erreur chargement JSON: {e}")
    raise

questions, ground_truths, answers, retrieved_contexts = [], [], [], []

print(f"🚀 Simulation RAG sur {len(data_loaded)} questions...")

for item in data_loaded:
    q = item["question"]
    gt = item["ground_truths"][0] if item["ground_truths"] else ""
    
    # 1. RETRIEVAL : On interroge FAISS (comme dans l'app)
    search_results = v_manager.search(q, k=SEARCH_K)
    
    # Extraire le texte des chunks pour Ragas et pour le prompt
    contexts = [res['text'] for res in search_results]
    context_combined = "\n\n".join(contexts)
    
    # 2. GENERATION : On génère la réponse avec le prompt structuré
    prompt_final = SYSTEM_PROMPT.format(context_str=context_combined, question=q)
    try:
        ans = mistral_llm.invoke(prompt_final).content
    except Exception:
        ans = ""

    questions.append(q)
    ground_truths.append(gt)
    answers.append(ans)
    retrieved_contexts.append(contexts)

# Création du Dataset
evaluation_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": retrieved_contexts,
    "ground_truth": ground_truths
})

# ----------------- ÉVALUATION RAGAS ----------------- #
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

print("\n⚖️ Calcul des scores Ragas...")
results = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=mistral_llm,
    embeddings=mistral_embeddings
)

# Affichage des résultats
df = results.to_pandas()
print("\n=== RÉSULTATS PAR QUESTION ===")
print(df[['question', 'faithfulness', 'answer_relevancy']].to_string())

print("\n=== MOYENNES GLOBALES ===")
print(results)