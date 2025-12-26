# %% 1. ENV & CONFIG
import os
import json
import traceback
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# IMPORT DES SCHÉMAS PYDANTIC
from utils.schemas import RAGQuery # Pour valider la question entrante

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FILE_PATH = "eval_dataset.json"

# %% 2. INITIALISATION
try:
    mistral_llm = ChatMistralAI(
        mistral_api_key=MISTRAL_API_KEY,
        model="mistral-small-latest",
        temperature=0
    )
    mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
    
    from utils.vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    if vsm.index is None:
        raise FileNotFoundError("L'index FAISS est introuvable.")
except Exception as e:
    print(f"❌ Erreur initialisation : {e}"); raise

# %% 3. GÉNÉRATION DES DONNÉES AVEC VALIDATION PYDANTIC
# ... (votre code précédent)

for item in data_loaded:
    q_raw = item["question"]
    gt = item["ground_truths"][0] if item["ground_truths"] else ""
    
    try:
        validated_query = RAGQuery(query_text=q_raw)
        q = validated_query.query_text
    except Exception as e:
        print(f"⚠️ Question ignorée : {e}")
        continue

    # Recherche RAG
    search_results = vsm.search(q, k=3) 
    
    # --- CORRECTION ICI ---
    # On utilise .content (Pydantic) au lieu de ["text"] (Dict)
    current_contexts = [res.content for res in search_results if res.content and "NaN" not in res.content]
    
    context_text = "\n".join(current_contexts)
    prompt = f"En vous basant sur le contexte suivant :\n{context_text}\n\nQuestion : {q}\nRéponse précise :"
    
    try:
        ans = mistral_llm.invoke(prompt).content
        
        # Validation simple de la sortie
        if len(ans) < 5:
            ans = "Réponse rejetée : Cohérence insuffisante."
            
    except:
        ans = "Erreur technique de réponse."

    questions.append(q)
    ground_truths.append(gt)
    answers.append(ans)
    retrieved_contexts.append(current_contexts)

# %% 4. ÉVALUATION ET TABLEAU DE SYNTHÈSE


evaluation_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": retrieved_contexts,
    "ground_truth": ground_truths
})

try:
    print("\n⚖️ Analyse RAGAS (Faithfulness & Relevancy)...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=mistral_llm,
        embeddings=mistral_embeddings
    )
    
    results_df = results.to_pandas()

    if 'question' not in results_df.columns:
        results_df.insert(0, 'question', questions)

    # Préparation du tableau final
    metrics_cols = ['faithfulness', 'answer_relevancy']
    df_display = results_df[['question'] + metrics_cols].copy()
    mean_scores = df_display[metrics_cols].mean()

    mean_row = pd.DataFrame({
        'question': ['--- MOYENNE GÉNÉRALE ---'],
        'faithfulness': [mean_scores['faithfulness']],
        'answer_relevancy': [mean_scores['answer_relevancy']]
    })

    final_table = pd.concat([df_display, mean_row], ignore_index=True)

    print("\n" + "="*80)
    print("📊 SYNTHÈSE DES SCORES MÉTIERS (CONTRÔLÉS PAR PYDANTIC)")
    print("="*80)
    print(final_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*80)

    final_table.to_csv("RAGAS_BASE_Model_Validated.csv", index=False)
    print("\n✅ Rapport exporté : RAGAS_BASE_Model_Validated.csv")

except Exception as e:
    print(f"❌ Erreur évaluation : {e}")
    traceback.print_exc()