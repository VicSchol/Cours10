import os
import sys
import logging
from typing import List

# 1️⃣ CONFIGURATION DU CHEMIN SYSTÈME
# Ajoute le dossier parent (la racine) pour permettre l'import du module 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
import uvicorn
from dotenv import load_dotenv

# Imports des modules locaux (Assurez-vous que les fichiers existent dans /utils)
from utils.vector_store import VectorStoreManager
from utils.sql_tools import get_nba_sql_tool
from utils.schemas import RAGQuery

# LangChain & Mistral
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

# 2️⃣ INITIALISATION
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NBA_API")

app = FastAPI(
    title="NBA Analyst API",
    description="API REST exposant le système hybride SQL + RAG",
    version="1.1.0"
)

# Chargement des ressources lourdes en mémoire
try:
    vsm = VectorStoreManager()
    nba_sql_agent = get_nba_sql_tool()
    
    llm = ChatMistralAI(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-small-latest",
        temperature=0.3
    )
    logger.info("✅ Ressources initialisées avec succès.")
except Exception as e:
    logger.error(f"❌ Erreur lors de l'initialisation : {e}")
    raise

# 3️⃣ LOGIQUE MÉTIER
def doit_utiliser_sql(query: str) -> bool:
    """Détermine si la question nécessite une requête SQL."""
    keywords = ['pts', 'points', 'moyenne', 'classement', 'stat', '%', 'rebond', 'meilleur', 'joueur', 'score']
    return any(word in query.lower() for word in keywords)

SYSTEM_PROMPT_HYBRIDE = """
Tu es **NBA Analyst AI**, un analyste NBA expérimenté et pédagogue.

🎯 Objectif :
- Répondre clairement à la question.
- Utiliser les statistiques si elles sont pertinentes.
- Appuyer l’analyse avec le contexte documentaire si utile.

📊 Statistiques (SQL) :
{sql_context}

📚 Documents (RAG) :
{rag_context}

N'oublie pas : Sois dynamique et n'évoque jamais tes outils techniques (SQL, FAISS, etc.).
"""



# 4️⃣ ENDPOINTS API

@app.get("/")
async def health_check():
    return {"status": "online", "model": "Mistral-Small"}

@app.post("/api/v1/query")
async def chat_nba(request: RAGQuery):
    """
    Reçoit une question, interroge SQL + FAISS, et renvoie une réponse structurée.
    """
    prompt_user = request.query_text
    logger.info(f"Question reçue : {prompt_user}")

    try:
        # A. Récupération SQL
        sql_ctx = "Aucune statistique trouvée."
        if doit_utiliser_sql(prompt_user):
            res_sql = nba_sql_agent(prompt_user)
            if res_sql:
                sql_ctx = res_sql

        # B. Récupération RAG
        # Correction : On accède à .content de l'objet IndexedChunk
        search_results = vsm.search(prompt_user, k=3)
        rag_ctx = "\n".join([doc.content for doc in search_results]) if search_results else "Aucun document trouvé."

        # C. Génération Mistral AI
        messages = [
            SystemMessage(content=SYSTEM_PROMPT_HYBRIDE.format(
                sql_context=sql_ctx,
                rag_context=rag_ctx
            )),
            HumanMessage(content=prompt_user)
        ]
        
        response = llm.invoke(messages)
        answer = response.content

        # D. Retour JSON
        return {
            "answer": answer,
            "metadata": {
                "sql_used": doit_utiliser_sql(prompt_user),
                "sources": [doc.source for doc in search_results] if search_results else [],
                "page_numbers": [doc.page_number for doc in search_results] if search_results else []
            }
        }

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête : {e}")
        raise HTTPException(status_code=500, detail="Une erreur interne est survenue.")

# 5️⃣ LANCEMENT
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    "http://localhost:8000/docs"