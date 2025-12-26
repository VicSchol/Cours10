import streamlit as st
import os
import faiss
import logging
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- FIX OPENMP & FAISS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
faiss.omp_set_num_threads(1)

# --- CHARGEMENT ENV ---
load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- IMPORTS PROJET ---
try:
    from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K, APP_TITLE
    from utils.vector_store import VectorStoreManager
    from utils.sql_tools import get_nba_sql_tool 
except ImportError as e:
    st.error(f"Erreur d'importation : {e}")
    st.stop()

# --- INITIALISATION LLM ---
llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model=MODEL_NAME,
    temperature=0.3,  # style plus naturel
    timeout=60
)

# --- CHARGEMENT RESSOURCES ---
@st.cache_resource
def load_all_resources(_llm):
    try:
        manager = VectorStoreManager()
        nba_sql_agent = get_nba_sql_tool()  # retourne la fonction run_player_stats
        return manager, nba_sql_agent
    except Exception as e:
        st.error(f"Erreur ressources : {e}")
        return None, None

vector_store_manager, nba_sql_agent = load_all_resources(llm)

# --- LOGIQUE MÉTIER ---
def doit_utiliser_sql(query: str) -> bool:
    keywords = [
        'pts', 'points', 'moyenne', 'classement',
        'stat', '%', 'rebond', 'meilleur', 'joueur'
    ]
    return any(word in query.lower() for word in keywords)

# --- PROMPT SYSTÈME ---
SYSTEM_PROMPT_HYBRIDE = """
Tu es **NBA Analyst AI**, un analyste NBA expérimenté et pédagogue.

🎯 Objectif :
- Répondre clairement à la question
- Utiliser les statistiques si elles sont pertinentes
- Appuyer l’analyse avec le contexte documentaire si utile
- Fournir une réponse fluide et agréable à lire

🗣️ Style :
- Ton naturel et conversationnel
- Phrases claires et dynamiques
- Explique les chiffres avec des mots simples
- Évite le jargon technique
- Ne mentionne jamais SQL, base de données, RAG ou documents

📊 Statistiques disponibles :
{sql_context}

📚 Contexte NBA :
{rag_context}

❓ Question :
{question}

👉 Réponse :
"""

# --- INTERFACE STREAMLIT ---
st.title(APP_TITLE)
st.caption("Assistant Hybride NBA (SQL + RAG) – Mistral AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Input utilisateur
if prompt := st.chat_input("Posez votre question sur la NBA..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        sql_ctx = "Aucune statistique pertinente trouvée."
        rag_ctx = "Aucun contexte documentaire pertinent trouvé."
        
        with st.status("🚀 Analyse en cours...", expanded=True) as status_box:
            
            # 1️⃣ SQL
            if doit_utiliser_sql(prompt):
                st.write("📊 Analyse des statistiques NBA...")
                try:
                    sql_ctx = nba_sql_agent(prompt)
                except Exception as e:
                    sql_ctx = f"Erreur lors de l'analyse des stats : {e}"

            # 2️⃣ RAG
            st.write("🔍 Recherche de contexte NBA...")
            try:
                search_results = vector_store_manager.search(prompt, k=SEARCH_K)
                rag_ctx = "\n".join([doc["text"] for doc in search_results])
            except Exception:
                rag_ctx = rag_ctx
            
            status_box.update(label="✅ Analyse terminée", state="complete")

        # 3️⃣ Génération finale
        system_message = SystemMessage(
            content=SYSTEM_PROMPT_HYBRIDE.format(
                sql_context=sql_ctx,
                rag_context=rag_ctx,
                question=prompt
            )
        )

        messages = [
            system_message,
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        answer = response.content

        st.write(answer)
        st.session_state.messages.append(AIMessage(content=answer))
