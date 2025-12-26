import streamlit as st
import os
import logging
from dotenv import load_dotenv

# --- FIX OPENMP (Indispensable sur Windows pour FAISS) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- NOUVEAUX IMPORTS LANGCHAIN & MISTRAL V1 ---
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez votre structure de dossiers.")
    st.stop()

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration du modèle LangChain ---
if not MISTRAL_API_KEY:
    st.error("Clé API Mistral non trouvée dans le fichier .env.")
    st.stop()

# Initialisation du LLM via le wrapper LangChain
llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model=MODEL_NAME,
    temperature=0.1
)

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource 
def get_vector_store_manager():
    try:
        manager = VectorStoreManager()
        if manager.index is None:
            return None
        return manager
    except Exception as e:
        logging.error(f"Erreur chargement VectorStoreManager: {e}")
        return None

vector_store_manager = get_vector_store_manager()

# --- Prompt Système pour RAG ---
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue NBA.
Ta mission est de répondre aux questions en te basant sur les documents fournis.

CONTEXTE FOURNI :
{context_str}

QUESTION DU FAN :
{question}

RÉPONSE DE L'ANALYSTE NBA :"""

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Propulsé par LangChain & Mistral")



# Initialisation de l'historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content=f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions !")
    ]

# Affichage des messages de l'historique
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# Zone de saisie utilisateur
if prompt := st.chat_input("Posez votre question..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Vérifier le Vector Store
    if vector_store_manager is None:
        st.error("Base de connaissances indisponible.")
        st.stop()

    # 3. Logique RAG (Récupération et Génération)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status = message_placeholder.info("🔍 Recherche dans la base documentaire...")

        try:
            # Recherche de contexte
            search_results = vector_store_manager.search(prompt, k=SEARCH_K)
            
            if search_results:
                context_str = "\n\n".join([
                    f"Source: {res['metadata'].get('filename', 'Doc')} | Extrait: {res['text']}" 
                    for res in search_results
                ])
                status.info("✍️ Analyse des documents et rédaction...")
            else:
                context_str = "Aucune information trouvée dans les documents."
                status.warning("⚠️ Pas de documents trouvés, réponse basée sur mes connaissances générales.")

            # Construction du prompt final
            final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)

            # Appel au LLM via LangChain (.invoke remplace client.chat)
            response = llm.invoke(final_prompt)
            response_content = response.content

            # Affichage de la réponse
            message_placeholder.write(response_content)
            
            # Sauvegarde dans l'historique
            st.session_state.messages.append(AIMessage(content=response_content))

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
            logging.exception("Erreur lors du processus RAG")

st.markdown("---")
st.caption("Mode RAG pur | Données indexées via FAISS")