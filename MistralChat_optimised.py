import streamlit as st
import os
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
    # Import de votre nouveau Tool SQL
    from utils.sql_tool import get_nba_sql_tool 
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers.")
    st.stop()

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
load_dotenv()
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée.")
    st.stop()

client = MistralClient(api_key=api_key)

# --- Chargement des Ressources (mis en cache) ---
@st.cache_resource 
def load_all_resources():
    logging.info("Chargement des ressources (VectorStore + SQL Tool)...")
    try:
        manager = VectorStoreManager()
        sql_agent = get_nba_sql_tool()
        return manager, sql_agent
    except Exception as e:
        st.error(f"Erreur lors du chargement des ressources : {e}")
        return None, None

vector_store_manager, nba_sql_agent = load_all_resources()

# --- Fonctions de détection d'intention ---
def doit_utiliser_sql(query: str) -> bool:
    """Détecte si la question porte sur des données chiffrées/Excel."""
    keywords = [
        'pts', 'points', 'moyenne', 'classement', 'meilleur', 'stat', 
        'percent', '%', 'rebond', 'pie', 'efficacite', 'netrtg'
    ]
    return any(word in query.lower() for word in keywords)

# --- Prompt Système pour RAG ---
SYSTEM_PROMPT_RAG = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue NBA.
Réponds à la question en te basant sur le contexte documentaire fourni ci-dessous.

CONTEXTE:
{{context_str}}

QUESTION DU FAN:
{{question}}"""

# --- Initialisation de l'historique ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Je peux analyser vos fichiers PDF (RAG) ou vos statistiques Excel (SQL). Que souhaitez-vous savoir ?"}]

# --- Fonction Génération Mistral (RAG) ---
def generer_reponse_mistral(prompt_messages: list[ChatMessage]) -> str:
    try:
        response = client.chat(model=model, messages=prompt_messages, temperature=0.1)
        return response.choices[0].message.content
    except Exception as e:
        logging.exception("Erreur API Mistral")
        return "Erreur technique lors de la génération RAG."

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant hybride (RAG + SQL) pour {NAME} | Modèle: {model}")

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie
if prompt := st.chat_input(f"Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # --- ROUTAGE : SQL OU RAG ? ---
        if doit_utiliser_sql(prompt):
            # LOGIQUE SQL
            message_placeholder.info("📊 Analyse des statistiques (Base SQL/Excel)...")
            try:
                result = nba_sql_agent.invoke({"input": prompt})
                response_content = result["output"]
            except Exception as e:
                logging.error(f"Erreur SQL : {e}")
                response_content = "Désolé, je n'ai pas pu extraire ces données de la base SQL."
        
        else:
            # LOGIQUE RAG (VOTRE CODE ORIGINAL)
            message_placeholder.info("🔍 Recherche dans les documents d'analyse...")
            if vector_store_manager:
                search_results = vector_store_manager.search(prompt, k=SEARCH_K)
                
                context_str = "\n\n---\n\n".join([
                    f"Source: {res['metadata'].get('source', 'Inconnue')}\nContenu: {res['text']}"
                    for res in search_results
                ])
                
                if not search_results:
                    context_str = "Aucune information trouvée dans les documents."

                final_prompt = SYSTEM_PROMPT_RAG.format(context_str=context_str, question=prompt)
                messages_for_api = [ChatMessage(role="user", content=final_prompt)]
                response_content = generer_reponse_mistral(messages_for_api)
            else:
                response_content = "Le moteur de recherche documentaire n'est pas prêt."

        # Affichage final
        message_placeholder.write(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})


# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # === Début de la logique RAG ===

    # 2. Vérifier si le Vector Store est disponible
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        logging.error("VectorStoreManager non disponible pour la recherche.")
        # On arrête ici car on ne peut pas faire de RAG
        st.stop()

    # 3. Rechercher le contexte dans le Vector Store
    try:
        logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
        logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
        search_results = [] # On continue sans contexte si la recherche échoue

    # 4. Formater le contexte pour le prompt LLM
    context_str = "\n\n---\n\n".join([
        f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
        for res in search_results
    ])

    if not search_results:
        context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
        logging.warning(f"Aucun contexte trouvé pour la query: {prompt}")

    # 5. Construire le prompt final pour l'API Mistral en utilisant le System Prompt RAG
    final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)

    # Créer la liste de messages pour l'API (juste le prompt système/utilisateur combiné)
    messages_for_api = [
        # On pourrait séparer system et user, mais Mistral gère bien un long message user structuré
        ChatMessage(role="user", content=final_prompt_for_llm)
    ]

    # === Fin de la logique RAG ===


    # 6. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant le prompt augmenté
        response_content = generer_reponse(messages_for_api)

        # Affichage de la réponse complète
        message_placeholder.write(response_content)

    # 7. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")