# indexer.py
import os, sys
import argparse
import logging
from typing import Optional



from utils.config import INPUT_DIR
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager
from dotenv import load_dotenv
# Assurez-vous que Logfire utilise le nom de variable attendu
# Si vous avez LOGFIRE_API_KEY dans le .env, l'étape ci-dessous est cruciale
if os.getenv("LOGFIRE_API_KEY") and not os.getenv("LOGFIRE_TOKEN"):
    os.environ["LOGFIRE_TOKEN"] = os.getenv("LOGFIRE_API_KEY")
    
load_dotenv()

# Import de Logfire
import logfire 
# --- Initialisation de Logfire ---
# Logfire utilise la clé LOGFIRE_API_KEY dans votre environnement .env
logfire.configure(project_name="Assistant-RAG-Mistral-Indexer")
# --- Fin Initialisation Logfire ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utiliser le décorateur logfire.instrument pour tracer la fonction principale
@logfire.instrument("RAG Indexing Pipeline")
def run_indexing(input_directory: str, data_url: Optional[str] = None):
    """Exécute le processus complet d'indexation."""
    logging.info("--- Démarrage du processus d'indexation ---")
    
    # Enregistrer des attributs de trace (Logfire)
    logfire.info("Configuration de l'indexation", input_dir=input_directory, data_url=data_url)

    # # --- Étape 1: Téléchargement et Extraction (Optionnel) ---
    if data_url:
        with logfire.span("Download and Extract Data"): # Trace cette étape
            logging.info(f"Tentative de téléchargement depuis l'URL: {data_url}")
            success = download_and_extract_zip(data_url, input_directory)
            if not success:
                logging.error("Échec du téléchargement ou de l'extraction. Arrêt.")
                return
        
    # --- Étape 2: Chargement et Parsing des Fichiers (Incluant Validation Pydantic) ---
    with logfire.span("Load and Validate Documents"): # Trace cette étape
        logging.info(f"Chargement et parsing des fichiers depuis: {input_directory}")
        # Note: load_and_parse_files DOIT maintenant valider la sortie avec SourceDocument Pydantic
        documents = load_and_parse_files(input_directory) 

        if not documents:
            logging.warning("Aucun document n'a été chargé ou parsé.")
            return

        # Enregistrer le nombre de documents chargés
        logfire.info("Documents chargés", count=len(documents))
        
    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
    with logfire.span("Build Vector Index"): # Trace cette étape
        logging.info("Initialisation du gestionnaire de Vector Store...")
        vector_store = VectorStoreManager() 

        logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
        # Cette méthode (build_index) DOIT maintenant utiliser Pydantic pour valider 
        # les chunks (DocumentChunk) et les chunks indexés (IndexedChunk).
        vector_store.build_index(documents) 
        
    # --- Fin du Processus ---
    logging.info("--- Processus d'indexation terminé avec succès ---")
    if vector_store.index:
        logfire.info("Index FAISS créé", chunks_count=vector_store.index.ntotal)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant les fichiers sources (par défaut: {INPUT_DIR})"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        # default=INPUT_DATA_URL, # Décommentez pour utiliser la valeur du .env par défaut
        default=None,
        help="URL optionnelle pour télécharger et extraire un fichier inputs.zip"
    )
    args = parser.parse_args()

    # Vérifier si l'URL est passée en argument, sinon prendre celle du .env (si définie)
    # final_data_url = args.data_url if args.data_url is not None else INPUT_DATA_URL
    # Simplification: on utilise seulement l'argument --data-url pour l'instant
    final_data_url = args.data_url

    run_indexing(input_directory=args.input_dir, data_url=final_data_url)