#%%
import os
import argparse
import logging
import sys # Ajouté pour gérer les arguments système
from typing import Optional, List
from dotenv import load_dotenv

# 1. Chargement de l'environnement et patch Logfire
load_dotenv()
if os.getenv("LOGFIRE_API_KEY") and not os.getenv("LOGFIRE_TOKEN"):
    os.environ["LOGFIRE_TOKEN"] = os.getenv("LOGFIRE_API_KEY")

import logfire
from utils.config import INPUT_DIR
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager
# Import des schémas Pydantic
from utils.schemas import DocumentChunk 

# --- Initialisation Logfire ---
# Suppression de project_name (déprécié)
logfire.configure()

# Configuration logging standard
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@logfire.instrument("RAG Indexing Pipeline")
def run_indexing(input_directory: str, data_url: Optional[str] = None):
    """Exécute le processus complet d'indexation avec validation Pydantic."""
    
    logging.info("--- Démarrage du processus d'indexation ---")
    logfire.info("Configuration", input_dir=input_directory, data_url=data_url)

    # --- Étape 1: Acquisition des données ---
    if data_url:
        with logfire.span("Download and Extract"):
            logging.info(f"Téléchargement : {data_url}")
            if not download_and_extract_zip(data_url, input_directory):
                logging.error("Échec acquisition data. Arrêt.")
                return
        
    # --- Étape 2: Chargement et Validation Pydantic ---
    with logfire.span("Load and Validate Documents"):
        logging.info(f"Analyse du répertoire : {input_directory}")
        
        # Ici, load_and_parse_files doit retourner une List[SourceDocument] (Pydantic)
        documents = load_and_parse_files(input_directory) 

        if not documents:
            logging.warning("Aucun document valide trouvé.")
            return

        logfire.info("Statistiques Documents", count=len(documents))

    # --- Étape 3: Création de l'index Vectoriel ---
    with logfire.span("Build Vector Index"):
        logging.info("Initialisation de FAISS...")
        vector_store = VectorStoreManager() 

        try:
            vector_store.build_index(documents)
            logging.info("Indexation terminée.")
        except Exception as e:
            logging.error(f"Erreur lors de la création de l'index : {e}")
            logfire.error("Indexation failed", error=str(e))
            return
        
    # --- Résumé final ---
    if vector_store.index:
        chunks_indexed = vector_store.index.ntotal
        logging.info(f"--- Succès : {chunks_indexed} chunks indexés ---")
        logfire.info("Indexing Complete", total_chunks=chunks_indexed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'indexation RAG")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR)
    parser.add_argument("--data-url", type=str, default=None)
    
    # CORRECTIF JUPYTER : On utilise parse_known_args() au lieu de parse_args()
    # Cela permet d'ignorer les arguments --f=... injectés par le kernel Jupyter
    args, unknown = parser.parse_known_args()
    
    run_indexing(input_directory=args.input_dir, data_url=args.data_url)