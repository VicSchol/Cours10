# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Optional
from mistralai import Mistral, models
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logfire
from .schemas import DocumentChunk, IndexedChunk
from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, any]] = []

        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY manquante !")
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        logging.info(f"Découpage de {len(documents)} documents en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{doc['metadata'].get('filename', 'N/A')}' découpé en {len(chunks)} chunks.")
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc_idx}_{i}",
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id_in_doc": i,
                        "start_index": chunk.metadata.get("start_index", -1)
                    }
                })
        logging.info(f"Total de {len(all_chunks)} chunks créés.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=texts_to_embed # 
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except models.MistralError as e:
                logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
                return None
            except Exception as e:
                # Cette erreur devrait maintenant être résolue après la mise à jour du SDK
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
                return None

        if not all_embeddings:
            logging.error("Aucun embedding n'a pu être généré.")
            return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, any]]):
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        with logfire.span("Indexing Steps"):
            raw_chunks_list = self._split_documents_to_chunks(documents)
            if not raw_chunks_list:
                logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
                return

            embeddings_array = self._generate_embeddings(raw_chunks_list)
            if embeddings_array is None or embeddings_array.shape[0] != len(raw_chunks_list):
                logging.error("Problème de génération d'embeddings. Nettoyage de l'index...")
                self.document_chunks = []
                self.index = None
                return

            all_validated_chunks: List[IndexedChunk] = []
            with logfire.span("Pydantic Validation and Embedding Association"):
                for i, raw_chunk_dict in enumerate(raw_chunks_list):
                    try:
                        metadata = raw_chunk_dict.get("metadata", {})
                        
                        # Récupère l'ID généré lors du découpage (dans _split_documents_to_chunks)
                        chunk_id = raw_chunk_dict.get("id")
                        
                        validated_chunk_metadata = DocumentChunk(
                            content=raw_chunk_dict["text"],
                            source=metadata.get("source", metadata.get("filename", "unknown_source")),
                            page_number=metadata.get("page", metadata.get("page_number"))
                        )
                        # 2. Créez l'IndexedChunk en passant TOUS les champs requis
                        final_indexed_chunk = IndexedChunk(
                            **validated_chunk_metadata.model_dump(),
                            # Ces deux champs sont désormais obligatoires et doivent être passés :
                            id=chunk_id, 
                            embedding=embeddings_array[i] 
                        )
                        all_validated_chunks.append(final_indexed_chunk)
                    except Exception as e:
                        logfire.error("Erreur de validation Pydantic pour un chunk",
                                     error=str(e),
                                     chunk_id=raw_chunk_dict.get("id", "N/A"))
                        continue

            if not all_validated_chunks:
                logging.error("Aucun chunk n'a passé la validation Pydantic. Index Faiss non créé.")
                return

            self.document_chunks = [
                {"text": c.content, "metadata": {"source": c.source, "page_number": c.page_number, "id": c.id}}
                for c in all_validated_chunks
            ]

            with logfire.span("FAISS Index Creation"):
                validated_embeddings = np.array([c.embedding for c in all_validated_chunks]).astype('float32')
                dimension = validated_embeddings.shape[1]
                logging.info(f"Création de l'index Faiss avec dimension {dimension}...")
                faiss.normalize_L2(validated_embeddings)
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(validated_embeddings)
                logfire.info("Index FAISS complété", num_vectors=self.index.ntotal)

            self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
            return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text] # <-- REVERT : 'texts' remplacé par 'input'
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)

            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            continue
                        results.append({
                            "score": similarity,
                            "raw_score": raw_score,
                            "text": chunk["text"],
                            "metadata": chunk["metadata"]
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k] if len(results) > k else results

        except models.MistralError as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []