import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Optional
from mistralai import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from .schemas import DocumentChunk, IndexedChunk
from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)

class VectorStoreManager:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[IndexedChunk] = []
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index chargé : {len(self.document_chunks)} chunks.")
            except Exception as e:
                logging.error(f"Erreur chargement index : {e}")

    def _split_documents(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        all_chunks = []
        for doc in documents:
            texts = splitter.split_text(doc.content)
            for text in texts:
                all_chunks.append(DocumentChunk(
                    content=text, 
                    source=doc.source, 
                    page_number=doc.page_number
                ))
        return all_chunks

    def build_index(self, documents: List[DocumentChunk]):
        """Pipeline complet : Split -> Embed -> Index -> Save."""
        raw_chunks = self._split_documents(documents)
        
        # Génération des embeddings par batch
        all_embeddings = []
        for i in range(0, len(raw_chunks), EMBEDDING_BATCH_SIZE):
            batch = raw_chunks[i : i + EMBEDDING_BATCH_SIZE]
            res = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[c.content for c in batch]
            )
            all_embeddings.extend([d.embedding for d in res.data])

        embeddings_np = np.array(all_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)

        # Création des objets IndexedChunk validés
        self.document_chunks = [
            IndexedChunk(
                **raw_chunks[i].model_dump(),
                id=f"chunk_{i}",
                embedding=embeddings_np[i]
            ) for i in range(len(raw_chunks))
        ]

        # FAISS Index
        self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        
        # Sauvegarde
        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
            pickle.dump(self.document_chunks, f)

    def search(self, query: str, k: int = 4) -> List[IndexedChunk]:
        if not self.index: return []
        
        res = self.mistral_client.embeddings.create(model=EMBEDDING_MODEL, inputs=[query])
        query_vec = np.array([res.data[0].embedding]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec, k)
        return [self.document_chunks[i] for i in indices[0] if i != -1]