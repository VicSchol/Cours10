# utils/schemas.py (Version Corrigée)

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import numpy as np

class DocumentChunk(BaseModel):
    """Schéma Pydantic pour un morceau de document (chunk) avec ses métadonnées de base."""
    content: str = Field(..., description="Le contenu textuel du chunk.")
    source: str = Field(..., description="Le chemin du fichier source.")
    page_number: Optional[int] = Field(default=None, description="Le numéro de page (si applicable).")

class IndexedChunk(DocumentChunk):
    """Schéma pour un chunk indexé, incluant le vecteur d'embedding généré par Mistral et un ID unique."""
    id: str = Field(..., description="Identifiant unique du chunk.")
    
    # 💥 CHANGEMENT CRUCIAL : Utilisez np.ndarray au lieu de List[float]
    embedding: np.ndarray = Field(..., description="Le vecteur d'embedding généré par Mistral.")

    # ⚠️ Nécessaire pour accepter np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Si vous utilisez Pydantic v1, utilisez :
    # class Config:
    #     arbitrary_types_allowed = True

class RAGQuery(BaseModel):
    """Schéma pour valider la requête entrante."""
    query_text: str = Field(..., description="La requête de l'utilisateur.")