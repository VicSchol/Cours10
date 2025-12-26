from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any
import numpy as np

# --- Modèles pour l'indexation ---

class DocumentChunk(BaseModel):
    """Schéma Pydantic pour un morceau de document (chunk) avec ses métadonnées de base."""
    content: str = Field(..., description="Le contenu textuel du chunk.")
    source: str = Field(..., description="Le chemin du fichier source.")
    page_number: Optional[int] = Field(default=None, description="Le numéro de page (si applicable).")

class IndexedChunk(DocumentChunk):
    """Schéma pour un chunk indexé, incluant le vecteur d'embedding."""
    id: str = Field(..., description="Identifiant unique du chunk.")
    embedding: np.ndarray = Field(..., description="Le vecteur d'embedding généré.")

    # Autorise numpy.ndarray qui n'est pas un type Python standard
    model_config = ConfigDict(arbitrary_types_allowed=True)

# --- Modèles pour la requête et la validation ---

class RAGQuery(BaseModel):
    """Schéma pour valider la requête entrante."""
    query_text: str = Field(..., description="La requête de l'utilisateur.")

class NBAQueryContext(BaseModel):
    """Contrôle le schéma des entrées du système RAG"""
    question: str
    sql_context: str
    rag_context: str
    has_sql_data: bool

class NBAResponseValidation(BaseModel):
    """Contrôle la cohérence de la sortie"""
    answer: str = Field(..., min_length=10)
    contains_stats: bool = Field(description="Indique si la réponse contient des chiffres/stats")

    @field_validator('answer')
    @classmethod
    def check_hallucination_phrases(cls, v: str) -> str:
        forbidden = ["je n'ai pas d'informations", "données non trouvées"]
        if any(phrase in v.lower() for phrase in forbidden):
            # On peut lever une erreur ou transformer la réponse ici
            pass
        return v