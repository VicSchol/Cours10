# Assistant RAG avec Mistral

🏀 Assistant RAG & SQL avec Mistral AI
Ce projet implémente un assistant virtuel avancé basé sur le modèle Mistral, capable de répondre à des questions complexes en combinant deux approches : le RAG (Retrieval-Augmented Generation) pour les documents textuels (PDF) et un Agent SQL pour l'analyse de données statistiques structurées (Excel/NBA).

🌟 Fonctionnalités
🔍 Approche Hybride : Routage intelligent des requêtes vers la base vectorielle (FAISS) ou la base relationnelle (SQL).

📊 Analyse de données NBA : Ingestion et interrogation de statistiques complexes via un pipeline Excel-to-SQL.

✅ Validation de Données : Utilisation de Pydantic et Pydantic AI pour garantir l'intégrité des flux d'entrée et de sortie.

📈 Évaluation de Performance : Framework de test intégré avec RAGAS pour calculer la précision et la fidélité des réponses.

🪵 Observabilité : Tracing complet des appels LLM avec Pydantic Logfire.


## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.9+ 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer un environnement virtuel**

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
.
├── MistralChat.py             # Interface utilisateur Streamlit originale
├── MistralChat_optimised.py   # Interface optimisée avec support hybride
├── indexer.py                 # Script d'indexation vectorielle (FAISS)
├── load_excel_to_db.py        # Migration des données Excel vers SQL
├── evaluate_ragas.py          # Évaluation RAG standard
├── evaluate_hybrid_ragas.py   # Évaluation du système hybride (RAG + SQL)
├── eval_dataset.json          # Jeu de tests (questions/réponses métiers)
├── requirements.txt           # Dépendances du projet
│
├── inputs/                    # Sources de données brutes
│   ├── *.pdf                  # Rapports et documents textuels
│   └── regular NBA.xlsx       # Données statistiques structurées
│
├── utils/                     # Logique métier et outils
│   ├── config.py              # Paramètres API et modèles
│   ├── data_loader.py         # Chargement des différents formats
│   ├── sql_tools.py           # Agent de génération de requêtes SQL
│   ├── schemas.py             # Validation des données (Pydantic)
│   └── vector_store.py        # Gestion de l'index vectoriel
│
├── vector_db/                 # Stockage des bases de données
│   ├── faiss_index.idx        # Index vectoriel pour la recherche sémantique
│   ├── document_chunks.pkl    # Chunks de texte sauvegardés
│   └── nba_analytics.db       # Base de données SQLite générée

```

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 4. Lancer la création de la base de données SQL

```bash
python utils/load_excel_to_db.py
```

### 3. Lancer l'application

```bash
streamlit run MistralChat_optimised.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


## Modules principaux

### `utils/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

### `utils/query_classifier.py`

Détermine si une requête nécessite une recherche RAG :
- Analyse des mots-clés
- Classification avec le modèle Mistral
- Détection des questions spécifiques vs générales

### `utils/database.py`

Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques

### inputs/regular NBA.xlsx 

Ce fichier sert de source principale pour le volet analytique (SQL).

### utils/schemas.py 

Contient les classes Pydantic garantissant que les données importées depuis Excel respectent le format attendu avant l'insertion en base.

### evaluate_hybrid_ragas.py 

Ce script calcule des métriques spécifiques pour comparer la précision du système lorsqu'il doit choisir entre chercher dans un document PDF ou interroger la base SQL.

### evaluate_ragas.py 

Ce script calcule des métriques spécifiques pour comparer la précision du système standard.

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation