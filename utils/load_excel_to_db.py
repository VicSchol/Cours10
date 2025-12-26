# %%
import pandas as pd
import sqlite3
import os
import re
from pydantic import create_model, ValidationError
from typing import Optional

# --- CONFIGURATION ---
DB_PATH = "vector_db/nba_analytics.db"
FILE_EXCEL = "inputs/regular NBA.xlsx"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def clean_column_name(name: str) -> str:
    """Nettoie les noms pour SQL (ex: 3P% -> n_3P_Pct)"""
    name = str(name).strip().replace('%', '_Pct').replace('+', 'Plus').replace('-', 'Minus')
    name = name.replace('/', '_').replace(' ', '_').replace(':', '')
    if re.match(r'^\d', name): name = "n_" + name
    return name

def get_dynamic_model(df_columns):
    """Crée dynamiquement un modèle Pydantic basé sur les colonnes du Excel."""
    fields = {}
    for col in df_columns:
        if col in ['Player', 'Team']:
            continue
        # On définit que chaque stat est optionnelle et doit être un float
        fields[col] = (Optional[float], None)
    
    # Ajout des champs obligatoires pour la jointure SQL
    fields['player_id'] = (int, ...)
    fields['team_code'] = (str, ...)
    
    return create_model('NBAStatRow', **fields)

def run_full_pipeline():
    print(f"📖 Lecture de {FILE_EXCEL}...")
    try:
        xls = pd.ExcelFile(FILE_EXCEL)
        df_stats_raw = pd.read_excel(xls, sheet_name="Données NBA", header=1)
        df_teams_raw = pd.read_excel(xls, sheet_name="Equipe")
        df_dict_raw = pd.read_excel(xls, sheet_name="Dictionnaire des données")
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}"); return

    # --- 1. PRÉPARATION DU DICTIONNAIRE ---
    df_dict_raw['sql_column'] = df_dict_raw.iloc[:, 0].apply(clean_column_name)

    # --- 2. CRÉATION DU SCHÉMA SQL ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.executescript("""
        DROP TABLE IF EXISTS stats; DROP TABLE IF EXISTS players; 
        DROP TABLE IF EXISTS teams; DROP TABLE IF EXISTS dictionary;

        CREATE TABLE teams (code TEXT PRIMARY KEY, full_name TEXT);
        
        CREATE TABLE players (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT UNIQUE, 
            team_code TEXT,
            FOREIGN KEY(team_code) REFERENCES teams(code)
        );

        CREATE TABLE dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            acronym TEXT, definition TEXT, sql_column TEXT
        );
    """)

    # --- 3. INSERTIONS ---

    # A. Dictionnaire & Teams
    df_dict_raw.columns = ['acronym', 'definition', 'sql_column']
    df_dict_raw.to_sql('dictionary', conn, if_exists='append', index=False)
    df_teams_raw.columns = ['code', 'full_name']
    df_teams_raw.to_sql('teams', conn, if_exists='append', index=False)

    # B. Préparation des Stats
    mapping_stats = {old: clean_column_name(old) for old in df_stats_raw.columns}
    df_stats_raw.rename(columns=mapping_stats, inplace=True)
    
    cols_stats = [c for c in df_stats_raw.columns if c not in ['Player', 'Team']]
    create_stats_query = f"CREATE TABLE stats (player_id INTEGER, team_code TEXT, {', '.join([f'{c} REAL' for c in cols_stats])});"
    cursor.execute(create_stats_query)

    # C. Validation avec Pydantic
    print(f"🛠 Validation Pydantic et structuration...")
    NBAStatModel = get_dynamic_model(df_stats_raw.columns)
    validated_stats_list = []
    
    for _, row in df_stats_raw.iterrows():
        try:
            # Gestion du joueur (Insertion/Récupération ID)
            cursor.execute("INSERT OR IGNORE INTO players (name, team_code) VALUES (?, ?)", 
                           (row['Player'], row['Team']))
            cursor.execute("SELECT id FROM players WHERE name = ?", (row['Player'],))
            p_id = cursor.fetchone()[0]
            
            # Préparation des données pour Pydantic
            raw_data = row.to_dict()
            raw_data['player_id'] = p_id
            raw_data['team_code'] = str(raw_data.pop('Team'))
            raw_data.pop('Player', None)
            
            # Nettoyage des NaN pour Pydantic (convertit NaN en None)
            clean_data = {k: (v if pd.notna(v) else None) for k, v in raw_data.items()}

            # Validation effective
            validated_row = NBAStatModel(**clean_data)
            validated_stats_list.append(validated_row.model_dump())

        except ValidationError as e:
            print(f"⚠️ Erreur de données pour {row.get('Player', 'Inconnu')} : {e.errors()}")
            continue
        except Exception as e:
            print(f"❌ Erreur imprévue : {e}")
            continue

    # D. Insertion finale des stats validées
    if validated_stats_list:
        pd.DataFrame(validated_stats_list).to_sql('stats', conn, if_exists='append', index=False)

    conn.commit()
    conn.close()
    print(f"✅ Base de données synchronisée et validée avec succès.")
    
if __name__ == "__main__":
    run_full_pipeline()