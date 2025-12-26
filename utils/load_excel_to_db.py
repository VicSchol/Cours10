# %%
import pandas as pd
import sqlite3
import os
import re
import unicodedata
from pydantic import BaseModel, create_model, ValidationError
from typing import Optional, Dict, Any

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

def run_full_pipeline():
    print(f"📖 Lecture de {FILE_EXCEL}...")
    try:
        xls = pd.ExcelFile(FILE_EXCEL)
        df_stats_raw = pd.read_excel(xls, sheet_name="Données NBA", header=1)
        df_teams_raw = pd.read_excel(xls, sheet_name="Equipe")
        df_analyse_raw = pd.read_excel(xls, sheet_name="Analyse")
        df_dict_raw = pd.read_excel(xls, sheet_name="Dictionnaire des données")
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}"); return

    # --- 1. PRÉPARATION DU DICTIONNAIRE ---
    df_dict_raw['sql_column'] = df_dict_raw.iloc[:, 0].apply(clean_column_name)

    # --- 2. CRÉATION DU SCHÉMA SQL (CORRIGÉ) ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # On reconstruit proprement
    cursor.executescript("""
        DROP TABLE IF EXISTS stats; DROP TABLE IF EXISTS players; 
        DROP TABLE IF EXISTS teams; DROP TABLE IF EXISTS dictionary;
        DROP TABLE IF EXISTS reports;

        CREATE TABLE teams (
            code TEXT PRIMARY KEY, 
            full_name TEXT
        );
        
        CREATE TABLE players (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT UNIQUE, 
            team_code TEXT,
            FOREIGN KEY(team_code) REFERENCES teams(code)
        );

        CREATE TABLE dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            acronym TEXT, 
            definition TEXT, 
            sql_column TEXT
        );
        
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            category TEXT, 
            content TEXT
        );
    """)

    # --- 3. INSERTIONS ---

    # A. Dictionnaire
    df_dict_raw.columns = ['acronym', 'definition', 'sql_column']
    df_dict_raw.to_sql('dictionary', conn, if_exists='append', index=False)

    # B. Teams
    df_teams_raw.columns = ['code', 'full_name']
    df_teams_raw.to_sql('teams', conn, if_exists='append', index=False)

    # C. Préparation des Stats & Players
    mapping_stats = {old: clean_column_name(old) for old in df_stats_raw.columns}
    df_stats_raw.rename(columns=mapping_stats, inplace=True)
    
    # On définit les colonnes de la table stats dynamiquement pour éviter l'erreur "no column named..."
    # On ajoute player_id et team_code (on garde Team mais renommé en team_code pour la cohérence)
    cols_stats = [c for c in df_stats_raw.columns if c not in ['Player', 'Team']]
    create_stats_query = f"CREATE TABLE stats (player_id INTEGER, team_code TEXT, {', '.join([f'{c} REAL' for c in cols_stats])});"
    cursor.execute(create_stats_query)

    print(f"🛠 Validation et structuration...")
    validated_stats_list = []
    
    for _, row in df_stats_raw.iterrows():
        try:
            # 1. Gestion du joueur
            cursor.execute("INSERT OR IGNORE INTO players (name, team_code) VALUES (?, ?)", 
                           (row['Player'], row['Team']))
            cursor.execute("SELECT id FROM players WHERE name = ?", (row['Player'],))
            p_id = cursor.fetchone()[0]
            
            # 2. Préparation de la ligne de stats
            stat_entry = row.to_dict()
            stat_entry['player_id'] = p_id
            stat_entry['team_code'] = stat_entry.pop('Team') # On renomme Team en team_code
            stat_entry.pop('Player', None) # On enlève le nom (déjà dans table players)
            
            validated_stats_list.append(stat_entry)
        except Exception as e:
            continue

    # D. Insertion finale
    if validated_stats_list:
        pd.DataFrame(validated_stats_list).to_sql('stats', conn, if_exists='append', index=False)

    conn.commit()
    conn.close()
    print(f"✅ Base de données synchronisée avec succès.")
    
if __name__ == "__main__":
    run_full_pipeline()