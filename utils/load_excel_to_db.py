import pandas as pd
import sqlite3
import os
import re
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
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}"); return

    # --- 1. PRÉPARATION DES DONNÉES ---
    # Nettoyage des colonnes stats
    df_stats_raw.columns = [str(c) for c in df_stats_raw.columns]
    df_stats_raw = df_stats_raw.loc[:, ~df_stats_raw.columns.str.contains('^Unnamed')]
    
    # Mapping des noms de colonnes pour SQL
    mapping_stats = {old: clean_column_name(old) for old in df_stats_raw.columns}
    df_stats_raw.rename(columns=mapping_stats, inplace=True)
    
    # Modèle Pydantic Dynamique pour validation
    fields = {col: (str, ...) if col in ['Player', 'Team'] else (Optional[Any], None) for col in df_stats_raw.columns}
    DynamicStatModel = create_model('DynamicStatModel', **fields)

    # --- 2. CRÉATION DU SCHÉMA SQL ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript("""
        DROP TABLE IF EXISTS stats; DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS players; DROP TABLE IF EXISTS teams;
        DROP TABLE IF EXISTS reports;
        
        CREATE TABLE teams (code TEXT PRIMARY KEY, full_name TEXT);
        CREATE TABLE players (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, team_code TEXT, FOREIGN KEY(team_code) REFERENCES teams(code));
        CREATE TABLE matches (id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE, home_team TEXT, away_team TEXT, score TEXT);
        CREATE TABLE reports (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, content TEXT);
    """)

    # --- 3. INSERTIONS ---

    # A. Insertion TEAMS (Correction du renommage ici)
    df_teams_raw.columns = ['code', 'full_name'] # On aligne sur le schéma SQL
    df_teams_raw.to_sql('teams', conn, if_exists='append', index=False)

    # B. Validation et Insertion PLAYERS & STATS
    print(f"🛠 Validation et structuration de {len(df_stats_raw)} joueurs...")
    validated_stats_list = []
    
    for _, row in df_stats_raw.iterrows():
        try:
            data = row.to_dict()
            valid_row = DynamicStatModel(**data)
            
            # Gestion de la table PLAYERS
            cursor.execute("INSERT OR IGNORE INTO players (name, team_code) VALUES (?, ?)", 
                           (valid_row.Player, valid_row.Team))
            cursor.execute("SELECT id FROM players WHERE name = ?", (valid_row.Player,))
            p_id = cursor.fetchone()[0]
            
            # Préparation de la ligne STATS
            stat_entry = data.copy()
            stat_entry['player_id'] = p_id
            stat_entry.pop('Player', None) # Normalisation : on retire le nom
            validated_stats_list.append(stat_entry)
        except ValidationError: continue

    # Insertion massive dans la table stats
    if validated_stats_list:
        pd.DataFrame(validated_stats_list).to_sql('stats', conn, if_exists='append', index=False)

    # C. Insertion REPORTS (Analyse)
    try:
        df_top15 = df_analyse_raw.iloc[91:107, [0, 1, 7]].dropna()
        for _, r in df_top15.iterrows():
            cursor.execute("INSERT INTO reports (category, content) VALUES (?, ?)", 
                           ("Top 15", f"Joueur: {r.iloc[0]}, Points: {r.iloc[1]}, PIE: {r.iloc[2]}"))
    except: print("⚠️ Section Analyse ignorée (format non reconnu).")

    conn.commit()
    conn.close()
    print(f"✅ Ingestion réussie dans {DB_PATH}")

if __name__ == "__main__":
    run_full_pipeline()