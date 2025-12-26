# %%
import os, sys
from dotenv import load_dotenv
import re

load_dotenv()

import sqlite3
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "vector_db/nba_analytics.db"


def get_nba_sql_tool():
    """
    Outil SQL fiable pour la NBA :
    - Extraction du joueur ou de l'équipe depuis la question
    - SQL réel (pas généré par le LLM)
    - Mapping via table dictionary
    """

    def extract_player_name(question: str) -> str | None:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM players")
        players = [row[0] for row in cursor.fetchall()]
        conn.close()

        for p in players:
            if p.lower() in question.lower():
                return p
        return None

    def run_player_stats(question: str) -> str:
        player_name = extract_player_name(question)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1️⃣ Cas joueur spécifique
        if player_name:
            cursor.execute("""
                SELECT PTS, FG_pct, PIE
                FROM stats s
                JOIN players p ON p.id = s.player_id
                WHERE LOWER(p.name) = LOWER(?)
            """, (player_name,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return f"Aucune statistique trouvée pour {player_name}."

            sql_ctx = f"Statistiques exactes de {player_name} :\n"
            sql_ctx += f"- PTS : {row[0]}\n"
            sql_ctx += f"- FG% : {row[1]}\n"
            sql_ctx += f"- PIE : {row[2]}"
            conn.close()
            return sql_ctx

        # 2️⃣ Cas meilleur marqueur (joueur)
        elif "meilleur marqueur" in question.lower() or "points total" in question.lower():
            cursor.execute("""
                SELECT p.name, SUM(s.PTS) AS total_points
                FROM stats s
                JOIN players p ON p.id = s.player_id
                GROUP BY p.name
                ORDER BY total_points DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if not row:
                conn.close()
                return "Aucune statistique trouvée."

            sql_ctx = f"Statistiques exactes du meilleur marqueur :\n"
            sql_ctx += f"- Joueur : {row[0]}\n"
            sql_ctx += f"- total_points : {row[1]}"
            conn.close()
            return sql_ctx

        # 3️⃣ Cas meilleure équipe (total points)
        elif "meilleure équipe" in question.lower() or "plus grand nombre total de points" in question.lower():
            cursor.execute("""
                SELECT t.full_name, SUM(s.PTS) AS total_points
                FROM stats s
                JOIN teams t ON s.team_code = t.code
                GROUP BY t.full_name
                ORDER BY total_points DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if not row:
                conn.close()
                return "Aucune statistique trouvée pour les équipes."

            sql_ctx = f"L'équipe avec le plus grand nombre de points est :\n"
            sql_ctx += f"- Équipe : {row[0]}\n"
            sql_ctx += f"- Points marqués : {row[1]}"
            conn.close()
            return sql_ctx
            
        # 4️⃣ Cas inconnu
        else:
            conn.close()
            return "Je n'ai pas identifié clairement le joueur ou l'équipe concerné(e)."

    return run_player_stats




# --- BLOC DE TEST ---
def test_nba_player_stats(question: str) -> str:
    """
    Fonction de test réutilisable pour interroger les stats NBA.
    """
    run_stats = get_nba_sql_tool()
    return run_stats(question)


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    question_test = "Quelle équipe possède le plus grand nombre total de points marqués sur la saison ?"
    print(test_nba_player_stats(question_test))