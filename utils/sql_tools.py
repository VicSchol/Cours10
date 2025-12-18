from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_nba_sql_tool(db_path="vector_db/nba_analytics.db"):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    # 1. Utilisation de gemini-1.5-flash (nom standard)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    system_prefix = """
    Tu es un expert en data analyse NBA. Base-toi sur les tables 'players', 'stats', 'teams' et 'reports'.
    
    RÈGLES IMPORTANTES :
    - Pour l'adresse à 3 points, utilise 'n_3P_Pct'.
    - Pour l'impact, utilise 'pie'.
    - Toujours faire une JOIN entre 'players' et 'stats' sur 'players.id = stats.player_id'.
    """

    # 2. Utilisation de tool-calling (le plus moderne pour Gemini)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling", # CHANGEMENT ICI
        verbose=True,
        prefix=system_prefix
    )
    
    return agent_executor

# --- BLOC DE TEST ---
if __name__ == "__main__":
    print("🔍 Test du SQL Tool NBA...")
    
    # Initialisation
    nba_agent = get_nba_sql_tool()
    
    # Test 1 : Requête de classement (Jointure + Tri)
    print("\n--- TEST 1 : TOP SCORERS ---")
    nba_agent.invoke({"input": "Qui sont les 3 meilleurs marqueurs (pts) de l'équipe de Oklahoma City (OKC) ?"})

    # Test 2 : Requête d'efficacité (Variable renommée)
    print("\n--- TEST 2 : ADRESSE 3 POINTS ---")
    nba_agent.invoke({"input": "Quel est le pourcentage à 3 points de Stephen Curry ?"})

    # Test 3 : Requête complexe (Jointure triple + Agrégation)
    print("\n--- TEST 3 : MOYENNE PAR ÉQUIPE ---")
    nba_agent.invoke({"input": "Quel est le PIE moyen des joueurs des Los Angeles Lakers ?"})