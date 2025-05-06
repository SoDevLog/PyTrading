""" Details of error message send back from YahooFinance
"""
import yfinance

def check_yfinance_server_status( ticker="AAPL", period="1d" ):
    """
    Check if yfinance is working properly by attempting to download data
    and analyzing the results/potential errors.
    """
    try:
        # Tentative de téléchargement de données
        data = yfinance.download( ticker, period=period, progress=True, auto_adjust=True )
        
        # Vérifie si le DataFrame est vide
        if data.empty:
            print("ERREUR: Données vides reçues. Problème potentiel avec les serveurs Yahoo Finance.")
            return False
        
        # Vérifie si le nombre de colonnes est correct (typiquement 6 pour OHLCV+Adj Close)
        if len(data.columns) < 5:  # On s'attend à avoir au moins 5 colonnes
            print(f"AVERTISSEMENT: Nombre de colonnes inhabituel ({len(data.columns)}). Données potentiellement incomplètes.")
            print(f"Colonnes reçues: {data.columns.tolist()}")
            return False
            
        # Vérifier si les données sont récentes (pour period='1d')
        if period == "1d" and len(data) == 0:
            print("ERREUR: Aucune donnée récente disponible. Problème possible avec les serveurs.")
            return False
            
        print("Les serveurs Yahoo Finance semblent fonctionner correctement.")
        print(f"Données reçues: {len(data)} entrées")
        return True
        
    except Exception as e:
        print(f"ERREUR lors de la connexion aux serveurs Yahoo Finance: {str(e)}")
        
        # Analyser les messages d'erreur courants
        error_msg = str(e).lower()
        if "connection" in error_msg or "timeout" in error_msg:
            print("Problème de connexion détecté. Vérifiez votre connexion internet ou les serveurs peuvent être indisponibles.")
        elif "not found" in error_msg or "404" in error_msg:
            print("Le ticker spécifié n'a pas été trouvé. Vérifiez que le symbole est correct.")
        elif "rate limit" in error_msg or "429" in error_msg:
            print("Limite de taux dépassée. Yahoo Finance a temporairement bloqué votre adresse IP.")
            
        return False

if __name__ == "__main__":
    check_yfinance_server_status()