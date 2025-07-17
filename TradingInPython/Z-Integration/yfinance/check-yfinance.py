""" Details of error message send back from YahooFinance

    Totalement inutile quand il y a un problème de mise à jour de yfinance 
    car le message le taux de requêtes est dépassé est totalement faux il faut faire :
    comme pour le passage de la v0.2.54 à 0.2.58
    
    >pip install --upgrade yfinance
    
"""
import yfinance
import pandas as pd
import time

def check_yfinance_server_status( ticker="AAPL" ):
    """
    Vérifie si les serveurs Yahoo Finance répondent correctement
    en utilisant l'API actuelle de yfinance v0.2.54+
    """
    try:
        # Créer un objet Ticker (approche recommandée dans les versions récentes)
        ticker_obj = yfinance.Ticker( ticker )
        
        # Essayer de récupérer les données historiques
        hist = ticker_obj.history(period="1d")
        
        # Vérifier si les données sont vides
        if hist.empty:
            print(f"ERREUR: Données vides reçues pour {ticker}. Problème potentiel avec les serveurs Yahoo Finance.")
            return False
        
        # Vérifier les métadonnées du ticker pour confirmer que l'API fonctionne
        try:
            info = ticker_obj.info
            if not info:
                print(f"AVERTISSEMENT: Métadonnées vides pour {ticker}. Accès partiel aux serveurs.")
                
        except Exception as e:
            print(f"Impossible d'accéder aux métadonnées: {str(e)}")
            print("L'API peut fonctionner partiellement (données historiques mais pas d'infos)")
        
        print(f"Les serveurs Yahoo Finance semblent fonctionner correctement pour {ticker}.")
        print(f"Données récentes récupérées: {len(hist)} lignes")
        return True
        
    except Exception as e:
        print(f"ERREUR lors de la connexion aux serveurs Yahoo Finance: {str(e)}")
        
        # Analyse des erreurs courantes avec yfinance récent
        error_msg = str(e).lower()
        if "connection" in error_msg or "timeout" in error_msg:
            print("Problème de connexion détecté. Vérifiez votre connexion internet ou les serveurs peuvent être indisponibles.")
        elif "not found" in error_msg or "404" in error_msg:
            print("Le ticker spécifié n'a pas été trouvé. Vérifiez que le symbole est correct.")
        elif "too many requests" in error_msg or "429" in error_msg:
            print("Limite de taux dépassée. Yahoo Finance a temporairement bloqué vos requêtes.")
        
        return False

def check_download_functionality( tickers=["AAPL", "MSFT"], period="1d" ):
    """
    Teste spécifiquement la fonctionnalité download() avec plusieurs tickers
    """
    try:
        # Utiliser download() directement (toujours disponible dans v0.2.54)
        data = yfinance.download( tickers=tickers, period=period, group_by="ticker", progress=False )
        
        if isinstance( data, pd.DataFrame ):
            if data.empty:
                print("ERREUR: La fonction download() a retourné un DataFrame vide")
                return False
                
            # Si plusieurs tickers, la structure est différente
            if len( tickers ) > 1:
                # Vérifier si on a bien des données pour chaque ticker
                missing_tickers = []
                for ticker in tickers:
                    if ticker not in data.columns.levels[0] if hasattr(data.columns, 'levels') else True:
                        missing_tickers.append(ticker)
                
                if missing_tickers:
                    print(f"AVERTISSEMENT: Données manquantes pour les tickers: {missing_tickers}")
                    return False
            
            print(f"La fonction download() fonctionne correctement. Données récupérées: {data.shape}")
            return True
        else:
            print(f"ERREUR: Format de données inattendu: {type(data)}")
            return False
            
    except Exception as e:
        print(f"ERREUR avec yf.download(): {str(e)}")
        return False

# Exécution des tests
print("Test #1: Vérification de base avec Ticker...")
check_yfinance_server_status()
time.sleep(1)

print("\nTest #2: Vérification avec yfinance.download()...")
check_download_functionality()