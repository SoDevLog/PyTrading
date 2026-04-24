""" Montrer comment utiliser l'API des scripts utilisateur.

    Pour intégrer l'API dans votre application, 
    insérez dans notre script uniquement le code :
    
    from user_scripts.api import api, UserScriptAPI
    
    Et utilisez l'API comme dans la fonction main()
        
"""
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
from user_scripts.api import api, UserScriptAPI

def main():
    
    print( "=== Test de l'API pour les scripts utilisateur ===" )
    
    print( f"Nom : {api.name}" )
    print( f"Ticker : {api.ticker}" )
    print( f"Période : {api.period}" )
    print( f"Intervalle : {api.interval}" )
    print( f"Tickers : {api.tickers}" )
    
    # Dataframe
    data = api.df.copy()
    if data.empty:
        print("Dataframe : aucune donnée disponible.")
        return

    print( data.tail(int(2)) )

    print(f"\n{len(data)} barres chargées.")
    print(f"Dernier prix : {api.df['Close'].iloc[-1]:.2f}")
    print(f"Dernière bougie : {api.df.iloc[-1].name}")

    # Callbacks
    api.on_bar( lambda bar: print(f"[OnBar] close={bar['Close']:.2f}") )
    api.on_close( lambda: print("[OnClose] Script terminé proprement.") )

if __name__ == "__main__":

    api_context = {
            'name': 'Microsoft',
            'ticker': 'MSFT',
            'period': '1d',
            'interval': '1m',
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'IT' ],
            'df': None
        }
    
    api = UserScriptAPI()
    api.update( **api_context )            
    
    main()