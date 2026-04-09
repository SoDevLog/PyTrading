""" Tool_Finance - Résoudre le problème du taux de change
"""
import pandas
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

DISPLAY_GRAPH = True

date_start = "2023-01-01"
date_end = dt.datetime.now().strftime('%Y-%m-%d')

# 1. Récupérer les données d'une action en dollars
symbol = "PLTR" # PALANTIR # "AAPL"  # exemple avec Apple
data = yf.download( symbol, start=date_start, end=date_end ).droplevel( 1, axis=1 )

# 2. Récupérer le taux de change EUR/USD
eurusd = yf.download( "EURUSD=X", start=date_start, end=date_end ).droplevel( 1, axis=1 )

# 3. Conversion des prix en euros
data['Close_EUR'] = data['Close'] / eurusd['Close']

# Afficher les 5 dernières valeurs
print( data[['Close', 'Close_EUR']].tail() )

# Conversion de la date string en datetime pour l'indexation
date_stock = "20/03/2025"
date_formattee = pandas.to_datetime( date_stock, format="%d/%m/%Y" )

# Afficher les valeurs à cette date spécifique
try:
    valeurs_date = data.loc[ date_formattee, ['Close', 'Close_EUR'] ]
    print( f"Valeurs pour la date du : {date_stock}:" )
    print( valeurs_date )
except KeyError:
    # Si la date exacte n'existe pas (weekend/jour férié ou hors période)
    print( f"Pas de données disponibles pour le {date_stock}. Voici la date la plus proche disponible:" )
    # Trouver la date la plus proche
    date_proche = data.index[ data.index.get_indexer([date_formattee], method='nearest')[0] ]
    valeurs_date = data.loc[ date_proche, ['Close', 'Close_EUR'] ]
    print( f"Valeurs pour la date du : {date_proche.strftime('%d/%m/%Y')}:" )
    print( valeurs_date )
    
# Optionnel: visualiser les prix en USD et EUR
if DISPLAY_GRAPH:
    plt.figure(figsize=(12, 6))
    plt.plot( data.index, data['Close'], label=f'{symbol} en USD')
    plt.plot( data.index, data['Close_EUR'], label=f'{symbol} en EUR')
    plt.title( f'Prix de {symbol} en USD et EUR')
    plt.legend()
    plt.grid(True)
    plt.show()