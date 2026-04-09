""" Tool_Finance - Indice - S&P Standard & Poor's SP 500
"""
import sys
import yfinance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

# make it independant from user's name
from pathlib import Path
base = Path(__file__).resolve().parent.parent.parent
sys.path.append( str(base) )

import helper as h

style.use( 'dark_background' )

# Apply common style sheet
plt.style.use('.\\styles\\darktradingplot.mplstyle')

# ---------------------------------------------------------------------------

date_start = '2020-01-01'
date_end = h.datetime_now()

# Télécharger les données historiques du S&P 500
sp500 = yfinance.Ticker( "^GSPC" ) # indice S&P 500
#market = "^AXJO" # Australian Securities Exchange (ASX)
cac_40 = yfinance.Ticker( "^FCHI" ) # Symbole de l'indice CAC 40 sur Yahoo Finance
euronext = yfinance.Ticker( "^AEX" ) # AEX (Euronext Amsterdam)

# Récupérer les données historiques (ici, depuis 2020)
historical_sp500 = sp500.history( start=date_start, end=date_end )
historical_cac_40 = cac_40.history( start=date_start, end=date_end )
historical_aex = euronext.history( start=date_start, end=date_end )

# Afficher les 5 premières lignes des données pour voir leur structure
#print(historical_data.head())

# Tracer les valeurs de clôture
historical_sp500['Close'].plot( figsize=(10, 6), title="Évolution du S&P 500" )
historical_cac_40['Close'].plot( figsize=(10, 6), title="S&P 500 et CAC 40" )
#historical_aex['Close'].plot( figsize=(10, 6), title="S&P 500 - CAC 40 - EuronextA" )

plt.xlabel( "Date" )
plt.ylabel( "Valeur de clôture" )
plt.gca().xaxis.set_major_formatter( mdates.DateFormatter('%Y-%m') )
plt.grid(True)
plt.subplots_adjust( left=0.12, right=0.9, bottom=0.13, top=0.9, wspace=0.2 )
plt.show()
