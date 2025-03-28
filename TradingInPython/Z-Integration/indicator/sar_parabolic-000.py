""" SAR parabolique - Parabolic Stop And Reverse développé par J. Welles Wilder

    Il est principalement utilisé pour détecter les tendances 
    et définir des points de retournement potentiels sur un graphique.

    - Si les points sont sous le prix -> Tendance haussière.
    - Si les points sont au-dessus du prix -> Tendance baissière.
    
    Lorsque le prix croise le SAR, cela peut indiquer un changement de tendance.
    
"""
import numpy
import pandas
import yfinance
import matplotlib.pyplot as plt
import sys
sys.path.append( 'C:\\Users\\Mabyre\\Documents\\GitHub\\PythonAdvanced\\' )
import helper as h

# -----------------------------------------------------------------------------

ticker = { 'name': "DASSAULT AVIATION", 'symbol': "AM.PA" }

# Date de début et date actuelle pour la fin
date_start = '2025-01-01' # h.datetime_past(0)
date_end = h.datetime_now()
interval_dates = '1d'

# Conserver la colonne 'Adj Close' pour les graphiques mais en intraday la colonne n'existe plus
auto_adjust = False 
    
data = yfinance.download(
    ticker['symbol'], 
    start=date_start, 
    end=date_end, 
    interval=interval_dates, 
    auto_adjust=auto_adjust,
    prepost=True
)

# Since v0.2.54 of yfinance needed to drop 'Ticker' level 
data.columns = data.columns.droplevel( 1 )

# -----------------------------------------------------------------------------

def parabolic_sar( highs, lows, af_start=0.02, af_increment=0.02, af_max=0.2 ):
    """
    Calcule le SAR parabolique à partir des plus hauts et plus bas.

    Args:
        highs (pandas.Series): Séries des plus hauts.
        lows (pandas.Series): Séries des plus bas.
        af_start (float): Facteur d'accélération initial.
        af_increment (float): Incrément du facteur d'accélération.
        af_max (float): Facteur d'accélération maximal.

    Returns:
        pandas.Series: Valeurs du SAR.
    """
    sar = numpy.zeros_like( highs )  # tableau pour stocker les valeurs du SAR
    trend = 1  # 1 = tendance haussière, -1 = tendance baissière
    ep = highs.iloc[0]  # foint extrême (Extreme Point), initialisé au premier plus haut
    af = af_start  # facteur d'accélération initial

    # Initialisation du premier SAR
    sar[0] = lows.iloc[0] if trend == 1 else highs.iloc[0]

    for i in range(1, len( highs )):
        # Calcul du SAR suivant la tendance actuelle
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        # Mise à jour du EP et AF
        #
        # Tendance haussière
        if trend == 1:  
            if highs.iloc[i] > ep:
                ep = highs.iloc[i]
                af = min( af + af_increment, af_max )  # Augmenter AF sans dépasser af_max
        # Tendance baissière
        else:  
            if lows.iloc[i] < ep:
                ep = lows.iloc[i]
                af = min( af + af_increment, af_max )

        # Changement de tendance si le SAR dépasse le prix
        if ( trend == 1 and sar[i] > lows.iloc[i] ) or ( trend == -1 and sar[i] < highs.iloc[i] ):
            trend *= -1  # inversion de tendance
            sar[i] = ep  # réinitialisation du SAR au dernier EP
            af = af_start  # réinitialisation de AF

    return pandas.Series( sar, index=highs.index )

# -----------------------------------------------------------------------------
# Amélioration avec AF dynamique, filtrage ATR, et la moyenne mobile sur le SAR
# -----------------------------------------------------------------------------

def enhanced_parabolic_sar( highs, lows, af_start=0.02, af_max=0.2, window_smoothing=3 ):
    sar = numpy.zeros_like( highs )
    trend = 1
    ep = highs.iloc[0]
    af = af_start

    # Ajustement Dynamique du Facteur d'Accélération (AF)
    # Idée : Plus la tendance est forte, plus l'AF augmente rapidement.
    #
    def dynamic_af( highs, lows ):
        atr = highs - lows
        af_dynamic = af_start + (atr / atr.max()) * (af_max - af_start)
        return numpy.clip( af_dynamic, af_start, af_max )

    sar[0] = lows.iloc[0] if highs.iloc[0] > lows.iloc[0] else highs.iloc[0]

    for i in range( 1, len(highs) ):
        af = dynamic_af( highs[:i], lows[:i] ).iloc[-1]
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        if trend == 1:
            if highs.iloc[i] > ep:
                ep = highs.iloc[i]
            if sar[i] > lows.iloc[i]:  
                trend = -1  
                sar[i] = ep
                af = af_start # restart af
        else:
            if lows.iloc[i] < ep:
                ep = lows.iloc[i]
            if sar[i] < highs.iloc[i]:  
                trend = 1  
                sar[i] = ep
                af = af_start # restart af

    return pandas.Series( sar, index=highs.index ).rolling( window=window_smoothing ).mean()

# -----------------------------------------------------------------------------

def enhanced_parabolic_sar_2( highs, lows, af_start=0.02, af_max=0.2, window_smoothing=3 ):
    n = len( highs )
    
    # Init variables
    sar = pandas.Series( index=highs.index, dtype=float )
    trend = 1  
    ep = highs.iloc[0]  
    af = af_start  

    def dynamic_af( highs, lows ):
        atr = max( highs.iloc[-1] - lows.iloc[-1], 1e-6 )  
        return min( af_start + (atr / max(atr, 1e-6) ), af_max )  

    sar.iloc[0] = lows.iloc[0] if highs.iloc[0] > lows.iloc[0] else highs.iloc[0]  

    for i in range( 1, n ):
        _low, _high = lows.iloc[:i+1], highs.iloc[:i+1]  
        af = dynamic_af( _high, _low )  
        sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])

        if ( trend == 1 and sar.iloc[i] > lows.iloc[i] ) or ( trend == -1 and sar.iloc[i] < highs.iloc[i] ):
            trend *= -1  # inversion du trend
            sar.iloc[i] = ep  # réinitialisation du SAR
            af = af_start  # réinitialisation du facteur d'accélération
        
        ep = max(ep, highs.iloc[i]) if trend == 1 else min(ep, lows.iloc[i])

    return sar.rolling( window=window_smoothing, min_periods=1 ).mean()

# -----------------------------------------------------------------------------
def enhanced_parabolic_sar_3( highs, lows, af_start=0.02, af_max=0.2, window_smoothing=3, atr_window=14 ):
    n = len(highs)
    
    # Init variables
    sar = pandas.Series(index=highs.index, dtype=float)
    trend = 1  # 1 pour tendance haussière, -1 pour tendance baissière
    ep = highs.iloc[0]  # extreme point (point extrême)
    af = af_start  # acceleration Factor (facteur d'accélération)

    # Calcul de l'ATR
    def calculate_atr( highs, lows, closes, window=atr_window ):
        high_low = highs - lows
        high_close = abs( highs - closes.shift(1) )
        low_close = abs( lows - closes.shift(1) )
        tr = pandas.concat( [high_low, high_close, low_close], axis=1 ).max( axis=1 )
        return tr.rolling( window=window, min_periods=1 ).mean()

    # Calcul du facteur d'accélération dynamique
    def dynamic_af( highs, lows, closes, i ):
        _atr = calculate_atr( highs, lows, closes )
        atr = _atr.iloc[-1]  # ATR du jour actuel
        return min(af_start + (atr / (atr + 1e-6)), af_max)

    sar.iloc[0] = lows.iloc[0] if highs.iloc[0] > lows.iloc[0] else highs.iloc[0]  # Initialisation SAR

    for i in range(1, n):
        closes = lows.copy()  # La fermeture est juste la valeur la plus basse ici
        af = dynamic_af(highs, lows, closes, i)
        sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])

        # Condition pour changement de tendance
        if (trend == 1 and sar.iloc[i] > lows.iloc[i]) or (trend == -1 and sar.iloc[i] < highs.iloc[i]):
            trend *= -1  # Changement de tendance
            sar.iloc[i] = ep  # Réinitialisation du SAR
            af = af_start  # Réinitialisation du facteur d'accélération
        
        ep = max(ep, highs.iloc[i]) if trend == 1 else min(ep, lows.iloc[i])

    return sar.rolling(window=window_smoothing, min_periods=1).mean()

# Calculate SAR
data["SAR"] = parabolic_sar( data["High"], data["Low"] )
data["SAR_enhanced"] = enhanced_parabolic_sar( data["High"], data["Low"] )
data["SAR_enhanced_2"] = enhanced_parabolic_sar_2( data["High"], data["Low"] )
data["SAR_enhanced_3"] = enhanced_parabolic_sar_3( data["High"], data["Low"] )

plt.figure( figsize=(10, 5) )
plt.plot( data.index, data["High"], label="High", color="blue", linestyle="dotted" )
plt.plot( data.index, data["Low"], label="Low", color="red", linestyle="dotted" )
plt.plot( data.index, data["SAR"], label="SAR Parabolique", color="green", marker="o", linestyle="None", alpha=0.4 )
plt.plot( data.index, data["SAR_enhanced"], label="SAR Enhanced", color="orange", marker="s", linestyle="None", alpha=0.6 )
plt.plot( data.index, data["SAR_enhanced_2"], label="SAR Enhanced 2.0", color="blue", marker="^", linestyle="None", alpha=0.8 )
plt.plot( data.index, data["SAR_enhanced_3"], label="SAR Enhanced 3.0", color="red", marker="x", linestyle="None", alpha=0.9 )
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.subplots_adjust( top=0.92, right=0.96 ) 
plt.title( f"{ticker['name']} - SAR Parabolique" )
plt.show()
