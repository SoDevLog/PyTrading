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
ticker = { 'name': "PALANTIR", 'symbol': "PLTR" }
ticker = { 'name': "SAFRAN", 'symbol': "SAF.PA" }

# Date de début et date actuelle pour la fin
date_start = '2024-01-01' # h.datetime_past(0)
date_end = h.datetime_now() # '2024-04-03' 
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

# -----------------------------------------------------------------------------
# Donné par "Claude"
# -----------------------------------------------------------------------------

def claude_calculate_sar( price_data: pandas.DataFrame, step: float = 0.02, max_step: float = 0.2):
    """
    Calcule le Parabolic SAR avec des opérations pandas vectorisées
    
    Parameters:
    -----------
    price_data : pandas.DataFrame
        DataFrame avec colonnes 'High', 'Low', 'close'
    step : float, default 0.02
        Pas d'accélération
    max_step : float, default 0.2
        Pas maximum d'accélération
    
    Returns:
    --------
    pandas.Series
        Valeurs du SAR
    """
    # Copie des données pour éviter les modifications
    df = price_data.copy()
    
    # Initialisation des colonnes
    df.loc[:, 'sar'] = numpy.nan
    df.loc[:, 'trend'] = numpy.nan
    df.loc[:, 'af'] = numpy.nan
    
    # Premiers paramètres
    df.loc[df.index[0], 'sar'] = df.loc[ df.index[0], 'Low' ]
    df.loc[df.index[0], 'trend'] = 1
    df.loc[df.index[0], 'af'] = step
    
    # Calcul itératif
    for i in range(1, len(df)):
        prev = df.index[i-1]
        curr = df.index[i]
        
        prev_sar = df.loc[prev, 'sar']
        prev_trend = df.loc[prev, 'trend']
        prev_af = df.loc[prev, 'af']
        
        if prev_trend == 1:  # Tendance haussière
            if prev_sar > df.loc[ prev, 'Low' ]:
                # Inversion de tendance
                df.loc[curr, 'trend'] = -1
                df.loc[curr, 'sar'] = df.loc[ prev, 'High' ]
                df.loc[curr, 'af'] = step
            else:
                df.loc[curr, 'trend'] = prev_trend
                
                # Point le plus haut
                extreme_point = max(df.loc[prev, 'High'], df.loc[curr, 'High'])
                
                # Mise à jour du facteur d'accélération
                if extreme_point > df.loc[ prev, 'High' ]:
                    df.loc[curr, 'af'] = min( prev_af + step, max_step )
                else:
                    df.loc[curr, 'af'] = prev_af
                
                df.loc[curr, 'sar'] = prev_sar + df.loc[curr, 'af'] * (extreme_point - prev_sar)
        
        else:  # Tendance baissière
            if prev_sar < df.loc[prev, 'High']:
                # Inversion de tendance
                df.loc[curr, 'trend'] = 1
                df.loc[curr, 'sar'] = df.loc[prev, 'Low']
                df.loc[curr, 'af'] = step
            else:
                df.loc[curr, 'trend'] = prev_trend
                
                # Point le plus bas
                extreme_point = min(df.loc[prev, 'Low'], df.loc[curr, 'Low'])
                
                # Mise à jour du facteur d'accélération
                if extreme_point < df.loc[prev, 'Low']:
                    df.loc[curr, 'af'] = min(prev_af + step, max_step)
                else:
                    df.loc[curr, 'af'] = prev_af
                
                df.loc[curr, 'sar'] = prev_sar - df.loc[curr, 'af'] * (prev_sar - extreme_point)
    
    return df[ 'sar']

# -----------------------------------------------------------------------------

def calculate_dynamic_acceleration_factor(
        df: pandas.DataFrame,
        base_step: float = 0.02, 
        max_step: float = 0.2, 
        volatility_window: int = 20
    ) -> pandas.Series:
    """
    Calcule un facteur d'accélération dynamique basé sur la volatilité du marché
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame avec colonnes 'High', 'Low', 'close'
    base_step : float, default 0.02
        Pas d'accélération de base
    max_step : float, default 0.2
        Pas maximum d'accélération
    volatility_window : int, default 20
        Fenêtre de calcul de la volatilité
    
    Returns:
    --------
    pandas.Series
        Facteur d'accélération dynamique pour chaque période
    """
    # Calcul de la volatilité historique
    volatility = df['Close'].pct_change().rolling(window=volatility_window).std()
    
    # Normalisation de la volatilité
    normalized_volatility = (volatility - volatility.mean()) / volatility.std()
    
    # Calcul du facteur d'accélération dynamique
    dynamic_af = base_step + numpy.abs(normalized_volatility) * (max_step - base_step)
    
    # Limiter le facteur d'accélération entre base_step et max_step
    dynamic_af = dynamic_af.clip(base_step, max_step)
    
    # Gérer les premières périodes sans volatilité calculée
    dynamic_af.fillna(base_step, inplace=True)
    
    return dynamic_af

# -----------------------------------------------------------------------------

def calculate_sar_with_dynamic_af(
        df: pandas.DataFrame, 
        base_step: float = 0.02, 
        max_step: float = 0.2, 
        volatility_window: int = 20
    ) -> pandas.Series:
    """
    Calcul du SAR avec facteur d'accélération dynamique
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame avec colonnes 'High', 'Low', 'close'
    base_step : float, default 0.02
        Pas d'accélération de base
    max_step : float, default 0.2
        Pas maximum d'accélération
    volatility_window : int, default 20
        Fenêtre de calcul de la volatilité
    
    Returns:
    --------
    pandas.Series
        Valeurs du SAR
    """
    # Calculer le facteur d'accélération dynamique
    df['dynamic_af'] = calculate_dynamic_acceleration_factor(
        df, base_step, max_step, volatility_window
    )
    
    # Initialisation
    sar_df = df.copy()
    sar_df['sar'] = numpy.nan
    sar_df['ep'] = numpy.nan  # Extreme Point
    sar_df['is_long'] = True
    
    # Valeurs initiales
    sar_df.loc[sar_df.index[0], 'sar'] = sar_df.loc[sar_df.index[0], 'Low']
    sar_df.loc[sar_df.index[0], 'ep'] = sar_df.loc[sar_df.index[0], 'High']
    
    # Calcul séquentiel
    for i in range(1, len(sar_df)):
        prev = sar_df.index[i-1]
        curr = sar_df.index[i]
        
        # Récupération des valeurs précédentes
        prev_sar = sar_df.loc[prev, 'sar']
        prev_ep = sar_df.loc[prev, 'ep']
        prev_is_long = sar_df.loc[prev, 'is_long']
        curr_dynamic_af = sar_df.loc[curr, 'dynamic_af']
        
        # Calcul du SAR selon la tendance
        if prev_is_long:
            # Tendance haussière
            sar = prev_sar + curr_dynamic_af * (prev_ep - prev_sar)
            
            # Vérification d'inversion
            if sar > sar_df.loc[curr, 'Low']:
                # Inversion de tendance
                sar = prev_ep
                is_long = False
                new_ep = sar_df.loc[curr, 'Low']
            else:
                # Continuation de la tendance
                is_long = True
                new_ep = max(prev_ep, sar_df.loc[curr, 'High'])
        else:
            # Tendance baissière
            sar = prev_sar - curr_dynamic_af * (prev_sar - prev_ep)
            
            # Vérification d'inversion
            if sar < sar_df.loc[curr, 'High']:
                # Inversion de tendance
                sar = prev_ep
                is_long = True
                new_ep = sar_df.loc[curr, 'High']
            else:
                # Continuation de la tendance
                is_long = False
                new_ep = min(prev_ep, sar_df.loc[curr, 'Low'])
        
        # Mise à jour des valeurs
        sar_df.loc[curr, 'sar'] = sar
        sar_df.loc[curr, 'ep'] = new_ep
        sar_df.loc[curr, 'is_long'] = is_long
    
    return sar_df['sar']

# Calculate SAR
data["SAR"] = parabolic_sar( data["High"], data["Low"] )
data["SAR_enhanced"] = enhanced_parabolic_sar( data["High"], data["Low"] )
data["SAR_enhanced_2"] = enhanced_parabolic_sar_2( data["High"], data["Low"] )
data["SAR_enhanced_3"] = enhanced_parabolic_sar_3( data["High"], data["Low"] )
data["SAR_claude"] = claude_calculate_sar( data )
data["SAR_claude_2"] = calculate_sar_with_dynamic_af( data )

plt.figure( figsize=(16, 8) )
plt.plot( data.index, data["High"], label="High", color="blue", linestyle="dotted" )
plt.plot( data.index, data["Low"], label="Low", color="red", linestyle="dotted" )
plt.plot( data.index, data["SAR"], label="SAR Parabolique", color="green", marker="o", linestyle="None", alpha=0.4 )
plt.plot( data.index, data["SAR_enhanced"], label="SAR Enhanced", color="orange", marker="s", linestyle="None", alpha=0.6 )
plt.plot( data.index, data["SAR_enhanced_2"], label="SAR Enhanced 2", color="gold", marker="^", linestyle="None", alpha=0.8 )
plt.plot( data.index, data["SAR_enhanced_3"], label="SAR Enhanced 3", color="red", marker="p", linestyle="None", alpha=0.9 )
plt.plot( data.index, data["SAR_claude"], label="SAR Claude", color="red", marker="x", linestyle="None", alpha=0.9 )
plt.plot( data.index, data["SAR_claude_2"], label="SAR Claude 2", color="blue", marker=".", linestyle="-", linewidth=0.5, markersize=12, alpha=1 )
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.subplots_adjust( top=0.92, right=0.96 ) 
plt.title( f"{ticker['name']} - SAR Parabolique" )
plt.show()
