""" Secondary indicators plateform TradingInPython

    - rsi
    - rsi_welles_wilder
    - macd
    - cmf
    - accdist
    - stochastic_oscillator
    - volume_weighted_average_price
    - bollinger_bands
    - atr
    - atr_rolling
    - calculate_sar_with_dynamic_af
    - calculate_balance_of_power
    - coppock
    - macd_zero_lag
    - obv
    - adx

"""
import pandas
import numpy


""" RSI - Relative Strength Index - Indice de force relative

	Evaluer la force ou la faiblesse d'un actif financier en comparant les gains et les pertes
 	récents sur une période spécifique. Il a été développé par l'analyste financier Welles Wilder.

	Le RSI est généralement calculé sur une période de 14 jours et oscille entre 0 et 100. 

  	Les valeurs supérieures à 70 indiquent que l'actif est suracheté, ce qui signifie 
  	qu'il pourrait être dû pour une correction à la baisse.

   	Les valeurs inférieures à 30 indiquent souvent que l'actif est survendu, ce qui peut suggérer 
    une opportunité d'achat.

	Cependant, il est important de noter que le RSI est un indicateur de momentum et ne doit pas 
 	être utilisé seul pour prendre des décisions d'investissement.

	Un "indicateur de momentum" évalue la vitesse ou la force du mouvement des prix dans une direction donnée. 
 	Il fournit une information sur la force d'un tendance.
"""
def rsi( data, window=14 ):
    delta = data['Close'].diff(1)
    delta[0] = delta[1] # 0 is always NaN, delta[1] is better than NaN
    
    gain = ( delta.where(delta > 0, 0) )
    loss = ( -delta.where(delta < 0, 0) )

    avg_gain = gain.rolling( window=window, min_periods=1 ).mean()
    avg_loss = loss.rolling( window=window, min_periods=1 ).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

""" RSI plus fidèle à son implémentation originale par J. Welles Wilder. 
    Utilise la méthode de calcul de moyenne exponentielle pondérée (EMA) 
    comme Wilder l'a conçue initialement.
"""
def rsi_welles_wilder( data, window=14 ):
    # Le alpha original de Wilder est 1/window
    alpha = 1.0 / window
    
    # Calculer les variations de prix
    delta = data['Close'].diff(1)
    
    # Séparer les hausses et les baisses
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = abs(down)
    
    # Calculer les moyennes mobiles exponentielles pondérées
    # Le adjust=False reproduit l'approche de Wilder
    avg_up = up.ewm( alpha=alpha, adjust=False ).mean()
    avg_down = down.ewm( alpha=alpha, adjust=False ).mean()
    
    # Calculer le RS et le RSI
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

""" Indicateur MACD - Moving Average Convergence Divergence

    Exemple d'utilisation:
        Supposons que vous ayez un DataFrame appelé 'df' avec une colonne 'Close' contenant les prix de clôture
        Vous pouvez calculer le MACD ainsi :
        macd, signal_line = calculate_macd(df)
        df: pandas.DataFrame

    EWMA (Exponential Weighted Moving Average) 

    Lorsque le MACD (Moving Average Convergence Divergence) est inférieur à sa ligne de signal, 
    cela indique généralement une tendance à la baisse dans le marché. 
    Cette configuration est souvent considérée comme un signal de vente 
    ou comme une confirmation que la tendance à la baisse pourrait se poursuivre.
    
    Explications:
    Croisement haussier (Bullish crossover) : Lorsque la ligne MACD croise au-dessus de la ligne de signal, 
    cela peut indiquer que le momentum devient haussier, suggérant un signal d'achat.
    Croisement baissier (Bearish crossover) : Lorsque la ligne MACD croise en dessous de la ligne de signal, 
    cela peut signifier que le momentum devient baissier, suggérant un signal de vente.

"""
def macd( data, histo=False, short_window=12, long_window=26, signal_window=9 ):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    :data: DataFrame with 'Close' prices
    :histo: Calculate or not histogram, to assure descendant compatibility 
    :short_window: Short moving average window, default is 12
    :long_window: Long moving average window, default is 26
    :signal_window: Signal line moving average window, default is 9
    :return: DataFrame with MACD and Signal line values
    """
    short_ema = data['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    if histo:
        histogram = macd - signal_line
        macd_range = macd.max() - macd.min()
        hist_range = histogram.max() - histogram.min()
        if (hist_range != 0).all():
             scaling_factor = macd_range / hist_range
        else:
            scaling_factor = 1
        histogram = (macd - signal_line) * scaling_factor
        return macd, signal_line, histogram
    else:
        return macd, signal_line


""" L'indicateur Chaikin Money Flow (CMF) est un outil d'analyse technique qui mesure la pression d'achat (accumulation) 
    par rapport à la pression de vente (distribution) d'un titre sur une période donnée.
    
    Il a été développé par l'analyste boursier Mark Chaikin.

    Le CMF repose sur l'idée que plus le cours de clôture est proche du sommet d'un titre, plus la pression d'achat 
    est forte (davantage d'accumulation s'est produite). Au contraire, plus le cours de clôture est proche du creux, 
    plus la distribution est forte.

    Sur un graphique, l'indicateur Chaikin Money Flow peut être évalué entre +100 et -100. Les zones comprises 
    entre 0 et 100 représentent l'accumulation, tandis que celles inférieures à 0 représentent la distribution.

    Les situations où l'indicateur se situe au-dessus ou en dessous de 0 pendant une période de 6 à 9 mois 
    (connues sous le nom de persistance des flux monétaires) peuvent être des signes de pressions d'achat 
    ou de vente significatives par de grandes institutions. De telles situations ont un impact 
    beaucoup plus prononcé sur l'action des prix.

    Le CMF est similaire à l'indicateur MACD (Moving Average Convergence Divergence-convergence et divergence des 
    moyennes mobiles), qui est plus populaire parmi les investisseurs et les analystes. Il utilise deux moyennes 
    mobiles individuelles pondérées exponentiellement (MME) pour mesurer le momentum. Le CMF analyse la différence 
    entre une MME de 3 jours et la MME de 10 jours de la ligne d'accumulation/distribution, qui est en fait un 
    indicateur distinct créé par Chaikin pour mesurer les entrées d'argent et la façon dont elles impactent les prix 
    des titres.
"""
def cmf( data, period=20 ):
    """ Calculate Chaikin Money Flow (CMF) financial indicator
        This operator is NaN at beginning during period steps
        
        Parameters
        ----------
        - **data**: DataFrame with 'High', 'Low', 'Close' and 'Volume' columns
        - **period**: Period to calculate CMF over, default is 20
        
        Returns
        -------
        Series with CMF values 
    """
    money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    money_flow_multiplier = money_flow_multiplier.fillna(0)
    money_flow_volume = money_flow_multiplier * data['Volume']
    money_flow_volume = money_flow_volume.fillna(0)
    cmf = money_flow_volume.rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
    return cmf.fillna(0)

""" ACCDIST - Accumulation/Distribution Line

	L'indicateur Accumulation/Distribution ACCDIST met en relation les cours et les volumes. Il a été développé
 	par Larry Williams, un célèbre trader sur contrats à terme. 

	Cet indicateur mesure la force entre l'offre et la demande en détectant si les investisseurs sont généralement
 	en Accumulation (acheteur) ou en Distribution (vendeur).

	L'Accumulation/Distribution est calculé en utilisant le prix de clôture, le prix le plus haut, le prix le plus
 	bas et le volume de la période. Il permet de repérer les phases d'accumulation et de distribution. Une valeur 
  	négative représente une sortie de capital et une valeur positive représente une entrée de capital.

	ACCDIST augmente, cela suggère une accumulation nette, les investisseurs achètent plus d'actions
 	qu'ils n'en vendent. Cela peut indiquer un sentiment positif à l'égard de l'actif financier, 
  	car il y a une pression à la hausse sur les prix.

	Si l'ACCDIST diminue, cela indique une distribution nette, avec plus de ventes que d'achats. 
 	Cela peut suggérer un sentiment négatif à l'égard de l'actif financier, 
  	car il y a une pression à la baisse sur les prix.

   	Résumé : l'ACCDIST est utilisé pour évaluer le flux de capitaux, il fournit une indication 
    sur la tendances de fond du marché. 

    Il peut être utilisé seul.
"""
def accdist( data ):
    # Close Location Volume
    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    clv = clv.fillna(0) # remplacer les valeurs NaN par 0
    accdist = clv * data['Volume']
    accdist = accdist.cumsum()
    #accdist_normalized = ((accdist - accdist.min()) / (accdist.max() - accdist.min())) * 200 - 100
    return accdist #accdist_normalized

""" STOCH - Développé par George Lane, indicateur de momentum mesure de l'élan.

    STOCH > 80 l'actif est zone de surachat potentiellement surévalué.
    STOCH < 20 l'actif est en zone de survente potentiellement sous-évalué.

    Signaux Achat/Vente :
    - Si la ligne rapide %K croise la ligne lente %D vers le haut, cela peut indiquer un signal d'achat.
      encore plus si le croisement s'effectue dans la zone verte < 20
      
    - Si elle croise vers le bas, cela peut signaler une opportunité de vente.
      encore plus si le croisement se pase en zone rouge > 80

    Divergences :
        Une divergence entre l'indicateur et le prix (par exemple, des sommets décroissants sur le Stoch
        tandis que les prix forment des sommets croissants) indiquent une inversion imminente de tendance.
        
"""
def stochastic_oscillator( data, k=14, d=3, k_smooth=3, min_periods=1 ):
    low_min = data['Low'].rolling( window=k, min_periods=min_periods ).min()
    high_max = data['High'].rolling( window=k, min_periods=min_periods ).max()
    
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    stoch_d = stoch_k.rolling( window=d ).mean()

    smoothedstoch_k = stoch_k.rolling( window=k_smooth, min_periods=min_periods ).mean()
    smoothedstoch_d = stoch_d.rolling( window=k_smooth, min_periods=min_periods ).mean()
    
    return smoothedstoch_k, smoothedstoch_d

"""	
	VWAP (Volume Weighted Average Price )

	Fonctionnement du VWAP (Volume Weighted Average Price) :
	Le VWAP est calculé en prenant la somme cumulée des prix multipliés par les volumes des transactions (prix * volume) sur une période, 
 	puis en divisant cette somme par le volume total cumulé. En d'autres termes, il pondère le prix en fonction du volume pour donner une 
  	image plus précise de la valeur moyenne d'un actif à travers les échanges.

	Interprétation :
	  - Prix au-dessus du VWAP : Si le prix est au-dessus du VWAP, cela peut indiquer que l'actif est potentiellement surévalué, 
  	    car il se négocie à un prix supérieur à sa moyenne pondérée par le volume. Cela peut être perçu comme un signal de vente.
     
	  - Prix en dessous du VWAP : Si le prix est en dessous du VWAP, cela peut signaler qu'il est sous-évalué, car il se négocie à 
  	    un prix inférieur à sa moyenne. Cela peut être perçu comme une opportunité d'achat.
        
"""
def volume_weighted_average_price( data ):
    cum_price_vol = (data['Close'] * data['Volume']).cumsum()
    cum_volume = data['Volume'].cumsum()
    wap = cum_price_vol / cum_volume
    return wap

""" 
	La volatilité du marché peut-être étudier grâce aux Bandes de Bollinger.
 
 	La volatilité sur le marché boursier, désigne la fréquence et l'amplitude des mouvements du marché. 
  	Plus les variations de prix sont importantes et fréquentes, plus la volatilité est élevée. 
   	Sur le plan statistique, la volatilité représente l'écart-type des rendements annualisés d'un marché
    au cours d'une période donnée.
    
	Plus le prix est proche de la bande supérieure, plus l'actif est proche des conditions de surachat.
	Plus le prix est proche de la bande inférieure, plus l'actif est proche de la survente.
"""
def bollinger_bands( data, window, n_std=2 ):
    """ Calculer les bandes de Bollinger pour un nombre de périodes et des écarts-types donnés.
		**data** : data frame
		**window** : int largeur de la fenêtre
		**n_std** : multiplicateur de l'écart type
    """
    sma = data['Close'].rolling( window=window, min_periods=1 ).mean()  # Moyenne mobile simple
    std = data['Close'].rolling( window=window, min_periods=1 ).std()   # Ecart-type
    upper_band = sma + (std * n_std)                     # Bande supérieure
    lower_band = sma - (std * n_std)                     # Bande inférieure
    return sma, upper_band, lower_band

""" Average True Range (ATR) - Measurement of market volatility.
    
    Indicateur de volatilité développé par J. Welles Wilder. 
    Il mesure l'amplitude moyenne des mouvements de prix sur une période donnée, sans indiquer la direction du mouvement.
    
    Plus l'ATR est élevé, plus le marché est volatil.
    
    Une hausse de l'ATR signale une augmentation de la volatilité, souvent associée à un breakout.
    Une baisse de l'ATR indique une phase de consolidation ou de faible volatilité.
    
    Placement de stop-loss : utilisation l'ATR pour ajuster leurs stops dynamiquement. Par exemple :
    StopLoss à X*ATR en dessous (ou au-dessus) du prix d'entrée.
    
"""
def atr( data, period=14 ):
    high = numpy.array( data['High'] )
    low = numpy.array( data['Low'] )
    close = numpy.array( data['Close'] )

    high_low = numpy.abs( high - low )
    high_close = numpy.abs( high - numpy.roll(close, 1) )
    low_close = numpy.abs( low - numpy.roll(close, 1) )

    # Numpy ne supporte pas plus de deux arguments en une seule fois.
    true_range = numpy.maximum( high_low, numpy.maximum( high_close, low_close ) )

    atr = numpy.zeros_like( true_range )
    atr[ 0:period ] = numpy.mean( true_range[0:period] )
    
    for i in range( period, len(true_range) ):
        atr[i] = ( atr[i-1] * (period-1) + true_range[i]) / period
        
    return atr

# -----------------------------------------------------------------------------

def atr_rolling( data, period=14 ):
    """
    ATR avec des valeurs calculées dès la première période, similaire au comportement d'un rolling window.
    Version optimisée utilisant les fonctions numpy pour de meilleures performances
    """
    high = numpy.array(data['High'])
    low = numpy.array(data['Low'])
    close = numpy.array(data['Close'])

    # Calcul du True Range
    high_low = high - low
    high_close = numpy.concatenate([[0], numpy.abs(high[1:] - close[:-1])])
    low_close = numpy.concatenate([[0], numpy.abs(low[1:] - close[:-1])])

    true_range = numpy.maximum(high_low, numpy.maximum(high_close, low_close))
    
    atr = numpy.zeros_like(true_range)
    atr[0] = true_range[0]
    
    # Utiliser numpy.cumsum pour calculer les moyennes cumulatives
    cumsum_tr = numpy.cumsum(true_range)
    
    for i in range(1, len(true_range)):
        if i < period:
            # Moyenne simple des éléments disponibles
            atr[i] = cumsum_tr[i] / (i + 1)
        else:
            # Formule de Wilder
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
    
    return atr

# -----------------------------------------------------------------------------
# SAR (Stop and Reverse)
#
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

# -----------------------------------------------------------------------------
# Normalise data upon 'n'
#
def normalise( data, max=1, min=0 ):
    norm = ( data - data.min() ) / ( data.max() - data.min() )
    scaled = norm * (max - min) + min
    return scaled

# -----------------------------------------------------------------------------

def calculate_balance_of_power( open, high, low, close ):
    """ Balance of Power """
    bop = numpy.zeros_like( close )
    
    # BOP = (Close - Open) / (High - Low)
    for i in range( len(close) ):
        range_hl = high[i] - low[i]
        if range_hl != 0:  # Éviter la division par zéro
            bop[i] = ( close[i] - open[i] ) / range_hl
        else:
            bop[i] = 0
            
    return bop

# -----------------------------------------------------------------------------
# ROC: Rate of Change
# 
def coppock( close, roc_long=14, roc_short=11, ema_period=10, signal_period=5 ):
    roc1 = 100 * (close - close.shift(roc_long)) / close.shift(roc_long)
    roc2 = 100 * (close - close.shift(roc_short)) / close.shift(roc_short)
    coppock_raw = roc1 + roc2

    coppock = coppock_raw.ewm( span=ema_period, adjust=False ).mean()
    signal = coppock.ewm( span=signal_period, adjust=False ).mean()
    derivative = coppock.diff()

    return pandas.DataFrame({
        "Coppock": coppock,
        "Signal": signal,
        "Derivative": derivative
    })
    
# -----------------------------------------------------------------------------
# MACD classique : plus lent, plus lissé, plus conservateur.
# MACD Zéro Lag : plus réactif, signale les retournements plus tôt, 
# mais parfois plus sensible au bruit.
#
def zero_lag_ema( series, span ):
    """ EMA avec correction de retard (zéro lag) """
    ema = series.ewm( span=span, adjust=False ).mean()
    ema_of_ema = ema.ewm( span=span, adjust=False ).mean()
    zlema = 2 * ema - ema_of_ema
    return zlema

def macd_zero_lag( data, short_window=12, long_window=26, signal_window=9, histo=True ):
    close = data['Close']
    short_zlema = zero_lag_ema( close, short_window )
    long_zlema = zero_lag_ema( close, long_window )

    macd_zl = short_zlema - long_zlema
    signal_zl = zero_lag_ema( macd_zl, signal_window )

    if histo:
        histogram = macd_zl - signal_zl
        return pandas.DataFrame({
            'MACD_ZL': macd_zl,
            'Signal_ZL': signal_zl,
            'Histogram_ZL': histogram
        })
    else:
        return pandas.DataFrame({
            'MACD_ZL': macd_zl,
            'Signal_ZL': signal_zl
        })

""" OBV - On-Balance Volume

    Hausse de l’OBV : accumulation (volume acheteur > vendeur)
    Baisse de l’OBV : distribution (volume vendeur > acheteur)
    Divergence prix/OBV :
    Si les prix montent mais que l’OBV baisse : risque de retournement baissier.
    Si les prix baissent mais que l’OBV monte : possible retournement haussier.

    Quand préférer l’OBV ?
    Indicateur ultra léger et cumulatif basé sur le volume.
    Travaille avec les divergences prix/volume.
    En complément d’indicateurs de tendance ou momentum (MACD, RSI, ADX).
"""
def obv( prices, volumes ):
    
    # Sanity check
    if len( prices ) != len( volumes ):
        raise ValueError("Les listes prices et volumes doivent avoir la même longueur")
    
    if len(prices) < 2:
        raise ValueError("Il faut au moins 2 valeurs pour calculer l'OBV")
    
    obv = [0]  # Premier jour, OBV = 0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            # Prix en hausse : ajouter le volume
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i-1]:
            # Prix en baisse : soustraire le volume
            obv.append(obv[-1] - volumes[i])
        else:
            # Prix inchangé : OBV reste identique
            obv.append(obv[-1])
    
    return obv

""" ADX - Average Directional Index

    Indicateur de tendance développé par J. Welles Wilder. 
    Il fait partie de l'ensemble des Directional Movement Indicators (DMI), 
    Inclut également les courbes :
    +DI (Positive Directional Indicator) 
    -DI (Negative Directional Indicator).

    L'ADX mesure la force d'une tendance (0-100), tandis que +DI et -DI indiquent la direction. 
    Les valeurs ADX > 25 suggèrent généralement une tendance forte.
"""
def adx( high, low, close, period=14 ):
    
    # Conversion en numpy arrays si nécessaire
    high = numpy.array(high)
    low = numpy.array(low)
    close = numpy.array(close)
    
    # Calcul du True Range (TR)
    def true_range(h, l, c):
        tr1 = h[1:] - l[1:]  # High - Low
        tr2 = numpy.abs(h[1:] - c[:-1])  # |High - Close précédent|
        tr3 = numpy.abs(l[1:] - c[:-1])  # |Low - Close précédent|
        
        return numpy.maximum(tr1, numpy.maximum(tr2, tr3))
    
    # Calcul des mouvements directionnels
    def directional_movement(h, l):
        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]
        
        plus_dm = numpy.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = numpy.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        return plus_dm, minus_dm
    
    # Fonction de lissage (Wilder's smoothing)
    def wilder_smooth( data, period ):
        smoothed = numpy.zeros_like( data )
        smoothed[ period-1 ] = numpy.mean( data[:period] )
        
        for i in range( period, len(data) ):
            smoothed[i] = ( smoothed[i-1] * (period - 1) + data[i] ) / period
            
        return smoothed
    
    # Calculs
    tr = true_range(high, low, close)
    plus_dm, minus_dm = directional_movement(high, low)
    
    # Lissage avec la méthode de Wilder
    atr = wilder_smooth(tr, period)
    plus_di_smooth = wilder_smooth(plus_dm, period)
    minus_di_smooth = wilder_smooth(minus_dm, period)
    
    # Calcul des indicateurs directionnels
    # Calcul des indicateurs directionnels (éviter division par zéro)
    plus_di = numpy.zeros_like(atr)
    minus_di = numpy.zeros_like(atr)
    mask = atr != 0
    plus_di[mask] = 100 * plus_di_smooth[mask] / atr[mask]
    minus_di[mask] = 100 * minus_di_smooth[mask] / atr[mask]
    
    # Calcul de l'ADX (éviter division par zéro)
    sum_di = plus_di + minus_di
    dx = numpy.zeros_like( sum_di )
    mask = sum_di != 0
    dx[mask] = 100 * numpy.abs(plus_di[mask] - minus_di[mask]) / sum_di[mask]
    
    adx = wilder_smooth(dx, period)
    
    # Ajouter des NaN au début pour aligner avec les données originales
    padding = numpy.full(len(high) - len(adx), numpy.nan)
    
    return {
        'ADX': numpy.concatenate([padding, adx]),
        '+DI': numpy.concatenate([padding, plus_di]),
        '-DI': numpy.concatenate([padding, minus_di])
    }

"""
    Calcule les fractales de Bill Williams sur un DataFrame de prix.
    Version optimisée avec opérations vectorisées.

    Parameters:
    data : DataFrame avec colonnes 'High' et 'Low'
    period : int, nombre de bougies de chaque côté à vérifier (défaut: 2)

    Returns:
    DataFrame avec colonnes 'Fractal_Up' et 'Fractal_Down'
"""    
def fractales_williams( data, period=2 ):

    
    high = data['High']
    low = data['Low']
    
    # Initialiser les masques à True
    is_fractal_up = numpy.ones(len(data), dtype=bool)
    is_fractal_down = numpy.ones(len(data), dtype=bool)
    
    # Vérifier toutes les bougies de 1 à period de chaque côté
    for i in range(1, period + 1):
        # Fractale UP : high doit être strictement > tous ses voisins
        is_fractal_up &= (high > high.shift(i)) & (high > high.shift(-i))
        
        # Fractale DOWN : low doit être strictement < tous ses voisins
        is_fractal_down &= (low < low.shift(i)) & (low < low.shift(-i))
    
    # Appliquer les masques et créer les colonnes
    data['Fractal_Up'] = numpy.where(is_fractal_up, high, numpy.nan)
    data['Fractal_Down'] = numpy.where(is_fractal_down, low, numpy.nan)
    
    return data    