""" Secondary indicators

    - rsi
    - macd
    - cmf
    - accdist
    - stochastic_oscillator
    - volume_weighted_average_price
    - bollinger_bands
    - atr   

"""

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
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
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
        if hist_range != 0:
             scaling_factor = macd_range / hist_range
        else:
            scaling_factor = 1
        histogram = (macd - signal_line) * scaling_factor
        return macd, signal_line, histogram
    else:
        return macd, signal_line


""" L'indicateur Chaikin Money Flow (CMF) est un outil d'analyse technique qui mesure la pression d'achat (accumulation) par rapport à la pression de vente (distribution) d'un titre sur une période donnée1. 
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

""" Indicateur - ACCDIST Accumulation/Distribution Line

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
def stochastic_oscillator( data, k=14, d=3, k_smooth=3 ):
    low_min = data['Low'].rolling( window=k ).min()
    high_max = data['High'].rolling( window=k ).max()
    
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    stoch_d = stoch_k.rolling( window=d ).mean()

    data['STOCH_k'] = stoch_k.rolling( window=k_smooth ).mean()
    data['STOCH_d'] = stoch_d.rolling( window=k_smooth ).mean()
    

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
    data['VWAP'] = cum_price_vol / cum_volume

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
    sma = data['Close'].rolling( window=window ).mean()  # Moyenne mobile simple
    std = data['Close'].rolling( window=window ).std()   # Ecart-type
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
def atr( data, period ):
    atr_values = []
    for i in range( len(data) ):
        if i == 0:
            atr_values.append( 0 )
        else:
            high_low = data['High'].iloc[i] - data['Low'].iloc[i]
            high_close = abs( data['High'].iloc[i] - data['Close'].iloc[i - 1] )
            low_close = abs( data['Low'].iloc[i] - data['Close'].iloc[i - 1] )
            true_range = max( high_low, high_close, low_close )
            if i < period:
                # Simple average over TR values ​​from index 1 to i (ignoring the initial 0)
                atr_values.append( sum(atr_values) / len(atr_values) if atr_values else true_range )
            else:
                # Using Wilder's formula for ATR
                atr_values.append( (atr_values[-1] * (period - 1) + true_range) / period )
    return atr_values
