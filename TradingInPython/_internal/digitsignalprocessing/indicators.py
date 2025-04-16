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
import pandas
import numpy

""" RSI - Relative Strength Index - Relative Strength Index

    Assessing the strength or weakness of a financial asset by comparing recent gains and losses over a specific period. It was developed by financial analyst Welles Wilder.

    The RSI is usually calculated over a 14-day period and ranges from 0 to 100.

    Values ​​above 70 indicate that the asset is overbought, meaning
    that it could be due for a downward correction.

    Values ​​below 30 often indicate that the asset is oversold, which may suggest
    a buying opportunity.

    However, it is important to note that the RSI is a momentum indicator and should not be used alone to make investment decisions.

    A "momentum indicator" assesses the speed or strength of price movement in a given direction.
    It provides information on the strength of a trend.
    
"""
def rsi( data, window=14 ):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

""" MACD Indicator - Moving Average Convergence Divergence

    Example of use:
        Suppose you have a DataFrame called 'df' with a column 'Close' containing the closing prices
        You can calculate the MACD like this:
        macd, signal_line = calculate_macd(df)
        df: pandas.DataFrame

    EWMA (Exponential Weighted Moving Average)

    When the MACD (Moving Average Convergence Divergence) is below its signal line,
    it usually indicates a downtrend in the market.
    This pattern is often considered a sell signal
    or a confirmation that the downtrend may continue.

    Explanations:
        Bullish crossover: When the MACD line crosses above the signal line,
        it can indicate that the momentum is turning bullish, suggesting a buy signal.
        Bearish crossover: When the MACD line crosses below the signal line,
        it may mean that the momentum is turning bearish, suggesting a sell signal.

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


""" The Chaikin Money Flow (CMF) indicator is a technical analysis tool that measures the buying pressure 
    (accumulation) versus selling pressure (distribution) of a security over a given period1.
    It was developed by stock analyst Mark Chaikin.

    The CMF is based on the idea that the closer the closing price is to the high of a security, the greater the buying pressure (more accumulation has occurred). Conversely, the closer the closing price is to the low, the greater the distribution.

    On a chart, the Chaikin Money Flow indicator can be measured between +100 and -100. Areas
    between 0 and 100 represent accumulation, while areas below 0 represent distribution.

    Situations where the indicator is above or below 0 for a period of 6 to 9 months
    (known as money flow persistence) can be signs of significant buying or selling pressure by large institutions. Such situations have a much more pronounced impact on price action.

    The CMF is similar to the MACD (Moving Average Convergence Divergence) indicator, which is more popular among investors and analysts. It uses two individual exponentially weighted moving averages (EMAs) to measure momentum. The CMF analyzes the difference between a 3-day EMA and the 10-day EMA of the accumulation/distribution line, which is actually a separate indicator created by Chaikin to measure inflows and how they impact security prices.

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

""" Indicator - ACCDIST Accumulation/Distribution Line

    The Accumulation/Distribution ACCDIST indicator relates prices to volumes. It was developed
    by Larry Williams, a famous futures trader.

    This indicator measures the strength between supply and demand by detecting whether investors are
    generally in Accumulation (buying) or Distribution (selling).

    The Accumulation/Distribution is calculated using the closing price, the high price, the low price
    and the volume of the period. It helps to identify accumulation and distribution phases. A negative value
    represents an outflow of capital and a positive value represents an inflow of capital.

    ACCDIST is increasing, this suggests a net accumulation, investors are buying more shares
    than they are selling. This can indicate a positive sentiment towards the financial asset,
    as there is upward pressure on prices.

    If the ACCDIST is decreasing, this indicates a distribution net, with more sales than purchases.
    This may suggest a negative sentiment towards the financial asset,
    as there is downward pressure on prices.

    Summary: The ACCDIST is used to assess the flow of capital, it provides an indication
    of the underlying market trends.

    It can be used alone.
    
"""
def accdist( data ):
    # Close Location Volume
    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    clv = clv.fillna(0) # remplacer les valeurs NaN par 0
    accdist = clv * data['Volume']
    accdist = accdist.cumsum()
    #accdist_normalized = ((accdist - accdist.min()) / (accdist.max() - accdist.min())) * 200 - 100
    return accdist #accdist_normalized

""" STOCH - Developed by George Lane, momentum indicator measures momentum.

    STOCH > 80 the asset is in an overbought zone potentially overvalued.
    STOCH < 20 the asset is in an oversold zone potentially undervalued.

    Buy/Sell Signals:
    - If the fast line %K crosses the slow line %D upwards, this can indicate a buy signal.
    even more if the crossover occurs in the green zone < 20

    - If it crosses downwards, this can signal a sell opportunity.
    even more if the crossover occurs in the red zone > 80

    Divergences:
    A divergence between the indicator and the price (for example, decreasing highs on the Stoch
    while prices form increasing highs) indicates an imminent trend reversal.

"""
def stochastic_oscillator( data, k=14, d=3, k_smooth=3 ):
    low_min = data['Low'].rolling( window=k ).min()
    high_max = data['High'].rolling( window=k ).max()
    
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    stoch_d = stoch_k.rolling( window=d ).mean()

    data['STOCH_k'] = stoch_k.rolling( window=k_smooth ).mean()
    data['STOCH_d'] = stoch_d.rolling( window=k_smooth ).mean()
    

""" VWAP (Volume Weighted Average Price )

    How VWAP (Volume Weighted Average Price) works:
        VWAP is calculated by taking the cumulative sum of prices multiplied by trading volumes (price * volume) over a period,
        then dividing this sum by the cumulative total volume. In other words, it weights the price according to volume to give a
        more accurate picture of the average value of an asset across trades.

    Interpretation:
        - Price above VWAP: If the price is above VWAP, this may indicate that the asset is potentially overvalued,
        as it is trading at a price higher than its volume weighted average. This can be seen as a sell signal.

        - Price below VWAP: If the price is below VWAP, this may signal that it is undervalued, as it is
        trading at a price lower than its average. This can be seen as an opportunity purchase.

"""
def volume_weighted_average_price( data ):
    cum_price_vol = (data['Close'] * data['Volume']).cumsum()
    cum_volume = data['Volume'].cumsum()
    data['VWAP'] = cum_price_vol / cum_volume

""" Market volatility can be studied using Bollinger Bands.

    Volatility in the stock market refers to the frequency and amplitude of market movements.

    The greater and more frequent the price changes, the higher the volatility.
    Statistically, volatility represents the standard deviation of a market's annualized returns
    over a given period.

    The closer the price is to the upper band, the closer the asset is to overbought conditions.

    The closer the price is to the lower band, the closer the asset is to oversold conditions.
    
"""
def bollinger_bands( data, window, n_std=2 ):
    """ Calculate Bollinger Bands for a given number of periods and standard deviations.
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

    Volatility indicator developed by J. Welles Wilder.

    It measures the average amplitude of price movements over a given period, without indicating the direction of the movement.

    The higher the ATR, the more volatile the market.

    An increase in the ATR signals an increase in volatility, often associated with a breakout.

    A decrease in the ATR indicates a phase of consolidation or low volatility.

    Stop-loss placement: use the ATR to adjust their stops dynamically. For example:
    StopLoss at X*ATR below (or above) the entry price.

"""
def atr( data, period ):
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
