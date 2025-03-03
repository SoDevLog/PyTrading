""" Since there is a modern Ichimoku...

    The modernized Ichimoku Kinko Hyo indicator is based on a combination of several elements of the traditional Ichimoku indicator,
    adjusted by the Average True Range (ATR) to account for volatility.

    1. Buy Signal
    -------------
    Condition :
        The Tenkan-sen line (conversion line) crosses above the Kijun-sen (base line).
        The closing price is above both boundaries of the Ichimoku cloud (Senkou Span A and Senkou Span B).
    Interpretation :
        This indicates bullish strength in the market. A bullish crossover between Tenkan-sen and Kijun-sen, accompanied by the price position above the cloud,
        suggests that buyers are in control of the market.
        Adding bands based on ATR helps estimate volatility and adjust the probability of a continued upward move.
    Action :
        Traders may consider entering a long position (buy) or maintaining existing positions if these conditions are met.    

     2. Sell Signal
    --------------------------------
    Condition :
        The Tenkan-sen line crosses below the Kijun-sen.
        The closing price is below both boundaries of the Ichimoku cloud (Senkou Span A and Senkou Span B).
    Interpretation :
        This signal indicates market weakness and a possible downtrend. The bearish crossover between Tenkan-sen and Kijun-sen, in conjunction with price below the cloud,
        suggests that sellers are dominant.
        The fit with the ATR gives an idea of ​​volatility, which is useful in assessing the strength of the downtrend.
    Action :
        Traders may consider entering a short position (selling) or exiting existing long positions.
        
    The strength of the trend is greater if these signals occur when the price is well above (for an uptrend)
    or well below (for a downtrend) the cloud.
    
    3. Neutral or Undecided Zones
    -----------------------------
    Condition :
        The price is inside the Ichimoku cloud.
        Crossbreeding between Tenkan-sen and Kijun-sen is unclear or repeated over a short period of time.
    Interpretation :
        When the price is in the cloud, it indicates an area of indecision where neither buyers nor sellers are taking control. The market
        may be in a consolidation phase or transitioning to a new trend.
    Action :
        Traders may choose to stay out of the market until a clearer direction emerges.
        Orders can be placed outside the cloud to capture a potential breakout.
    
    4. Confirmation Signal and Volatility (via ATR and Kijun_sen_upper and Kijun_sen_lower)
    -----------------------------------------------------------------------------------------
    Condition :
        ATR based bands (Kijun_sen_upper and Kijun_sen_lower) can be used to confirm signals or to manage risks.
    Interprétation :
        If the price moves significantly away from the bands, it could indicate an over-volatility move. Traders could interpret this
        as a signal of trend exhaustion or use the information to adjust their stop-loss.
    Action :
        Adjust positions or stop orders based on the widening or narrowing of these bands, taking into account current market volatility.

        - Kijun_sen_upper : This upper band represents a dynamic resistance.
        The higher the volatility (ATR), the further this band will be from the Kijun-sen.
        When the price reaches this band, it can indicate an overbought zone or a resistance level
        where the price could reverse or correct downward.
        
        - Kijun_sen_lower : This lower band represents a dynamic support.
        If the price goes down to this band, it may indicate an oversold area or a support level
        where the price could bounce upwards.

    Market Volatility:
        When the bands are wide (i.e. ATR is high), it indicates that the market is volatile. Such a situation suggests caution,
        as larger price movements may occur.
        When the bands are closer together, it indicates a decrease in volatility, which may suggest a more stable market.

    Potential Reversal Points:
        If the price breaks above the upper band (Kijun_sen_upper), it may signal an overbought condition, and a downward correction may follow.

    If the price breaks below the lower band (Kijun_sen_lower), it may indicate an oversold condition, and an upward bounce may follow.

    Trend Confirmation:
        As long as the price moves between the bands, it confirms the current trend.
        If the price breaks above the bands, it may be a signal of a trend change or a sharp acceleration in the move.
    
    5. Kumo (cloud)
    ---------------
    Interpretation:
        The cloud is the central element of the Ichimoku indicator. It represents the support and resistance zone.
        The thicker the cloud, the stronger the support or resistance.
        When the price is above the cloud, the trend is considered bullish, and when it is below, the trend is bearish.
        If the price is inside the cloud, it indicates an indecisive or neutral trend.

        - Twist of the cloud: this is when Senkou Span A goes below or above Senkou Span B
        When the Ichimoku cloud twists after a trend, it indicates a weakening of the trend and the price enters a range. It may then be time to
        take profits.

    6. Chikou Span line delayed line (Lagging Span)
    --------------------------------------------------
    Condition:
        The Chikou Span is often used as confirmation. It shows the current price projected back 26 periods.
    Interpretation:
        If Chikou Span is above the current price, it confirms an uptrend. If it is below, it confirms a downtrend.
    Action:
        Use Chikou Span to confirm buy or sell signals, or to avoid taking positions against the prevailing trend.

    7. Indicator Philosophy
    ------------------------------
    Ichimoku is based on visual analysis and trend theory.
        The indicator assumes that prices already reflect all available market information,
        including volumes, even if this is not explicitly taken into account.

    - Advantages and limitations without volumes
        Advantages:
            Simplicity: The absence of volume data makes Ichimoku easier to use. Traders do not have to analyze
            additional data, which can make decision-making faster and less cluttered.

            Trend Identification: It is particularly effective in identifying trends, support and resistance levels,
            and generating trading signals in a variety of market conditions.

            Limitations:
            Lack of Volume Confirmation: Volume can often provide important clues about the strength or weakness of a trend.
            For example, high volume during an uptrend can confirm the strength of that trend,
            while low volume could indicate weakness or lack of market commitment.
            Difficulty in Spotting Reversals: Volume can help spot trend reversals by showing whether price
            movements are supported by significant participation. Without this data, some reversals can be more difficult to spot.

    8. Conclusion
    -------------
        Interpreting signals in Modernized Ichimoku must consider several factors simultaneously, including line crossovers,
        position relative to the cloud, and volatility adjustments via ATR. By combining these elements, traders can get a more complete overview of the market situation, which can help make more informed decisions about entries, exits and risk management.
"""
import warnings
import numpy as np
import pandas
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .indicators import atr

# 1. Calculate the components of Ichimoku with ATR
# ----------------------------------------------------
# - True Range (TR)
# TR = max( ∣High − Low∣, ∣High − Previous Close∣, ∣Low − Previous Close∣ )
#
# The idea is to capture the largest price gaps,
# taking into account not only the intraday range (High - Low),
# but also the opening gaps compared to the closing of the previous period.
#
# - ATR: Moving average over a number of periods
#
def ichimoku_modernise( data, period1=9, period2=26, period3=52, multiplier=1.5 ):
    # Tenkan-sen (conversion line)
    data['Tenkan_sen'] = (data['High'].rolling(window=period1).max() + data['Low'].rolling(window=period1).min()) / 2

   # Kijun-sen (baseline)
    data['Kijun_sen'] = (data['High'].rolling(window=period2).max() + data['Low'].rolling(window=period2).min()) / 2

    # Senkou Span A (first cloud boundary)
    data['Senkou_span_A'] = ((data['Tenkan_sen'] + data['Kijun_sen']) / 2).shift(period2)

    # Senkou Span B (second cloud boundary)
    data['Senkou_span_B'] = ((data['High'].rolling(window=period3).max() + data['Low'].rolling(window=period3).min()) / 2).shift(period2)

    # Chikou Span (delayed line)
    data['Chikou_span'] = data['Close'].shift(-period2)

    # Average True Range (ATR)
    data['ATR'] = atr( data, period2 )
    
    # Add ATR based bands
    data['Kijun_sen_upper'] = data['Kijun_sen'] + (data['ATR'] * multiplier)
    data['Kijun_sen_lower'] = data['Kijun_sen'] - (data['ATR'] * multiplier)
    
    return data

# 2. Generate Ichimoku-based signals and train a logistic regression model
# 
def generate_signals( data ):
    data['Signal'] = 0

    # Conditions for buy signals
    data.loc[(data['Tenkan_sen'] > data['Kijun_sen']) & (data['Close'] > data['Senkou_span_A']) & (data['Close'] > data['Senkou_span_B']), 'Signal'] = 1

    # Conditions for sell signals
    data.loc[(data['Tenkan_sen'] < data['Kijun_sen']) & (data['Close'] < data['Senkou_span_A']) & (data['Close'] < data['Senkou_span_B']), 'Signal'] = -1
    
    # To display this signal at the bottom of the graph
    mean = data['Close'].mean()
    data['Signal_display'] = data['Signal'] + mean
    
    return data

def train_predictive_model_0000(data):
    # We create features to train the model
    X = data[['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']]
    y = data['Signal']
    
    # We eliminate lines where the signal is not defined
    X = X.dropna()
    y = y[ X.index ]

    if X.shape[0] == 0:
        raise ValueError("Pas assez de données pour entraîner le modèle après suppression des NaN.")
    
    if X.shape[0] < 10:
        raise ValueError("Pas assez de données pour diviser en ensembles d'entraînement et de test.")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict( X_test )
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Précision du modèle: {accuracy:.2f}")
    
    return model

# The variable y = data['Signal'] is what the machine learning model is trying to predict.
# The machine learning model, for example LogisticRegression, is a supervised model.
# This means that it learns by observing the relationships between the input
# variables (X - the features)
# and the target variable (y - the buy/sell signals).
#
def train_predictive_model( data ):
    global y_test, y_pred
    
    # We create features to train the model
    X = data[['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']]
    y = data['Signal']
    
    # Checking NaNs
    X = X.dropna()
    y = y[ X.index ]
    
    if X.shape[0] == 0:
        raise ValueError("Pas assez de données pour entraîner le modèle après suppression des NaN.")
    
    if X.shape[0] < 10:
        raise ValueError("Pas assez de données pour diviser en ensembles d'entraînement et de test.")
    
    # Data normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame to preserve column names
    X_scaled_df = pandas.DataFrame( X_scaled, columns=X.columns, index=X.index )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split( X_scaled_df, y, test_size=0.3, random_state=42 )
    
    # Build and train the model with higher iterations and catching warnings
    with warnings.catch_warnings():
        warnings.simplefilter( "ignore", category=ConvergenceWarning )
        model = LogisticRegression( max_iter=1000, solver='lbfgs')  # Increase max_iter to 1000 other solvers 'liblinear', 'sag', 'saga'
        model.fit( X_train, y_train )
    
    # Predictions and evaluation
    y_pred = model.predict( X_test )
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Précision du modèle: {accuracy:.2f}")
    
    return model

# Apply the model to new data to get predictive signals
# honestly... I don't see what this function is for
def apply_model( model, data ):
    
    X_new = data[['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']].dropna()
    data['Signal_Forcasted'] = np.nan
    data.loc[ X_new.index, 'Signal_Forcasted'] = model.predict( X_new )

    # Can display on the graph
    mean = data['Close'].mean()
    data['Signal_Forcasted'] = data['Signal_Forcasted'] + mean
        
    return data

def plot_forcasts( df, y_test, y_pred ):
    # Create a column in the DataFrame for predictions, initially with NaNs
    df['Forcasts'] = float('nan')
    
    # Align predictions with indices in df
    df.loc[ y_test.index, 'Forcasts'] = y_pred

    # Trace the Kijun-sen
    #plt.plot( df['Kijun_sen'], label='Kijun-sen')

    # Plot buying and selling predictions
    plt.scatter( df.index, df['Forcasts'], label='Forcasts', marker='o', color='red')

def plot_future_predictions(df, y_pred):
    # Latest index of the historical series
    last_index = df.index[-1]
    
    # Calculating the number of future predictions
    future_indices = pandas.date_range( start=last_index, periods=len( y_pred ) + 1, freq='D')[1:]
    
    # Create a series for future predictions
    future_predictions = pandas.Series( y_pred, index=future_indices )
    
    # Tracer the Kijun-sen
    #plt.plot( df.index, df['Kijun_sen'], label='Kijun-sen' )
    
    # Plot future predictions
    plt.plot( future_predictions.index, future_predictions, label='Prédictions Futures', marker='o', linestyle='--', color='red')

def predict_future_signals(df, model, days_in_future, days_in_past):
    """
    Predict future signals using a model based on past data windows.

    Parameters:
    df (pd.DataFrame): DataFrame containing historical data.
    model: Trained machine learning model.
    days_in_future (int): Number of days to predict future signals for.
    days_in_past (int): Number of days of past data to use for each prediction.

    Returns:
    np.array: Signal predictions for future days.
    """
    
    # Assume df has a column 'Signal' which is the label to predict
    if 'Signal' not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'Signal'.")

    # Take only the 7 values ​​for creating the model
    features = ['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'ATR', 'Kijun_sen_upper', 'Kijun_sen_lower']
    _df = df[ features ].copy()
    _df.fillna( 0, inplace=True )
    
    # Normalize historical data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform( _df )
    
    # Prepare past data windows for predictions
    X = np.array([data_scaled[i:i + days_in_past] for i in range(len(data_scaled) - days_in_past)])
    
    # Ensure the model expects the data to be of good shape
    X = np.reshape(X, (X.shape[0], days_in_past, X.shape[2]))
    
    # Use the last data window to start future predictions
    last_window = pandas.DataFrame(X[-1], columns=features)

    predictions = []
    
    for _ in range( days_in_future ):
                
        # Predict future signal
        prediction = model.predict( last_window  )
        predictions.append( prediction[0] )
        
        # Prepare input for next prediction
        X_prime = np.roll( last_window, shift=-1, axis=0)  # Move window
        last_window = pandas.DataFrame( X_prime, columns=features )
    
    # Convert predictions to numpy array
    predictions = np.array( predictions )
    
    # Scaling for display
    mean = df['Close'].mean()
    predictions = predictions + mean

    # Invert the transformation to get the real signal values ​​if needed
    # Here we don't use `scaler.inverse_transform` because we predicted the signals directly
    return predictions