""" Digit Signal Processing generic functions

    Moving Average Calculation in Digital Signal Processing

    Noise Filtering: Rolling averaging can be used to filter out noise or random fluctuations in a signal. By smoothing the signal over a period of time, it can help highlight long-term trends while reducing short-term noise.

    Trend Analysis: By calculating the rolling average over larger windows, one can obtain a better representation of the long-term trends of a signal. This can be useful for identifying general patterns or behaviors in time series.

    Reducing rapid variations: If a signal exhibits rapid variations that are not of interest, rolling averaging can smooth them out, allowing clearer visualization of important features.

    Outlier Elimination: Rolling averaging can help eliminate outliers or sudden spikes that could be the result of measurement errors or corrupted data.

    Tracking temporal evolution: By smoothing a signal, rolling averaging can make it easier to track large temporal changes while minimizing unnecessary fluctuations.

    Improved visualization: When graphing temporal data, using the rolling average can make the visualization more readable by reducing noise and highlighting trends.
    
    mode='same' : Le résultat aura la même taille que le signal d'entrée.
    mode='valid' : Le résultat aura une taille plus petite, uniquement les parties du signal où la fenêtre complète peut être appliquée.
    mode='full' : Le résultat aura une taille plus grande, en incluant les bords où la fenêtre est partiellement appliquée.
    
    min_periods :
    Si le nombre de valeurs dans la fenêtre est inférieur à min_periods, la méthode .rolling().mean() renverra NaN pour cette position, car elle n'a pas assez de données pour calculer la moyenne.
    Si le nombre de valeurs dans la fenêtre est supérieur ou égal à min_periods, la méthode calculera la moyenne des valeurs disponibles dans la fenêtre.

    Utilisation typique de min_periods :
    min_periods=1 : Cela signifie que la moyenne mobile sera calculée dès le premier point. La première valeur sera donc égale à elle-même, la deuxième valeur sera la moyenne des deux premiers points, et ainsi de suite, jusqu'à ce que la fenêtre soit entièrement remplie.
    min_periods=window_size : Dans ce cas, la moyenne mobile ne sera calculée qu'une fois que la fenêtre complète est disponible. 
    Avant cela, les premières valeurs seront NaN.

"""
import numpy
from collections import namedtuple

# -----------------------------------------------------------------------------
# DIY (Do It Yourself) SMA (Single Mouving Average)
#
def moving_average_diy( signal, window_size ):
    
    sma = numpy.zeros( len(signal) )
    
    # Initialize first values
    for i in range( window_size - 1 ):
        sma[i] = numpy.mean( signal[:i+1] )  # Partial window averaging

    # Calcul de la moyenne pour les autres fenêtres
    for i in range(window_size - 1, len(signal)):
        sma[i] = numpy.mean( signal[i - window_size + 1 : i + 1] )  # Average over the sliding window

    return sma

# -----------------------------------------------------------------------------
# signal is shifted to window_size / 2
#
def moving_average_pandas( signal, window_size ):
    return signal.rolling( window=window_size, min_periods=1 ).mean()

# ewm: exponential window
#
def moving_average_pandas_exp( signal, window_size ):
    return signal.ewm( span=window_size, min_periods=1, adjust=False ).mean()

# ----------------------------------------------------------------------------
# Can choose mode
#
def moving_average( signal, window_size, mode='valid' ):
    """ Use convolve to calculate moving average
        - window_size: is calculate as integral calculation of the function equal to 1
          sigma(weights) = 1
        - mode: 'same' for sma has same lenght as signal
    """
    weights = numpy.ones( window_size ) / window_size
    sma = numpy.convolve( signal, weights, mode )
    return sma

# ----------------------------------------------------------------------------
# Mode always valid
#
def moving_average_extended( signal, window_size, mode='reflect' ):
    """ Use convolve to calculate moving average
        - window_size: is calculate as integral calculation of the function equal to 1
          sigma(weights) = 1
    """
    weights = numpy.ones( window_size ) / window_size
    pad_width = window_size - 1
    extended_signal = numpy.pad( signal, (pad_width//2, pad_width - pad_width//2), mode=mode)
    sma = numpy.convolve( extended_signal, weights, 'valid' )
    return sma

# ----------------------------------------------------------------------------

def moving_average_exp(signal, window_size, mode='valid'):
    """ Use convolve to calculate moving average with an exponential window
        - window_size: is calculate as integral calculation of the function equal to 1
          sigma(weights) = 1
    """
    alpha = 0.2 # smoothing factor
    weights = numpy.exp( -alpha * numpy.arange(window_size) )
    weights /= weights.sum() # normalize weights for sigma(weights) equal to one
    nparray_values = numpy.asarray(signal, dtype=float)
    sma = numpy.convolve(nparray_values, weights, mode)
    return sma

# ----------------------------------------------------------------------------

def moving_average_exp_extended( signal, window_size ):
    """ Use convolve to calculate moving average with an exponential window
        - window_size: is calculate as integral calculation of the function equal to 1
          sigma(weights) = 1
    """
    alpha = 0.2 # smoothing factor
    weights = numpy.exp( -alpha * numpy.arange(window_size) )
    weights /= weights.sum() # normalize weights for sigma(weights) equal to one
    nparray_values = numpy.asarray( signal, dtype=float )
    pad_width = window_size - 1
    extended_signal = numpy.pad( nparray_values, (pad_width//2, pad_width - pad_width//2), mode='reflect')
    sma = numpy.convolve( extended_signal, weights, 'valid' )
    return sma

# ----------------------------------------------------------------------------

def reshape( signal1, signal2 ):
    """ Make signal2 as long as signal1
        For moving average function there are values erased
        to be displayed signal must be reshaped to the lenght
        of signal1
    """
    lg_reshape = len(signal1) - len(signal2)
    if lg_reshape < 0:
        print('ERROR: Reshape could \'not be possible!')
        return False
    else:
        s2reshaped = signal2
        for x in range( 0, lg_reshape ):
            s2reshaped = numpy.append(s2reshaped, s2reshaped[len(signal2)-1] )
        return s2reshaped

""" Le but est de na pas utiliser scipy.stats dont la taille est de 30 Mo
    Mais finalement scipy est utilisée par sklearn ... Grrr
"""


def linregress( x, y ):
    """
    Régression linéaire utilisant uniquement NumPy.
    Reproduit les fonctionnalités principales de scipy.stats.linregress.

    Parameters :
    - x : array-like, les données indépendantes.
    - y : array-like, les données dépendantes.

    Return :
    - slope : pente de la régression.
    - intercept : ordonnée à l'origine.
    - stderr : erreur standard de la pente.
    """

    # namedtuple for results
    LinregressResult = namedtuple(
        "LinregressResult",
        ["slope", "intercept", "stderr"]
    )
            
    # Conversion en tableaux NumPy
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)

    # Moyennes et tailles des données
    n = len(x)
    x_mean = numpy.mean(x)
    y_mean = numpy.mean(y)

    # Calcul de la pente et de l'intercept
    slope = numpy.sum((x - x_mean) * (y - y_mean)) / numpy.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    # Coefficient de corrélation
    #ss_tot = numpy.sum((y - y_mean) ** 2)
    ss_res = numpy.sum((y - (slope * x + intercept)) ** 2)
    #rvalue = numpy.sqrt(1 - (ss_res / ss_tot)) * numpy.sign(slope)

    # Coefficient de détermination (R^2)
    #r_squared = rvalue ** 2

    # Erreur standard de la pente
    stderr = numpy.sqrt(ss_res / (n - 2)) / numpy.sqrt(numpy.sum((x - x_mean) ** 2))

    # Calcul de la valeur-p (distribution t de Student)
    #t_stat = rvalue * numpy.sqrt((n - 2) / (1 - r_squared))
    #pvalue = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2))

    return LinregressResult( slope, intercept, stderr )
