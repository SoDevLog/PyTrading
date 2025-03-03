""" Digit Signal Processing generic functions

    Moving Average Calculation in Digital Signal Processing

    Noise Filtering: Rolling averaging can be used to filter out noise or random fluctuations in a signal. By smoothing the signal over a period of time, it can help highlight long-term trends while reducing short-term noise.

    Trend Analysis: By calculating the rolling average over larger windows, one can obtain a better representation of the long-term trends of a signal. This can be useful for identifying general patterns or behaviors in time series.

    Reducing rapid variations: If a signal exhibits rapid variations that are not of interest, rolling averaging can smooth them out, allowing clearer visualization of important features.

    Outlier Elimination: Rolling averaging can help eliminate outliers or sudden spikes that could be the result of measurement errors or corrupted data.

    Tracking temporal evolution: By smoothing a signal, rolling averaging can make it easier to track large temporal changes while minimizing unnecessary fluctuations.

    Improved visualization: When graphing temporal data, using the rolling average can make the visualization more readable by reducing noise and highlighting trends.

    mode='same' : The result will be the same size as the input signal.
    mode='valid' : The result will be smaller, only the parts of the signal where the full window can be applied.
    mode='full' : The result will be larger, including the edges where the window is partially applied.

    min_periods :
    If the number of values in the window is less than min_periods, the .rolling().mean() method will return NaN for that position, because it does not have enough data to calculate the average.
    If the number of values in the window is greater than or equal to min_periods, the method will calculate the average of the values ​​available in the window.

    Typical usage of min_periods :
    min_periods=1 : This means that the moving average will be calculated from the first point. The first value will therefore be equal to itself, the second value will be the average of the first two points, and so on, until the window is completely filled.
    min_periods=window_size : In this case, the moving average will only be calculated once the full window is available.
    Before that, the first values will be NaN.

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

    # Calculating the average for other windows
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

""" The goal is to not use scipy.stats which is 30 MB in size 
    But finally scipy is used by sklearn ... Grrr
"""


def linregress( x, y ):
    """
    Linear regression using only NumPy.

    Reproduces the main features of scipy.stats.linregress.

    Parameters :
    - x : array-like, the independent data.
    - y : array-like, the dependent data.

    Return :
    - slope : slope of the regression.
    - intercept : y-intercept.
    - stderr : standard error of the slope.
    """

    # namedtuple for results
    LinregressResult = namedtuple(
        "LinregressResult",
        ["slope", "intercept", "stderr"]
    )
            
    # Conversion to NumPy arrays
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)

    # Data averages and sizes
    n = len(x)
    x_mean = numpy.mean(x)
    y_mean = numpy.mean(y)

    # Calculation of slope and intercept
    slope = numpy.sum((x - x_mean) * (y - y_mean)) / numpy.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    # Correlation coefficient
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
