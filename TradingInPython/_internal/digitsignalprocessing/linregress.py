""" Le but est de na pas utiliser scipy.stats dont la taille est de 30 Mo
    Mais finalement scipy est utilisée par sklearn ... Grrr
"""
import numpy
from collections import namedtuple

def calculate( x, y ):
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
