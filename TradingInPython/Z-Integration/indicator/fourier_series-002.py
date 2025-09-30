""" Tool_Used - Utilisation de la décomposition en série de Fourier pour créer un indicateur de tendance.

    Indicateur boursier de tendance basé sur la décomposition en série de Fourier
    avec analyse des dérivées des composantes pour déterminer la direction du marché
    somme des dérivées des composantes de Fourier pour obtenir un indicateur de tendance.
    
"""
import pandas as pd
import numpy as np
import yfinance
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime

# Paramètres pour l'extraction des données
company = {'symbol': 'STMPA.PA', 'name' : 'STMICROELECTRONICS'}
#company = {'symbol': 'HO.PA', 'name' : 'THALES'}

date_start = '2025-07-01'
date_end = datetime.now()
interval_dates = '1d'

# Nombre de termes de la décomposition en séries de fourier
N_TERMS = 9 + 1 # pour avoir n composantes sinusoïdales

def fourier_series( data, n_terms, k=0, display=True ):
    """Calcule la série de Fourier pour les données données"""
    n = len(data)
    t = np.arange(n)
    result = np.zeros(n)
    coefficients = []

    if k == 0:  # toutes les composantes
        _range = range(0, n_terms)
    else:  # seulement la composante k
        _range = range(k, k + 1)

    for k in _range:
        a_k = 2/n * np.sum(data * np.cos(2 * np.pi * k * t / n))
        b_k = 2/n * np.sum(data * np.sin(2 * np.pi * k * t / n))
        result += a_k * np.cos(2 * np.pi * k * t / n) + b_k * np.sin(2 * np.pi * k * t / n)
        
        phi_k = np.arctan2(b_k, a_k)  # Utiliser arctan2 pour éviter les divisions par zéro
        amplitude = np.sqrt(a_k**2 + b_k**2)
        
        # Calcul de la période
        if k != 0:
            period = n / k
        else:
            period = np.inf
            
        coefficients.append({'k': k, 'a_k': a_k, 'b_k': b_k, 'amplitude': amplitude, 'period': period, 'phase': phi_k})
        
        if display:
            print(f"{k} ampli: {amplitude:.2f} cos: {a_k:.2f} sin: {b_k:.2f} phi: {np.degrees(phi_k):.2f}° period: {period:.2f}")

    return result, coefficients

def fourier_derivative( data, n_terms ):
    """Calcule les dérivées des composantes de Fourier"""
    n = len( data )
    t = np.arange(n)
    derivatives = []
    total_derivative = np.zeros(n)

    for k in range( 1, n_terms ):  # k=0 donne dérivée nulle (composante constante)
        a_k = 2/n * np.sum(data * np.cos(2 * np.pi * k * t / n))
        b_k = 2/n * np.sum(data * np.sin(2 * np.pi * k * t / n))
        
        # Dérivée de a_k * cos(2πkt/n) + b_k * sin(2πkt/n)
        # = -a_k * (2πk/n) * sin(2πkt/n) + b_k * (2πk/n) * cos(2πkt/n)
        derivative_k = (-a_k * (2 * np.pi * k / n) * np.sin(2 * np.pi * k * t / n) + 
                       b_k * (2 * np.pi * k / n) * np.cos(2 * np.pi * k * t / n))
        
        derivatives.append( derivative_k )
        total_derivative += derivative_k

    return derivatives, total_derivative

def normalize_signal( y_data, min_val, max_val ):
    """Normalise le signal entre min_val et max_val"""
    y_min = y_data.min()
    y_max = y_data.max()
    if y_max - y_min == 0:
        return np.full_like( y_data, (min_val + max_val) / 2 )
    return ( (y_data - y_min) / (y_max - y_min)) * (max_val - min_val) + min_val

def calculate_trend_indicator( derivative_signal, window=5 ):
    """Calcule un indicateur de tendance basé sur la dérivée"""
    # Moyenne mobile de la dérivée pour lisser le signal
    smoothed_derivative = pd.Series( derivative_signal ).rolling( window=window, center=True ).mean()
    smoothed_derivative = smoothed_derivative.bfill().ffill()
    
    # Indicateur de tendance (-1: baissière, 0: neutre, 1: haussière)
    trend_indicator = np.sign( smoothed_derivative.values )  # Convertir en array numpy
    
    # Force de la tendance (valeur absolue de la dérivée normalisée)
    trend_strength = np.abs( smoothed_derivative.values ) / ( np.abs( smoothed_derivative.values ).max() + 1e-8 )
    
    return trend_indicator, trend_strength, smoothed_derivative.values

# -----------------------------------------------------------------------------
# Récupération des données
# -----------------------------------------------------------------------------

ticker = yfinance.Ticker( company['symbol'] )
data = ticker.history(
    start=date_start,
    end=date_end,
    interval=interval_dates,
    prepost=True
)

print(f"Nombre de points de données: {len(data)}")

# Vérification de la taille minimale des données
if len(data) < N_TERMS * 2:
    print(f"Attention: Pas assez de données ({len(data)}) pour {N_TERMS} termes de Fourier.")
    print(f"Réduction du nombre de termes à {len(data)//2}")
    N_TERMS = max(2, len(data)//2)

if len(data) < 10:
    print("Erreur: Dataset trop petit pour l'analyse. Minimum 10 points requis.")
    exit()

# Calcul de la série de Fourier et des dérivées
fourier_result, coefficients = fourier_series( data['Close'].values, N_TERMS )
derivatives, total_derivative = fourier_derivative( data['Close'].values, N_TERMS )

# Normalisation
min_price = data['Close'].min()
max_price = data['Close'].max()
data['FourierNormalize'] = normalize_signal( fourier_result, min_price, max_price )

# Calcul de l'indicateur de tendance
trend_indicator, trend_strength, smoothed_derivative = calculate_trend_indicator( total_derivative )

# -----------------------------------------------------------------------------
# Création des graphiques avec axe continu
# -----------------------------------------------------------------------------

#fig = plt.figure( figsize=(16, 12) )
fig = plt.figure( figsize=(14, 9) )
params = {"left": 0.06, "bottom": 0.08, "right": 0.95, "top": 0.95, "hspace": 0.25}
grid_spec = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 2], **params)

ax1 = fig.add_subplot( grid_spec[0] )
ax2 = fig.add_subplot( grid_spec[1], sharex=ax1 )
ax3 = fig.add_subplot( grid_spec[2], sharex=ax1 )

# Utiliser l'axe continu pour les tracés
# Création d'un axe d'indices continus (sans les discontinuités des weekends)
data['index_continuous'] = np.arange(len(data))
x_axis = data['index_continuous'].values

# Graphique 1: Prix et série de Fourier
ax1.plot( x_axis, data['Close'], label='Prix de clôture', color='blue', linewidth=1.5 )
ax1.plot( x_axis, data['FourierNormalize'], label=f'Série de Fourier', color='orange', linewidth=1.5 )
ax1.set_title( f'{company["name"]} - Prix et Reconstruction Fourier k={N_TERMS}', fontsize=14, fontweight='bold' )
ax1.legend()
ax1.grid( True, alpha=0.3 )
ax1.set_ylabel( 'Prix (€)' )

# Graphique 2: Composantes individuelles
colors = plt.cm.Set3( np.linspace(0, 1, N_TERMS-1) )
for i in range( 1, N_TERMS ):
    fourier_c, _ = fourier_series( data['Close'].values, N_TERMS, i, display=False )
    ax2.plot( x_axis, fourier_c, label=f'Composante {i}', color=colors[i-1], alpha=0.7 )

ax2.set_title( 'Composantes individuelles de Fourier', fontsize=12, fontweight='bold' )
ax2.legend( bbox_to_anchor=(1.05, 1), loc='upper left' )
ax2.grid( True, alpha=0.3 )
ax2.set_ylabel( 'Amplitude' )

# Graphique 3: Indicateur de tendance basé sur les dérivées
ax3_twin = ax3.twinx()

# Dérivée totale lissée
ax3.plot( x_axis, smoothed_derivative, label='Dérivée totale (lissée)', color='purple', linewidth=2 )
ax3.axhline( y=0, color='black', linestyle='-', alpha=0.5 )
ax3.fill_between( x_axis, 0, smoothed_derivative,
                  where=(smoothed_derivative > 0), color='green', alpha=0.3, label='Tendance haussière' )
ax3.fill_between( x_axis, 0, smoothed_derivative,
                  where=(smoothed_derivative < 0), color='red', alpha=0.3, label='Tendance baissière' )

# Force de la tendance
ax3_twin.plot( x_axis, trend_strength, label='Force de la tendance', color='orange', linestyle=':', linewidth=1.5 )

ax3.set_title( 'Indicateur de Tendance - Somme des Dérivées des Composantes Fourier', fontsize=12, fontweight='bold' )
ax3.set_xlabel( 'Jours de trading (continus)' )
ax3.set_ylabel( 'Dérivée (Vitesse de changement)' )
ax3_twin.set_ylabel( 'Force de la tendance (0-1)', color='orange' )
ax3.grid( True, alpha=0.3 )
ax3.legend( loc='upper left' )
ax3_twin.legend( loc='upper right' )

# Configuration des ticks de l'axe x pour afficher les dates
# Sélectionner quelques points de repère pour les dates
n_ticks = min(10, len(data))  # Maximum 10 ticks
tick_indices = np.linspace(0, len(data)-1, n_ticks, dtype=int)
tick_positions = data['index_continuous'].values[tick_indices]
tick_labels = [data.index[i].strftime('%Y-%m-%d') for i in tick_indices]

ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha='right')

# Masquer les étiquettes x pour les graphiques supérieurs
plt.setp( ax1.get_xticklabels(), visible=False )
plt.setp( ax2.get_xticklabels(), visible=False )

# Ajustement manuel de la mise en page
plt.subplots_adjust( left=0.06, bottom=0.12, right=0.94, top=0.95, hspace=0.25 )

# -----------------------------------------------------------------------------
# Analyse de l'indicateur de tendance
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ANALYSE DE L'INDICATEUR DE TENDANCE")
print("="*60)

# Statistiques sur la tendance actuelle
n_recent = min( 10, len( trend_indicator ) )
recent_trend = trend_indicator[-n_recent:] if n_recent > 0 else []
recent_strength = trend_strength[-n_recent:] if n_recent > 0 else []

if len( recent_trend ) > 0:
    current_trend = "Haussière" if recent_trend[-1] > 0 else "Baissière" if recent_trend[-1] < 0 else "Neutre"
    avg_strength = np.mean(recent_strength)
else:
    current_trend = "Indéterminé"
    avg_strength = 0.0

print( f"Tendance actuelle: {current_trend}")
print( f"Force moyenne ({n_recent} derniers jours): {avg_strength:.3f}")

# Nombre de changements de tendance récents
if len( recent_trend ) > 1:
    trend_changes = np.sum( np.diff( recent_trend ) != 0 )
    print( f"Changements de tendance ({n_recent} derniers jours): {trend_changes}" )
else:
    print( "Pas assez de données pour analyser les changements de tendance" )

# Analyse des composantes dominantes
amplitudes = [coeff['amplitude'] for coeff in coefficients if coeff['k'] > 0]
periods = [coeff['period'] for coeff in coefficients if coeff['k'] > 0]

if amplitudes:
    dominant_idx = np.argmax( amplitudes )
    print( f"Composante dominante: k={dominant_idx+1}, amplitude={amplitudes[dominant_idx]:.2f}, période={periods[dominant_idx]:.1f} jours de trading" )

plt.show()