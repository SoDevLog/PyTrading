""" 
    Manipulation des phases des composantes de Fourier
    pour créer des indicateurs avancés et prédictions de tendance

    Le décalage de phase doit être proportionnel à la fréquence (k) pour que les hautes fréquences soient plus avancées que les basses fréquences.
    Ainsi, les hautes fréquences (petites périodes) réagissent plus rapidement aux changements de prix.
    Cela permet à l'indicateur avancé d'anticiper les mouvements du prix.
    Inversement, pour un indicateur retardé, les basses fréquences sont plus retardées que les hautes fréquences.
    Cela permet à l'indicateur retardé de confirmer les mouvements du prix après coup.
    Le suiveur de tendance renforce les basses fréquences (tendances longues) et atténue les hautes fréquences (bruit).
    L'oscillateur supprime la composante DC et les basses fréquences pour mettre en évidence les cycles de trading.

    Claude se trompe
    J'ai créé fourier_inverse pour reconstruire le signal original sans appliquer de décalage de phase.
    Et j'ai trouvé l'erreur dans la méthode de reconstruction du signal.
    La formule correcte pour la reconstruction est:
        result += amplitude * np.cos( 2 * np.pi * k * t / self.n - phase )

    réglage du trend_follower par : K_MODULATION
    
    Idicateurs de tendance gradient:
    - divergence_signal : signaux de divergence entre indicateur avancé et prix
    - leading_trend_norm : tendance de l'indicateur avancé (normalisée)

    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.
    
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

date_start = '2024-10-01'
date_end = datetime.now()
interval_dates = '1d'

# Nombre de termes de la décomposition en séries de fourier
N_TERMS = 9 + 1
N_LEAD_DAYS = 3 # Nombre de jours pour l'indicateur à avance de phase
N_LAG_DAYS = 3  # Nombre de jours pour l'indicateur à retard de phase

class FourierAnalyzer:
    def __init__( self, data, n_terms ):
        self.data = data
        self.n_terms = n_terms
        self.n = len(data)
        self.coefficients = self._compute_coefficients()
        
    def _compute_coefficients( self ):
        """Calcule les coefficients de Fourier originaux"""
        coefficients = []
        t = np.arange(self.n)
        
        for k in range(self.n_terms):
            a_k = 2/self.n * np.sum( self.data * np.cos(2 * np.pi * k * t / self.n) )
            b_k = 2/self.n * np.sum( self.data * np.sin(2 * np.pi * k * t / self.n) )

            # Conversion en amplitude et phase
            phase = np.arctan2( b_k, a_k )
            amplitude = np.sqrt( a_k**2 + b_k**2 )
            
            if k != 0:
                period = self.n / k
                frequency = 1 / period
            else:
                period = np.inf
                frequency = 0
                
            coefficients.append({
                'k': k,
                'a_k': a_k,
                'b_k': b_k,
                'amplitude': amplitude,
                'phase': phase,
                'period': period,
                'frequency': frequency
            })

            print(f"{k} ampli: {amplitude:.2f} cos: {a_k:.2f} sin: {b_k:.2f} phi: {np.degrees(phase):.2f}° period: {period:.2f}")

        return coefficients
    
    def reconstruct_signal( self, phase_shifts=None, amplitude_weights=None, t_range=None ):
        """
        Reconstruit le signal avec des modifications optionnelles
        
        Parameters:
        - phase_shifts: dict {k: shift_in_radians} pour modifier les phases
        - amplitude_weights: dict {k: weight} pour pondérer les amplitudes
        - t_range: range de temps (par défaut: longueur originale)
        """
        if t_range is None:
            t = np.arange(self.n)
        else:
            t = t_range
            
        result = np.zeros(len(t))
        
        for coeff in self.coefficients:
            k = coeff['k']
            a_k = coeff['a_k']
            amplitude = coeff['amplitude']
            phase = coeff['phase']
            
            # Appliquer les modifications de phase si spécifiées
            if phase_shifts and k in phase_shifts:
                phase += phase_shifts[k]
                
            # Appliquer les pondérations d'amplitude si spécifiées
            if amplitude_weights and k in amplitude_weights:
                amplitude *= amplitude_weights[k]
            
            # Ajouter la composante modifiée
            if k == 0:
                result += a_k / 2  # moyenne (composante DC)
            else:
                result += amplitude * np.cos( 2 * np.pi * k * t / self.n - phase )
                
        return result
    
    def fourier_inverse( self ):

        t = np.arange( self.n )
        result = np.zeros( len(t) )
        
        for coeff in self.coefficients:
            k = coeff['k']
            a_k = coeff['a_k']
            #b_k = coeff['b_k']
            amplitude = coeff['amplitude']
            phase = coeff['phase']
            
            if k == 0:
                result += a_k / 2  # moyenne (composante DC)
            else:
                result += amplitude * np.cos( 2 * np.pi * k * t / self.n - phase )
                
        return result

    def create_leading_indicator( self, lead_days=5 ):
        """
        Crée un indicateur avancé en décalant les phases selon la fréquence
        Les hautes fréquences sont plus avancées que les basses fréquences
        """
        phase_shifts = {}
        for coeff in self.coefficients:
            k = coeff['k']
            if k > 0:
                # Plus k est grand (haute fréquence), plus le décalage est important
                # Décalage en fonction de la période pour avoir un effet proportionnel
                phase_advance = (2 * np.pi * lead_days * k) / self.n
                phase_shifts[k] = phase_advance
                
        return self.reconstruct_signal( phase_shifts=phase_shifts )

    def create_lagging_indicator( self, lag_days=5 ):
        """Crée un indicateur retardé"""
        phase_shifts = {}
        for coeff in self.coefficients:
            k = coeff['k']
            if k > 0:
                phase_lag = -(2 * np.pi * lag_days * k) / self.n
                phase_shifts[k] = phase_lag

        return self.reconstruct_signal( phase_shifts=phase_shifts )

    # -------------------------------------------------------------------------
    
    def amplitude_corrector( self, k_factor, k_default=0.0 ):
    
        amplitude_weights = {}
        
        for coeff in self.coefficients:
            k = coeff['k']
            if k in k_factor:
                amplitude_weights[k] = k_factor[k]
            else:
                amplitude_weights[k] = k_default
                
        return amplitude_weights
    
    # -------------------------------------------------------------------------

    def get_phase_analysis( self ):
        """Analyse des phases pour identifier les cycles dominants"""
        analysis = []
        for coeff in self.coefficients[1:]:  # Exclure k=0
            k = coeff['k']
            phase_deg = np.degrees(coeff['phase'])
            
            # Interprétation de la phase
            if -45 <= phase_deg <= 45:
                phase_interp = "Phase croissante"
            elif 45 < phase_deg <= 135:
                phase_interp = "Phase culminante"
            elif 135 < phase_deg <= 180 or -180 <= phase_deg <= -135:
                phase_interp = "Phase décroissante"
            else:
                phase_interp = "Phase basse"
                
            analysis.append({
                'composante': k,
                'periode_jours': coeff['period'],
                'amplitude': coeff['amplitude'],
                'phase_deg': phase_deg,
                'interpretation': phase_interp
            })
            
        return analysis

def normalize_signal( y_data, min_val, max_val ):
    """Normalise le signal entre min_val et max_val"""
    y_min = y_data.min()
    y_max = y_data.max()
    if y_max - y_min == 0:
        return np.full_like( y_data, (min_val + max_val) / 2 )
    return ((y_data - y_min) / (y_max - y_min)) * (max_val - min_val) + min_val

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

# Continus graph
data['index_continuous'] = np.arange(len(data))
x_axis = data['index_continuous'].values

n_ticks = min(10, len(data))  # Maximum 10 ticks
tick_indices = np.linspace(0, len(data)-1, n_ticks, dtype=int)
tick_positions = data['index_continuous'].values[tick_indices]
tick_labels = [data.index[i].strftime('%Y-%m-%d') for i in tick_indices]

# -----------------------------------------------------------------------------
# Initialisation de l'analyseur Fourier
# -----------------------------------------------------------------------------

print( "\n" + "="*70 )
print(f"ANALYSE DE FOURIER SUR N points : {len(data)}")
print( "="*70 )

analyzer = FourierAnalyzer( data['Close'].values, N_TERMS )

# Reconstruction du signal original
original_reconstruction = analyzer.fourier_inverse()

# Création des indicateurs avec manipulation des phases
leading_indicator = analyzer.create_leading_indicator( lead_days=N_LEAD_DAYS )
lagging_indicator = analyzer.create_lagging_indicator( lag_days=N_LAG_DAYS )

# -------------------------------------------------------------------------
# Analyseur de tendance
# -------------------------------------------------------------------------
# Correction des amplitudes pour le suiveur de tendance
# Renforcer les basses fréquences et atténuer les hautes fréquences
#
K_MODULATION = {
    0: 1.0,   # Composante DC
    1: 1.3,   # Basses fréquences - tendances longues 
    2: 1.3,
    3: 1.3,   # Moyennes fréquences
    4: 0.8,
    5: 0.3,   # Hautes fréquences
    6: 0.2,   
    7: 0.2,
    8: 0.2,
    9: 0.2
}
weights = analyzer.amplitude_corrector( K_MODULATION, k_default=0.0 )
trend_follower = analyzer.reconstruct_signal( amplitude_weights=weights )

# Supression de la composante DC et des basses fréquences pour l'oscillateur
K_MODULATION_OSC = {
    0: 0.0,   # Supression de la composante DC
    1: 0.1,   # Basses fréquences
    2: 0.1,
    3: 0.1,
    4: 1.6,
    5: 1.2,
    5: 1.2,
    6: 1.2,
    7: 1.0
}

weights = analyzer.amplitude_corrector( K_MODULATION_OSC, k_default=1.0 )
oscillator = analyzer.reconstruct_signal( amplitude_weights=weights )

# Normalisation pour l'affichage
min_price = data['Close'].min()
max_price = data['Close'].max()

leading_norm = normalize_signal( leading_indicator, min_price, max_price )
lagging_norm = normalize_signal( lagging_indicator, min_price, max_price )
trend_norm = normalize_signal( trend_follower, min_price, max_price )
oscillator_norm = normalize_signal( oscillator, min_price * 0.95, max_price * 1.05 )

# -----------------------------------------------------------------------------
# Création des graphiques
# -----------------------------------------------------------------------------

#fig = plt.figure( figsize=(16, 14) )
fig = plt.figure( figsize=(16, 9) )

params = {"left": 0.05, "bottom": 0.08, "right": 0.95, "top": 0.95, "hspace": 0.3}
#grid_spec = gridspec.GridSpec( 4, 1, height_ratios=[3, 2, 2, 2], **params)
grid_spec = gridspec.GridSpec( 3, 1, height_ratios=[3, 2, 2], **params)

ax1 = fig.add_subplot( grid_spec[0] )
#ax2 = fig.add_subplot( grid_spec[1], sharex=ax1 )
ax3 = fig.add_subplot( grid_spec[1], sharex=ax1 )
ax4 = fig.add_subplot( grid_spec[2], sharex=ax1 )

# Graphique 1: Prix et reconstructions
ax1.plot( x_axis, data['Close'], label='Prix original', color='black', linewidth=2, alpha=0.7 )
ax1.plot( x_axis, normalize_signal(original_reconstruction, min_price, max_price), 
         label='Reconstruction Fourier', color='blue', linewidth=1.5, linestyle='--' )
_label=f"Indicateur à avance de phase (+{N_LEAD_DAYS}j)"
ax1.plot( x_axis, leading_norm, label=_label, color='green', linewidth=1.5 )
_label=f"Indicateur à retard de phase (-{N_LAG_DAYS}j)"
ax1.plot( x_axis, lagging_norm, label=_label, color='red', linewidth=1.5 )
ax1.plot( x_axis, trend_norm, label="Tendance par Correcteur d'amplitude", color='purple', linewidth=2 )

ax1.set_title( f"{company['name']} [data={len(data)}] - Modulation des phases de Fourier [k={N_TERMS-1}]" ) #, fontsize=14, fontweight='bold' )
ax1.legend( loc='upper left' )
ax1.grid( True, alpha=0.3 )
ax1.set_ylabel( 'Prix (€)' )

# Graphique 2: Suiveur de tendance
# ax2.plot( x_axis, data['Close'], label='Prix original', color='black', alpha=0.5, linewidth=1 )
# ax2.plot( x_axis, trend_norm, label='Suiveur de tendance (BF renforcées)', color='purple', linewidth=2 )
# ax2.set_title( 'Suiveur de Tendance - Basses Fréquences Renforcées', fontsize=12 )
# ax2.legend()
# ax2.grid( True, alpha=0.3 )
# ax2.set_ylabel( 'Prix (€)' )

# Graphique 3: Oscillateur
ax3.plot( x_axis, oscillator_norm, label='Oscillateur (HF renforcées)', color='orange', linewidth=1.5 )
ax3.axhline( y=np.mean(oscillator_norm), color='gray', linestyle=':', alpha=0.7 )
ax3.fill_between( x_axis, np.mean(oscillator_norm), oscillator_norm,
                 where=(oscillator_norm > np.mean(oscillator_norm)), 
                 color='green', alpha=0.3, label='Zone positive')
ax3.fill_between( x_axis, np.mean(oscillator_norm), oscillator_norm,
                 where=(oscillator_norm < np.mean(oscillator_norm)),
                 color='red', alpha=0.3, label='Zone négative')
ax3.set_title( f"Oscillateur Rapide - Modulation d'amplitude mod:{K_MODULATION_OSC}", fontsize=12 )
ax3.legend()
ax3.grid( True, alpha=0.3 )
ax3.set_ylabel( 'Oscillation' )

# Graphique 4: Analyse des signaux de trading
# Création de signaux basés sur les croisements
leading_trend = np.gradient( leading_norm )
price_trend = np.gradient( data['Close'].values )

# Normalisation des tendances pour comparaison équitable
def normalize_trend( trend, window=5 ):
    """Normalise et lisse la tendance pour comparaison"""
    # Lissage avec moyenne mobile
    trend_smooth = pd.Series(trend).rolling(window=window, center=True).mean().bfill().ffill().values
    # Normalisation par l'écart-type pour avoir des ordres de grandeur similaires
    trend_std = np.std(trend_smooth)
    if trend_std > 0:
        return trend_smooth / trend_std
    return trend_smooth

leading_trend_norm = normalize_trend( leading_trend )
price_trend_norm = normalize_trend( price_trend )

# Signal de divergence
divergence_threshold = 1  # Seuil pour éviter les faux signaux
strong_leading = np.abs( leading_trend_norm ) > divergence_threshold
strong_price = np.abs( price_trend_norm ) > divergence_threshold
significant_move = strong_leading | strong_price

# Divergence quand les signes sont opposés ET qu'il y a un mouvement significatif
divergence_signal = ( np.sign( leading_trend_norm ) != np.sign( price_trend_norm ) ) & significant_move

ax4.plot( x_axis, leading_trend_norm, label='Tendance indicateur avancé (normalisée)', color='green', linewidth=1.5 )
ax4.plot( x_axis, price_trend_norm, label='Tendance prix (normalisée)', color='blue', linewidth=1.5 )
ax4.scatter( x_axis[divergence_signal], np.zeros( np.sum(divergence_signal) ), 
           color='red', s=50, alpha=0.8, label=f'Signaux divergence ({np.sum(divergence_signal)})', zorder=5)

# Ajouter des zones de divergence
ax4.fill_between(x_axis, -2, 2, where=divergence_signal, 
                color='yellow', alpha=0.6, label='Zones de divergence')

ax4.axhline( y=0, color='black', linestyle='-', alpha=0.5 )
ax4.axhline( y=divergence_threshold, color='gray', linestyle='--', alpha=0.5, label=f'Seuil ±{divergence_threshold}' )
ax4.axhline( y=-divergence_threshold, color='gray', linestyle='--', alpha=0.5 )
ax4.set_ylim(-3, 3)
ax4.set_title( 'Signaux de Trading - Divergences entre Indicateur Avancé et Prix', fontsize=12 )

ax4.legend()
ax4.grid( True, alpha=0.3 )
ax4.set_ylabel( 'Dérivée' )
ax4.set_xlabel( 'Date' )
ax4.set_xticks( tick_positions )
ax4.set_xticklabels( tick_labels, rotation=45, ha='right' )

# Masquer les étiquettes x pour les graphiques supérieurs
plt.setp( ax1.get_xticklabels(), visible=False )
#plt.setp( ax2.get_xticklabels(), visible=False )
plt.setp( ax3.get_xticklabels(), visible=False )
plt.setp( ax4.get_xticklabels(), rotation=45, ha='right' )

plt.subplots_adjust( **params )

# ------------------------------------------------------------------------
# Création des lignes verticales pour le curseur (initialement invisibles)
# ------------------------------------------------------------------------
cursor_lines = []
for ax in [ax1, ax3, ax4]:
    line = ax.axvline( x=x_axis[0], color='black', linestyle='--', linewidth=0.9, alpha=0.7, visible=False )
    cursor_lines.append( line )

# Annotation pour afficher la date
annotation = ax1.annotate('', xy=(0, 0), xytext=(10, 10),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                         visible=False)

def on_mouse_move( event ):
    # Vérifier si la souris est dans un des axes
    for i, ax in enumerate([ax1, ax3, ax4]):
        if event.inaxes == ax and event.xdata is not None:
            # Afficher toutes les lignes verticales
            for line in cursor_lines:
                line.set_xdata([event.xdata, event.xdata])
                line.set_visible(True)
            
            # Mettre à jour l'annotation
            try:
                # Trouver l'index le plus proche
                idx = np.argmin(np.abs(x_axis - event.xdata))
                date_str = x_axis[idx].strftime('%Y-%m-%d')
                price = data['Close'].iloc[idx]
                
                annotation.xy = (event.xdata, event.ydata)
                annotation.set_text(f'{date_str}\nPrix: {price:.2f}€')
                annotation.set_visible(True)
            except:
                pass
            
            fig.canvas.draw_idle()
            return
    
    # Si la souris n'est dans aucun axe, masquer les lignes et l'annotation
    for line in cursor_lines:
        line.set_visible(False)
    annotation.set_visible(False)
    fig.canvas.draw_idle()

# Connecter l'événement de mouvement de la souris
fig.canvas.mpl_connect( 'motion_notify_event', on_mouse_move )

# -----------------------------------------------------------------------------
# Analyse des phases
# -----------------------------------------------------------------------------

print( "\n" + "="*70 )
print( "ANALYSE DES PHASES DES COMPOSANTES DE FOURIER" )
print( "="*70 )

phase_analysis = analyzer.get_phase_analysis()
for analysis in phase_analysis:
    print( f"Composante {analysis['composante']:2d} | "
           f"Période: {analysis['periode_jours']:6.1f}j | "
           f"Amplitude: {analysis['amplitude']:6.2f} | "
           f"Phase: {analysis['phase_deg']:6.1f}° | "
           f"{analysis['interpretation']}" )

print( "\n" + "="*70 )
print( "STRATÉGIES DE MANIPULATION DES PHASES" )
print( "="*70 )
print( "1. INDICATEUR AVANCÉ: Décalage positif des phases" )
print( f"   → Anticipe les mouvements de {N_LEAD_DAYS} jours" )
print( f"   → Utile pour les signaux d'entrée précoces" )

print( "\n2. INDICATEUR RETARDÉ: Décalage négatif des phases" )
print( f"   → Confirme les mouvements après {N_LAG_DAYS} jours" )
print( f"   → Utile pour confirmer les tendances" )

print( "\n3. SUIVEUR DE TENDANCE: Pondération des amplitudes" )
print( "   → Renforce les basses fréquences (tendances longues)" )
print( "   → Atténue les hautes fréquences (bruit)" )

print( "\n4. OSCILLATEUR: Suppression de la tendance" )
print( "   → Supprime la composante DC et les BF" )
print( "   → Met en évidence les cycles de trading" )

print( "\n5. SIGNAUX DE DIVERGENCE:" )
divergence_count = np.sum( divergence_signal )
print( f"   → {divergence_count} signaux de divergence détectés" )
print( "   → Quand l'indicateur avancé diverge du prix" )

print( f"    K_MODULATION: {K_MODULATION}" )
print( f"K_MODULATION_OSC: {K_MODULATION_OSC}" )

plt.show()