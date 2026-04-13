""" MACRO ISM (Institute for Supply Management) PRO — Indice synthétique de santé économique FRED (Federal Reserve Bank of St. Louis)

    Pour créer une clef API FRED :
    - https://fred.stlouisfed.org/docs/api/api_key.html

    Les fichiers :
    - macro_ism_pro.csv : données tabulaires (score, momentum, proba de récession, régime, Z-scores par composante)
    - macro_ism_pro.png : graphique multi-panneaux (score, momentum, proba de récession, contributions)

    Sont exportés dans le dossier courant :
    >C:\\Users\\{UserName}\\Documents\\GitHub\\PythonAdvanced

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fredapi import Fred
from matplotlib.patches import Patch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Configuration -----------------------------------------------------------

API_KEY = 'YOUR_API_KEY_HERE'  # Remplacez par votre clé API FRED
START = '1995-01-01'
END = datetime.now().strftime('%Y-%m-%d')
ROLL_ZSCORE = 36 # fenêtre rolling Z-score (mois)
SMOOTH_SHORT = 3 # lissage court (momentum)
SMOOTH_LONG  = 6 # lissage long (régime)

fred = Fred(api_key=API_KEY)

# --- Séries & métadonnées ----------------------------------------------------
#  Chaque entrée : (ticker_fred, label, pct_change_lag, inversion, poids)
#    pct_change_lag : 0 = diff simple, N = pct_change(N)
#    inversion      : True si hausse = négatif pour la macro (chômage, inflation…)
#    poids          : relatifs, normalisés à 1 automatiquement

SERIES_CONFIG = [
    # Ticker          Label              lag  inv    poids
    ('INDPRO',   'Production indus.',    3, False,  0.20),
    ('DGORDER',  'Commandes durables',   3, False,  0.15),
    ('PAYEMS',   'Emploi non-agricole',  1, False,  0.20),
    ('UNRATE',   'Taux de chômage',      1, True,   0.10),
    ('RSAFS',    'Ventes au détail',     3, False,  0.10),
    ('HOUST',    'Mises en chantier',    3, False,  0.10),
    ('UMCSENT',  'Conf. consommateur',   1, False,  0.10),
    ('CPIAUCSL', 'CPI (inflation)',      3, True,   0.05),
]

# Normalisation des poids
total_weight = sum(r[4] for r in SERIES_CONFIG)
SERIES_CONFIG = [(t, l, lag, inv, w / total_weight) for t, l, lag, inv, w in SERIES_CONFIG]

# --- Téléchargement ----------------------------------------------------------

print(f"[{END}] Téléchargement des séries FRED…")
raw = {}
for ticker, label, *_ in SERIES_CONFIG:
    print(f"  • {ticker} ({label})")
    raw[ticker] = fred.get_series(ticker, observation_start=START, observation_end=END)

df_raw = pd.concat(raw.values(), axis=1)
df_raw.columns = [t for t, *_ in SERIES_CONFIG]
df_raw = df_raw.resample('MS').last()   # mensualisation uniforme (début de mois)
df_raw = df_raw.interpolate('time')     # combler les trous ponctuels

# --- Transformations ---------------------------------------------------------

df = pd.DataFrame(index=df_raw.index)

for ticker, label, lag, inv, _ in SERIES_CONFIG:
    s = df_raw[ticker]
    transformed = s.pct_change(lag) if lag > 1 else s.diff(1)
    df[f'T_{ticker}'] = -transformed if inv else transformed

# --- Z-score ROLLING (anti look-ahead) ---------------------------------------

for ticker, *_ in SERIES_CONFIG:
    col = f'T_{ticker}'
    roll = df[col].rolling(ROLL_ZSCORE, min_periods=12)
    df[f'Z_{ticker}'] = (df[col] - roll.mean()) / roll.std()

# --- Score composite pondéré -------------------------------------------------

df['MACRO_ISM'] = sum(
    w * df[f'Z_{ticker}']
    for ticker, _, _lag, _inv, w in SERIES_CONFIG
)

df['MACRO_ISM_S']  = df['MACRO_ISM'].rolling(SMOOTH_LONG).mean()    # lissé (régime)
df['MACRO_ISM_MO'] = df['MACRO_ISM'].rolling(SMOOTH_SHORT).mean()   # court (momentum)

# Momentum = dérivée du score lissé (différence 1 mois)
df['MOMENTUM'] = df['MACRO_ISM_S'].diff(1)

# --- Probabilité de récession (sigmoïde inversée) ----------------------------

df['RECESSION_PROB'] = 1 / (1 + np.exp( 2 * df['MACRO_ISM_S']) )  # plus le score est bas, plus la probabilité de récession est élevée

# --- Régimes (5 états) -------------------------------------------------------
#
#  forte_expansion   > +1.0
#  expansion         > +0.4
#  neutre            > -0.4
#  contraction       > -1.0
#  forte_contraction ≤ -1.0

REGIME_COLORS = {
    'forte_expansion':   ('#1a7a2e', '#c8f5d4'),
    'expansion':         ('#4caf50', '#e8f5e9'),
    'neutre':            ('#90a4ae', '#f0f4f8'),
    'contraction':       ('#ef9a9a', '#fff3f3'),
    'forte_contraction': ('#c62828', '#ffebee'),
}

def classify(score):
    if score > 1.0:  return 'forte_expansion'
    if score > 0.4:  return 'expansion'
    if score > -0.4: return 'neutre'
    if score > -1.0: return 'contraction'
    return 'forte_contraction'

df['REGIME'] = df['MACRO_ISM_S'].apply(classify)
df = df.dropna(subset=['MACRO_ISM_S'])

# --- Export CSV --------------------------------------------------------------

# export_cols = (
#     ['MACRO_ISM', 'MACRO_ISM_S', 'MACRO_ISM_MO', 'MOMENTUM', 'RECESSION_PROB', 'REGIME']
#     + [f'Z_{t}' for t, *_ in SERIES_CONFIG]
# )
# df[ export_cols ].to_csv( 'macro_ism_pro.csv' )
# print( "Export CSV → macro_ism_pro.csv" )

# --- Résumé terminal ----------------------------------------------------------

last = df.iloc[-1]
print(f"""
╔══════════════════════════════════════════════╗
║         MACRO ISM — DERNIER POINT            ║
╠══════════════════════════════════════════════╣
║  Date            : {df.index[-1].strftime('%Y-%m')}
║  Score brut      : {last['MACRO_ISM']:+.3f}
║  Score lissé     : {last['MACRO_ISM_S']:+.3f}
║  Momentum        : {last['MOMENTUM']:+.3f}
║  Proba récession : {last['RECESSION_PROB']:.1%}
║  Régime          : {last['REGIME'].upper()}
╚══════════════════════════════════════════════╝
""")

# --- Graphique multi-panneaux ------------------------------------------------

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45, wspace=0.30)

ax_main  = fig.add_subplot(gs[0, :])   # panneau principal (ligne pleine)
ax_mom   = fig.add_subplot(gs[1, :])   # momentum
ax_rec   = fig.add_subplot(gs[2, :])   # proba récession
ax_comp  = fig.add_subplot(gs[3, :])   # contributions Z par composante

# --- Panneau 1 : Macro ISM ---------------------------------------------------

for i in range( 1, len(df) ):
    regime = df['REGIME'].iloc[i]
    _, bg = REGIME_COLORS[regime]
    ax_main.axvspan(df.index[i-1], df.index[i], color=bg, alpha=0.9, linewidth=0)

ax_main.plot(df.index, df['MACRO_ISM'],    color='#9e9e9e', lw=0.8, alpha=0.6, label='Brut')
ax_main.plot(df.index, df['MACRO_ISM_S'],  color='#1565c0', lw=2.2, label=f'Lissé ({SMOOTH_LONG}m)')
ax_main.axhline( 1.0, ls=':', color='#2e7d32', lw=1)
ax_main.axhline( 0.4, ls='--', color='#4caf50', lw=0.8)
ax_main.axhline( 0,   ls='-',  color='black',   lw=1)
ax_main.axhline(-0.4, ls='--', color='#ef9a9a', lw=0.8)
ax_main.axhline(-1.0, ls=':', color='#c62828', lw=1)

legend_patches = [Patch(facecolor=bg, label=r.replace('_', ' ').title())
                  for r, (_, bg) in REGIME_COLORS.items()]
ax_main.legend(handles=legend_patches + ax_main.get_lines()[:2],
               loc='lower left', fontsize=8, ncol=4)
ax_main.set_title('Macro ISM — Indice composite pondéré (FRED)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Z-score pondéré')

# --- Panneau 2 : Momentum ----------------------------------------------------

colors_mom = ['#c62828' if v < 0 else '#2e7d32' for v in df['MOMENTUM']]
ax_mom.bar(df.index, df['MOMENTUM'], color=colors_mom, width=20, alpha=0.8)
ax_mom.axhline(0, color='black', lw=0.8)
ax_mom.set_title('Momentum du score (Δ lissé 1m)', fontsize=10)
ax_mom.set_ylabel('Δ score')

# --- Panneau 3 : Probabilité de récession ------------------------------------

ax_rec.fill_between(df.index, df['RECESSION_PROB'], alpha=0.6, color='#c62828')
ax_rec.axhline(0.5, ls='--', color='black', lw=1)
ax_rec.set_ylim(0, 1)
ax_rec.set_title('Probabilité de récession (sigmoïde)', fontsize=10)
ax_rec.set_ylabel('Probabilité')
ax_rec.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

# ── Panneau 4 : Contributions par composante ---------------------------------

comp_cols  = [f'Z_{t}' for t, *_ in SERIES_CONFIG]
labels     = [l for _, l, *_ in SERIES_CONFIG]
weights    = [w for *_, w in SERIES_CONFIG]

# Afficher les 36 derniers mois
recent     = df[comp_cols].tail(36)
weighted   = recent.multiply(weights)

bottom_pos = pd.Series(0.0, index=recent.index)
bottom_neg = pd.Series(0.0, index=recent.index)

colors_comp = plt.cm.tab10(np.linspace(0, 1, len(comp_cols)))

for col, label, color in zip(weighted.columns, labels, colors_comp):
    vals = weighted[col]
    pos  = vals.clip(lower=0)
    neg  = vals.clip(upper=0)
    ax_comp.bar(recent.index, pos, bottom=bottom_pos, width=20,
                label=label, color=color, alpha=0.85)
    ax_comp.bar(recent.index, neg, bottom=bottom_neg, width=20,
                color=color, alpha=0.85)
    bottom_pos = bottom_pos + pos
    bottom_neg = bottom_neg + neg

ax_comp.axhline(0, color='black', lw=0.8)
ax_comp.set_title('Contributions pondérées par composante (36 derniers mois)', fontsize=10)
ax_comp.set_ylabel('Contribution au score')
ax_comp.legend(loc='upper left', fontsize=7, ncol=4)

# --- Titre global ---------------------------------------------------------------

fig.suptitle( f'Macro ISM Pro | Données FRED | Généré le {END}',
             fontsize=11, style='italic', color='#555')

# plt.savefig('macro_ism_pro.png', dpi=150, bbox_inches='tight')
# print("Export graphique → macro_ism_pro.png")
plt.show()