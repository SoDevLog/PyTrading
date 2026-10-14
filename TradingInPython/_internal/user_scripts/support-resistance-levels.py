""" Détection de Supports / Résistances dynamiques guidée par le profil de volume

    1. Profil de volume Acheteurs / Vendeurs (sur les 'lookback' dernières barres)
       - Le volume de chaque bougie est réparti sur les lignes de prix touchées par son range
         High-Low, au prorata du recouvrement (et non pas affecté à un seul prix).
       - Acheteurs / Vendeurs sont ESTIMÉS d'après la position de la clôture dans le range VRAI
         (de min(Low, clôture précédente) à max(High, clôture précédente), donc gap inclus) :
         clôture près du haut → surtout acheteurs, près du bas → surtout vendeurs.
         Ce n'est pas du vrai bid/ask : il faudrait le flux de transactions avec le sens de
         l'agresseur (footprint), absent des datas d'un OHLCV.

    2. Supports / Résistances dynamiques
       a) pivots High/Low confirmés (swing) : connus seulement 'swing' barres après leur formation
       b) autour du pivot, recherche du meilleur noeud du profil (reconstruit à la date de
          confirmation du pivot) avec un score d'évidence :
            force du noeud / POC, relief local, dominance RELATIVE acheteurs (support) ou vendeurs
            (résistance), proximité du pivot. La dominance est mesurée par rapport au biais global
            du profil : en tendance, presque toutes les clôtures sont près du même côté, ce qui
            fait pencher tout le profil et ne dit rien d'un noeud en particulier
       c) le niveau final = pivot attiré vers le noeud (plus l'évidence est forte, plus l'attraction
          est forte) : le prix propose le niveau, le volume l'affine
          Le niveau est une ZONE, pas une ligne précise : elle s'étend du pivot brut au niveau
          final (largeur minimale 0.1 x ATR). La ligne affichée n'en est que le prix de référence
       d) score final 0-100 (évidence + absorption + volume du pivot + rejet par la mèche)
       e) cycle de vie : un niveau reste actif jusqu'à une clôture au-delà du bord lointain de sa
          zone (bord bas d'un support, bord haut d'une résistance) + 'break_atr' x ATR ;
          il est alors CASSÉ et change de rôle (support → résistance, ou l'inverse) : un niveau
          retourné naît à la barre de cassure, au même prix, avec le même score. Un niveau ne se
          retourne qu'une fois : s'il est cassé à son tour, il s'arrête ;
          fusion des niveaux de même sens proches ('merge_atr') ; au-delà de 'max_levels'
          le plus faible est supprimé

    Aucune donnée future n'est utilisée : un pivot n'est évalué qu'à sa barre de confirmation.

    Graphique : bougies + zones (bandes) et niveaux (trait plein = actif, tirets = rôle inversé, pointillé = cassé),
    croix à chaque cassure, score du niveau dans un marqueur sur le pivot, prix exact à
    l'extrémité des niveaux actifs.
    Le profil de volume sert uniquement au calcul, il n'est pas dessiné.

    Axe X : positionnel (entier séquentiel), pas temporel - même principe que le script BSI :
    une position par barre tradée, donc pas de trou le week-end. Les vraies dates sont
    réinjectées via un FuncFormatter (ticks) et format_xdata (survol NavigationToolbar2Tk).

    Entrée : DataFrame avec les colonnes Open, High, Low, Close, Volume
    Dépendances : pandas, numpy, matplotlib, yfinance, digitsignalprocessing.indicators (atr_rolling)
"""
import sys
import numpy as np
import pandas as pd
import yfinance
import matplotlib
import matplotlib.style
matplotlib.style.use("seaborn-v0_8-notebook")
import matplotlib.ticker as mticker

from dataclasses import dataclass
from typing import Optional
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pathlib import Path
base = Path(__file__).resolve().parent.parent
sys.path.append( str(base) )

import digitsignalprocessing.indicators as dsp
from figure.candlestick_ohlc import candlestick_ohlc

# ---------------------------------------------
# THEME - True = fond noir / False = fond blanc
# ---------------------------------------------

DARK_MODE: bool = False

_THEME = {
    True: { # --- dark ---
        "bg":           "#0d1117",
        "bg_legend":    "#1a1a1a",
        "text":         "#ffffff",
        "text_muted":   "#aaaaaa",
        "grid":         "#333333",
        "bull":         "#26a69a",
        "bear":         "#ef5350",
    },
    False: { # --- light ---
        "bg":           "#ffffff",
        "bg_legend":    "#f5f5f5",
        "text":         "#111111",
        "text_muted":   "#555555",
        "grid":         "#A5A5A5",
        "bull":         "#0a8a7e",
        "bear":         "#c62828",
    },
}

# ---------------------------------------------
# PARAMÈTRES
# ---------------------------------------------

@dataclass
class SR_Settings:
    # --- Profil de volume (reconstruit à la confirmation de chaque pivot) --- #
    lookback:       int   = 200    # nombre de barres du profil
    rows:           int   = 40     # nombre de lignes de prix
    # --- Moteur de pivots guidé par le profil --- #
    swing:          int   = 10     # confirmation gauche/droite des pivots
    atr_period:     int   = 14
    search_atr:     float = 1.5    # rayon de recherche d'un noeud autour du pivot (x ATR)
    min_node:       float = 0.25   # volume mini du noeud / POC
    min_evidence:   float = 0.45   # évidence minimale (0-1)
    pull:           float = 0.7    # attraction du pivot vers le noeud (0 = pivot brut, 1 = noeud)
    min_score:      float = 45.0   # score final minimal (0-100)
    max_levels:     int   = 8      # nombre max de niveaux actifs
    merge_atr:      float = 0.5    # fusion des niveaux de même sens plus proches que ça (x ATR)
    break_atr:      float = 0.3    # tampon d'invalidation sur la clôture (x ATR)
    flip_levels:    bool  = True   # un niveau cassé change de rôle (support <-> résistance), une seule fois
    show_broken:    bool  = True   # affiche aussi les niveaux cassés (pointillés + croix à la cassure)
    show_cross:     bool  = False  # affiche les croisements de niveaux

# Pondérations
# noeud, relief local, dominance directionnelle, proximité du pivot
_W_EVIDENCE = np.array( [0.35, 0.25, 0.20, 0.20] )
# évidence profil, absorption, volume du pivot, rejet mèche
_W_SCORE = np.array( [0.55, 0.15, 0.15, 0.15] )
# largeur minimale d'une zone (x ATR), pour qu'un 'pull' faible ne redonne pas une simple ligne
_MIN_ZONE_ATR = 0.1

@dataclass
class Level:
    is_support:  bool
    pivot_idx:   int           # barre du pivot (début de la ligne)
    pivot_price: float         # prix brut du pivot (High ou Low)
    price:       float         # niveau final, attiré vers le noeud du profil (ligne affichée)
    lo:          float         # bord bas de la zone
    hi:          float         # bord haut de la zone
    score:       float         # 0-100
    end_idx:     Optional[int] = None   # barre de cassure, None = niveau actif
    flipped:     bool = False  # True = niveau né de la cassure d'un niveau de rôle opposé

# ---------------------------------------------
# 1. PROFIL DE VOLUME ACHETEURS / VENDEURS
# ---------------------------------------------

def volume_profile( high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    prev_close: np.ndarray, volume: np.ndarray, rows: int ):
    """
    Profil acheteurs / vendeurs sur les barres fournies.
    Retourne (edges, buy, sell) : bords des lignes (rows + 1) et volumes estimés par ligne.
    """
    edges = np.linspace(low.min(), high.max(), rows + 1)
    bot, top = edges[:-1], edges[1:]

    rng  = high - low
    flat = rng < 1e-12 # bougie sans range : High == Low
    rng  = np.where( flat, 1.0, rng )

    # part du range de chaque bougie qui recouvre chaque ligne  (barres x lignes)
    overlap = np.minimum(high[:, None], top) - np.maximum(low[:, None], bot)
    share = np.clip( overlap, 0, None) / rng[:, None] # plus rapide que np.round(overlap / rng[:, None], 6)

    # bougie plate : tout son volume sur la ligne qui contient la clôture
    row_close = np.clip( np.searchsorted( edges, close, side="right" ) - 1, 0, rows - 1 )
    share[flat] = 0.0
    share[flat, row_close[flat]] = 1.0

    # sens du volume : position de la clôture dans le range VRAI (gap avec la clôture précédente inclus)
    # clôture près du haut -> acheteurs ; aucun mouvement du tout -> 50/50
    tr_low  = np.minimum( low, prev_close )
    tr_rng  = np.maximum( high, prev_close ) - tr_low
    tr_flat = tr_rng < 1e-12
    buy_ratio = np.where( tr_flat, 0.5, (close - tr_low) / np.where( tr_flat, 1.0, tr_rng ) )

    vol  = volume[:, None] * share
    buy  = (vol * buy_ratio[:, None]).sum(axis=0)
    sell = (vol * (1.0 - buy_ratio)[:, None]).sum(axis=0)

    return edges, buy, sell

# ---------------------------------------------
# 2. SUPPORTS / RÉSISTANCES DYNAMIQUES
# ---------------------------------------------

def _best_node( edges, buy, sell, price, atr_v, is_support, cfg: SR_Settings ):
    """
    Meilleur noeud du profil autour du pivot.
    Retourne (évidence 0-1, prix du noeud) ou None si aucune ligne ne qualifie.
    """
    total = buy + sell
    poc = total.max()
    centers = (edges[:-1] + edges[1:]) / 2
    reach = cfg.search_atr * atr_v
    if poc <= 0 or reach <= 0:
        return None

    safe  = np.maximum(total, 1e-12)
    node  = total / poc # force du noeud vs POC

    kernel = np.array([1, 1, 0, 1, 1], dtype=float) # 2 lignes de chaque côté
    neigh  = np.convolve(total, kernel, "same") / np.convolve(np.ones_like(total), kernel, "same")
    peak   = np.clip((total - neigh) / safe, 0, 1)  # relief local

    # dominance RELATIVE : excès local d'acheteurs (support) ou de vendeurs (résistance) par rapport
    # au biais global du profil, recentré sur 0.5 (0.5 = ligne dans la moyenne du profil)
    tilt      = buy.sum() / total.sum()
    excess    = buy / safe - tilt if is_support else sell / safe - (1 - tilt)
    direction = np.clip( 0.5 + excess, 0, 1 )
    proximity = np.clip(1 - np.abs(centers - price) / reach, 0, 1)

    evidence = _W_EVIDENCE @ np.vstack( [node, peak, direction, proximity] )
    eligible = ( np.abs(centers - price) <= reach) & (node >= cfg.min_node )
    if not eligible.any():
        return None

    r = np.where( eligible, evidence, -1.0 ).argmax()

    return evidence[r], centers[r]

# -----------------------------------------------------------------------------

def _pivot_quality( o, h, l, c, v, i, swing, is_support ):
    """Absorption, volume relatif du pivot et rejet par la mèche — trois valeurs entre 0 et 1."""
    lo, hi = max(0, i - swing), min(len(c), i + swing + 1)
    recent = v[max(0, i - 20):i]
    avg = max(recent.mean() if len(recent) else v[i], 1e-12)

    rng = np.maximum(h[lo:hi] - l[lo:hi], 1e-12)
    body = np.abs(c[lo:hi] - o[lo:hi])
    
    # gros volume + petit corps = absorption
    absorption = np.max( np.clip(v[lo:hi] / avg / 2, 0, 1) * (1 - body / rng) )

    volume_rel = min(v[i] / avg / 2, 1.0)
    wick = (min(o[i], c[i]) - l[i]) if is_support else (h[i] - max(o[i], c[i]))
    wick_rel = np.clip(wick / max(h[i] - l[i], 1e-12), 0, 1)
    return absorption, volume_rel, wick_rel

# -----------------------------------------------------------------------------

def _is_broken( lv: Level, close: float, buffer: float ) -> bool:
    """Cassé = clôture au-delà du bord lointain de la zone (bas pour un support, haut pour une résistance)."""
    return close < lv.lo - buffer if lv.is_support else close > lv.hi + buffer

# -----------------------------------------------------------------------------

def dynamic_levels( df: pd.DataFrame, cfg: Optional[SR_Settings] = None ) -> list:
    """
    Calcule tous les niveaux S/R sur l'historique (actifs et invalidés).
    Rien n'utilise de donnée future : un pivot n'est évalué qu'à sa barre de confirmation,
    avec le profil et l'ATR connus à cette date.
    Un niveau actif a end_idx = None ; sinon end_idx est la barre de sa cassure.
    Un niveau cassé reste dans la liste ; avec cfg.flip_levels il donne naissance à un niveau
    de rôle opposé (flipped=True) qui démarre à cette barre, au même prix.
    """
    cfg = cfg or SR_Settings()
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df)
    s = cfg.swing
    atr_np = dsp.atr_rolling( df, cfg.atr_period )
    pc = np.r_[c[:1], c[:-1]] # clôture précédente (1re barre : sa propre clôture, donc pas de gap)

    # --- a) pivots confirmés : niveaux candidats, rangés par barre de confirmation --- #
    pending = {}
    for i in range(s, n - s):
        for is_support, hit in ((True,  l[i - s:i + s + 1].argmin() == s),
                                (False, h[i - s:i + s + 1].argmax() == s)):
            if not hit:
                continue
            t = i + s # barre de confirmation
            lo = max(0, t - cfg.lookback + 1)
            price = l[i] if is_support else h[i]
            edges, buy, sell = volume_profile( h[lo:t + 1], l[lo:t + 1], c[lo:t + 1], pc[lo:t + 1], v[lo:t + 1], cfg.rows )

            # --- b) meilleur noeud du profil autour du pivot --- #
            best = _best_node( edges, buy, sell, price, atr_np[t], is_support, cfg )
            if best is None or best[0] < cfg.min_evidence:
                continue
            evidence, node_price = best

            # --- d) score final --- #
            quality = _pivot_quality( o, h, l, c, v, i, s, is_support )
            score = 100 * float( _W_SCORE @ np.array([evidence, *quality]) )
            if score < cfg.min_score:
                continue

            # --- c) le profil tire le pivot vers le noeud --- #
            level = price + cfg.pull * evidence * (node_price - price)
            half = _MIN_ZONE_ATR * atr_np[t] / 2
            lo, hi = min( price, level - half ), max( price, level + half ) # zone : pivot brut <-> niveau final
            pending.setdefault( t, []).append( Level(is_support, i, price, level, lo, hi, score) )

    # --- e) cycle de vie : invalidation, fusion, plafond --- #
    active, levels = [], []
    for t in range(n):
        
        for lv in active[:]:
            if _is_broken( lv, c[t], cfg.break_atr * atr_np[t] ):
                lv.end_idx = t
                active.remove(lv)
                
                # changement de rôle, traité plus bas comme un nouveau niveau
                if cfg.flip_levels and not lv.flipped: 
                    pending.setdefault( t, [] ).append(
                        Level( not lv.is_support, t, lv.price, lv.price, lv.lo, lv.hi, lv.score, flipped=True ) )

        for lv in pending.get( t, [] ):
            if _is_broken( lv, c[t], cfg.break_atr * atr_np[t] ):
                continue # déjà percé à sa naissance
            
            twin = next((a for a in active
                         if a.is_support == lv.is_support
                         and abs(a.price - lv.price) <= cfg.merge_atr * atr_np[t]), None)
            
            if twin is not None:
                if twin.score >= lv.score:
                    continue
                active.remove( twin )
                levels.remove( twin )
                
            active.append( lv )
            levels.append( lv )

        while len( active ) > cfg.max_levels:
            weakest = min( active, key=lambda a: a.score )
            active.remove( weakest )
            levels.remove( weakest )

    return levels

# --------------------------------------------------
# GRAPHIQUE - Supports / Résistances dynamiques
# --------------------------------------------------

def _fmt_price( p: float ) -> str:
    """Prix lisible : 4 décimales sous 10 (forex, petits cours), 2 décimales au-delà."""
    return f"{p:.4f}" if p < 10 else f"{p:.2f}"

def plot_supports_resistances(
        ticker: str = "AAPL",
        title: str = "",
        period: str = "1y",
        interval: str = "1d",
        df = None,
        cfg: Optional[SR_Settings] = None
    ) -> Figure:

    T   = _THEME[DARK_MODE] # palette active selon la constante
    cfg = cfg or SR_Settings()

    if df is None or df.empty:
        df = yfinance.download( ticker, period=period, interval=interval,
            auto_adjust=True, progress=False )

    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}")

    # Aplatir les colonnes si MultiIndex (yfinance ≥ 0.2)
    if isinstance( df.columns, pd.MultiIndex ):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df["Volume"].sum() <= 0:
        raise ValueError(f"{ticker} : pas de volume, détection impossible")

    o, h, l, c = ( df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close") )
    n = len(df)
    levels = dynamic_levels( df, cfg )
    gap = max( 6, round( 0.06 * n ) ) # place pour l'étiquette prix des niveaux actifs

    # --- Axe X positionnel (voir docstring) --- #
    idx   = np.arange(n)
    dates = df.index
    daily = bool((dates.normalize() == dates).all()) # barres journalières : pas d'heure à afficher
    fmt   = "%Y-%m-%d" if daily else "%d/%m %H:%M"

    # --- Layout --- #
    fig = Figure( figsize=(12, 8) )
    ax = fig.subplots()

    fig.patch.set_facecolor(T["bg"])
    ax.set_facecolor(T["bg"])
    ax.tick_params(colors=T["text_muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(T["grid"])

    # --- Bougies --- #
    ohlc = np.column_stack( [ idx, o, h, l, c ] )
    candlestick_ohlc( ax, ohlc, width=0.48, colorup= T["bull"], colordown= T["bear"] )

    # --- Niveaux S/R dynamiques --- #
    for lv in levels:
        active = lv.end_idx is None
        if not (active or cfg.show_broken):
            continue
        
        color = T["bull"] if lv.is_support else T["bear"]
        x_end = n - 1 if active else lv.end_idx
        style = ( "--" if lv.flipped else "-" ) if active else ":" # actif / rôle inversé / cassé
        line_width = ( 0.8 if lv.flipped else 1.5 ) if active else 0.8
        ax.fill_between( [lv.pivot_idx, x_end], lv.lo, lv.hi, color=color, linewidth=0, zorder=1,
                         alpha=0.15 if active else 0.06 ) # zone
        ax.hlines( lv.price, lv.pivot_idx, x_end, colors=color, zorder=3, 
                   lw=line_width, linestyles=style, alpha=1.0 if active else 0.55 )

        # croix à la cassure, sur le bord franchi de la zone
        if not active and cfg.show_cross:
            ax.plot( lv.end_idx, lv.lo if lv.is_support else lv.hi, marker="x", color=color, ms=6, mew=1.5, zorder=6 )

        # marqueur du pivot brut, score à l'intérieur (pas pour un niveau retourné : il n'a pas de pivot)
        if not lv.flipped:
            ax.text( lv.pivot_idx, lv.pivot_price, f"{lv.score:.0f}", ha="center",
                    va="top" if lv.is_support else "bottom", fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"), zorder=6 )
        
        # prix exact à l'extrémité du niveau
        if active: 
            ax.text( x_end + 0.6, lv.price, _fmt_price( lv.price ),
                     ha="left", va="center", fontsize=8, color=color, zorder=6 )

    ax.set_xlim( -1, n - 1 + gap )
    ax.set_title( title, color=T["text"], fontsize=13, pad=8 )
    handles = [
        Line2D([], [], color=T["bull"], label="Support"),
        Line2D([], [], color=T["bear"], label="Résistance"),
    ]
    if cfg.flip_levels:
        handles.append( Line2D([], [], color=T["text_muted"], ls="--", label="Rôle inversé") )
    if cfg.show_broken:
        handles.append( Line2D([], [], color=T["text_muted"], ls=":", marker="x", label="Cassé") )
    ax.legend( handles=handles, loc="upper left", fontsize=8, facecolor=T["bg_legend"], labelcolor=T["text"] )

    # --- Ticks de l'axe X : ré-injecter les vraies dates --- #

    # Les positions au-delà de la dernière barre (marge des étiquettes) n'ont pas de date : "".
    def format_xdata( x ):
        i = int( round( x ) )
        return dates[i].strftime(fmt) if 0 <= i < n else ""

    ax.xaxis.set_major_formatter( mticker.FuncFormatter(lambda x, pos=None: format_xdata(x)) )
    ax.xaxis.set_major_locator( mticker.MaxNLocator(integer=True, nbins=8) )
    ax.format_xdata = format_xdata # coordonnées affichées par la toolbar
    ax.grid( True, alpha=0.3, color='#A5A5A5', linestyle='--', linewidth=0.5 )
    fig.autofmt_xdate()

    return fig

# --------------------------------------------------
# main() appelé par script_runner de TradingInPython
#
def main():
    matplotlib.use("Agg") # script_runner : thread secondaire, rendu différé via api.show_figure()
    from styles.watermark import Watermark
    from user_scripts.api import api

    if not api.check_parameters( ['ticker', 'period', 'interval'] ):
        exit(1)

    title = f"{api.name or api.ticker} - Dynamic S/R"
    
    fig = plot_supports_resistances( 
                api.ticker, title=title, period=api.period, 
                interval=api.interval, df=api.df 
            )
    
    Watermark.apply( fig )

    fig.tight_layout()
    api.show_figure( fig, title )

if __name__ == "__main__":
    matplotlib.use("TkAgg") # standalone : thread principal, affichage direct OK

    TICKER   = "NVDA"
    PERIOD   = "5y" # le profil est reconstruit à chaque pivot : prévoir de l'historique avant
    INTERVAL = "1d"

    fig = plot_supports_resistances( 
                TICKER, title="Nvidia - Dynamic S/R", 
                period=PERIOD, interval=INTERVAL 
            )
    
    fig.tight_layout()

    # Créer la fenêtre Tkinter
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title("Dynamic S/R")
    canvas = FigureCanvasTkAgg( fig, master=root )
    toolbar = NavigationToolbar2Tk( canvas, root )
    toolbar.update()

    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    root.mainloop()
