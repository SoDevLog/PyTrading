""" Volume Structure Analyzer + Dynamic S/R

    1. Profil de volume glissant Acheteurs / Vendeurs (sur les `lookback` dernières barres)
       - Le volume de chaque bougie est réparti sur les lignes de prix touchées par son range
         High-Low, au prorata du recouvrement (et non pas affecté à un seul prix).
       - Acheteurs / Vendeurs sont ESTIMÉS d'après la position de la clôture dans le range :
         clôture près du High → surtout acheteurs, près du Low → surtout vendeurs.
         Ce n'est pas du vrai bid/ask ni du footprint.
       - POC (Point of Control) = ligne au volume maximum.

    2. Supports / Résistances dynamiques guidés par le profil
       a) pivots High/Low confirmés (swing) : connus seulement `swing` barres après leur formation
       b) autour du pivot, recherche du meilleur noeud du profil (reconstruit à la date de
          confirmation du pivot) avec un score d'évidence :
            force du noeud / POC, relief local, dominance acheteurs (support) ou vendeurs
            (résistance), proximité du pivot
       c) le niveau final = pivot attiré vers le noeud (plus l'évidence est forte, plus l'attraction
          est forte) : le prix propose le niveau, le volume l'affine
       d) score final 0-100 (évidence + absorption + volume du pivot + rejet par la mèche)
       e) cycle de vie : un niveau reste actif jusqu'à une clôture au-delà de 'break_atr' x ATR ;
          fusion des niveaux de même sens proches ('merge_atr') ; au-delà de 'max_levels'
          le plus faible est supprimé

    Axe X : positionnel (entier séquentiel), pas temporel - même principe que le script BSI :
    une position par barre tradée, donc pas de trou le week-end. Les vraies dates sont
    réinjectées via un FuncFormatter (ticks) et format_xdata (survol NavigationToolbar2Tk).
    Le profil est dessiné à droite de la dernière barre, sur le même axe.

    Dépendances : pandas, numpy, matplotlib, yfinance (colonnes OHLCV requises, Volume compris)
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
from matplotlib.patches import Patch

from pathlib import Path
base = Path(__file__).resolve().parent.parent
sys.path.append( str(base) )

import digitsignalprocessing.indicators as dsp

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
        "poc":          "#ffb300",
    },
    False: { # --- light ---
        "bg":           "#ffffff",
        "bg_legend":    "#f5f5f5",
        "text":         "#111111",
        "text_muted":   "#555555",
        "grid":         "#A5A5A5",
        "bull":         "#0a8a7e",
        "bear":         "#c62828",
        "poc":          "#e69100",
    },
}

# ---------------------------------------------
# PARAMÈTRES
# ---------------------------------------------

@dataclass
class ProfileSettings:
    # --- Profil glissant --- #
    lookback:       int   = 200    # nombre de barres du profil
    rows:           int   = 40     # nombre de lignes de prix
    width:          int   = 40     # largeur max du profil (en barres)
    offset:         int   = 6      # distance entre la dernière barre et le profil
    show_profile:   bool  = True
    show_poc:       bool  = True
    show_level_rows: bool = True   # surligne les lignes du profil qui portent un niveau actif
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
    show_broken:    bool  = True   # affiche aussi les niveaux déjà invalidés (pointillés)

# Pondérations (choix de ce script, pas celles de l'original)
# noeud, relief local, dominance directionnelle, proximité du pivot
_W_EVIDENCE = np.array([0.35, 0.25, 0.20, 0.20])
# évidence profil, absorption, volume du pivot, rejet mèche
_W_SCORE = np.array([0.55, 0.15, 0.15, 0.15])

@dataclass
class Level:
    is_support:  bool
    pivot_idx:   int               # barre du pivot (début de la ligne)
    pivot_price: float             # prix brut du pivot (High ou Low)
    price:       float             # niveau final, attiré vers le noeud du profil
    score:       float             # 0-100
    end_idx:     Optional[int] = None   # barre d'invalidation, None = niveau actif

# ---------------------------------------------
# 1. PROFIL DE VOLUME ACHETEURS / VENDEURS
# ---------------------------------------------

def volume_profile( high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   volume: np.ndarray, rows: int ):
    """
    Profil acheteurs / vendeurs sur les barres fournies.
    Retourne (edges, buy, sell) : bords des lignes (rows + 1) et volumes estimés par ligne.
    """
    edges = np.linspace(low.min(), high.max(), rows + 1)
    bot, top = edges[:-1], edges[1:]

    rng  = high - low
    flat = rng < 1e-12                                   # bougie sans range : High == Low
    rng  = np.where( flat, 1.0, rng )

    # part du range de chaque bougie qui recouvre chaque ligne  (barres x lignes)
    overlap = np.minimum(high[:, None], top) - np.maximum(low[:, None], bot)
    share   = np.clip(overlap, 0, None) / rng[:, None]

    # bougie plate : tout son volume sur la ligne qui contient la clôture
    row_close = np.clip(np.searchsorted(edges, close, side="right") - 1, 0, rows - 1)
    share[flat] = 0.0
    share[flat, row_close[flat]] = 1.0

    # clôture près du High -> acheteurs ; bougie plate -> 50/50
    buy_ratio = np.where( flat, 0.5, (close - low) / rng )

    vol  = volume[:, None] * share
    buy  = (vol * buy_ratio[:, None]).sum(axis=0)
    sell = (vol * (1.0 - buy_ratio)[:, None]).sum(axis=0)
    
    return edges, buy, sell

# ---------------------------------------------
# 2. SUPPORTS / RÉSISTANCES DYNAMIQUES
# ---------------------------------------------

def _best_node( edges, buy, sell, price, atr_v, is_support, cfg: ProfileSettings ):
    """
    Meilleur noeud du profil autour du pivot.
    Retourne (évidence 0-1, prix du noeud) ou None si aucune ligne ne qualifie.
    """
    total   = buy + sell
    poc     = total.max()
    centers = (edges[:-1] + edges[1:]) / 2
    reach   = cfg.search_atr * atr_v
    if poc <= 0 or reach <= 0:
        return None

    safe  = np.maximum(total, 1e-12)
    node  = total / poc                                           # force du noeud vs POC

    kernel = np.array([1, 1, 0, 1, 1], dtype=float)               # 2 lignes de chaque côté
    neigh  = np.convolve(total, kernel, "same") / np.convolve(np.ones_like(total), kernel, "same")
    peak   = np.clip((total - neigh) / safe, 0, 1)                # relief local

    direction = (buy if is_support else sell) / safe              # dominance acheteurs / vendeurs
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
    avg    = max(recent.mean() if len(recent) else v[i], 1e-12)

    rng   = np.maximum(h[lo:hi] - l[lo:hi], 1e-12)
    body  = np.abs(c[lo:hi] - o[lo:hi])
    # gros volume + petit corps = absorption
    absorption = np.max(np.clip(v[lo:hi] / avg / 2, 0, 1) * (1 - body / rng))

    volume_rel = min(v[i] / avg / 2, 1.0)
    wick = (min(o[i], c[i]) - l[i]) if is_support else (h[i] - max(o[i], c[i]))
    wick_rel = np.clip(wick / max(h[i] - l[i], 1e-12), 0, 1)
    
    return absorption, volume_rel, wick_rel

# -----------------------------------------------------------------------------

def _is_broken( lv: Level, close: float, buffer: float ) -> bool:
    return close < lv.price - buffer if lv.is_support else close > lv.price + buffer

# -----------------------------------------------------------------------------

def dynamic_levels( df: pd.DataFrame, cfg: ProfileSettings ) -> list:
    """
    Calcule tous les niveaux S/R sur l'historique (actifs et invalidés).
    Rien n'utilise de donnée future : un pivot n'est évalué qu'à sa barre de confirmation,
    avec le profil et l'ATR connus à cette date.
    """
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df)
    atr_np = dsp.atr_rolling( df, cfg.atr_period )
    s = cfg.swing
    
    # Sanity check
    if n < cfg.lookback:
        print( f"Attention lookback:{cfg.lookback} trop long > {n} et sera écretté")

    # --- a) pivots confirmés → niveaux candidats, rangés par barre de confirmation --- #
    pending = {}
    for i in range(s, n - s):
        for is_support, hit in ((True,  l[i - s:i + s + 1].argmin() == s),
                                (False, h[i - s:i + s + 1].argmax() == s)):
            if not hit:
                continue
            t     = i + s                                   # barre de confirmation
            lo    = max(0, t - cfg.lookback + 1)
            price = l[i] if is_support else h[i]
            edges, buy, sell = volume_profile( h[lo:t + 1], l[lo:t + 1], c[lo:t + 1], v[lo:t + 1], cfg.rows )

            # --- b) meilleur noeud du profil autour du pivot ---
            best = _best_node(edges, buy, sell, price, atr_np[t], is_support, cfg)
            if best is None or best[0] < cfg.min_evidence:
                continue
            evidence, node_price = best

            # --- d) score final ---
            quality = _pivot_quality(o, h, l, c, v, i, s, is_support)
            score   = 100 * float(_W_SCORE @ np.array([evidence, *quality]))
            if score < cfg.min_score:
                continue

            # --- c) le profil tire le pivot vers le noeud ---
            level = price + cfg.pull * evidence * (node_price - price)
            pending.setdefault(t, []).append(Level(is_support, i, price, level, score))

    # --- e) cycle de vie : invalidation, fusion, plafond --- #
    active, levels = [], []
    for t in range(n):
        for lv in active[:]:
            if _is_broken(lv, c[t], cfg.break_atr * atr_np[t]):
                lv.end_idx = t
                active.remove(lv)

        for lv in pending.get(t, []):
            if _is_broken(lv, c[t], cfg.break_atr * atr_np[t]):
                continue                                    # déjà percé à sa naissance
            twin = next((a for a in active
                         if a.is_support == lv.is_support
                         and abs(a.price - lv.price) <= cfg.merge_atr * atr_np[t]), None)
            if twin is not None:
                if twin.score >= lv.score:
                    continue
                active.remove(twin)
                levels.remove(twin)
            active.append(lv)
            levels.append(lv)

        while len(active) > cfg.max_levels:
            weakest = min(active, key=lambda a: a.score)
            active.remove(weakest)
            levels.remove(weakest)

    return levels

def _fmt_price( p: float ) -> str:
    """Prix lisible : 4 décimales sous 10 (forex, petits cours), 2 décimales au-delà."""
    return f"{p:.4f}" if p < 10 else f"{p:.2f}"

# --------------------------------------------------
# GRAPHIQUE - Buyers & Sellers Profile + Dynamic S/R
# --------------------------------------------------

def plot_bsp( ticker: str = "AAPL",
              title: str = "",
              period: str = "1y",
              interval: str = "1d",
              df = None,
              cfg: Optional[ProfileSettings] = None) -> Figure:

    T   = _THEME[DARK_MODE]   # palette active selon la constante
    cfg = cfg or ProfileSettings()

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
        raise ValueError(f"{ticker} : pas de volume, profil impossible")

    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df)
    levels = dynamic_levels( df, cfg )

    # Profil affiché : les 'lookback' dernières barres
    lo = max(0, n - cfg.lookback)
    edges, buy, sell = volume_profile( h[lo:], l[lo:], c[lo:], v[lo:], cfg.rows )
    centers = (edges[:-1] + edges[1:]) / 2
    row_h   = edges[1] - edges[0]
    total   = buy + sell
    scale   = cfg.width / total.max()
    gap     = max( cfg.offset, round( 0.06 * (n + cfg.width) ) )  # place pour l'étiquette prix (score) des niveaux
    x0      = n - 1 + gap                                # bord gauche du profil

    # --- Axe X positionnel (voir docstring) --- #
    idx   = np.arange(n)
    dates = df.index
    daily = bool((dates.normalize() == dates).all())     # barres journalières : pas d'heure à afficher
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
    colors = [T["bull"] if cl >= op else T["bear"] for op, cl in zip(o, c)]
    ax.vlines( idx, l, h, colors=colors, lw=0.8, alpha=0.8 )
    ax.bar( idx, np.abs(c - o), bottom=np.minimum(o, c), color=colors, width=0.6 )

    # --- Profil acheteurs / vendeurs --- #
    if cfg.show_profile:
        bar_h = row_h * 0.92
        ax.barh( centers, buy * scale,  left=x0, height=bar_h, color=T["bull"], alpha=0.55, lw=0 )
        ax.barh( centers, sell * scale, left=x0 + buy * scale, height=bar_h, color=T["bear"], alpha=0.55, lw=0 )

        if cfg.show_poc:
            p = total.argmax()
            ax.barh( centers[p], total[p] * scale, left=x0, height=bar_h,
                    fill=False, edgecolor=T["poc"], lw=1.5, zorder=4 )
            ax.hlines( centers[p], lo, x0, colors=T["poc"], linestyles=":", lw=0.9, zorder=3 )
            
        if cfg.show_level_rows:
            for lv in levels:
                if lv.end_idx is None and edges[0] <= lv.price <= edges[-1]:
                    r = min( np.searchsorted(edges, lv.price, side="right") - 1, cfg.rows - 1 )
                    ax.barh( centers[r], total[r] * scale, left=x0, height=bar_h, fill=False,
                            edgecolor=T["bull"] if lv.is_support else T["bear"], lw=1.2, zorder=4 )

    # --- Niveaux S/R dynamiques --- #
    for lv in levels:
        active = lv.end_idx is None
        if not ( active or cfg.show_broken ):
            continue
        
        color = T["bull"] if lv.is_support else T["bear"]
        x_end = n - 1 if active else lv.end_idx
        ax.hlines(lv.price, lv.pivot_idx, x_end, colors=color, zorder=3,
                  lw=1.5 if active else 0.8, linestyles="-" if active else ":",
                  alpha=1.0 if active else 0.55)

        # marqueur du pivot brut, score à l'intérieur
        ax.text(lv.pivot_idx, lv.pivot_price, f"{lv.score:.0f}", ha="center",
                va="top" if lv.is_support else "bottom", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"), zorder=6)
        
        # prix exact + score à l'extrémité du niveau
        if active:
            ax.text( x_end + 0.6, lv.price, f"{_fmt_price( lv.price )}",
                     ha="left", va="center", fontsize=8, color=color, zorder=6 )

    ax.set_xlim( -1, x0 + cfg.width + 2 )
    ax.set_title( title, color=T["text"], fontsize=13, pad=8 )
    ax.legend(
        handles=[
            Patch(color=T["bull"], alpha=0.55, label="Acheteurs (estimés)"),
            Patch(color=T["bear"], alpha=0.55, label="Vendeurs (estimés)"),
            Line2D([], [], color=T["poc"],  ls=":", label="POC"),
            Line2D([], [], color=T["bull"], label="Support"),
            Line2D([], [], color=T["bear"], label="Résistance"),
        ], loc="upper left", fontsize=8, facecolor=T["bg_legend"], labelcolor=T["text"]
    )

    # --- Ticks de l'axe X : ré-injecter les vraies dates --- #
    
    # Les positions au-delà de la dernière barre (zone du profil) n'ont pas de date : "".
    def format_xdata( x ):
        i = int( round( x ) )
        return dates[i].strftime(fmt) if 0 <= i < n else ""

    ax.xaxis.set_major_formatter( mticker.FuncFormatter(lambda x, pos=None: format_xdata(x)) )
    ax.xaxis.set_major_locator( mticker.MaxNLocator(integer=True, nbins=8) )
    ax.format_xdata = format_xdata # coordonnées affichées par la toolbar
    ax.grid( True, alpha=0.3, color='#A5A5A5', linestyle='--', linewidth=0.5 )
    fig.autofmt_xdate()

    return fig

# ---------------------------------------------
# main() appelé par script_runner
#
def main():
    matplotlib.use("Agg") # script_runner : thread secondaire, rendu différé via api.show_figure()
    from styles.watermark import Watermark
    from user_scripts.api import api

    if not api.check_parameters( ['ticker', 'period', 'interval'] ):
        exit(1)

    title = f"{api.name or api.ticker} - Buyers & Sellers Profile + Dynamic S/R"
    fig = plot_bsp( api.ticker, title=title, period=api.period, interval=api.interval, df=api.df )
    Watermark.apply( fig )

    fig.tight_layout()
    api.show_figure( fig, title )

if __name__ == "__main__":
    matplotlib.use("TkAgg") # standalone : thread principal, affichage direct OK

    TICKER   = "NVDA"
    PERIOD   = "2y"      # le profil est reconstruit à chaque pivot : prévoir de l'historique avant
    INTERVAL = "1d"

    fig = plot_bsp( TICKER, title="Nvidia - Buyers & Sellers Profile + Dynamic S/R", period=PERIOD, interval=INTERVAL )
    fig.tight_layout()

    # Créer la fenêtre Tkinter
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title(f"Buyers & Sellers Profile + Dynamic S/R")
    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    root.mainloop()
