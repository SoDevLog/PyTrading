""" Indicateurs BSI (Bull/Bear Strength Index)

    Reproduction des composants visibles sur le graphique :
    1. Cloud BSI    : deux EMA formant un nuage coloré (style Ichimoku)
    2. Stop BSI     : ATR Trailing Stop (style SuperTrend)
    3. Synergie     : alignement multi-timeframe (fond coloré)
    4. Histogrammes : oscillateur binaire, sur 3 échelles empilées
       - BSI LT (long terme)  : la boussole, biais de marché de fond
       - BSI MT (moyen terme) : indicateur stratégique de prise de bénéfices
       - BSI CT (court terme) : indicateur chirurgical de timing

    Pandas : infer_objects()
    C'est une précaution "anti-warning / anti-ambiguïté de dtype" liée au fait que reindex(..., method="ffill")
    peut casser le dtype booléen en y insérant des NaN. Elle rend le pipeline robuste et 
    silencieux face à ce changement de comportement annoncé.

    Axe X : positionnel (entier séquentiel), pas temporel.
    Un DatetimeIndex réserve de l'espace proportionnel au temps réel écoulé, ce qui crée
    des trous visuels les week-ends / jours fériés (pas de séance tradée). En traçant sur
    un axe entier (une position par barre tradée), ces trous disparaissent. Les vraies
    dates sont réinjectées a posteriori via un FuncFormatter (ticks) et format_xdata
    (coordonnées affichées par la NavigationToolbar2Tk au survol de la souris).

    Dépendances : pandas, numpy, matplotlib, yfinance
"""
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)  # supprime le FutureWarning fillna/ffill/bfill
import matplotlib
import matplotlib.style
matplotlib.style.use("seaborn-v0_8-darkgrid")
import matplotlib.ticker as mticker
import yfinance

# ------------------------------------------------------
# 1. CLOUD BSI  (deux EMA lissées = nuage Ichimoku-like)
# ------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def cloud_bsi(close: pd.Series,
              fast: int = 9,
              slow: int = 26) -> pd.DataFrame:
    """
    Retourne deux lignes EMA.
    La zone entre elles est colorée verte (fast > slow) ou rouge.
    Analogue à la conversion line / base line d'Ichimoku.
    """
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    return pd.DataFrame({
        "cloud_fast": fast_line,
        "cloud_slow": slow_line,
        "cloud_bull": fast_line > slow_line,   # True = haussier
    })

# ---------------------------------------------
# 2. STOP BSI  (ATR Trailing Stop / SuperTrend)
# ---------------------------------------------

def atr(high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10) -> pd.Series:
    """Average True Range."""
    h_l  = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low  - close.shift(1)).abs()
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def stop_bsi(high: pd.Series,
             low: pd.Series,
             close: pd.Series,
             atr_period: int = 10,
             multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend-style trailing stop.
    Retourne la ligne de stop et la direction (1 = haussier, -1 = baissier).
    """
    _atr = atr(high, low, close, atr_period)
    hl2  = (high + low) / 2

    basic_upper = hl2 + multiplier * _atr
    basic_lower = hl2 - multiplier * _atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(close)):
        fu_prev = final_upper.iloc[i - 1]
        fl_prev = final_lower.iloc[i - 1]
        c_prev  = close.iloc[i - 1]

        final_upper.iloc[i] = (
            basic_upper.iloc[i]
            if basic_upper.iloc[i] < fu_prev or c_prev > fu_prev
            else fu_prev
        )
        final_lower.iloc[i] = (
            basic_lower.iloc[i]
            if basic_lower.iloc[i] > fl_prev or c_prev < fl_prev
            else fl_prev
        )

    direction = pd.Series(1, index=close.index)
    stop_line = pd.Series(np.nan, index=close.index)

    for i in range(1, len(close)):
        d_prev = direction.iloc[i - 1]
        c      = close.iloc[i]
        fl     = final_lower.iloc[i]
        fu     = final_upper.iloc[i]

        if d_prev == 1:
            direction.iloc[i] = -1 if c < fl else 1
        else:
            direction.iloc[i] =  1 if c > fu else -1

        stop_line.iloc[i] = fl if direction.iloc[i] == 1 else fu

    return pd.DataFrame({
        "stop":      stop_line,
        "direction": direction,       # 1 = hausse, -1 = baisse
    })

# -----------------------------------------
# 3. SYNERGIE  (alignement multi-timeframe)
# -----------------------------------------

def _to_bool(series: pd.Series, ref_index) -> pd.Series:
    """Ré-indexe sur ref_index et garantit un dtype booléen propre."""
    return (
        series
        .reindex(ref_index)
        .ffill()
        .infer_objects(copy=False)
        .fillna(False)
        .astype(bool)
    )

def synergie(close: pd.Series,
             fast_ct: int = 9,   slow_ct: int = 26,
             fast_mt: int = 21,  slow_mt: int = 55,
             fast_lt: int = 50,  slow_lt: int = 200) -> pd.Series:
    """
    Synergie 3 horizons CT / MT / LT — système d'autorisation / interdiction.

    Règle d'unanimité (les 3 horizons doivent être dans le même sens) :
      → +1  Synergie Verte  : LT + MT + CT tous haussiers → autorisation long
      → -1  Synergie Rouge  : LT + MT + CT tous baissiers → autorisation short
      →  0  Zone blanche    : discordance → situation incertaine, on ne trade pas

    C'est un filtre strict : mieux vaut rater une entrée que d'entrer en désaccord.
    """
    bull_lt = _to_bool(ema(close, fast_lt) > ema(close, slow_lt), close.index)
    bull_mt = _to_bool(ema(close, fast_mt) > ema(close, slow_mt), close.index)
    bull_ct = _to_bool(ema(close, fast_ct) > ema(close, slow_ct), close.index)

    votes_bull = bull_lt.astype(int) + bull_mt.astype(int) + bull_ct.astype(int)

    result = pd.Series(0, index=close.index)   # 0 = zone blanche par défaut
    result[votes_bull == 3] =  1               # unanimité haussière
    result[votes_bull == 0] = -1               # unanimité baissière
    return result

# ---------------------------------------------
# 4. HISTOGRAMME BSI  (oscillateur binaire)
# ---------------------------------------------

def histogramme_bsi(close: pd.Series,
                    fast: int  = 9,
                    slow: int  = 26,
                    signal: int = 9) -> pd.DataFrame:
    """
    Oscillateur style MACD coloré en vert/rouge selon la direction.
    Retourne la valeur de l'histogramme et sa couleur.
    """
    macd_line   = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist        = macd_line - signal_line

    color = np.where(hist >= 0, "bull", "bear")
    return pd.DataFrame({
        "macd":   macd_line,
        "signal": signal_line,
        "hist":   hist,
        "color":  color,
    })

# ---------------------------------------------
# THÈME - True = fond noir / False = fond blanc
# ---------------------------------------------

DARK_MODE: bool = False

_THEME = {
    True: { # --- dark ---
        "bg":           "#0d1117",
        "bg_legend":    "#1a1a1a",
        "text":         "#ffffff",
        "text_muted":   "#aaaaaa",
        "grid":         "#333333",
        "zero_line":    "#555555",
        "bull":         "#26a69a",
        "bull_bsi":     "#0a8a59",
        "bear":         "#ef5350",
        "bear_bsi":     "#d83633",
        "syn_bull":     "#1a3a1a",
        "syn_bear":     "#3a1a1a",
    },
    False: { # --- light ---
        "bg":           "#ffffff",
        "bg_legend":    "#f5f5f5",
        "text":         "#111111",
        "text_muted":   "#555555",
        "grid":         "#dddddd",
        "zero_line":    "#aaaaaa",
        "bull":         "#0a8a7e",
        "bull_bsi":     "#0a8a59",
        "bear":         "#c62828",
        "bear_bsi":     "#d83633",
        "syn_bull":     "#d6f0eb",
        "syn_bear":     "#fce8e8",
    },
}

# ------------------------------------
# GRAPHIQUE - Bull/Bear Strength Index
# ------------------------------------

def plot_bsi(ticker: str = "AAPL",
             period: str = "6mo",
             interval: str = "1d",
             df = None) -> None:

    T = _THEME[DARK_MODE]   # palette active selon la constante

    if df is None or df.empty:
        df = yfinance.download( ticker, period=period, interval=interval,
            auto_adjust=True, progress=False )

    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}")

    # Aplatir les colonnes si MultiIndex (yfinance ≥ 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Calcul des indicateurs
    cloud   = cloud_bsi(close)
    stop    = stop_bsi(high, low, close)
    hist_ct = histogramme_bsi(close, fast=9,   slow=26,  signal=9)   # court terme
    hist_mt = histogramme_bsi(close, fast=21,  slow=55,  signal=9)   # moyen terme
    hist_lt = histogramme_bsi(close, fast=50,  slow=100, signal=9)   # long terme

    # Synergie 3 horizons CT/MT/LT — une seule série close, périodes différentes
    syn = synergie( close,
                    fast_ct=9,  slow_ct=26,
                    fast_mt=21, slow_mt=55,
                    fast_lt=50, slow_lt=100 )

    # --- Axe X positionnel --------------------------------
    # idx : position entière (0, 1, 2...) utilisée pour tout le tracé —
    #       une barre tradée = une position, donc pas de trou le week-end.
    # dates : vraies dates conservées à part, uniquement pour l'affichage
    #         (ticks de l'axe + coordonnées de la toolbar).
    idx   = np.arange(len(df))
    dates = df.index

    # --- Layout --------------------------------
    from matplotlib.figure import Figure
    
    # Checking witch matplotlib style is used
    #print( matplotlib.rcParams['axes.facecolor'], matplotlib.rcParams['axes.grid'] )
    
    fig = Figure(figsize=(12, 9))
    axes = fig.subplots(
        4, 1,
        gridspec_kw={"height_ratios": [6, 1, 1, 1]},
        sharex=True,
    )
    
    fig.patch.set_facecolor(T["bg"])
    for ax in axes:
        ax.set_facecolor(T["bg"])
        ax.tick_params(colors=T["text_muted"])
        ax.yaxis.label.set_color(T["text_muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(T["grid"])

    ax_price, ax_hist_ct, ax_hist_mt, ax_hist_lt = axes

    # --- Fond Synergie — unanimité 3/3 : vert / rouge / blanc (neutre) ---
    for i in range(len(syn)):
        val = syn.iloc[i]
        if val == 0:
            continue                                         # zone blanche : fond inchangé
        color = T["syn_bull"] if val == 1 else T["syn_bear"]
        ax_price.axvspan(idx[i], idx[min(i + 1, len(idx) - 1)],
                         color=color, alpha=0.6, lw=0)

    # --- Bougies simplifiées ---------------------------------------------
    for i, (ts, row) in enumerate(df.iterrows()):
        o, h_, l_, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = T["bull"] if c >= o else T["bear"]
        ax_price.plot([idx[i], idx[i]], [l_, h_], color=color, lw=0.8, alpha=0.8)
        ax_price.bar(idx[i], abs(c - o), bottom=min(o, c),
                     color=color, width=0.6, align="center")

    # -- Cloud BSI ------------------------------------------------------
    ax_price.plot(idx, cloud["cloud_fast"], color=T["bull"], lw=1.2,
                  label="Cloud rapide")
    ax_price.plot(idx, cloud["cloud_slow"], color=T["bear"], lw=1.2,
                  label="Cloud lent")
    ax_price.fill_between(
        idx,
        cloud["cloud_fast"], cloud["cloud_slow"],
        where=cloud["cloud_bull"],
        color=T["bull"], alpha=0.2,
    )
    ax_price.fill_between(
        idx,
        cloud["cloud_fast"], cloud["cloud_slow"],
        where=~cloud["cloud_bull"],
        color=T["bear"], alpha=0.2,
    )

    # --- Stop BSI --------------------------------
    bull_mask = stop["direction"] == 1
    bear_mask = stop["direction"] == -1
    ax_price.scatter(idx[bull_mask], stop["stop"][bull_mask],
                     color=T["bull_bsi"], s=6, zorder=5, label="Stop BSI (hausse)")
    ax_price.scatter(idx[bear_mask], stop["stop"][bear_mask],
                     color=T["bear_bsi"], s=6, zorder=5, label="Stop BSI (baisse)")

    ax_price.set_title(f"{ticker} - Bull/Bear Strength Index", color=T["text"],
                       fontsize=13, pad=8)
    ax_price.legend(loc="upper left", fontsize=8,
                    facecolor=T["bg_legend"], labelcolor=T["text"])

    # -- Histogrammes BSI CT / MT / LT — force du mouvement ----------------
    # Chaque axe combine :
    #   • un fond de couleur pleine (vert/rouge) indiquant la direction de l'EMA
    #   • des barres MACD dont la hauteur traduit la force / l'accélération
    bull_ct_sig = (ema(close, 9)   > ema(close, 26)).values
    bull_mt_sig = (ema(close, 21)  > ema(close, 55)).values
    bull_lt_sig = (ema(close, 50)  > ema(close, 100)).values

    def draw_hist(ax, hist_df, bull_sig, label):
        """
        Fond de direction (axvspan pâle) + barres MACD colorées selon le signe.
        La couleur de la barre = signe de l'histogramme (pas forcément la direction EMA),
        ce qui permet de voir les divergences et le ralentissement du momentum.
        """
        # Fond de direction : bande pâle indiquant la tendance EMA
        for i in range(len(bull_sig)):
            bg = T["syn_bull"] if bull_sig[i] else T["syn_bear"]
            ax.axvspan(idx[i], idx[min(i + 1, len(idx) - 1)],
                       color=bg, alpha=0.55, lw=0)

        # Barres MACD : couleur selon signe de l'histogramme
        colors = [T["bull"] if v >= 0 else T["bear"] for v in hist_df["hist"]]
        ax.bar(idx, hist_df["hist"], color=colors, width=0.8, zorder=2)
        ax.axhline(0, color=T["zero_line"], lw=0.6, zorder=3)

        ax.set_ylabel( label, fontsize=9, color=T["text_muted"] )
        for spine in ax.spines.values():
            spine.set_edgecolor(T["grid"])

    draw_hist(ax_hist_ct, hist_ct, bull_ct_sig, "BSI CT  9/26")
    draw_hist(ax_hist_mt, hist_mt, bull_mt_sig, "BSI MT  21/55")
    draw_hist(ax_hist_lt, hist_lt, bull_lt_sig, "BSI LT  50/100")

    # --- Ticks de l'axe X : ré-injecter les vraies dates -------------------
    # x est une position entière ; on va chercher la date correspondante
    # dans "dates" (le DatetimeIndex d'origine) pour l'affichage du tick.
    def format_date(x, pos=None):
        i = int(round(x))
        if 0 <= i < len(dates):
            return dates[i].strftime("%Y-%m-%d")
        return ""

    ax_hist_ct.xaxis.set_major_formatter(mticker.FuncFormatter(format_date))
    ax_hist_ct.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    fig.autofmt_xdate()  # incline les labels pour éviter le chevauchement

    # --- Coordonnées affichées par la NavigationToolbar2Tk au survol -------
    # ax.format_xdata n'est PAS partagé entre axes même avec sharex=True :
    # sans ceci, seul un survol sur ax_price afficherait une date lisible
    # (via le tick formatter d'ax_hist_ct qui ne s'applique qu'à son propre
    # axe) — les trois histogrammes du bas resteraient en position entière brute.
    def format_xdata(x):
        i = int(round(x))
        if 0 <= i < len(dates):
            return dates[i].strftime("%Y-%m-%d")
        return ""

    for ax in axes:
        ax.format_xdata = format_xdata

    return fig

# ---------------------------------------------
# POINTS D'ENTRÉE 
# ---------------------------------------------

# ---------------------------------------------
# main() appelé explicitement par script_runner
#
def main():
    matplotlib.use("Agg") # script_runner : thread secondaire, rendu différé via api.show_figure()
    from user_scripts.api import api

    if not api.check_parameters( ['ticker', 'period', 'interval'] ):
        exit(1)

    fig = plot_bsi( api.ticker, period=api.period, interval=api.interval, df=api.df )
    fig.tight_layout()
    api.show_figure( fig, title=f"{api.ticker} - Bull/Bear Strength Index" )

if __name__ == "__main__":
    matplotlib.use("TkAgg") # standalone : thread principal, affichage direct OK
    
    TICKER   = "AAPL"
    PERIOD   = "1y"
    INTERVAL = "1d"

    fig = plot_bsi(TICKER, period=PERIOD, interval=INTERVAL)
    fig.tight_layout()

    # Créer la fenêtre Tkinter    
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title(f"{TICKER} - Bull/Bear Strength Index")
    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    root.mainloop()