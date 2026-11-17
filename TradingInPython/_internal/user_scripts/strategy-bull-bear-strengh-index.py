""" Indicateurs BSI (Bull/Bear Strength Index)

    Reproduction des composants visibles sur le graphique :
    1. cloud_bsi    : deux EMA formant un nuage coloré (style Ichimoku)
    2. Stop BSI     : ATR Trailing Stop (style SuperTrend)
    3. Synergie     : alignement multi-timeframe (fond coloré)
    4. histo_bsi : oscillateur binaire, sur 3 échelles empilées
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
from dataclasses import dataclass
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)  # supprime le FutureWarning fillna/ffill/bfill
import matplotlib
import matplotlib.style
#matplotlib.style.use("seaborn-v0_8-darkgrid")
matplotlib.style.use("seaborn-v0_8-notebook")
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import yfinance
run_standalone = (__name__ == "__main__")

if run_standalone:
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    sys.path.append( str(base_dir) )
from styles.watermark import Watermark

# ---------------------------
# Paramètrage des signaux BSI
# ---------------------------

FAST_CT: int = 9
SLOW_CT: int = 26

FAST_MT: int = 21
SLOW_MT: int = 60

FAST_LT: int = 50
SLOW_LT: int = 200

# Nuage BSI cloud_bsi
CLOUD_EMA_FAST: int = 9
CLOUD_EMA_SLOW: int = 26

# Super Trend / Stop BSI
SUPER_TREND_PERIOD: int = 26
SUPER_TREND_MULTIPLIER: float = 3.2

ZONE_RULES: bool = False  # True = interdiction d'acheter en zone rouge / vendre en zone verte
EXIT_ON_WHITE_ZONE: bool = True  # True = clôture dès que la synergie passe en zone blanche
BREAK_EVEN_R: float = 1.0        # le trade passe à break-even une fois +1R atteint (0 = désactivé)
WARMUP_BARS: int = SUPER_TREND_PERIOD  # barres ignorées au début (chauffe du Stop BSI)

ALLOW_LONG: bool = True          # True = prend les signaux d'achat (positions longues)
ALLOW_SHORT: bool = False         # True = prend les signaux de vente à découvert (positions short)
 
# ------------------------------------------------------------
# 1. CLOUD BSI  (nuage de tendance et de volatilité : EMA ± ATR)
# ------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def cloud_bsi( close: pd.Series,
               fast: int,
               slow: int) -> pd.DataFrame:
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

def stop_bsi( high: pd.Series,
              low: pd.Series,
              close: pd.Series,
              atr_period: int,
              multiplier: float ) -> pd.DataFrame:
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

def histo_bsi( close: pd.Series,
               fast: int,
               slow: int,
               signal: int) -> pd.DataFrame:
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

# ----------------------------------------------------------
# 5. SIGNAUX D'ENTRÉE / SORTIE ET SPREAD
# ----------------------------------------------------------

@dataclass
class Trade:
    side:    int            # +1 = long, -1 = short
    i_entry: int            # position (barre) d'entrée
    entry:   float
    risk:    float          # distance entrée -> Stop BSI au moment de l'entrée
    i_exit:  int = -1
    exit:    float = np.nan
    is_open: bool = True    # True = trade encore ouvert, valorisé au dernier cours

    @property
    def pnl(self) -> float:
        return self.side * (self.exit - self.entry)

    @property
    def r(self) -> float:
        return self.pnl / self.risk if self.risk > 0 else np.nan

def bsi_trades(close: pd.Series,
               stop: pd.DataFrame,
               syn: pd.Series) -> list[Trade]:
    """
    Signaux BSI, évalués à la clôture de chaque barre.

    Entrée : retournement du Stop BSI (direction -1 -> 1 = long, 1 -> -1 = short).
      ZONE_RULES : pas d'achat en synergie rouge, pas de vente en synergie verte.

    Sortie, la première de ces conditions :
      - le Stop BSI se retourne contre la position
      - break-even : +BREAK_EVEN_R atteint, puis retour au prix d'entrée
      - EXIT_ON_WHITE_ZONE : la synergie passe en zone blanche
      ZONE_RULES : une sortie est reportée tant qu'elle serait une vente en synergie
      verte (position longue) ou un achat en synergie rouge (position short).
    """
    c         = close.values
    direction = stop["direction"].values
    stop_line = stop["stop"].values
    zone      = syn.values

    trades: list[Trade] = []
    trade: Trade | None = None
    be_armed = False

    for i in range( max(1, WARMUP_BARS), len(c) ):

        # --- position ouverte : faut-il sortir ? ---
        if trade is not None:
            side  = trade.side
            gain  = (c[i] - trade.entry) * side
            if BREAK_EVEN_R > 0 and gain >= BREAK_EVEN_R * trade.risk:
                be_armed = True

            want_exit = (
                direction[i] == -side
                or (be_armed and gain <= 0)
                or (EXIT_ON_WHITE_ZONE and zone[i] == 0 and zone[i - 1] != 0)
            )
            blocked = ZONE_RULES and zone[i] == side   # long en zone verte / short en zone rouge

            if want_exit and not blocked:
                trade.i_exit, trade.exit, trade.is_open = i, c[i], False
                trades.append(trade)
                trade, be_armed = None, False
            continue                                   # pas de nouvelle entrée sur cette barre

        # --- à plat : retournement du Stop BSI ? ---
        if direction[i] == direction[i - 1]:
            continue
        side = int(direction[i])
        if (side == 1 and not ALLOW_LONG) or (side == -1 and not ALLOW_SHORT):
            continue
        if ZONE_RULES and zone[i] == -side:            # achat en rouge / vente en vert interdits
            continue
        risk = abs(c[i] - stop_line[i])
        if not np.isfinite(risk) or risk == 0:
            continue
        trade = Trade(side=side, i_entry=i, entry=c[i], risk=risk)

    if trade is not None:                              # trade en cours : valorisé au dernier cours
        trade.i_exit, trade.exit = len(c) - 1, c[-1]
        trades.append(trade)
    return trades

def print_trades(ticker: str, trades: list[Trade], dates) -> float:
    """Affiche le détail des trades et retourne le Spread (somme des P&L)."""
    print(f"\n=== BSI {ticker} - LONG={ALLOW_LONG}  SHORT={ALLOW_SHORT}  ZONE_RULES={ZONE_RULES}  "
          f"EXIT_ON_WHITE_ZONE={EXIT_ON_WHITE_ZONE}  BREAK_EVEN_R={BREAK_EVEN_R} ===")
    print(f"{'#':>2}  {'Sens':<5} {'Entrée':<10} {'Prix':>9}  {'Sortie':<10} {'Prix':>9}  "
          f"{'Risque':>8} {'P&L':>9} {'R':>6}")
    for n, t in enumerate(trades, 1):
        print(f"{n:>2}  {'long' if t.side == 1 else 'short':<5} "
              f"{dates[t.i_entry]:%Y-%m-%d} {t.entry:>9.2f}  "
              f"{dates[t.i_exit]:%Y-%m-%d} {t.exit:>9.2f}  "
              f"{t.risk:>8.2f} {t.pnl:>+9.2f} {t.r:>+6.2f}"
              f"{'  (en cours)' if t.is_open else ''}")
    spread = sum(t.pnl for t in trades)
    if trades:
        gagnants = sum(t.pnl > 0 for t in trades)
        r_moyen  = np.nanmean([t.r for t in trades])
        print(f"Spread = {spread:+.2f}   trades = {len(trades)}   "
              f"gagnants = {gagnants}/{len(trades)}   R moyen = {r_moyen:+.2f}")
    else:
        print("Aucun trade.")
    return spread

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
    cloud   = cloud_bsi( close, fast=CLOUD_EMA_FAST, slow=CLOUD_EMA_SLOW )
    stop    = stop_bsi( high, low, close, atr_period=SUPER_TREND_PERIOD, multiplier=SUPER_TREND_MULTIPLIER )
    hist_ct = histo_bsi( close, fast=FAST_CT, slow=SLOW_CT, signal=9 ) # court terme
    hist_mt = histo_bsi( close, fast=FAST_MT, slow=SLOW_MT, signal=9 ) # moyen terme
    hist_lt = histo_bsi( close, fast=FAST_LT, slow=SLOW_LT, signal=9 ) # long terme


    # Synergie 3 horizons CT/MT/LT — une seule série close, périodes différentes
    syn = synergie( close,
                    fast_ct=FAST_CT, slow_ct=SLOW_CT,
                    fast_mt=FAST_MT, slow_mt=SLOW_MT,
                    fast_lt=FAST_LT, slow_lt=SLOW_LT )

    # Signaux d'entrée / sortie et Spread
    trades = bsi_trades( close, stop, syn )
    spread = print_trades( ticker, trades, df.index )

    # --- Axe X positionnel --------------------------------
    # idx : position entière (0, 1, 2...) utilisée pour tout le tracé —
    #       une barre tradée = une position, donc pas de trou le week-end.
    # dates : vraies dates conservées à part, uniquement pour l'affichage
    #         (ticks de l'axe + coordonnées de la toolbar).
    idx   = np.arange(len(df))
    dates = df.index

    # --- Layout --------------------------------
   
    # Checking witch matplotlib style is used
    #print( matplotlib.rcParams['axes.facecolor'], matplotlib.rcParams['axes.grid'] )
    
    fig = Figure(figsize=(12, 9))
    Watermark.apply( fig )
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

    # --- Signaux d'entrée / sortie ---------------------------------------
    labels = set()
    def _label(name):
        if name in labels:
            return None
        labels.add(name)
        return name

    for t in trades:
        x_in, x_out = idx[t.i_entry], idx[t.i_exit]
        col_trade = T["bull"] if t.pnl >= 0 else T["bear"]
        ax_price.plot([x_in, x_out], [t.entry, t.exit], ls="--", lw=0.9,
                      color=col_trade, zorder=6)
        ax_price.scatter(x_in, t.entry, marker="^" if t.side == 1 else "v", s=80,
                         color=T["bull"] if t.side == 1 else T["bear"],
                         edgecolors=T["text"], linewidths=0.6, zorder=8,
                         label=_label("Entrée long" if t.side == 1 else "Entrée short"))
        if not t.is_open:
            ax_price.scatter(x_out, t.exit, marker="o", s=45, color=col_trade,
                             edgecolors=T["text"], linewidths=0.6, zorder=8,
                             label=_label("Sortie"))

    spread = sum(t.pnl for t in trades)
    if trades:
        gagnants = sum(t.pnl > 0 for t in trades)
    _title = f"{ticker} ({interval}/{period}) · Bull/Bear Strength Index · Cloud {CLOUD_EMA_FAST}/{CLOUD_EMA_SLOW}"
    _title += f" · Super Trend {SUPER_TREND_PERIOD}/{SUPER_TREND_MULTIPLIER}"
    _title += f" · {gagnants}/{len(trades)} trades gagnants"
    _title += f" · Spread {spread:+.2f}"
    ax_price.set_title( _title , color=T["text"], fontsize=13, pad=8 )
    ax_price.legend(loc="upper left", fontsize=8,
                    facecolor=T["bg_legend"], labelcolor=T["text"])

    # -- Histogrammes BSI CT / MT / LT — force du mouvement ----------------
    # Chaque axe combine :
    #   • un fond de couleur pleine (vert/rouge) indiquant la direction de l'EMA
    #   • des barres MACD dont la hauteur traduit la force / l'accélération
    bull_ct_sig = (hist_ct["macd"] > 0).values
    bull_mt_sig = (hist_mt["macd"] > 0).values
    bull_lt_sig = (hist_lt["macd"] > 0).values

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

    draw_hist( ax_hist_ct, hist_ct, bull_ct_sig, f"BSI CT  {FAST_CT}/{SLOW_CT}" )
    draw_hist( ax_hist_mt, hist_mt, bull_mt_sig, f"BSI MT  {FAST_MT}/{SLOW_MT}" )
    draw_hist( ax_hist_lt, hist_lt, bull_lt_sig, f"BSI LT  {FAST_LT}/{SLOW_LT}" )

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

if run_standalone:
    matplotlib.use("TkAgg") # standalone : thread principal, affichage direct OK
    
    TICKER   = "NVDA" #  AI.PA BN.PA MSFT NVDA AIR.PA AAPL AMZN GOOGL TSLA META NFLX
    PERIOD   = "1y"
    INTERVAL = "1d"

    fig = plot_bsi( TICKER, period=PERIOD, interval=INTERVAL )
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