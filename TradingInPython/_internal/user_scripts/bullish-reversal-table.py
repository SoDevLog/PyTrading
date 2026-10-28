""" Filtre les actions en retournement haussier après une baisse.

Scénario recherché : une baisse marquée, un creux (survente), puis un
retournement confirmé par le volume, sans courir après un rebond déjà avancé.

6 critères, un point chacun :
  C1  Baisse préalable    : clôture < SMA50 et chute >= MIN_DROP (du plus haut au creux)   [obligatoire]
  C2  Survente            : RSI(14) passé sous RSI_OVERSOLD dans les RSI_WINDOW dernières séances
  C3  Momentum            : histogramme MACD en hausse depuis MACD_RISING séances
  C4  Retournement        : clôture > EMA10 et plus-bas plus haut (pas de nouveau creux
                            depuis HL_RECENT séances)                                      [obligatoire]
  C5  Volume              : volume récent >= VOLUME_MULTIPLIER x référence, et volume des
                            séances haussières > celui des séances baissières
  C6  Entrée pas tardive  : rebond depuis le creux <= MAX_REBOUND

Statut OK : score >= MIN_SCORE, avec C1 et C4 remplis.
"""
import yfinance
import numpy
import pandas as pd
from matplotlib.figure import Figure

run_standalone = (__name__ == "__main__")

if run_standalone:
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    sys.path.append( str(base_dir) )
    from user_scripts.api import UserScriptAPI

from user_scripts.api import api
from styles.watermark import Watermark
from digitsignalprocessing.indicators import rsi_welles_wilder, macd

# ------------------------------------------------------------------------------
# Paramètres
# ------------------------------------------------------------------------------

# Historique
HISTORY_PERIOD   = "6mo"
MIN_BARS         = 80      # nombre minimum de séances pour évaluer un ticker

# C1 : baisse préalable
LOOKBACK         = 40       # fenêtre de recherche du plus haut et du creux (séances)
MIN_DROP         = 0.15     # chute minimale du plus haut au creux (0.15 = -15%)
SMA_PERIOD       = 50

# C2 : survente
RSI_PERIOD       = 14
RSI_OVERSOLD     = 30
RSI_WINDOW       = 10       # le RSI doit être passé sous le seuil dans ces dernières séances

# C3 : momentum
MACD_RISING      = 3        # nombre de séances consécutives de hausse de l'histogramme MACD

# C4 : retournement
EMA_TREND        = 10
HL_RECENT        = 5        # le plus bas des N dernières séances doit rester au-dessus du creux précédent

# C5 : volume
VOL_RECENT       = 3        # séances récentes
VOL_REF          = 20       # séances de référence (précédant les séances récentes)
VOLUME_MULTIPLIER = 1.3
UPDOWN_WINDOW    = 5        # fenêtre de comparaison volume haussier / baissier

# C6 : entrée pas tardive
MAX_REBOUND      = 0.12     # rebond maximal depuis le creux (0.12 = +12%)

# Statut et rapport
MIN_SCORE        = 5        # score minimal (sur 6) pour le statut OK
REPORT_MIN_SCORE = 2        # score minimal pour apparaître dans le tableau du rapport

COLOR_HEADER   = "#14599e"  # bleu foncé
COLOR_POSITIVE = "#9ee9b1"  # vert clair  : statut OK
COLOR_NEAR     = "#f3e3a0"  # jaune clair : il manque un seul critère
COLOR_NEUTRAL  = "#f2f2f2"  # gris clair  : autres

CRITERIA = ["C1", "C2", "C3", "C4", "C5", "C6"]

# ------------------------------------------------------------------------------

def evaluate_reversal( df: pd.DataFrame ) -> dict | None:
    """
    Évalue les 6 critères de retournement haussier sur un historique OHLCV.
    Retourne None si l'historique est insuffisant.
    """
    df = df.dropna( subset=["Close", "High", "Low"] ).copy()
    df["Volume"] = df["Volume"].fillna( 0 )

    if len( df ) < MIN_BARS:
        print( f"len( df ): {len( df )}  < MIN_BARS: {MIN_BARS}")
        return None

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    last = close.iloc[-1]

    # --- Creux et plus haut précédant le creux (fenêtre LOOKBACK) --- #
    w_low      = low.iloc[ -LOOKBACK: ]
    w_high     = high.iloc[ -LOOKBACK: ]
    trough_pos = int( numpy.argmin( w_low.values ) )
    trough     = w_low.iloc[ trough_pos ]
    peak       = w_high.iloc[ :trough_pos + 1 ].max()
    drop       = (peak - trough) / peak if peak > 0 else 0.0

    # --- C1 : baisse préalable --- #
    sma = close.rolling( SMA_PERIOD ).mean().iloc[-1]
    c1  = bool( last < sma and drop >= MIN_DROP )

    # --- C2 : survente récente --- #
    rsi_min = rsi_welles_wilder( df, RSI_PERIOD ).iloc[ -RSI_WINDOW: ].min()
    c2 = bool( rsi_min < RSI_OVERSOLD )

    # --- C3 : histogramme MACD en hausse --- #
    #hist = compute_macd_hist( close )
    a, b, hist = macd( df, histo=True )
    
    c3 = bool( (hist.diff().iloc[ -MACD_RISING: ] > 0).all() )

    # --- C4 : retournement (EMA + plus-bas plus haut) --- #
    ema         = close.ewm( span=EMA_TREND, adjust=False ).mean().iloc[-1]
    prev_trough = low.iloc[ -(LOOKBACK + HL_RECENT) : -HL_RECENT ].min()
    recent_low  = low.iloc[ -HL_RECENT: ].min()
    c4 = bool( last > ema and recent_low > prev_trough )

    # --- C5 : volume --- #
    vol_recent = volume.iloc[ -VOL_RECENT: ].mean()
    vol_ref    = volume.iloc[ -(VOL_REF + VOL_RECENT) : -VOL_RECENT ].mean()
    vol_ratio  = vol_recent / vol_ref if vol_ref > 0 else 0.0

    chg      = close.diff().iloc[ -UPDOWN_WINDOW: ]
    vol_w    = volume.iloc[ -UPDOWN_WINDOW: ]
    up_vol   = vol_w[ chg > 0 ].sum()
    down_vol = vol_w[ chg < 0 ].sum()
    c5 = bool( vol_ratio >= VOLUME_MULTIPLIER and up_vol > down_vol )

    # --- C6 : rebond pas trop avancé --- #
    rebound = (last - trough) / trough if trough > 0 else 0.0
    c6 = bool( 0 <= rebound <= MAX_REBOUND )

    flags = [ c1, c2, c3, c4, c5, c6 ]
    score = sum( flags )
    ok    = score >= MIN_SCORE and c1 and c4

    return {
        "price":     last,
        "drop":      drop,
        "rsi_min":   rsi_min,
        "rebound":   rebound,
        "vol_ratio": vol_ratio,
        "flags":     flags,
        "score":     score,
        "ok":        ok,
    }

# ------------------------------------------------------------------------------

def filter_bullish_reversal( tickers: list[str] ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cherche les actions en retournement haussier après une baisse (voir l'en-tête du fichier).

    Returns:
        (df_qualified, df_all) : DataFrame des actions qualifiées, et DataFrame de
        l'ensemble des tickers analysés (triés par score puis ratio de volume décroissants).
    """
    tickers = list( dict.fromkeys( tickers ) )  # supprime les doublons en gardant l'ordre
    rows = []

    for ticker in tickers:
        try:
            _ticker = yfinance.Ticker( ticker )
            df = _ticker.history( period=HISTORY_PERIOD )

            res = evaluate_reversal( df ) if not df.empty else None
            if res is None:
                print(f"[{ticker}] Données insuffisantes, ignoré.")
                continue

            try:
                _short_name = _ticker.get_info().get( 'shortName' ) or ticker
            except Exception:
                _short_name = ticker

            row = {
                "Ticker":    ticker,
                "Nom":       str( _short_name )[:22],
                "Score":     res["score"],
                "Chute":     f"-{res['drop']:.0%}",
                "RSI min":   round( res["rsi_min"], 1 ) if pd.notna( res["rsi_min"] ) else None,
                "Rebond":    f"{res['rebound']:+.1%}",
                "Vol ratio": round( res["vol_ratio"], 2 ),
            }
            for name, flag in zip( CRITERIA, res["flags"] ):
                row[name] = "+" if flag else "-"
            row["Statut"] = "OK" if res["ok"] else "-"
            rows.append( row )

            print( f"{_short_name} ({ticker}) : score {res['score']}/6 | chute -{res['drop']:.0%} | "
                   f"rebond {res['rebound']:+.1%} | vol x{res['vol_ratio']:.2f}"
                   + (" -> OK" if res["ok"] else "") )
        except Exception as e:
            print(f"[{ticker}] Erreur : {e}")

    if not rows:
        print("Aucune donnée exploitable.")
        return pd.DataFrame(), pd.DataFrame()

    df_all = ( pd.DataFrame( rows )
                 .sort_values( ["Score", "Vol ratio"], ascending=False )
                 .reset_index( drop=True ) )
    df_qualified = df_all[ df_all["Statut"] == "OK" ].reset_index( drop=True )

    if df_qualified.empty:
        print("Aucune action ne remplit les critères.")

    return df_qualified, df_all

# -----------------------------------------------------------------------------

def show_scan_report( df_all: pd.DataFrame, nb_tickers: int ) -> Figure | None:
    """
    Affiche le rapport du scan dans une figure Matplotlib : paramètres utilisés,
    détail des tickers dont le score atteint REPORT_MIN_SCORE, et résumé final.
    """
    if df_all.empty:
        print("Aucune société qualifiée !")
        return None

    df_view      = df_all[ df_all["Score"] >= REPORT_MIN_SCORE ].reset_index( drop=True )
    nb_qualifies = ( df_all["Statut"] == "OK" ).sum()
    nb_rows      = len( df_view ) + 1  # +1 pour la ligne d'en-tête

    fig = Figure( figsize=(12, 0.35 * nb_rows + 2.6) )
    Watermark.apply( fig )
    ax = fig.add_subplot( 111 )
    ax.axis( "off" )

    header = (
        f"Filtre des actions en retournement haussier - {nb_tickers} tickers analysés\n"
        f"Chute ≥ {MIN_DROP*100:.0f}%  |  RSI < {RSI_OVERSOLD} ({RSI_WINDOW} séances)  |  "
        f"MACD histo en hausse ({MACD_RISING})  |  Volume ≥ {VOLUME_MULTIPLIER} x moy. {VOL_REF}j  |  "
        f"Rebond ≤ {MAX_REBOUND*100:.0f}% / {LOOKBACK} |  OK score ≥ {MIN_SCORE}/6 Rapport:{REPORT_MIN_SCORE}"
    )
    ax.set_title( header, fontsize=10, fontweight="bold", pad=0, loc="left" )

    if df_view.empty:
        ax.text( 0.5, 0.5, f"Aucune action avec un score ≥ {REPORT_MIN_SCORE}/6.",
                 transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold" )
        return fig

    table = ax.table(
        cellText  = df_view.values,
        colLabels = df_view.columns,
        cellLoc   = "center",
        loc       = "upper center",
        bbox      = [0, 0.08, 1, 0.82],
    )
    table.auto_set_font_size( False )
    table.set_fontsize( 9 )
    table.auto_set_column_width( col=list( range( len( df_view.columns ) ) ) )

    # Style de l'en-tête du tableau
    for col in range( len( df_view.columns ) ):
        cell = table[0, col]
        cell.set_facecolor( COLOR_HEADER )
        cell.set_text_props( color="white", fontweight="bold" )

    # Lignes : vert = OK, jaune = il manque un point pour le statut OK, gris = autres
    for row in range( 1, nb_rows ):
        rec = df_view.iloc[ row - 1 ]
        if rec["Statut"] == "OK":
            bg_color = COLOR_POSITIVE
        elif rec["Score"] == MIN_SCORE - 1:
            bg_color = COLOR_NEAR
        else:
            bg_color = COLOR_NEUTRAL
        for col in range( len( df_view.columns ) ):
            table[row, col].set_facecolor( bg_color )

    footer = (
        f"{nb_qualifies} action(s) retenue(s)" if nb_qualifies
        else "Aucune action ne remplit les critères."
    )
    ax.text( 0.5, 0.0, footer, transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold" )

    return fig

# -----------------------------------------------------------------------------

def main():

    # Check parameters for the script
    print( "--- Check API's parameters ---" )
    if not api.check_parameters( ['tickers'] ):
        exit(1)

    tickers = api.tickers

    print( "--- Filtre des actions en retournement haussier ---" )
    print( f"Chute ≥ {MIN_DROP*100:.0f}% | RSI < {RSI_OVERSOLD} | Volume ≥ {VOLUME_MULTIPLIER} x moyenne {VOL_REF}j | "
           f"Rebond ≤ {MAX_REBOUND*100:.0f}% | OK si score ≥ {MIN_SCORE}/6\n" )

    df_qualified, df_all = filter_bullish_reversal( tickers )

    if not df_qualified.empty:
        print( "\n" + df_qualified.to_string( index=False ) )

    fig = show_scan_report( df_all, len( set( tickers ) ) )
    if fig is None:
        return
    fig.tight_layout()
    fig_title = f"Filtre en retournement haussier - {len(tickers)} tickers analysés"

    if run_standalone == False:
        api.show_figure( fig, 'table', fig_title )  # rendu thread-safe dans le thread Tkinter principal
    else:
        # Créer la fenêtre Tkinter, avec un ascenseur vertical si la table dépasse l'écran
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        root = tk.Tk()
        root.title( fig_title )

        # La fenêtre ne dépasse pas la hauteur de l'écran ; au-delà, le contenu devient scrollable
        fig_w_px = int( fig.get_figwidth() * fig.dpi )
        fig_h_px = int( fig.get_figheight() * fig.dpi )
        win_h = min( fig_h_px + 20, root.winfo_screenheight() - 100 )
        root.geometry( f"{fig_w_px}x{win_h}" )

        outer = tk.Frame( root )
        outer.pack( fill="both", expand=True )

        v_scroll = tk.Scrollbar( outer, orient="vertical" )
        v_scroll.pack( side="right", fill="y" )

        scroll_area = tk.Canvas( outer, yscrollcommand=v_scroll.set, highlightthickness=0 )
        scroll_area.pack( side="left", fill="both", expand=True )
        v_scroll.config( command=scroll_area.yview )

        # NB : toolbar/scrollbar doivent cibler root, pas inner, si réactivés (zone fixe hors défilement)
        inner = tk.Frame( scroll_area )
        scroll_area.create_window( (0, 0), window=inner, anchor="nw" )

        canvas = FigureCanvasTkAgg(fig, master=inner)
        # toolbar = NavigationToolbar2Tk(canvas, root)
        # toolbar.update()

        canvas.draw()
        canvas.get_tk_widget().pack()

        inner.bind( "<Configure>", lambda e: scroll_area.configure( scrollregion=scroll_area.bbox("all") ) )
        scroll_area.bind_all( "<MouseWheel>", lambda e: scroll_area.yview_scroll( int(-1 * (e.delta / 120)), "units" ) )

        root.mainloop()

if run_standalone:

    api_context = {
        'tickers': [ 'HO.PA', 'AM.PA', 'AIR.PA', 'GE', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'IT', 'BABA', 'AMD', 'BA', 'BIDU', 'BP', 'RACE', 'FTNT', 'F', 'GM', 'INTC', 'JD', 'BMW.DE', 'CROX' ],
    }

    api = UserScriptAPI()
    api.update( **api_context )

    main()
