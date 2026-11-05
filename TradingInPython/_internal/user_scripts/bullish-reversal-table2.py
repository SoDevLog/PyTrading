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
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

run_standalone = (__name__ == "__main__")

if run_standalone:
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    sys.path.append( str(base_dir) )
    from user_scripts.api import UserScriptAPI

import settings.default as settings
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

_THEME = {
    True: { # --- DARK --- #
        "bg":           "#0d1117",
        "bg_legend":    "#0D397A",
        "text":         "#C4BBBB",
        "grid":         "#333333",
        "bull":         "#4EFF2F",
        "bear":         "#FF3B62",
        "poc":          "#ffb300",
        "text_strong":  "#e6edf3",
        "text_muted":   "#929db1",
    },
    False: { # --- LIGHT --- #
        "bg":           "#ffffff",
        "bg_legend":    "#f5f5f5",
        "text":         "#111111",
        "grid":         "#A5A5A5",
        "bull":         "#26B6A3",
        "bear":         "#D83C49",
        "poc":          "#ffb129",
        "text_strong":  "#000000",
        "text_muted":   "#6b7280",
    },
}

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
    Rapport du scan sous forme de carte moderne (dessinée en Matplotlib, sans ax.table) :
    score en mini-barres, critères en pastilles, badge de statut, valeurs colorées.
    Thème clair/sombre selon settings.MODE.dark. Seuls les tickers dont le score
    atteint REPORT_MIN_SCORE sont affichés.
    """
    if df_all.empty:
        print("Aucune société qualifiée !")
        return None

    T = _THEME[ settings.MODE.dark ]   # lu à l'appel, pas à l'import (sinon le thème reste figé)

    df_view = df_all[ df_all["Score"] >= REPORT_MIN_SCORE ].reset_index( drop=True )
    nb_qualifies = int( ( df_all["Statut"] == "OK" ).sum() )
    n = len( df_view )

    # --- Colonnes : (clé, titre, largeur, alignement) --- #
    columns = [
        ("Ticker",    "TICKER",     90, "left"),
        ("Nom",       "SOCIÉTÉ",   190, "left"),
        ("Score",     "SCORE",     150, "left"),
        ("Chute",     "CHUTE",      80, "right"),
        ("RSI min",   "RSI MIN",    80, "right"),
        ("Rebond",    "REBOND",     90, "right"),
        ("Vol ratio", "VOL. RATIO", 90, "right"),
        ("C1",        "BAISSE*",    66, "center"),
        ("C2",        "SURVENTE",   66, "center"),
        ("C3",        "MACD",       66, "center"),
        ("C4",        "RETOUR*",    66, "center"),
        ("C5",        "VOLUME",     66, "center"),
        ("C6",        "ENTRÉE",     66, "center"),
        ("Statut",    "STATUT",    100, "center"),
    ]

    # --- Géométrie (1 unité = 0.01 pouce) --- #
    OUT, IN, CELL_PAD = 20, 16, 10
    TITLE_H, HEADER_H, ROW_H, FOOTER_H = 92, 46, 44, 104
    W = 2 * (OUT + IN) + sum( c[2] for c in columns )
    card_h = HEADER_H + max( n, 1 ) * ROW_H + 10
    H = TITLE_H + card_h + FOOTER_H

    fig = Figure( figsize=(W / 100, H / 100), facecolor=T["bg"] )
    Watermark.apply( fig )
    ax = fig.add_axes( [0, 0, 1, 1] )
    ax.set_xlim( 0, W )
    ax.set_ylim( H, 0 )   # axe Y vers le bas : plus simple pour dessiner un tableau
    ax.axis( "off" )

    def pill( x: float, y: float, w: float, h: float, color: str, alpha: float, zorder: float = 1 ):
        ax.add_patch( FancyBboxPatch( (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                      fc=color, ec=color, alpha=alpha, zorder=zorder ) )

    # --- Titre --- #
    ax.text( OUT, 34, "Retournement haussier", fontsize=17, fontweight="bold",
             color=T["text_strong"], va="center" )
    
    #--- Sous-titre --- #
    ax.text( OUT, 64,
             f"{nb_tickers} tickers analysés   ·   {datetime.today():%d/%m/%Y}   ·   "
             f"chute ≥ {MIN_DROP*100:.0f}% ({LOOKBACK} séances)   ·   RSI < {RSI_OVERSOLD} ({RSI_WINDOW} séances)   ·   "
             f"volume ≥ {VOLUME_MULTIPLIER}× moy. {VOL_REF}j   ·   rebond ≤ {MAX_REBOUND*100:.0f}%   ·   OK si score ≥ {MIN_SCORE}/6",
             fontsize=9, color=T["text_muted"], va="center" )

    # Badge de synthèse en haut à droite
    label = f"{nb_qualifies} action(s) retenue(s)" if nb_qualifies else "Aucune action retenue"
    bw = 12 + 7.2 * len( label )
    pill( W - OUT - bw, 20, bw, 30, T["bull"] if nb_qualifies else T["grid"], 0.18 )
    ax.text( W - OUT - bw / 2, 35, label, fontsize=9.5, fontweight="bold",
             color=T["bull"] if nb_qualifies else T["text"], ha="center", va="center" )

    # --- Carte principale --- #
    ax.add_patch( FancyBboxPatch( (OUT, TITLE_H), W - 2 * OUT, card_h,
                                  boxstyle="round,pad=0,rounding_size=14",
                                  fc=T["bg"], ec=T["grid"], lw=1, zorder=0 ) )

    # Position X de chaque colonne
    x_pos, x = {}, OUT + IN
    for key, title, width, align in columns:
        x_pos[key] = ( x, width, align )
        x += width

    def anchor( key: str ):
        x0, width, align = x_pos[ key ]
        if align == "left":
            return x0 + CELL_PAD, "left"
        if align == "right":
            return x0 + width - CELL_PAD, "right"
        return x0 + width / 2, "center"

    # --- En-tête --- #
    pill( OUT + 8, TITLE_H + 8, W - 2 * OUT - 16, HEADER_H - 12, T["bg_legend"], 1.0 )
    y_head = TITLE_H + HEADER_H / 2 + 2
    for key, title, width, align in columns:
        tx, ha = anchor( key )
        ax.text( tx, y_head, title, fontsize=8, fontweight="bold", color=T["text_muted"], ha=ha, va="center" )

    # --- Aucune ligne à afficher --- #
    if n == 0:
        ax.text( W / 2, TITLE_H + HEADER_H + ROW_H / 2, f"Aucune action avec un score ≥ {REPORT_MIN_SCORE}/6.",
                 fontsize=11, color=T["text_muted"], ha="center", va="center" )

    # --- Lignes --- #
    for i in range( n ):
        rec     = df_view.iloc[ i ]
        yc      = TITLE_H + HEADER_H + i * ROW_H + ROW_H / 2
        is_ok   = rec["Statut"] == "OK"
        is_near = (not is_ok) and rec["Score"] == MIN_SCORE - 1
        accent  = T["bull"] if is_ok else ( T["poc"] if is_near else None )

        # Surlignage en vert léger si retenue, ambre léger s'il manque un point sinon simple séparateur
        if accent:
            ax.add_patch( FancyBboxPatch( (OUT + 8, yc - ROW_H / 2 + 3), W - 2 * OUT - 16, ROW_H - 6,
                                          boxstyle="round,pad=0,rounding_size=8",
                                          fc=accent, ec="none", alpha=0.10 ) )
            ax.add_patch( FancyBboxPatch( (OUT + 8, yc - 9), 3.5, 18,
                                          boxstyle="round,pad=0,rounding_size=1.7", fc=accent, ec="none" ) )
        elif i < n - 1:
            ax.plot( [OUT + IN, W - OUT - IN], [yc + ROW_H / 2] * 2, color=T["grid"], lw=0.8, alpha=0.5 )

        # Ticker / Nom
        tx, ha = anchor( "Ticker" )
        ax.text( tx + 6, yc, rec["Ticker"], fontsize=10, fontweight="bold", color=T["text_strong"], ha=ha, va="center" )
        tx, ha = anchor( "Nom" )
        ax.text( tx, yc, rec["Nom"], fontsize=9, color=T["text"], ha=ha, va="center" )

        # Score : mini-barres + "5/6"
        score     = int( rec["Score"] )
        bar_color = accent if accent else T["text_muted"]
        bx, _     = anchor( "Score" )
        for k in range( len( CRITERIA ) ):
            ax.add_patch( FancyBboxPatch( (bx + k * 19, yc - 4), 16, 8, boxstyle="round,pad=0,rounding_size=2.5",
                                          fc=bar_color if k < score else T["grid"],
                                          ec="none", alpha=1.0 if k < score else 0.45 ) )
        ax.text( bx + len( CRITERIA ) * 19 + 6, yc, f"{score}/{len( CRITERIA )}",
                 fontsize=9, fontweight="bold", color=T["text_strong"], ha="left", va="center" )

        # Chute / RSI min / Rebond / Volume
        tx, ha = anchor( "Chute" )
        ax.text( tx, yc, rec["Chute"], fontsize=9.5, fontweight="bold", color=T["bear"], ha=ha, va="center" )

        tx, ha = anchor( "RSI min" )
        if pd.isna( rec["RSI min"] ):
            ax.text( tx, yc, "–", fontsize=9.5, color=T["text_muted"], ha=ha, va="center" )
        else:
            oversold = rec["RSI min"] < RSI_OVERSOLD
            ax.text( tx, yc, f"{rec['RSI min']:.0f}", fontsize=9.5, fontweight="bold" if oversold else "normal",
                     color=T["poc"] if oversold else T["text"], ha=ha, va="center" )

        tx, ha = anchor( "Rebond" )
        ax.text( tx, yc, rec["Rebond"], fontsize=9.5, fontweight="bold", color=T["bull"], ha=ha, va="center" )

        tx, ha = anchor( "Vol ratio" )
        strong = rec["Vol ratio"] >= VOLUME_MULTIPLIER
        ax.text( tx, yc, f"×{rec['Vol ratio']:.2f}", fontsize=9.5, fontweight="bold" if strong else "normal",
                 color=T["bull"] if strong else T["text"], ha=ha, va="center" )

        # Critères : pastille pleine = rempli, anneau vide = non rempli
        for c in CRITERIA:
            cx, _ = anchor( c )
            if rec[ c ] == "+":
                ax.scatter( [cx], [yc], s=70, marker="o", c=T["bull"], edgecolors=T["bull"], linewidths=1, zorder=3 )
            else:
                ax.scatter( [cx], [yc], s=70, marker="o", facecolors="none", edgecolors=T["grid"], linewidths=1.2, zorder=3 )

        # Statut : badge
        sx, _ = anchor( "Statut" )
        if accent:
            text, pw = ( "OK", 62 ) if is_ok else ( "PRESQUE", 80 )
            pill( sx - pw / 2, yc - 11, pw, 22, accent, 0.20 )
            ax.text( sx, yc, text, fontsize=8.5, fontweight="bold",
                     color=T["bull"] if is_ok else T["text_strong"], ha="center", va="center" )
        else:
            ax.text( sx, yc, "–", fontsize=10, color=T["text_muted"], ha="center", va="center" )

    # --- Pied de page : légende --- #
    y_foot = TITLE_H + card_h + 28
    ax.text( OUT, y_foot,
             "* critère obligatoire   ·   pastille pleine : critère rempli   ·   PRESQUE : il manque un point pour le statut OK",
             fontsize=8.5, color=T["text_muted"], va="center" )
    ax.text( OUT, y_foot + 22,
             "Baisse : clôture < SMA50 et chute ≥ seuil   ·   Survente : RSI sous le seuil   ·   MACD : histogramme en hausse",
             fontsize=8, color=T["text_muted"], va="center" )
    ax.text( OUT, y_foot + 40,
             "Retour : > EMA10 et plus-bas plus haut   ·   Volume : récent > référence, séances haussières dominantes   ·   Entrée : rebond non tardif",
             fontsize=8, color=T["text_muted"], va="center" )

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
        api.show_figure( fig, 'table', fig_title )  # rendu thread-safe dans le thread Tkinter principal )  # rendu thread-safe dans le thread Tkinter principal
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
        'tickers': [ 
            'HO.PA', 'AM.PA', 'AIR.PA', 'GE', 'AAPL', 'MSFT', 'GOOGL', 'AMZN',
            'NVDA', 'IT', 'BABA', 'AMD', 'BA', 'BIDU', 'BP', 'RACE', 'FTNT',
            'F', 'GM', 'INTC', 'JD', 'BMW.DE', 'CROX' ],
    }

    api = UserScriptAPI()
    api.update( **api_context )

    main()
