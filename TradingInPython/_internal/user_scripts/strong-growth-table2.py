""" Filtre les actions en forte croissance sur n jours.
"""
import yfinance
import pandas as pd
from datetime import datetime, timedelta
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

# ------------------------------------------------------------------------------

NB_DAYS = 5
PRICE_THRESHOLD = 0.05 # +5%
VOLUME_MULTIPLIER = 1.2

_THEME = {
    True: { # --- DARK --- #
        "BG":        "#0F1117",
        "BG_LEGEND" :"#195BBE",
        "TXT_LEGEND":"#8DABDF",
        "ROW":       "#16191F",
        "ROW_ALT":   "#1F232B",
        "BORDER":    "#252B35",
        "TEXT":      "#F4F7FB",
        "MUTED":     "#858E9E",
        "GRID":      "#303744",
        "BAR_OFF":   "#2A303A",
        "GREEN":     "#00E676",
        "RED":       "#FF4D67",
        "ORANGE":    "#FFB020",
        "BLUE":      "#4DA3FF",
    },
    False: { # --- LIGHT --- #
        "BG":        "#FFFFFF",
        "BG_LEGEND": "#AEC8FF",
        "TXT_LEGEND":"#414D66",
        "ROW":       "#FAFAFA",
        "ROW_ALT":   "#F3F3F3",
        "BORDER":    "#173363",
        "TEXT":      "#6b7280",
        "MUTED":     "#858E9E",
        "GRID":      "#C8CDD5",
        "BAR_OFF":   "#BFC6D2",
        "GREEN":     "#26B6A3",
        "RED":       "#D83C49",
        "ORANGE":    "#FFB020",
        "BLUE":      "#4DA3FF",
    },
}

# Colonnes du rapport : (clé, titre, largeur en pixels, alignement)
COLUMNS = [
    ("Ticker",      "TICKER",              82, "left"),
    ("Nom",         "SOCIÉTÉ",            200, "left"),
    ("Prix",        "PRIX DÉBUT → ACTUEL",160, "center"),
    ("Performance", "PERF.",               95, "right"),
    ("Vol ratio",   "VOLUME ×",           215, "left"),
    ("Statut",      "STATUT",             123, "center"),
]
VOL_BAR_W = 115  # largeur de la barre de volume, en pixels

# ------------------------------------------------------------------------------

def filter_strong_growth(
        tickers: list[str], 
        nb_days: int = 20,
        price_threshold: float = 0.10,
        volume_multiplier: float = 1.5
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtre les actions en forte croissance sur n jours.

    Critères combinés :
      - Performance prix >= price_threshold (défaut : +10%)
      - Volume moyen sur n jours >= volume_multiplier x volume moyen long terme (défaut : 1.5x)

    Args:
        tickers          : liste de symboles boursiers
        nb_days          : fenêtre d'analyse en séances de bourse
        price_threshold  : seuil de hausse prix (0.10 = +10%)
        volume_multiplier: ratio volume récent / volume long terme

    Returns:
        (df_qualified, df_all) : DataFrame des actions qualifiées (triées par performance
        décroissante), et DataFrame de l'ensemble des tickers analysés avec leur statut.
    """
    end_date = datetime.today()
    # Historique long terme pour calculer le volume de référence (1 an)
    lt_start = end_date - timedelta(days=365)

    rows = []

    for ticker in tickers:
        try:
            _ticker = yfinance.Ticker( ticker )
            _stock_info = _ticker.get_info()
            _short_name = _stock_info.get( 'shortName' ) or ticker

            # Données long terme (volume de référence)
            df_lt = _ticker.history( start=lt_start, end=end_date )
            if df_lt.empty or len( df_lt ) < nb_days:
                print(f"[{ticker}] Données insuffisantes, ignoré.")
                continue

            # Fenêtre récente
            df_recent = df_lt.iloc[ -nb_days: ]  # dernières n sessions ouvrées

            # --- Critère 1 : performance prix --- #
            # dropna() écarte notamment la dernière valeur si la période n'est pas clôturée
            closes      = df_recent["Close"].dropna()
            price_start = closes.iloc[0]
            price_end   = closes.iloc[-1]
            perf        = (price_end - price_start) / price_start

            # --- Critère 2 : volume anormal --- #
            avg_vol_recent = df_recent["Volume"].mean()
            avg_vol_lt     = df_lt["Volume"].mean()  # référence long terme
            vol_ratio      = avg_vol_recent / avg_vol_lt if avg_vol_lt > 0 else 0

            # --- Filtre combiné --- #
            qualifie = perf >= price_threshold and vol_ratio >= volume_multiplier

            rows.append({
                "Ticker":        ticker,
                "Nom":           _short_name,
                "Prix départ":   round(price_start, 2),
                "Prix actuel":   round(price_end, 2),
                "Performance":   f"{perf:+.2%}",
                "Perf (float)":  perf,
                "Vol ratio":     round(vol_ratio, 2),
                "Vol moyen (n)": int(avg_vol_recent),
                "Vol moyen LT":  int(avg_vol_lt),
                "Statut":        "OK" if qualifie else "-",
            })

            _result = f"{_short_name} ({ticker}) : {perf:+.2%} | Vol: {vol_ratio:.2f}"
            print( _result + (" -> OK" if qualifie else "") )
        except Exception as e:
            print(f"[{ticker}] Erreur : {e}")

    if not rows:
        print("Aucune action ne remplit les critères.")
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.DataFrame(rows).sort_values( "Perf (float)", ascending=False ).reset_index(drop=True)
    df_qualified = df_all[ df_all["Statut"] == "OK" ].drop(columns=["Perf (float)"]).reset_index(drop=True)

    if df_qualified.empty:
        print("Aucune action ne remplit les critères.")

    return df_qualified, df_all

# -----------------------------------------------------------------------------

def show_scan_report(
        df_all: pd.DataFrame,
        tickers: list[str],
        nb_days: int,
        price_threshold: float,
        volume_multiplier: float
    ) -> Figure:
    """Rapport du scan Croissance Forte
    Colonnes définies en pixels, axe Y orienté vers le bas, carte arrondie,
    en-tête en pilule, séparateurs fins, barre de volume et badge de statut.
    Le thème reste piloté par settings.MODE.dark et les deux palettes _THEME.
    """
    T = _THEME[ settings.MODE.dark ]  # lire le mode courant à chaque rendu

    records   = df_all.to_dict( "records" )
    nb_ok     = sum( r["Statut"] == "OK" for r in records )
    nb_price  = sum( r["Perf (float)"] >= price_threshold for r in records )
    nb_volume = sum( r["Vol ratio"] >= volume_multiplier for r in records )
    vol_max   = max( [2 * volume_multiplier] + [r["Vol ratio"] for r in records] )  # échelle commune des barres

    # --- Géométrie (1 unité = 0.01 pouce) --- #
    OUT, IN, CELL_PAD = 20, 14, 8
    TITLE_H, HEADER_H, ROW_H, FOOTER_H = 110, 44, 42, 70
    W = 2 * (OUT + IN) + sum( column[2] for column in COLUMNS )
    card_h = HEADER_H + max( len( records ), 1 ) * ROW_H + 10
    H = TITLE_H + card_h + FOOTER_H

    fig = Figure( figsize=(W / 100, H / 100), dpi=110, facecolor=T["BG"] )
    Watermark.apply( fig )
    ax = fig.add_axes( [0, 0, 1, 1] )
    ax.set_xlim( 0, W )
    ax.set_ylim( H, 0 )  # axe Y vers le bas : mêmes coordonnées que le tableau de référence
    ax.axis( "off" )

    def pill( x: float, y: float, w: float, h: float, color: str,
              alpha: float = 1.0, zorder: float = 1 ):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                fc=color, ec="none", alpha=alpha, zorder=zorder,
            )
        )

    # Calcul des positions de chaque colonne.
    x_pos, x = {}, OUT + IN
    for key, title, width, align in COLUMNS:
        x_pos[key] = (x, width, align)
        x += width

    def anchor( key: str ):
        x0, width, align = x_pos[key]
        if align == "left":
            return x0 + CELL_PAD, "left"
        if align == "right":
            return x0 + width - CELL_PAD, "right"
        return x0 + width / 2, "center"

    # --- Titre --- #
    ax.text(
        OUT, 31, "Croissance forte - Strong Growth", fontsize=17, fontweight="bold",
        color=T["TEXT"], va="center",
    )

    # --- Sous-titre --- #
    ax.text(
        OUT, 61,
        f"{len( tickers )} tickers analysés   ·   {datetime.today():%d/%m/%Y}   ·   "
        f"fenêtre {nb_days} séances   ·   prix ≥ +{price_threshold:.0%}   ·   "
        f"volume ≥ ×{volume_multiplier:.1f} la moyenne long terme",
        fontsize=8.5, color=T["MUTED"], va="center",
    )

    # Badge de synthèse à droite
    badge_label = f"{nb_ok} action(s) retenue(s)"
    badge_w = max( 150, 16 + 7.0 * len( badge_label ) )
    badge_color = T["GREEN"] if nb_ok else T["GRID"]
    pill( W - OUT - badge_w, 17, badge_w, 30, badge_color, 0.16 )
    ax.text(
        W - OUT - badge_w / 2, 32, badge_label, fontsize=9,
        fontweight="bold", color=badge_color, ha="center", va="center",
    )

    # Résumé compact : combien de tickers passent chaque critère.
    kpis = [
        ("RETENUES",       nb_ok,                T["GREEN"]),
        ("PRIX ≥ SEUIL",   nb_price,             T["BLUE"]),
        ("VOLUME ≥ SEUIL", nb_volume,            T["ORANGE"]),
        ("ÉCARTÉES",       len( records ) - nb_ok, T["MUTED"]),
    ]
    kpi_x = OUT
    for label, value, color in kpis:
        ax.text( kpi_x, 88, label, fontsize=7, fontweight="bold", color=T["MUTED"], va="center" )
        # Décale le nombre selon la longueur du libellé pour éviter les collisions.
        value_x = kpi_x + 7.2 * len( label ) + 12
        ax.text( value_x, 88, str( value ), fontsize=9, fontweight="bold", color=color, va="center" )
        kpi_x = value_x + 70

    # --- Carte principale et en-tête arrondi --- #
    ax.add_patch(
        FancyBboxPatch(
            (OUT, TITLE_H), W - 2 * OUT, card_h,
            boxstyle="round,pad=0,rounding_size=14",
            fc=T["BG"], ec=T["GRID"], lw=1.0, zorder=0,
        )
    )

    # --- En-tête --- #
    pill( OUT + 8, TITLE_H + 8, W - 2 * OUT - 16, HEADER_H - 12, T["BG_LEGEND"], 1.0 )
    y_head = TITLE_H + HEADER_H / 2 + 2
    for key, title, width, align in COLUMNS:
        tx, ha = anchor( key )
        ax.text(
            tx, y_head, title, fontsize=7.2, fontweight="bold",
            color=T["TXT_LEGEND"], ha=ha, va="center",
        )

    # --- Aucune ligne --- #
    if not records:
        ax.text(
            W / 2, TITLE_H + HEADER_H + ROW_H / 2,
            "Aucun ticker n'a pu être analysé.", fontsize=11,
            color=T["MUTED"], ha="center", va="center",
        )

    # --- Lignes --- #
    for i, rec in enumerate( records ):
        ok = rec["Statut"] == "OK"
        yc = TITLE_H + HEADER_H + i * ROW_H + ROW_H / 2

        # Fond alterné ROW / ROW_ALT ; les actions retenues y ajoutent un accent vert.
        row_box = dict( xy=(OUT + 8, yc - ROW_H / 2 + 3), width=W - 2 * OUT - 16, height=ROW_H - 6 )
        ax.add_patch(
            FancyBboxPatch(
                **row_box, boxstyle="round,pad=0,rounding_size=8",
                fc=T["ROW"] if i % 2 == 0 else T["ROW_ALT"], ec="none",
            )
        )
        if ok:
            ax.add_patch(
                FancyBboxPatch(
                    **row_box, boxstyle="round,pad=0,rounding_size=8",
                    fc=T["GREEN"], ec="none", alpha=0.09,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    (OUT + 8, yc - 9), 3.5, 18,
                    boxstyle="round,pad=0,rounding_size=1.7",
                    fc=T["GREEN"], ec="none",
                )
            )

        # Ticker, nom, prix de début et de fin de fenêtre.
        tx, ha = anchor( "Ticker" )
        ax.text( tx + 2, yc, rec["Ticker"], fontsize=8.7, fontweight="bold",
                 color=T["TEXT"], ha=ha, va="center" )
        tx, ha = anchor( "Nom" )
        ax.text( tx, yc, str( rec["Nom"] )[:28], fontsize=7.6,
                 color=T["TEXT"], ha=ha, va="center" )
        tx, ha = anchor( "Prix" )
        ax.text( tx, yc, f"{rec['Prix départ']:.2f} → {rec['Prix actuel']:.2f}", fontsize=7.8,
                 color=T["MUTED"], ha=ha, va="center" )

        # Performance : vert si positive, rouge si négative.
        tx, ha = anchor( "Performance" )
        ax.text( tx, yc, rec["Performance"], fontsize=8.4, fontweight="bold",
                 color=T["GREEN"] if rec["Perf (float)"] >= 0 else T["RED"], ha=ha, va="center" )

        # Barre de volume + repère du seuil + ratio.
        ratio = rec["Vol ratio"]
        bx, _ = anchor( "Vol ratio" )
        fill  = max( VOL_BAR_W * min( ratio / vol_max, 1 ), 7 )
        pill( bx, yc - 3.5, VOL_BAR_W, 7, T["BAR_OFF"], 0.55 )
        pill( bx, yc - 3.5, fill, 7, T["GREEN"] if ratio >= volume_multiplier else T["MUTED"] )
        x_threshold = bx + VOL_BAR_W * volume_multiplier / vol_max
        ax.plot( [x_threshold] * 2, [yc - 8, yc + 8], color=T["TEXT"], lw=0.9, zorder=4 )
        ax.text( bx + VOL_BAR_W + 10, yc, f"×{ratio:.2f}", fontsize=7.8,
                 fontweight="bold" if ratio >= volume_multiplier else "normal",
                 color=T["GREEN"] if ratio >= volume_multiplier else T["TEXT"], va="center" )

        # Statut : badge pour les actions retenues.
        sx, _ = anchor( "Statut" )
        if ok:
            pill( sx - 40, yc - 11, 80, 22, T["GREEN"], 0.16 )
            ax.text( sx, yc, "RETENU", fontsize=7.6, fontweight="bold",
                     color=T["GREEN"], ha="center", va="center" )
        else:
            ax.text( sx, yc, "—", fontsize=8, color=T["MUTED"], ha="center", va="center" )

    # --- Pied de page / légende --- #
    y_foot = TITLE_H + card_h + 22
    ax.text(
        OUT, y_foot,
        "RETENU = prix ≥ seuil ET volume ≥ seuil   ·   barre : volume moyen de la fenêtre / volume moyen sur 1 an   ·   trait : seuil de volume",
        fontsize=8.2, color=T["MUTED"], va="center",
    )
    ax.text(
        OUT, y_foot + 19,
        "PERF. = clôture actuelle / clôture au début de la fenêtre   ·   la dernière bougie est écartée tant qu'elle n'est pas clôturée",
        fontsize=7.7, color=T["MUTED"], va="center",
    )

    return fig

# -----------------------------------------------------------------------------

def show_in_window( fig: Figure, title: str ) -> None:
    """
    Mode standalone : fenêtre Tkinter, avec un ascenseur vertical si la table dépasse l'écran.
    """
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    bg = _THEME[ settings.MODE.dark ]["BG"]

    root = tk.Tk()
    root.title( title )
    root.configure( bg=bg )

    # La fenêtre ne dépasse pas la hauteur de l'écran ; au-delà, le contenu devient scrollable
    fig_w_px = int( fig.get_figwidth() * fig.dpi )
    fig_h_px = int( fig.get_figheight() * fig.dpi )
    win_h = min( fig_h_px + 20, root.winfo_screenheight() - 100 )
    root.geometry( f"{fig_w_px + 20}x{win_h}" )  # +20 : largeur de l'ascenseur

    outer = tk.Frame( root, bg=bg )
    outer.pack( fill="both", expand=True )

    v_scroll = tk.Scrollbar( outer, orient="vertical" )
    v_scroll.pack( side="right", fill="y" )

    scroll_area = tk.Canvas( outer, yscrollcommand=v_scroll.set, highlightthickness=0, bg=bg )
    scroll_area.pack( side="left", fill="both", expand=True )
    v_scroll.config( command=scroll_area.yview )

    # NB : si une toolbar est réactivée, elle doit cibler root, pas inner (zone fixe hors défilement)
    inner = tk.Frame( scroll_area, bg=bg )
    scroll_area.create_window( (0, 0), window=inner, anchor="nw" )

    canvas = FigureCanvasTkAgg( fig, master=inner )
    canvas.draw()
    canvas.get_tk_widget().pack()

    inner.bind( "<Configure>", lambda e: scroll_area.configure( scrollregion=scroll_area.bbox("all") ) )
    scroll_area.bind_all( "<MouseWheel>", lambda e: scroll_area.yview_scroll( int(-1 * (e.delta / 120)), "units" ) )

    root.mainloop()

# -----------------------------------------------------------------------------

def main():

    # Check parameters for the script
    print( "--- Check API's parameters ---" )
    if not api.check_parameters( ['tickers'] ):
        exit(1)

    tickers = api.tickers

    print( "--- Filtre des actions à Croissance Forte ---" )
    print( f"Fenêtre : {NB_DAYS} jours | Prix ≥ +{PRICE_THRESHOLD*100:.0f}% | Volume ≥ {VOLUME_MULTIPLIER} x la moyenne long terme\n" )
        
    df_qualified, df_all = filter_strong_growth(
        tickers,
        nb_days=NB_DAYS,
        price_threshold=PRICE_THRESHOLD,
        volume_multiplier=VOLUME_MULTIPLIER
    )

    if not df_qualified.empty:
        print( df_qualified.to_string(index=False) )

    fig = show_scan_report( df_all, tickers, NB_DAYS, PRICE_THRESHOLD, VOLUME_MULTIPLIER )

    if run_standalone:
        show_in_window( fig, f"Filtre à Croissance Forte - {len(tickers)} tickers analysés" )
    else:
        api.show_figure( fig, type='table', title="Strong Growth Filter" )  # rendu thread-safe dans le thread Tkinter principal

if run_standalone:

    api_context = {
        'tickers': [ 'HO.PA', 'AM.PA', 'AIR.PA', 'GE', 'AAPL', 'MSFT', 'GOOGL',
            # 'AMZN', 'NVDA', 'IT', 'BABA', 'AMD', 'BA', 'BIDU', 'BP', 'RACE', 
            # 'FTNT', 'F', 'GM', 'INTC', 'JD', 'BMW', 'CROX' 
            ],
    }
        
    api = UserScriptAPI()
    api.update( **api_context )
    
    main()
