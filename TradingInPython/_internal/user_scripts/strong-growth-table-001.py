""" Filtre les actions en forte croissance sur n jours.
"""
import yfinance
import numpy
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.figure import Figure

if __name__ == "__main__":
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    sys.path.append( str(base_dir) )
    from user_scripts.api import UserScriptAPI
    
from user_scripts.api import api
from styles.watermark import Watermark

# ------------------------------------------------------------------------------

NB_DAYS = 10
PRICE_THRESHOLD = 0.05 # +5%
VOLUME_MULTIPLIER = 1.2

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
        n_days           : fenêtre d'analyse en jours calendaires
        price_threshold  : seuil de hausse prix (0.10 = +10%)
        volume_multiplier: ratio volume récent / volume long terme

    Returns:
        (df_qualified, df_all) : DataFrame des actions qualifiées (triées par performance
        décroissante), et DataFrame de l'ensemble des tickers analysés avec leur statut.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=nb_days)
    # Historique long terme pour calculer le volume de référence (1 an)
    lt_start   = end_date - timedelta(days=365)

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
            price_start = df_recent["Close"].iloc[0]
            
            # Exclut la dernière valeur si elle correspond à une période non clôturée
            y = df_recent["Close"]
            if numpy.isnan( y[-1] ):
                y = y[:-1]

            price_end = y.iloc[-1]
            perf      = (price_end - price_start) / price_start

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
    ) -> None:
    """
    Affiche le rapport complet du scan dans une figure Matplotlib : paramètres utilisés,
    détail de chaque ticker analysé (qualifié ou non) et résumé final.
    """
    if df_all.empty:
        return

    cols = ["Ticker", "Nom", "Performance", "Vol ratio", "Statut"]
    df_view = df_all[ cols ]
    nb_qualifies = ( df_all["Statut"] == "OK" ).sum()
    nb_rows = len( df_view ) + 1  # +1 pour la ligne d'en-tête

    fig = Figure( figsize=(9, 0.35 * nb_rows + 2.2) )
    Watermark.apply( fig )
    ax  = fig.add_subplot( 111 )
    ax.axis( "off" )

    header = (
        f"Filtre des actions à Croissance Forte - {len(tickers)} tickers analysés\n"
        f"Fenêtre : {nb_days} jours  |  Prix ≥ +{price_threshold*100:.0f}%  |  "
        f"Volume ≥ {volume_multiplier} x la moyenne long terme"
    )
    ax.set_title( header, fontsize=11, fontweight="bold", pad=0, loc="left" )

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
        cell.set_facecolor( "#14599e" )
        cell.set_text_props( color="white", fontweight="bold" )

    # Lignes : vert clair si performance positive (croissant), rouge clair si négative (décroissant)
    for row in range( 1, nb_rows ):
        perf = df_all.iloc[ row - 1 ]["Perf (float)"]
        bg_color = "#ebfff0" if perf >= 0 else "#ffe9eb"
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
    fig.tight_layout()

    if __name__ != "__main__":
        api.show_figure( fig )  # rendu thread-safe dans le thread Tkinter principal
    else:
        # Créer la fenêtre Tkinter, avec un ascenseur vertical si la table dépasse l'écran
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        root = tk.Tk()
        root.title(f"Filtre à Croissance Forte - {len(tickers)} tickers analysés")

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

if __name__ == "__main__":

    api_context = {
        'tickers': [ 'HO.PA', 'AM.PA', 'AIR.PA', 'GE', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'IT', 'BABA', 'AMD', 'BA', 'BIDU', 'BP', 'RACE', 'FTNT', 'F', 'GE', 'GM', 'INTC', 'JD', 'BMW', 'CROX' ],
    }
        
    api = UserScriptAPI()
    api.update( **api_context )
    
    main()