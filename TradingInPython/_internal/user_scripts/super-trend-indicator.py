""" Indicateur SuperTrend base sur l'ATR (atr_rolling).
"""

import sys
import numpy
import pandas
import matplotlib.style
matplotlib.style.use("seaborn-v0_8-notebook")
import matplotlib.pyplot as plt

from pathlib import Path
base = Path(__file__).resolve().parent.parent
sys.path.append( str(base) )

from matplotlib.figure import Figure
import digitsignalprocessing.indicators as dsp

# -----------------------------------------------------------------------------

def plot_supertrend( ax, data, st_df, price_color='black', show_markers=True, dot_size=15 ):
    """
    Affiche le SuperTrend sur un axe matplotlib existant (ex: un axe
    embarque dans un canvas Tkinter via FigureCanvasTkAgg).

    ax     : matplotlib.axes.Axes deja cree
    data   : DataFrame OHLC d'origine (meme index que st_df)
    st_df  : DataFrame retourne par supertrend()
    """
    direction = st_df['Direction'].values
    st = st_df['SuperTrend'].values

    # Segmentation par tendance : les valeurs hors-tendance passent a NaN.
    # ax.scatter ignore nativement les points NaN (rien n'est dessine pour
    # ces indices), donc pas besoin de filtrer les tableaux au prealable.
    up_line = numpy.where(direction == 1, st, numpy.nan)
    down_line = numpy.where(direction == -1, st, numpy.nan)

    ax.plot( data.index, data['Close'], color=price_color, linewidth=1, label='Close')
    ax.scatter( data.index, up_line, color='tab:green', s=dot_size, label='SuperTrend (haussier)' )
    ax.scatter( data.index, down_line, color='tab:red', s=dot_size, label='SuperTrend (baissier)' )

    if show_markers:
        flips = numpy.where(numpy.diff(direction) != 0)[0] + 1
        up_labeled = False
        down_labeled = False
        for idx in flips:
            if direction[idx] == 1:
                ax.scatter(data.index[idx], st[idx], marker='^', color='tab:green', s=60, zorder=5, label='Retournement haussier' if not up_labeled else None )
                up_labeled = True
            else:
                ax.scatter(data.index[idx], st[idx], marker='v', color='tab:red', s=60, zorder=5, label='Retournement baissier' if not down_labeled else None )
                down_labeled = True

    return ax

# ---------------------------------------------
# POINTS D'ENTRÉE 
# ---------------------------------------------

# ---------------------------------------------
# main() appelé explicitement par script_runner
#
def main():
    matplotlib.use("Agg") # script_runner thread secondaire rendu différé via api.show_figure()
    from user_scripts.api import api

    if not api.check_parameters( ['ticker', 'period', 'interval'] ):
        exit(1)

    if api.df is None or api.df.empty:
        df = yfinance.download( api.ticker, period=api.period, interval=api.interval,
            auto_adjust=True, progress=False )
    else:
        df = api.df

    if df.empty:
        raise ValueError(f"Aucune donnée pour {api.ticker}")

    st_df = dsp.super_trend( df, period=10, multiplier=2.0 )
    
    fig = Figure( figsize=(12, 6) )
    axe = fig.add_subplot( 111 )
    plot_supertrend( axe, df, st_df, dot_size=10 )
    axe.set_title( f"{api.name} - SuperTrend" )
    axe.legend(loc='upper left')
    fig.tight_layout()
    
    api.show_figure( fig, title=f"{api.ticker} - SuperTrend" )

if __name__ == '__main__':
    # Exemple minimal
    import yfinance

    TICKER   = "NVDA"
    
    df = yfinance.download( TICKER, period='6mo', interval='1d' )
    if isinstance( df.columns, pandas.MultiIndex ):
        df.columns = df.columns.get_level_values(0)

    st_df = dsp.super_trend( df, period=10, multiplier=2.0 )

    # Create figure without 'plt'
    fig = Figure( figsize=(12, 6) )
    axe = fig.add_subplot( 111 )
    plot_supertrend( axe, df, st_df, dot_size=10 )
    axe.legend( loc='upper left' )
    axe.set_title( f"{TICKER} - SuperTrend" )
    fig.tight_layout()

    # Créer la fenêtre Tkinter    
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title(f"{TICKER} - SuperTrend")
    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    root.mainloop()    
