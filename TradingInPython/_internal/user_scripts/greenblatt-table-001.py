""" Tool_Finance - Greenblatt Magic Formula

    La formule magique de Greenblatt est une stratégie d'investissement qui combine deux indicateurs financiers:        
        - Le rendement des bénéfices (Earnings Yield)
        - Le rendement du capital investi (Return on Capital)
    
    La formule magique de Greenblatt classe les actions en fonction de ces deux indicateurs, en attribuant un rang à chaque action pour chaque indicateur.
    Ensuite, les rangs sont additionnés pour obtenir un score global, et les actions avec les scores les plus bas sont considérées comme les meilleures opportunités d'investissement selon cette stratégie. 
        
"""
import pandas as pd
import yfinance
from matplotlib.figure import Figure

run_standalone = (__name__ == "__main__")

if run_standalone:
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    sys.path.append( str(base_dir) )

from user_scripts.api import api
from user_scripts.api import UserScriptAPI
from styles.watermark import Watermark

COLOR_HEADER   = "#2c3e50"
COLOR_POSITIVE = "#d4edda"
COLOR_NEGATIVE = "#f8d7da"

# Retrieve financial data from Yahoo Finance
def get_financial_data( ticker ):
    stock = yfinance.Ticker( ticker )
    try:
        stock_info = stock.info.copy()
        industry_key = stock_info.get( 'industryKey' )
        short_name = stock_info.get( 'shortName' )
        sector = stock_info.get( 'sector' )
        print( f'{short_name} Sector: {sector} Industry key: {industry_key}' )
            
        financials = stock.financials.copy()
        financials_transposed = financials.transpose()
        balance_sheet = stock.balance_sheet.copy()
        history_6mo = stock.history( period='6mo' ).copy()  # 6 derniers mois
        
        # Capitalisation boursière
        market_cap = stock_info['marketCap']  
        
        # EBIT (bénéfice avant impôts et intérêts)
        if 'EBIT' in financials.index:
            ebit = financials.loc['EBIT'].iloc[0]
        elif 'Net Income' in financials.index:
            ebit = financials.loc['Net Income'].iloc[0]  # Pour se secteur Bancaire Revenu net = EBIT
        else:
            ebit = None
        
        # Trésorerie différentes clés possibles
        cash_keys = ['Cash', 'Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']
        cash = next( (balance_sheet.loc[key].iloc[0] for key in cash_keys if key in balance_sheet.index), 0)

        # Dette différentes clés possibles
        debt_keys = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
        total_debt = next((balance_sheet.loc[key].iloc[0] for key in debt_keys if key in balance_sheet.index), 0)
        
        # Total des actifs et passifs
        total_assets = balance_sheet.loc['Total Assets'].iloc[0]  # Actifs totaux
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]  # Passifs

        # Calcul de la valeur d'entreprise (EV = Market Cap + Debt - Cash)
        enterprise_value = market_cap + total_debt - cash

        # Calcul des indicateurs
        earnings_yield = ebit / enterprise_value if enterprise_value > 0 else None  # EY = EBIT / EV
        return_on_capital = ebit / (total_assets - total_liabilities) if total_assets > total_liabilities else None  # ROC = EBIT / Capital Investi

        # Calcule du momentum basé sur le rendement des 6 derniers mois
        historical_data = history_6mo  
        momentum = ( historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[0] ) / historical_data["Close"].iloc[0]  # Rendement sur 6 mois

        # Croissance du chiffre d'affaires
        revenue_growth = None
        if 'Total Revenue' in financials_transposed.columns:
            revenue_current_year = financials.loc['Total Revenue'].iloc[0]  # Chiffre d'Affaires de l'année la plus récente
            revenue_previous_year = financials.loc['Total Revenue'].iloc[1]  # Chiffre d'Affaires de l'année précédente
            
            # Calcul de la croissance du chiffre d'affaires
            revenue_growth = (revenue_current_year - revenue_previous_year) / revenue_previous_year * 100

        return {
            'Short name': short_name,
            'Ticker': ticker,
            'Earnings Yield': earnings_yield, # Rendement des bénéfices
            'Return on Capital': return_on_capital, # ROC
            'Momentum (6 months)': momentum,
            'Revenue growth': revenue_growth
        }
        
    except Exception as e:
        print(f'Erreur avec le symbol {ticker}: {e}')
        return None


def compute_magic_formula( tickers ) -> pd.DataFrame:
    """ Calcule les indicateurs, les rangs et le score Magic Formula (trié du meilleur au moins bon). """
    data = [ get_financial_data( ticker ) for ticker in tickers ]
    data = [ d for d in data if d ]  # Supprimer les valeurs None

    df = pd.DataFrame( data )
    if df.empty:
        return df

    # Classement des actions selon EY et ROC
    df['EY Rank']  = df['Earnings Yield'].rank( ascending=False )
    df['ROC Rank'] = df['Return on Capital'].rank( ascending=False )

    # Score final = Somme des rangs (plus bas = mieux classé)
    df['Magic Formula Score'] = df['EY Rank'] + df['ROC Rank']
    return df.sort_values( by='Magic Formula Score' ).reset_index( drop=True )


def _fmt_pct( value, factor: float = 100 ) -> str:
    """ Formate un ratio en pourcentage ; 'N/A' si la valeur est absente. """
    return "N/A" if pd.isna( value ) else f"{value * factor:+.1f} %"


def _fmt_rank( value ) -> str:
    return "N/A" if pd.isna( value ) else f"{value:.0f}"


def show_magic_formula_report( df: pd.DataFrame, nb_tickers: int ) -> Figure:
    """
    Construit le rapport de la Formule Magique dans une figure Matplotlib :
    un tableau classé par score, une ligne par ticker analysé.
    """
    df_view = pd.DataFrame( {
        "Rang"        : range( 1, len( df ) + 1 ),
        "Ticker"      : df['Ticker'],
        "Nom"         : df['Short name'],
        "Earnings Yield" : df['Earnings Yield'].map( lambda v: _fmt_pct( v ) ),
        "ROC"         : df['Return on Capital'].map( lambda v: _fmt_pct( v ) ),
        "Momentum 6M" : df['Momentum (6 months)'].map( lambda v: _fmt_pct( v ) ),
        "Croissance CA" : df['Revenue growth'].map( lambda v: _fmt_pct( v, 1 ) ),
        "Rang EY"     : df['EY Rank'].map( _fmt_rank ),
        "Rang ROC"    : df['ROC Rank'].map( _fmt_rank ),
        "Score"       : df['Magic Formula Score'].map( _fmt_rank ),
    } )
    nb_rows = len( df_view ) + 1  # +1 pour la ligne d'en-tête

    fig = Figure( figsize=(12, 0.35 * nb_rows + 2.2) )
    Watermark.apply( fig )
    ax = fig.add_subplot( 111 )
    ax.axis( "off" )

    header = (
        f"Formule Magique de Greenblatt - {nb_tickers} tickers analysés\n"
        f"Score = Rang Earnings Yield + Rang ROC  |  Plus faible = mieux classé"
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
        cell.set_facecolor( COLOR_HEADER )
        cell.set_text_props( color="white", fontweight="bold" )

    # Lignes : vert clair si momentum positif, rouge clair si négatif (ou inconnu)
    for row in range( 1, nb_rows ):
        momentum = df.iloc[ row - 1 ]['Momentum (6 months)']
        bg_color = COLOR_POSITIVE if momentum >= 0 else COLOR_NEGATIVE
        for col in range( len( df_view.columns ) ):
            table[row, col].set_facecolor( bg_color )

    footer = f"{len( df )} action(s) classée(s)  |  Vert : momentum 6 mois positif"
    ax.text( 0.5, 0.0, footer, transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold" )

    return fig

# -----------------------------------------------------------------------------

def main():
    print( "--- Check API's parameters ---" )
    if not api.check_parameters( ['tickers'] ):
        exit(1)

    tickers = api.tickers
    df = compute_magic_formula( tickers )
    if df.empty:
        print( "Aucune donnée exploitable." )
        return

    fig = show_magic_formula_report( df, len( tickers ) )
    fig.tight_layout()
    
    if run_standalone == False:
        api.show_figure( fig, "Greenblatt Magic Formula" )
    else:
        # Créer la fenêtre Tkinter    
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        root = tk.Tk()
        root.title(f"Greenblatt magic formula")
        canvas = FigureCanvasTkAgg(fig, master=root)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        root.mainloop()    

if run_standalone:

    api_context = {
        'tickers': [ 'GOOG', 'AMZN', 'AAPL', 'IT', 'MSFT', 'NVDA' ],
    }
        
    api = UserScriptAPI()
    api.update( **api_context )
        
    main()
