"""
    - Génération du scénario
    - Exécution de la Tkinter App
    
    Build:
    - pyinstaller smc.spec --clean
"""
import pandas
import yfinance
from datetime import datetime

from smc_generateur_scenario import generate_sample_data, generate_smc_scenario
from smc_ui import SMC_Tkinter_UI

# -----------------------------------------------------------------------------

stock_name = None

def generate_data_yfinance():
    global stock_name
    
    #symbol = 'AI.PA' # AIR LIQUIDE
    #symbol = 'STLAP.PA' # STLAP
    #symbol = 'AAPL' # APLE
    #symbol = 'AM.PA' # DASSAULT AVIATION
    #symbol = 'SAF.PA' # SAFRAN
    #symbol = 'WMT' # WALMART
    symbol = 'HO.PA' # THALES

    ticker = yfinance.Ticker( symbol )

    date_start = '2025-01-01'     # Date de début
    date_end = datetime.now() # '2026-01-02'       # Date de fin
    interval_fetch = '1d'         # Intervalle de temps (1d, 1wk, 1mo, etc.)
    data = ticker.history(
        start=date_start, 
        end=date_end, 
        interval=interval_fetch, 
        prepost=True,
        auto_adjust=False
    )
    
    # interval_fecth = '15m' # '1m' '2m' '5m' '15m' '30m' '60m' '90m' '1h' '1d' '5d' '1wk' '1mo' '3mo'
    # period_select = '1mo' # '1d' '5d' '1mo' '3mo' '6mo' '1y' '2y' '5y' '10y' 'ytd' 'max'
    # data = ticker.history( 
    #     period=period_select, 
    #     interval=interval_fecth, 
    #     prepost=False, 
    #     auto_adjust=False
    # )

    stock_info = ticker.get_info()
    stock_name = stock_info.get( 'shortName' )
    print( f"Stock name: {stock_name}" )

    # Create a DataFrame
    #data['DateSaved'] = data.index
    #data = data.reset_index() # make date as a 'straight' column
    df = pandas.DataFrame( data ) # df.iloc[-1] last values
    print(f"len: {len( data )}")
    
    return df

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from smc_ui import SMC_Tkinter_UI
    df = generate_data_yfinance()
    # For using generated sample data in smc_generateur_scenario.py
    #df = generate_sample_data( seed=43 ) # for CHoCH detection
    #df, _ = generate_smc_scenario( lg_data=150, start_price=100.0, seed=42 )
    app = SMC_Tkinter_UI( df )
    app.run_smc() # create SMC_Engine apply parameters
    app.plot( stock_name ) # afficher un premier graphique
    app.run()
