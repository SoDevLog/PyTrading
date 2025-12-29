"""
    - Génération du scénario
    - Exécution de la Tkinter App
    
    Build:
    - pyinstaller smc.spec --clean
"""
import pandas
import yfinance
from smc_generateur_scenario import generate_sample_data, generate_smc_scenario
from smc_ui import SMC_Tkinter_UI

# -----------------------------------------------------------------------------

stock_name = None

def generate_data_yfinance():
    global stock_name
    
    symbol = 'AI.PA' # AIR LIQUIDE
    #symbol = 'AAPL' # APLE
    #symbol = 'AM.PA' # DASSAULT AVIATION
                
    date_start = '2023-12-24'     # Date de début
    date_end = '2025-12-24'       # Date de fin
    interval_fetch = '1d'         # Intervalle de temps (1d, 1wk, 1mo, etc.)

    ticker = yfinance.Ticker( symbol )
    data = ticker.history( 
        start=date_start, 
        end=date_end, 
        interval=interval_fetch, 
        prepost=True,
        auto_adjust=False
    )

    stock_info = ticker.get_info()
    stock_name = stock_info.get( 'shortName' )
    print( f"Stock name: {stock_name}" )

    # Create a DataFrame
    data['DateSaved'] = data.index
    data = data.reset_index() # make date as a 'straight' column
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
